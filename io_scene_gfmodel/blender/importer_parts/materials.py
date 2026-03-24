

from __future__ import annotations

import json
import os
import struct
from typing import Dict, List, Optional, Sequence, Tuple

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, StringProperty
from bpy_extras.io_utils import ImportHelper, axis_conversion
from mathutils import Matrix, Quaternion, Vector

from ...core.io import _load_any
from ...core.patch_plan import PatchPlan, steps_to_breadcrumb
from ...core.pica import (
    _bgra_to_rgba_floats,
    _decode_rgba_u32,
    _decode_texenv_update_buffer,
    _flip_bgra_y,
    _pica_decode_bitmap_to_bgra,
)
from ...core.types import (
    _GFMaterial,
    _GFModel,
    _GFMotion,
    _GFShader,
    _GFSubMesh,
    _GFTexture,
)
from ..anim import (
    _apply_uv_anim_enable,
    _apply_visibility_anim_enable,
    _compute_rest_world_mats,
    _euler_to_quat_xyz,
    _gf_runtime_cache_armature,
    _mot_eval,
    _transform_quat_basis,
)


def _make_image(tex: _GFTexture) -> bpy.types.Image:
    img = bpy.data.images.get(tex.name)
    if img is not None:
        return img
    img = bpy.data.images.new(tex.name, width=tex.width, height=tex.height, alpha=True)
    bgra = _pica_decode_bitmap_to_bgra(tex.raw, tex.width, tex.height, tex.fmt)
    bgra = _flip_bgra_y(bgra, tex.width, tex.height)
    img.pixels = _bgra_to_rgba_floats(bgra)
    try:
        if _tex_is_non_color(tex.name):
            img.colorspace_settings.name = 'Non-Color'
    except Exception:
        pass
    img.pack()
    return img


def _tex_is_non_color(name: str) -> bool:
    n = str(name or '').lower()
    return (
        n.endswith('nor.tga')
        or n.endswith('nor.png')
        or '_nor' in n
        or 'normal' in n
        or 'mask' in n
        or n.endswith('_n.tga')
        or n.endswith('_n.png')
        or n.endswith('_m.tga')
        or n.endswith('_m.png')
    )


def _map_wrap_mode_to_blender(wrap: int) -> str:


    w = int(wrap)
    if w == 2:
        return 'REPEAT'
    if w == 3:
        return 'MIRROR'
    if w == 1:
        return 'CLIP'
    return 'EXTEND'


def _is_linear_filter(mag: int, minf: int) -> bool:


    m = int(mag)
    mi = int(minf)
    return (m == 1) or (mi >= 3)




def _shader_key_variants(name: str) -> list[str]:
    raw = str(name or '').strip()
    if not raw:
        return []
    low = raw.lower()

    low2 = low.replace('\\', '/')
    base = low2.rsplit('/', 1)[-1]
    keys = {raw, low, base}
    if '.' in base:
        keys.add(base.rsplit('.', 1)[0])
    return [k for k in keys if k]


def _shader_lookup(shader_by_name: dict[str, object], *candidates: str) -> object | None:
    for cand in candidates:
        for k in _shader_key_variants(cand):
            sh = shader_by_name.get(k)
            if sh is not None:
                return sh
    return None

def _set_gfmodel_material_props(
    mat: bpy.types.Material,
    mat_def: _GFMaterial,
    shader_by_name: Dict[str, _GFShader],
) -> None:
    try:
        mat["gfmodel_face_culling"] = int(mat_def.face_culling or 0)
    except Exception:
        pass

    try:
        sh = _shader_lookup(shader_by_name, mat_def.frag_shader, mat_def.shader_name, mat_def.vtx_shader)
        mat["gfmodel_pica"] = json.dumps(
            {
                "material": {
                    "name": mat_def.name,
                    "shader_name": mat_def.shader_name,
                    "vtx_shader": mat_def.vtx_shader,
                    "frag_shader": mat_def.frag_shader,
                    "tex": {"0": mat_def.tex0, "1": mat_def.tex1, "2": mat_def.tex2},
                    "tex_units": [
                        {
                            "name": tu.name,
                            "unit_index": int(tu.unit_index),
                            "mapping_type": int(tu.mapping_type),
                            "scale": [float(tu.scale.x), float(tu.scale.y)],
                            "rotation": float(tu.rotation),
                            "translation": [float(tu.translation.x), float(tu.translation.y)],
                            "sampler_words": [int(w) for w in (tu.sampler_words or [])],
                        }
                        for tu in (mat_def.tex_units or [])
                    ],
                    "bump_texture": int(mat_def.bump_texture),
                    "const_assignments": [int(x) for x in (mat_def.const_assignments or [])],
                    "colors_rgba": [[int(c) for c in rgba] for rgba in (mat_def.colors_rgba or [])],
                    "alpha_test": {
                        "enabled": bool(mat_def.alpha_test_enabled),
                        "ref": float(mat_def.alpha_test_ref),
                        "func": int(mat_def.alpha_test_func),
                    },
                    "blend_func": mat_def.blend_func,
                    "blend_color_rgba": mat_def.blend_color_rgba,
                    "stencil_test": mat_def.stencil_test,
                    "stencil_op": mat_def.stencil_op,
                    "depth_test_enabled": mat_def.depth_test_enabled,
                    "depth_test_func": mat_def.depth_test_func,
                    "depth_write": mat_def.depth_write,
                    "color_write_mask": mat_def.color_write_mask,
                    "render_priority": int(mat_def.render_priority),
                    "render_layer": int(mat_def.render_layer),
                },
                "shader": {
                    "found": bool(sh is not None),
                    "name": (sh.name if sh else None),
                    "texenv_buffer_color": (int(sh.texenv_buffer_color) if (sh and sh.texenv_buffer_color is not None) else None),
                    "texenv_update_buffer": (int(sh.texenv_update_buffer) if (sh and sh.texenv_update_buffer is not None) else None),
                    "texenv_stages": [
                        {
                            "stage": int(s.stage),
                            "source": (int(s.source) if s.source is not None else None),
                            "operand": (int(s.operand) if s.operand is not None else None),
                            "combiner": (int(s.combiner) if s.combiner is not None else None),
                            "scale": (int(s.scale) if s.scale is not None else None),
                            "color": (int(s.color) if s.color is not None else None),
                        }
                        for s in (sh.texenv_stages if sh else [])
                    ],
                },
            },
            ensure_ascii=False,
        )
        if "gfmodel_pica_error" in mat:
            del mat["gfmodel_pica_error"]
    except Exception as e:
        mat["gfmodel_pica_error"] = f"{type(e).__name__}: {e}"


def _make_material(
    mat_def: _GFMaterial,
    textures: Dict[str, bpy.types.Image],
    shader_by_name: Dict[str, _GFShader],
) -> bpy.types.Material:
    mat = bpy.data.materials.get(mat_def.name)
    if mat is not None:
        _set_gfmodel_material_props(mat, mat_def, shader_by_name)
        return mat

    mat = bpy.data.materials.new(name=mat_def.name)
    mat.use_nodes = True
    nt = mat.node_tree
    if nt is None:
        _set_gfmodel_material_props(mat, mat_def, shader_by_name)
        return mat

    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (0, 0)
    out.location = (300, 0)
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    def make_tex_unit_nodes(tu, x: float, y: float):
        if not getattr(tu, 'name', None) or tu.name not in textures:
            return None

        texcoord = nt.nodes.get("GF_TEXCOORD")
        if texcoord is None:
            texcoord = nt.nodes.new("ShaderNodeTexCoord")
            texcoord.name = "GF_TEXCOORD"
            texcoord.label = "GF TexCoord"
            texcoord.location = (-1050, 0)

        mapping = nt.nodes.new("ShaderNodeMapping")
        mapping.location = (x - 350, y)
        mapping.label = f"GF UV Mapping (Unit {int(tu.unit_index)})"
        mapping.name = f"GF_MAPPING_{int(tu.unit_index)}"
        mapping.inputs["Scale"].default_value = (float(tu.scale.x), float(tu.scale.y), 1.0)
        mapping.inputs["Rotation"].default_value = (0.0, 0.0, float(tu.rotation))
        mapping.inputs["Location"].default_value = (float(tu.translation.x), float(tu.translation.y), 0.0)

        tex_node = nt.nodes.new("ShaderNodeTexImage")
        tex_node.image = textures[tu.name]
        tex_node.location = (x, y)

        wrap_u = int(tu.sampler_words[0]) if tu.sampler_words else 0
        wrap_v = int(tu.sampler_words[1]) if len(tu.sampler_words) > 1 else wrap_u
        mag = int(tu.sampler_words[2]) if len(tu.sampler_words) > 2 else 1
        minf = int(tu.sampler_words[3]) if len(tu.sampler_words) > 3 else 3

        tex_node.interpolation = 'Linear' if _is_linear_filter(mag, minf) else 'Closest'

        ext_u = _map_wrap_mode_to_blender(wrap_u)
        ext_v = _map_wrap_mode_to_blender(wrap_v)
        tex_node.extension = ext_u if ext_u == ext_v else ('MIRROR' if 'MIRROR' in (ext_u, ext_v) else 'REPEAT')

        nt.links.new(texcoord.outputs["UV"], mapping.inputs["Vector"])
        nt.links.new(mapping.outputs['Vector'], tex_node.inputs['Vector'])
        return tex_node

    tex_nodes: Dict[int, bpy.types.ShaderNodeTexImage] = {}
    for tu in mat_def.tex_units:
        if 0 <= tu.unit_index <= 2 and tu.name:
            tnode = make_tex_unit_nodes(tu, -300, -250 * int(tu.unit_index))
            if tnode is not None:
                tex_nodes[int(tu.unit_index)] = tnode


    sh = _shader_lookup(shader_by_name, mat_def.frag_shader, mat_def.shader_name, mat_def.vtx_shader)

    def const_rgba(stage) -> tuple[float, float, float, float]:
        if stage is None or stage.color is None:
            return (1.0, 1.0, 1.0, 1.0)
        return _decode_rgba_u32(int(stage.color))

    def node_rgb(rgb, x, y, label):
        n = nt.nodes.new('ShaderNodeRGB')
        n.location = (x, y)
        n.label = label
        n.outputs[0].default_value = (float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0)
        return n.outputs['Color']

    def node_val(v, x, y, label):
        n = nt.nodes.new('ShaderNodeValue')
        n.location = (x, y)
        n.label = label
        n.outputs[0].default_value = float(v)
        return n.outputs[0]

    if sh and getattr(sh, 'texenv_stages', None):
        prev_c = node_rgb((1.0, 1.0, 1.0), -850, 250, 'PrevInit')
        prev_a = node_val(1.0, -850, 220, 'PrevInitA')

        if sh.texenv_buffer_color is not None:
            bc = _decode_rgba_u32(int(sh.texenv_buffer_color))
            buf_c = node_rgb((bc[0], bc[1], bc[2]), -850, 160, 'BufInit')
            buf_a = node_val(bc[3], -850, 130, 'BufInitA')
        else:
            buf_c = node_rgb((0.0, 0.0, 0.0), -850, 160, 'BufInit')
            buf_a = node_val(0.0, -850, 130, 'BufInitA')

        def source_socket(src_id: int, stage, prev_c, prev_a, buf_c, buf_a):
            if src_id == 3 and 0 in tex_nodes:
                return tex_nodes[0].outputs['Color'], tex_nodes[0].outputs['Alpha']
            if src_id == 4 and 1 in tex_nodes:
                return tex_nodes[1].outputs['Color'], tex_nodes[1].outputs['Alpha']
            if src_id == 5 and 2 in tex_nodes:
                return tex_nodes[2].outputs['Color'], tex_nodes[2].outputs['Alpha']
            if src_id == 14:
                c = const_rgba(stage)
                return node_rgb((c[0], c[1], c[2]), -900, -950, 'Const'), node_val(c[3], -900, -980, 'ConstA')
            if src_id == 13:
                return buf_c, buf_a
            if src_id == 15:
                return prev_c, prev_a
            if src_id in (0, 1, 2):
                vc = nt.nodes.new('ShaderNodeVertexColor')
                vc.layer_name = 'Col'
                vc.location = (-900, -860)
                return vc.outputs['Color'], vc.outputs['Alpha']
            return node_rgb((1.0,1.0,1.0), -900, -820, 'White'), node_val(1.0, -900, -850, 'One')

        def op_color(op_id: int, c, a, x, y):
            if op_id == 0:
                return c
            if op_id == 1:
                one = node_rgb((1.0,1.0,1.0), x-200, y, 'One')
                sub = nt.nodes.new('ShaderNodeVectorMath'); sub.operation='SUBTRACT'; sub.location=(x,y)
                nt.links.new(one, sub.inputs[0]); nt.links.new(c, sub.inputs[1])
                return sub.outputs['Vector']
            if op_id == 2:
                comb = nt.nodes.new('ShaderNodeCombineXYZ'); comb.location=(x,y)
                nt.links.new(a, comb.inputs['X']); nt.links.new(a, comb.inputs['Y']); nt.links.new(a, comb.inputs['Z'])
                return comb.outputs['Vector']
            return c

        def op_alpha(op_id: int, c, a, x, y):
            if op_id == 0:
                return a
            if op_id == 1:
                one = node_val(1.0, x-200, y, 'OneA')
                sub = nt.nodes.new('ShaderNodeMath'); sub.operation='SUBTRACT'; sub.location=(x,y)
                nt.links.new(one, sub.inputs[0]); nt.links.new(a, sub.inputs[1])
                return sub.outputs[0]
            return a

        def combine_color(mode: int, a, b, c, x, y):
            if mode == 0:
                return a
            if mode == 1:
                mul = nt.nodes.new('ShaderNodeVectorMath'); mul.operation='MULTIPLY'; mul.location=(x,y)
                nt.links.new(a, mul.inputs[0]); nt.links.new(b, mul.inputs[1])
                return mul.outputs['Vector']
            if mode == 2:
                add = nt.nodes.new('ShaderNodeVectorMath'); add.operation='ADD'; add.location=(x,y)
                nt.links.new(a, add.inputs[0]); nt.links.new(b, add.inputs[1])
                return add.outputs['Vector']
            if mode == 4:

                inv = nt.nodes.new('ShaderNodeVectorMath'); inv.operation='SUBTRACT'; inv.location=(x-150,y-60)
                one = node_rgb((1.0,1.0,1.0), x-350, y-60, 'One')
                nt.links.new(one, inv.inputs[0]); nt.links.new(c, inv.inputs[1])
                mul1 = nt.nodes.new('ShaderNodeVectorMath'); mul1.operation='MULTIPLY'; mul1.location=(x,y)
                mul2 = nt.nodes.new('ShaderNodeVectorMath'); mul2.operation='MULTIPLY'; mul2.location=(x,y-60)
                nt.links.new(a, mul1.inputs[0]); nt.links.new(c, mul1.inputs[1])
                nt.links.new(b, mul2.inputs[0]); nt.links.new(inv.outputs['Vector'], mul2.inputs[1])
                add = nt.nodes.new('ShaderNodeVectorMath'); add.operation='ADD'; add.location=(x+150,y-30)
                nt.links.new(mul1.outputs['Vector'], add.inputs[0]); nt.links.new(mul2.outputs['Vector'], add.inputs[1])
                return add.outputs['Vector']
            return a

        def combine_alpha(mode: int, a, b, c, x, y):
            if mode == 0:
                return a
            if mode == 1:
                mul = nt.nodes.new('ShaderNodeMath'); mul.operation='MULTIPLY'; mul.location=(x,y)
                nt.links.new(a, mul.inputs[0]); nt.links.new(b, mul.inputs[1])
                return mul.outputs[0]
            if mode == 2:
                add = nt.nodes.new('ShaderNodeMath'); add.operation='ADD'; add.location=(x,y)
                nt.links.new(a, add.inputs[0]); nt.links.new(b, add.inputs[1])
                return add.outputs[0]
            if mode == 4:
                inv = nt.nodes.new('ShaderNodeMath'); inv.operation='SUBTRACT'; inv.location=(x-150,y)
                inv.inputs[0].default_value=1.0
                nt.links.new(c, inv.inputs[1])
                mul1 = nt.nodes.new('ShaderNodeMath'); mul1.operation='MULTIPLY'; mul1.location=(x,y)
                mul2 = nt.nodes.new('ShaderNodeMath'); mul2.operation='MULTIPLY'; mul2.location=(x,y-60)
                nt.links.new(a, mul1.inputs[0]); nt.links.new(c, mul1.inputs[1])
                nt.links.new(b, mul2.inputs[0]); nt.links.new(inv.outputs[0], mul2.inputs[1])
                add = nt.nodes.new('ShaderNodeMath'); add.operation='ADD'; add.location=(x+150,y-30)
                nt.links.new(mul1.outputs[0], add.inputs[0]); nt.links.new(mul2.outputs[0], add.inputs[1])
                return add.outputs[0]
            return a

        update_flags = _decode_texenv_update_buffer(int(sh.texenv_update_buffer or 0))
        y0 = 250
        for st in sh.texenv_stages:
            if st.source is None or st.operand is None or st.combiner is None or st.scale is None:
                continue
            src = int(st.source)
            op = int(st.operand)
            comb = int(st.combiner)
            sc = int(st.scale)

            col_mode = (comb >> 0) & 0xF
            alp_mode = (comb >> 16) & 0xF

            c0 = (src >> 0) & 0xF
            c1 = (src >> 4) & 0xF
            c2 = (src >> 8) & 0xF
            a0 = (src >> 16) & 0xF
            a1 = (src >> 20) & 0xF
            a2 = (src >> 24) & 0xF

            oc0 = (op >> 0) & 0xF
            oc1 = (op >> 4) & 0xF
            oc2 = (op >> 8) & 0xF
            oa0 = (op >> 12) & 0x7
            oa1 = (op >> 16) & 0x7
            oa2 = (op >> 20) & 0x7

            s0c, s0a = source_socket(c0, st, prev_c, prev_a, buf_c, buf_a)
            s1c, s1a = source_socket(c1, st, prev_c, prev_a, buf_c, buf_a)
            s2c, s2a = source_socket(c2, st, prev_c, prev_a, buf_c, buf_a)
            t0c = op_color(oc0, s0c, s0a, -400, y0)
            t1c = op_color(oc1, s1c, s1a, -400, y0 - 60)
            t2c = op_color(oc2, s2c, s2a, -400, y0 - 120)

            a0c, a0a = source_socket(a0, st, prev_c, prev_a, buf_c, buf_a)
            a1c, a1a = source_socket(a1, st, prev_c, prev_a, buf_c, buf_a)
            a2c, a2a = source_socket(a2, st, prev_c, prev_a, buf_c, buf_a)
            t0a = op_alpha(oa0, a0c, a0a, -400, y0 - 200)
            t1a = op_alpha(oa1, a1c, a1a, -400, y0 - 260)
            t2a = op_alpha(oa2, a2c, a2a, -400, y0 - 320)

            out_c = combine_color(col_mode, t0c, t1c, t2c, -60, y0 - 60)
            out_a = combine_alpha(alp_mode, t0a, t1a, t2a, -60, y0 - 260)


            col_scale = (sc >> 0) & 0x3
            alp_scale = (sc >> 16) & 0x3
            if col_scale in (1,2):
                fac = 2.0 if col_scale==1 else 4.0
                mul = nt.nodes.new('ShaderNodeVectorMath'); mul.operation='MULTIPLY'; mul.location=(120, y0-60)
                mul.inputs[1].default_value = (fac, fac, fac)
                nt.links.new(out_c, mul.inputs[0])
                out_c = mul.outputs['Vector']
            if alp_scale in (1,2):
                fac = 2.0 if alp_scale==1 else 4.0
                mul = nt.nodes.new('ShaderNodeMath'); mul.operation='MULTIPLY'; mul.location=(120, y0-260)
                mul.inputs[1].default_value = fac
                nt.links.new(out_a, mul.inputs[0])
                out_a = mul.outputs[0]

            prev_c = out_c
            prev_a = out_a

            uf = update_flags.get(int(st.stage), {"update_color_buffer": False, "update_alpha_buffer": False})
            if uf.get('update_color_buffer'):
                buf_c = prev_c
            if uf.get('update_alpha_buffer'):
                buf_a = prev_a

            y0 -= 420

        nt.links.new(prev_c, bsdf.inputs['Base Color'])
        nt.links.new(prev_a, bsdf.inputs['Alpha'])
    else:
        if 0 in tex_nodes:
            nt.links.new(tex_nodes[0].outputs['Color'], bsdf.inputs['Base Color'])
            nt.links.new(tex_nodes[0].outputs['Alpha'], bsdf.inputs['Alpha'])

    mat.blend_method = 'OPAQUE'
    if mat_def.alpha_test_enabled:
        mat.blend_method = 'CLIP'
        if hasattr(mat, 'alpha_threshold'):
            mat.alpha_threshold = float(mat_def.alpha_test_ref)

    try:
        mat.use_backface_culling = (int(mat_def.face_culling or 0) == 2)
    except Exception:
        pass

    _set_gfmodel_material_props(mat, mat_def, shader_by_name)
    return mat


