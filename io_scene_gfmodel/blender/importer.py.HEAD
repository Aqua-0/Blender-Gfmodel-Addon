"""Blender import implementation for GFModel.

Contains mesh/armature/material creation and the import operator.
"""

from __future__ import annotations

import json
import os
import struct
from typing import Dict, List, Optional, Sequence, Tuple

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, StringProperty
from bpy_extras.io_utils import ImportHelper, axis_conversion
from mathutils import Matrix, Quaternion, Vector

from ..core.io import _load_any
from ..core.pica import (
    _bgra_to_rgba_floats,
    _decode_rgba_u32,
    _decode_texenv_update_buffer,
    _flip_bgra_y,
    _pica_decode_bitmap_to_bgra,
)
from ..core.types import (
    _GFMaterial,
    _GFModel,
    _GFMotion,
    _GFShader,
    _GFSubMesh,
    _GFTexture,
)
from .anim import (
    _apply_uv_anim_enable,
    _apply_visibility_anim_enable,
    _compute_rest_world_mats,
    _euler_to_quat_xyz,
    _gf_runtime_cache_armature,
    _mot_eval,
    _transform_quat_basis,
)


def _bone_uses_ssc(flags: int) -> bool:
    return (int(flags) & 0x02) != 0


def _triangulate(indices: Sequence[int], prim: int) -> List[Tuple[int, int, int]]:
    if prim == 0:             
        tris: List[Tuple[int, int, int]] = []
        for i in range(0, len(indices) - 2, 3):
            a, b, c = indices[i], indices[i + 1], indices[i + 2]
            if a == b or b == c or a == c:
                continue
            tris.append((a, b, c))
        return tris
    if prim == 1:                 
        tris = []
        flip = False
        for i in range(len(indices) - 2):
            a, b, c = indices[i], indices[i + 1], indices[i + 2]
            if a == b or b == c or a == c:
                flip = not flip
                continue
            tris.append((a, c, b) if flip else (a, b, c))
            flip = not flip
        return tris
    if prim == 2:               
        if len(indices) < 3:
            return []
        center = indices[0]
        tris = []
        for i in range(1, len(indices) - 1):
            a, b = indices[i], indices[i + 1]
            if center == a or a == b or center == b:
                continue
            tris.append((center, a, b))
        return tris
    return []


def _read_vertices(
    submesh: _GFSubMesh,
) -> Tuple[
    List[Vector],
    List[Vector],
    List[Vector],
    List[Vector],
    List[List[Tuple[int, float]]],
]:
    buf = memoryview(submesh.raw_buffer)
    stride = submesh.vertex_stride
    if stride <= 0:
        return [], [], [], []
    count = len(buf) // stride

    pos: List[Vector] = [Vector((0.0, 0.0, 0.0))] * count
    nrm: List[Vector] = [Vector((0.0, 0.0, 1.0))] * count
    uv0: List[Vector] = [Vector((0.0, 0.0))] * count
    col: List[Vector] = [Vector((1.0, 1.0, 1.0, 1.0))] * count
    weights: List[List[Tuple[int, float]]] = [[] for _ in range(count)]

    fixed_indices = next((fa for fa in submesh.fixed_attributes if fa.name == 7), None)
    fixed_weights = next((fa for fa in submesh.fixed_attributes if fa.name == 8), None)
    fixed_color = next((fa for fa in submesh.fixed_attributes if fa.name == 3), None)

    for vi in range(count):
        o = vi * stride
                                                                               
        local = o
        bone_indices: List[int] = []
        bone_weights: List[float] = []

        def align2(fmt: int) -> None:
            nonlocal local
            if fmt not in (0, 1):
                local += local & 1

        def read_elem(fmt: int) -> float:
            nonlocal local
            if fmt == 0:
                v = struct.unpack_from("<b", buf, local)[0]
                local += 1
                return float(v)
            if fmt == 1:
                v = buf[local]
                local += 1
                return float(v)
            if fmt == 2:
                v = struct.unpack_from("<h", buf, local)[0]
                local += 2
                return float(v)
            v = struct.unpack_from("<f", buf, local)[0]
            local += 4
            return float(v)

        for attr in submesh.attributes:
            align2(attr.fmt)
            elems = [0.0, 0.0, 0.0, 0.0]
            for ei in range(attr.elements):
                elems[ei] = read_elem(attr.fmt)
            v4 = Vector((elems[0], elems[1], elems[2], elems[3])) * attr.scale
            if attr.name == 0:            
                pos[vi] = Vector((v4.x, v4.y, v4.z))
            elif attr.name == 1:          
                nrm[vi] = Vector((v4.x, v4.y, v4.z))
            elif attr.name == 3:         
                col[vi] = Vector((v4.x, v4.y, v4.z, v4.w))
            elif attr.name == 4:             
                uv0[vi] = Vector((v4.x, v4.y))
            elif attr.name == 7:             
                bone_indices.extend(
                    [int(v4.x), int(v4.y), int(v4.z), int(v4.w)][: attr.elements]
                )
            elif attr.name == 8:              
                bone_weights.extend(
                    [float(v4.x), float(v4.y), float(v4.z), float(v4.w)][
                        : attr.elements
                    ]
                )

        if fixed_color is not None and (col[vi].x, col[vi].y, col[vi].z, col[vi].w) == (
            1.0,
            1.0,
            1.0,
            1.0,
        ):
            col[vi] = Vector(
                (
                    float(fixed_color.x),
                    float(fixed_color.y),
                    float(fixed_color.z),
                    float(fixed_color.w),
                )
            )

        if (not bone_indices) and fixed_indices is not None:
            bone_indices = [
                int(fixed_indices.x),
                int(fixed_indices.y),
                int(fixed_indices.z),
                int(fixed_indices.w),
            ]
        if (not bone_weights) and fixed_weights is not None:
            bone_weights = [
                float(fixed_weights.x),
                float(fixed_weights.y),
                float(fixed_weights.z),
                float(fixed_weights.w),
            ]

        wlist: List[Tuple[int, float]] = []
        for bi, bw in zip(bone_indices, bone_weights):
            if bw <= 0:
                continue
            wlist.append((bi, bw))
        weights[vi] = wlist

    return pos, nrm, uv0, col, weights


def _make_image(tex: _GFTexture) -> bpy.types.Image:
    img = bpy.data.images.get(tex.name)
    if img is not None:
        return img
    img = bpy.data.images.new(tex.name, width=tex.width, height=tex.height, alpha=True)
    bgra = _pica_decode_bitmap_to_bgra(tex.raw, tex.width, tex.height, tex.fmt)
    bgra = _flip_bgra_y(bgra, tex.width, tex.height)
    img.pixels = _bgra_to_rgba_floats(bgra)
    img.pack()
    return img


def _make_material(
    mat_def: _GFMaterial,
    textures: Dict[str, bpy.types.Image],
    shader_by_name: Dict[str, _GFShader],
) -> bpy.types.Material:
    mat = bpy.data.materials.get(mat_def.name)
    if mat is not None:
        return mat
    mat = bpy.data.materials.new(name=mat_def.name)
    mat.use_nodes = True
    nt = mat.node_tree
    if nt is None:
        return mat
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (0, 0)
    out.location = (300, 0)
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    def make_tex_unit_nodes(
        unit_index: int, tex_name: str, x: float, y: float
    ) -> Optional[bpy.types.ShaderNodeTexImage]:
        if tex_name not in textures:
            return None
        texcoord = nt.nodes.get("GF_TEXCOORD")
        if texcoord is None:
            texcoord = nt.nodes.new("ShaderNodeTexCoord")
            texcoord.name = "GF_TEXCOORD"
            texcoord.label = "GF TexCoord"
            texcoord.location = (-1050, 0)
        mapping = nt.nodes.new("ShaderNodeMapping")
        mapping.location = (x - 350, y)
        mapping.label = f"GF UV Mapping (Unit {unit_index})"
        mapping.name = f"GF_MAPPING_{unit_index}"
        mapping.inputs["Scale"].default_value = (1.0, 1.0, 1.0)
        mapping.inputs["Rotation"].default_value = (0.0, 0.0, 0.0)
        mapping.inputs["Location"].default_value = (0.0, 0.0, 0.0)
        tex_node = nt.nodes.new("ShaderNodeTexImage")
        tex_node.image = textures[tex_name]
        tex_node.location = (x, y)
        tex_node.interpolation = "Linear"
                                                                                    
                                                                               
        tex_node.extension = "REPEAT"
        nt.links.new(texcoord.outputs["UV"], mapping.inputs["Vector"])
        nt.links.new(mapping.outputs["Vector"], tex_node.inputs["Vector"])
        return tex_node

    tex_nodes: Dict[int, bpy.types.ShaderNodeTexImage] = {}
    for tu in mat_def.tex_units:
        if 0 <= tu.unit_index <= 2 and tu.name:
            tnode = make_tex_unit_nodes(
                tu.unit_index, tu.name, -300, -250 * tu.unit_index
            )
            if tnode is not None:
                tex_nodes[tu.unit_index] = tnode
            mapping = nt.nodes.get(f"GF_MAPPING_{tu.unit_index}")
            if mapping is not None:
                mapping.inputs["Scale"].default_value = (tu.scale.x, tu.scale.y, 1.0)
                mapping.inputs["Rotation"].default_value = (0.0, 0.0, tu.rotation)
                mapping.inputs["Location"].default_value = (
                    tu.translation.x,
                    tu.translation.y,
                    0.0,
                )

    def _node_value(
        v: float, x: float, y: float, name: str
    ) -> bpy.types.ShaderNodeValue:
        n = nt.nodes.new("ShaderNodeValue")
        n.location = (x, y)
        n.label = name
        n.outputs[0].default_value = float(v)
        return n

    def _node_rgb(
        rgb: Tuple[float, float, float], x: float, y: float, name: str
    ) -> bpy.types.ShaderNodeRGB:
        n = nt.nodes.new("ShaderNodeRGB")
        n.location = (x, y)
        n.label = name
        n.outputs[0].default_value = (float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0)
        return n

    def _node_const_vec3(
        v: Tuple[float, float, float], x: float, y: float, name: str
    ) -> bpy.types.ShaderNodeCombineXYZ:
        cx = _node_value(v[0], x - 220, y + 60, f"{name}.x")
        cy = _node_value(v[1], x - 220, y + 30, f"{name}.y")
        cz = _node_value(v[2], x - 220, y + 0, f"{name}.z")
        comb = nt.nodes.new("ShaderNodeCombineXYZ")
        comb.location = (x, y)
        comb.label = name
        nt.links.new(cx.outputs[0], comb.inputs["X"])
        nt.links.new(cy.outputs[0], comb.inputs["Y"])
        nt.links.new(cz.outputs[0], comb.inputs["Z"])
        return comb

    def _vec_one(x: float, y: float, name: str) -> bpy.types.ShaderNodeCombineXYZ:
        return _node_const_vec3((1.0, 1.0, 1.0), x, y, name)

    def _op_color(
        operand_id: int,
        color: bpy.types.NodeSocket,
        alpha: bpy.types.NodeSocket,
        x: float,
        y: float,
    ) -> bpy.types.NodeSocket:
        if operand_id == 0:         
            return color
        if operand_id == 1:                 
            one = _vec_one(x - 200, y, "One")
            sub = nt.nodes.new("ShaderNodeVectorMath")
            sub.operation = "SUBTRACT"
            sub.location = (x, y)
            nt.links.new(one.outputs["Vector"], sub.inputs[0])
            nt.links.new(color, sub.inputs[1])
            return sub.outputs["Vector"]
        if operand_id == 2:         
            comb = nt.nodes.new("ShaderNodeCombineXYZ")
            comb.location = (x, y)
            nt.links.new(alpha, comb.inputs["X"])
            nt.links.new(alpha, comb.inputs["Y"])
            nt.links.new(alpha, comb.inputs["Z"])
            return comb.outputs["Vector"]
        if operand_id == 3:                 
            one = _node_value(1.0, x - 250, y, "1")
            inv = nt.nodes.new("ShaderNodeMath")
            inv.operation = "SUBTRACT"
            inv.location = (x - 50, y)
            nt.links.new(one.outputs[0], inv.inputs[0])
            nt.links.new(alpha, inv.inputs[1])
            comb = nt.nodes.new("ShaderNodeCombineXYZ")
            comb.location = (x, y)
            nt.links.new(inv.outputs[0], comb.inputs["X"])
            nt.links.new(inv.outputs[0], comb.inputs["Y"])
            nt.links.new(inv.outputs[0], comb.inputs["Z"])
            return comb.outputs["Vector"]

        sep = nt.nodes.new("ShaderNodeSeparateColor")
        sep.mode = "RGB"
        sep.location = (x - 250, y)
        nt.links.new(color, sep.inputs["Color"])
        channel = None
        inv = False
        if operand_id in (4, 5):
            channel = "Red"
            inv = operand_id == 5
        elif operand_id in (8, 9):
            channel = "Green"
            inv = operand_id == 9
        elif operand_id in (12, 13):
            channel = "Blue"
            inv = operand_id == 13
        if channel is None:
            return color
        out = sep.outputs[channel]
        if inv:
            one = _node_value(1.0, x - 160, y, "1")
            sub = nt.nodes.new("ShaderNodeMath")
            sub.operation = "SUBTRACT"
            sub.location = (x - 20, y)
            nt.links.new(one.outputs[0], sub.inputs[0])
            nt.links.new(out, sub.inputs[1])
            out = sub.outputs[0]
        comb = nt.nodes.new("ShaderNodeCombineXYZ")
        comb.location = (x, y)
        nt.links.new(out, comb.inputs["X"])
        nt.links.new(out, comb.inputs["Y"])
        nt.links.new(out, comb.inputs["Z"])
        return comb.outputs["Vector"]

    def _op_alpha(
        operand_id: int,
        color: bpy.types.NodeSocket,
        alpha: bpy.types.NodeSocket,
        x: float,
        y: float,
    ) -> bpy.types.NodeSocket:
        if operand_id == 0:         
            return alpha
        if operand_id == 1:                 
            one = _node_value(1.0, x - 160, y, "1")
            sub = nt.nodes.new("ShaderNodeMath")
            sub.operation = "SUBTRACT"
            sub.location = (x, y)
            nt.links.new(one.outputs[0], sub.inputs[0])
            nt.links.new(alpha, sub.inputs[1])
            return sub.outputs[0]
        sep = nt.nodes.new("ShaderNodeSeparateColor")
        sep.mode = "RGB"
        sep.location = (x - 200, y)
        nt.links.new(color, sep.inputs["Color"])
        channel = None
        inv = False
        if operand_id in (2, 3):
            channel = "Red"
            inv = operand_id == 3
        elif operand_id in (4, 5):
            channel = "Green"
            inv = operand_id == 5
        elif operand_id in (6, 7):
            channel = "Blue"
            inv = operand_id == 7
        if channel is None:
            return alpha
        out = sep.outputs[channel]
        if inv:
            one = _node_value(1.0, x - 160, y, "1")
            sub = nt.nodes.new("ShaderNodeMath")
            sub.operation = "SUBTRACT"
            sub.location = (x, y)
            nt.links.new(one.outputs[0], sub.inputs[0])
            nt.links.new(out, sub.inputs[1])
            out = sub.outputs[0]
        return out

    def _clamp01_vec(
        inp: bpy.types.NodeSocket, x: float, y: float
    ) -> bpy.types.NodeSocket:
                                                                                                 
        v0 = _node_const_vec3((0.0, 0.0, 0.0), x - 250, y + 0, "ClampMin0")
        v1 = _node_const_vec3((1.0, 1.0, 1.0), x - 250, y - 140, "ClampMax1")
        vmin = nt.nodes.new("ShaderNodeVectorMath")
        vmin.operation = "MINIMUM"
        vmin.location = (x, y)
        nt.links.new(inp, vmin.inputs[0])
        nt.links.new(v1.outputs["Vector"], vmin.inputs[1])
        vmax = nt.nodes.new("ShaderNodeVectorMath")
        vmax.operation = "MAXIMUM"
        vmax.location = (x + 240, y)
        nt.links.new(vmin.outputs["Vector"], vmax.inputs[0])
        nt.links.new(v0.outputs["Vector"], vmax.inputs[1])
        return vmax.outputs["Vector"]

    def _clamp01(inp: bpy.types.NodeSocket, x: float, y: float) -> bpy.types.NodeSocket:
        clamp = nt.nodes.new("ShaderNodeClamp")
        clamp.location = (x, y)
        nt.links.new(inp, clamp.inputs["Value"])
        clamp.inputs["Min"].default_value = 0.0
        clamp.inputs["Max"].default_value = 1.0
        return clamp.outputs["Result"]

    def _texenv_combine_color(
        mode: int,
        a: bpy.types.NodeSocket,
        b: bpy.types.NodeSocket,
        c: bpy.types.NodeSocket,
        x: float,
        y: float,
    ) -> bpy.types.NodeSocket:
        if mode == 0:           
            return a
        if mode == 1:            
            mul = nt.nodes.new("ShaderNodeVectorMath")
            mul.operation = "MULTIPLY"
            mul.location = (x, y)
            nt.links.new(a, mul.inputs[0])
            nt.links.new(b, mul.inputs[1])
            return mul.outputs["Vector"]
        if mode == 2:       
            add = nt.nodes.new("ShaderNodeVectorMath")
            add.operation = "ADD"
            add.location = (x, y)
            nt.links.new(a, add.inputs[0])
            nt.links.new(b, add.inputs[1])
            return add.outputs["Vector"]
        if mode == 3:                      
            add = nt.nodes.new("ShaderNodeVectorMath")
            add.operation = "ADD"
            add.location = (x - 120, y)
            nt.links.new(a, add.inputs[0])
            nt.links.new(b, add.inputs[1])
            half = _node_const_vec3((0.5, 0.5, 0.5), x - 120, y - 120, "Half")
            sub = nt.nodes.new("ShaderNodeVectorMath")
            sub.operation = "SUBTRACT"
            sub.location = (x, y)
            nt.links.new(add.outputs["Vector"], sub.inputs[0])
            nt.links.new(half.outputs["Vector"], sub.inputs[1])
            return sub.outputs["Vector"]
        if mode == 4:               
                           
            inv = nt.nodes.new("ShaderNodeVectorMath")
            inv.operation = "SUBTRACT"
            inv.location = (x - 240, y - 120)
            one = _vec_one(x - 460, y - 120, "One")
            nt.links.new(one.outputs["Vector"], inv.inputs[0])
            nt.links.new(c, inv.inputs[1])
            a_mul = nt.nodes.new("ShaderNodeVectorMath")
            a_mul.operation = "MULTIPLY"
            a_mul.location = (x - 120, y + 40)
            nt.links.new(a, a_mul.inputs[0])
            nt.links.new(c, a_mul.inputs[1])
            b_mul = nt.nodes.new("ShaderNodeVectorMath")
            b_mul.operation = "MULTIPLY"
            b_mul.location = (x - 120, y - 120)
            nt.links.new(b, b_mul.inputs[0])
            nt.links.new(inv.outputs["Vector"], b_mul.inputs[1])
            add = nt.nodes.new("ShaderNodeVectorMath")
            add.operation = "ADD"
            add.location = (x, y)
            nt.links.new(a_mul.outputs["Vector"], add.inputs[0])
            nt.links.new(b_mul.outputs["Vector"], add.inputs[1])
            return add.outputs["Vector"]
        if mode == 5:            
            sub = nt.nodes.new("ShaderNodeVectorMath")
            sub.operation = "SUBTRACT"
            sub.location = (x, y)
            nt.links.new(a, sub.inputs[0])
            nt.links.new(b, sub.inputs[1])
            return sub.outputs["Vector"]
        if mode == 8:           
            mul = nt.nodes.new("ShaderNodeVectorMath")
            mul.operation = "MULTIPLY"
            mul.location = (x - 120, y)
            nt.links.new(a, mul.inputs[0])
            nt.links.new(b, mul.inputs[1])
            add = nt.nodes.new("ShaderNodeVectorMath")
            add.operation = "ADD"
            add.location = (x, y)
            nt.links.new(mul.outputs["Vector"], add.inputs[0])
            nt.links.new(c, add.inputs[1])
            return add.outputs["Vector"]
        if mode == 9:           
            add = nt.nodes.new("ShaderNodeVectorMath")
            add.operation = "ADD"
            add.location = (x - 120, y)
            nt.links.new(a, add.inputs[0])
            nt.links.new(b, add.inputs[1])
            mul = nt.nodes.new("ShaderNodeVectorMath")
            mul.operation = "MULTIPLY"
            mul.location = (x, y)
            nt.links.new(add.outputs["Vector"], mul.inputs[0])
            nt.links.new(c, mul.inputs[1])
            return mul.outputs["Vector"]
        return a

    def _texenv_combine_alpha(
        mode: int,
        a: bpy.types.NodeSocket,
        b: bpy.types.NodeSocket,
        c: bpy.types.NodeSocket,
        x: float,
        y: float,
    ) -> bpy.types.NodeSocket:
        if mode == 0:           
            return a
        if mode == 1:            
            mul = nt.nodes.new("ShaderNodeMath")
            mul.operation = "MULTIPLY"
            mul.location = (x, y)
            nt.links.new(a, mul.inputs[0])
            nt.links.new(b, mul.inputs[1])
            return mul.outputs[0]
        if mode == 2:       
            add = nt.nodes.new("ShaderNodeMath")
            add.operation = "ADD"
            add.location = (x, y)
            nt.links.new(a, add.inputs[0])
            nt.links.new(b, add.inputs[1])
            return add.outputs[0]
        if mode == 3:                      
            add = nt.nodes.new("ShaderNodeMath")
            add.operation = "ADD"
            add.location = (x - 120, y)
            nt.links.new(a, add.inputs[0])
            nt.links.new(b, add.inputs[1])
            sub = nt.nodes.new("ShaderNodeMath")
            sub.operation = "SUBTRACT"
            sub.location = (x, y)
            nt.links.new(add.outputs[0], sub.inputs[0])
            sub.inputs[1].default_value = 0.5
            return sub.outputs[0]
        if mode == 4:               
                           
            inv = nt.nodes.new("ShaderNodeMath")
            inv.operation = "SUBTRACT"
            inv.location = (x - 240, y - 120)
            inv.inputs[0].default_value = 1.0
            nt.links.new(c, inv.inputs[1])
            a_mul = nt.nodes.new("ShaderNodeMath")
            a_mul.operation = "MULTIPLY"
            a_mul.location = (x - 120, y + 40)
            nt.links.new(a, a_mul.inputs[0])
            nt.links.new(c, a_mul.inputs[1])
            b_mul = nt.nodes.new("ShaderNodeMath")
            b_mul.operation = "MULTIPLY"
            b_mul.location = (x - 120, y - 120)
            nt.links.new(b, b_mul.inputs[0])
            nt.links.new(inv.outputs[0], b_mul.inputs[1])
            add = nt.nodes.new("ShaderNodeMath")
            add.operation = "ADD"
            add.location = (x, y)
            nt.links.new(a_mul.outputs[0], add.inputs[0])
            nt.links.new(b_mul.outputs[0], add.inputs[1])
            return add.outputs[0]
        if mode == 5:            
            sub = nt.nodes.new("ShaderNodeMath")
            sub.operation = "SUBTRACT"
            sub.location = (x, y)
            nt.links.new(a, sub.inputs[0])
            nt.links.new(b, sub.inputs[1])
            return sub.outputs[0]
        if mode == 8:           
            mul = nt.nodes.new("ShaderNodeMath")
            mul.operation = "MULTIPLY"
            mul.location = (x - 120, y)
            nt.links.new(a, mul.inputs[0])
            nt.links.new(b, mul.inputs[1])
            add = nt.nodes.new("ShaderNodeMath")
            add.operation = "ADD"
            add.location = (x, y)
            nt.links.new(mul.outputs[0], add.inputs[0])
            nt.links.new(c, add.inputs[1])
            return add.outputs[0]
        if mode == 9:           
            add = nt.nodes.new("ShaderNodeMath")
            add.operation = "ADD"
            add.location = (x - 120, y)
            nt.links.new(a, add.inputs[0])
            nt.links.new(b, add.inputs[1])
            mul = nt.nodes.new("ShaderNodeMath")
            mul.operation = "MULTIPLY"
            mul.location = (x, y)
            nt.links.new(add.outputs[0], mul.inputs[0])
            nt.links.new(c, mul.inputs[1])
            return mul.outputs[0]
        return a

    def _apply_scale_vec(
        scale_id: int, vec: bpy.types.NodeSocket, x: float, y: float
    ) -> bpy.types.NodeSocket:
        factor = 1.0
        if scale_id == 1:
            factor = 2.0
        elif scale_id == 2:
            factor = 4.0
        if factor == 1.0:
            return vec
        fac = _node_const_vec3((factor, factor, factor), x - 200, y, f"*{factor}")
        mul = nt.nodes.new("ShaderNodeVectorMath")
        mul.operation = "MULTIPLY"
        mul.location = (x, y)
        nt.links.new(vec, mul.inputs[0])
        nt.links.new(fac.outputs["Vector"], mul.inputs[1])
        return mul.outputs["Vector"]

    def _apply_scale(
        scale_id: int, val: bpy.types.NodeSocket, x: float, y: float
    ) -> bpy.types.NodeSocket:
        factor = 1.0
        if scale_id == 1:
            factor = 2.0
        elif scale_id == 2:
            factor = 4.0
        if factor == 1.0:
            return val
        mul = nt.nodes.new("ShaderNodeMath")
        mul.operation = "MULTIPLY"
        mul.location = (x, y)
        nt.links.new(val, mul.inputs[0])
        mul.inputs[1].default_value = factor
        return mul.outputs[0]

    def _source_sockets(
        source_id: int,
        stage_const_rgba: Tuple[float, float, float, float],
        prev_c: bpy.types.NodeSocket,
        prev_a: bpy.types.NodeSocket,
        buf_c: bpy.types.NodeSocket,
        buf_a: bpy.types.NodeSocket,
    ) -> Tuple[bpy.types.NodeSocket, bpy.types.NodeSocket]:
        if source_id == 3 and 0 in tex_nodes:
            return tex_nodes[0].outputs["Color"], tex_nodes[0].outputs["Alpha"]
        if source_id == 4 and 1 in tex_nodes:
            return tex_nodes[1].outputs["Color"], tex_nodes[1].outputs["Alpha"]
        if source_id == 5 and 2 in tex_nodes:
            return tex_nodes[2].outputs["Color"], tex_nodes[2].outputs["Alpha"]
        if source_id == 14:            
            rgb = (stage_const_rgba[0], stage_const_rgba[1], stage_const_rgba[2])
            col = _node_rgb(rgb, -900, -950, "TexEnv Constant").outputs["Color"]
            a = _node_value(
                stage_const_rgba[3], -900, -980, "TexEnv Constant A"
            ).outputs[0]
            return col, a
        if source_id == 13:                  
            return buf_c, buf_a
        if source_id == 15:            
            return prev_c, prev_a
                                                  
                                                                                   
        if source_id in (0, 1, 2):
            attr = nt.nodes.new("ShaderNodeVertexColor")
            attr.layer_name = "Col"
            attr.location = (-900, -860)
            return attr.outputs["Color"], attr.outputs["Alpha"]
                         
        white = _node_rgb((1.0, 1.0, 1.0), -900, -820, "White")
        one = _node_value(1.0, -900, -850, "1")
        return white.outputs["Color"], one.outputs[0]

                                                                                                    
                                                                                             
    sh_preview = shader_by_name.get(mat_def.frag_shader)
    if sh_preview and sh_preview.texenv_stages:
        stage_consts: List[Tuple[float, float, float, float]] = []
        for st in sh_preview.texenv_stages:
            if st.color is None:
                stage_consts.append((1.0, 1.0, 1.0, 1.0))
            else:
                stage_consts.append(_decode_rgba_u32(int(st.color)))

                                  
        prev_color = _node_rgb((1.0, 1.0, 1.0), -850, 250, "PrevInit").outputs["Color"]
        prev_alpha = _node_value(1.0, -850, 220, "PrevInitA").outputs[0]
        if sh_preview.texenv_buffer_color is not None:
            bc = _decode_rgba_u32(int(sh_preview.texenv_buffer_color))
            buf_color = _node_rgb((bc[0], bc[1], bc[2]), -850, 160, "BufInit").outputs[
                "Color"
            ]
            buf_alpha = _node_value(bc[3], -850, 130, "BufInitA").outputs[0]
        else:
            buf_color = _node_rgb((0.0, 0.0, 0.0), -850, 160, "BufInit").outputs[
                "Color"
            ]
            buf_alpha = _node_value(0.0, -850, 130, "BufInitA").outputs[0]

        update_flags = _decode_texenv_update_buffer(
            int(sh_preview.texenv_update_buffer or 0)
        )

        y0 = 250
        for st in sh_preview.texenv_stages:
            if (
                st.source is None
                or st.operand is None
                or st.combiner is None
                or st.scale is None
            ):
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

            const_rgba = (
                stage_consts[st.stage]
                if st.stage < len(stage_consts)
                else (1.0, 1.0, 1.0, 1.0)
            )

            s0c, s0a = _source_sockets(
                c0, const_rgba, prev_color, prev_alpha, buf_color, buf_alpha
            )
            s1c, s1a = _source_sockets(
                c1, const_rgba, prev_color, prev_alpha, buf_color, buf_alpha
            )
            s2c, s2a = _source_sockets(
                c2, const_rgba, prev_color, prev_alpha, buf_color, buf_alpha
            )
            t0c = _op_color(oc0, s0c, s0a, -400, y0)
            t1c = _op_color(oc1, s1c, s1a, -400, y0 - 60)
            t2c = _op_color(oc2, s2c, s2a, -400, y0 - 120)

                                            
            sa0c, sa0a = _source_sockets(
                a0, const_rgba, prev_color, prev_alpha, buf_color, buf_alpha
            )
            sa1c, sa1a = _source_sockets(
                a1, const_rgba, prev_color, prev_alpha, buf_color, buf_alpha
            )
            sa2c, sa2a = _source_sockets(
                a2, const_rgba, prev_color, prev_alpha, buf_color, buf_alpha
            )
            t0a = _op_alpha(oa0, sa0c, sa0a, -400, y0 - 200)
            t1a = _op_alpha(oa1, sa1c, sa1a, -400, y0 - 260)
            t2a = _op_alpha(oa2, sa2c, sa2a, -400, y0 - 320)

            out_c = _texenv_combine_color(col_mode, t0c, t1c, t2c, -60, y0 - 60)
            out_a = _texenv_combine_alpha(alp_mode, t0a, t1a, t2a, -60, y0 - 260)

            col_scale = (sc >> 0) & 0x3
            alp_scale = (sc >> 16) & 0x3
            out_c = _apply_scale_vec(col_scale, out_c, 120, y0 - 60)
            out_a = _apply_scale(alp_scale, out_a, 120, y0 - 260)
            out_c = _clamp01_vec(out_c, 320, y0 - 60)
            out_a = _clamp01(out_a, 320, y0 - 260)

            prev_color = out_c
            prev_alpha = out_a

            uf = update_flags.get(
                st.stage, {"update_color_buffer": False, "update_alpha_buffer": False}
            )
            if uf.get("update_color_buffer"):
                buf_color = prev_color
            if uf.get("update_alpha_buffer"):
                buf_alpha = prev_alpha

            y0 -= 420

        bsdf.inputs["Base Color"].default_value = (1.0, 1.0, 1.0, 1.0)
        nt.links.new(prev_color, bsdf.inputs["Base Color"])
        nt.links.new(prev_alpha, bsdf.inputs["Alpha"])
    else:
                                                           
        if 0 in tex_nodes:
            nt.links.new(tex_nodes[0].outputs["Color"], bsdf.inputs["Base Color"])

                                                                                                  
    mat.blend_method = "OPAQUE"

                                                                                
    resolved_alpha: Optional[float] = None
    sh = shader_by_name.get(mat_def.frag_shader)
    if sh and sh.texenv_stages and sh.texenv_stages[0].combiner is not None:
        comb = sh.texenv_stages[0].combiner
        src = sh.texenv_stages[0].source
        col = sh.texenv_stages[0].color
        if src is not None and col is not None:
            alpha_mode = (comb >> 16) & 0xF
            alpha_src_a = (src >> 16) & 0xF
            if alpha_mode == 0 and alpha_src_a == 14:
                resolved_alpha = _decode_rgba_u32(col)[3]

    if resolved_alpha is not None and 0.0 < resolved_alpha < 1.0:
        bsdf.inputs["Alpha"].default_value = float(resolved_alpha)
        mat.blend_method = "BLEND"
        if hasattr(mat, "shadow_method"):
            mat.shadow_method = "HASHED"
    elif mat_def.alpha_test_enabled:
        mat.blend_method = "CLIP"
        if hasattr(mat, "alpha_threshold"):
            mat.alpha_threshold = float(mat_def.alpha_test_ref)

                                     
                                                        
                                                                                                                 
    try:
        if mat_def.face_culling == 2:
            mat.use_backface_culling = True
        else:
            mat.use_backface_culling = False
        mat["gfmodel_face_culling"] = int(mat_def.face_culling or 0)
    except Exception:
        pass

                                                                 
    try:
        sh = shader_by_name.get(mat_def.frag_shader)
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
                            "unit_index": tu.unit_index,
                            "mapping_type": tu.mapping_type,
                            "scale": [tu.scale.x, tu.scale.y],
                            "rotation": tu.rotation,
                            "translation": [tu.translation.x, tu.translation.y],
                            "sampler_words": [int(w) for w in tu.sampler_words],
                        }
                        for tu in mat_def.tex_units
                    ],
                    "bump_texture": int(mat_def.bump_texture),
                    "const_assignments": [int(x) for x in mat_def.const_assignments],
                    "colors_rgba": [
                        [int(c) for c in rgba] for rgba in mat_def.colors_rgba
                    ],
                    "alpha_test": {
                        "enabled": mat_def.alpha_test_enabled,
                        "ref": mat_def.alpha_test_ref,
                        "func": mat_def.alpha_test_func,
                    },
                    "blend_func": mat_def.blend_func,
                    "blend_color_rgba": (
                        [int(c) for c in mat_def.blend_color_rgba]
                        if mat_def.blend_color_rgba is not None
                        else None
                    ),
                    "stencil_test": mat_def.stencil_test,
                    "stencil_op": mat_def.stencil_op,
                    "depth_test": {
                        "enabled": mat_def.depth_test_enabled,
                        "func": mat_def.depth_test_func,
                    },
                    "depth_write": mat_def.depth_write,
                    "color_write_mask": (
                        [bool(x) for x in mat_def.color_write_mask]
                        if mat_def.color_write_mask is not None
                        else None
                    ),
                    "face_culling": mat_def.face_culling,
                    "pica_commands_u32": mat_def.pica_commands,
                    "pica_regs": _jsonify_pica_regs(mat_def.pica_regs),
                    "lut_hashes": [int(x) for x in mat_def.lut_hashes],
                    "render_priority": int(mat_def.render_priority),
                    "render_layer": int(mat_def.render_layer),
                    "header_hashes": [int(x) for x in mat_def.header_hashes],
                    "unk_render": int(mat_def.unk_render),
                    "userdata": {
                        "edge_type": int(mat_def.edge_type),
                        "id_edge_enable": int(mat_def.id_edge_enable),
                        "edge_id": int(mat_def.edge_id),
                        "projection_type": int(mat_def.projection_type),
                        "rim_pow": float(mat_def.rim_pow),
                        "rim_scale": float(mat_def.rim_scale),
                        "phong_pow": float(mat_def.phong_pow),
                        "phong_scale": float(mat_def.phong_scale),
                        "id_edge_offset_enable": int(mat_def.id_edge_offset_enable),
                        "edge_map_alpha_mask": int(mat_def.edge_map_alpha_mask),
                        "bake_ops": [int(x) for x in mat_def.bake_ops],
                        "vertex_shader_type": int(mat_def.vertex_shader_type),
                        "shader_params": [float(x) for x in mat_def.shader_params],
                    },
                },
                "frag_shader": {
                    "name": sh.name if sh else mat_def.frag_shader,
                    "found": sh is not None,
                    "texenv_buffer_color": sh.texenv_buffer_color if sh else None,
                    "texenv_update_buffer": sh.texenv_update_buffer if sh else None,
                    "texenv_update_buffer_decoded": _decode_texenv_update_buffer(
                        int(sh.texenv_update_buffer or 0)
                    )
                    if sh and sh.texenv_update_buffer is not None
                    else None,
                    "texenv_stages": [
                        {
                            "stage": s.stage,
                            "source": s.source,
                            "operand": s.operand,
                            "combiner": s.combiner,
                            "color": s.color,
                            "scale": s.scale,
                            "decoded": {
                                "source": _decode_texenv_source(int(s.source or 0))
                                if s.source is not None
                                else None,
                                "operand": _decode_texenv_operand(int(s.operand or 0))
                                if s.operand is not None
                                else None,
                                "combiner": _decode_texenv_combiner(
                                    int(s.combiner or 0)
                                )
                                if s.combiner is not None
                                else None,
                                "color": list(_decode_rgba_u32(int(s.color or 0)))
                                if s.color is not None
                                else None,
                                "scale": _decode_texenv_scale(int(s.scale or 0))
                                if s.scale is not None
                                else None,
                            },
                        }
                        for s in (sh.texenv_stages if sh else [])
                    ],
                    "pica_commands_u32": sh.pica_commands if sh else None,
                    "pica_regs": _jsonify_pica_regs(sh.pica_regs) if sh else None,
                },
            }
        )
    except Exception:
        pass
    return mat


def _build_armature(
    ctx: bpy.types.Context,
    model: _GFModel,
    conv: Matrix,
    global_scale: float,
    collection: bpy.types.Collection,
) -> bpy.types.Object:
    arm_data = bpy.data.armatures.new(f"{model.name}_Armature")
    arm_obj = bpy.data.objects.new(arm_data.name, arm_data)
    collection.objects.link(arm_obj)

    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode="EDIT")

    bones_by_name: Dict[str, bpy.types.EditBone] = {}
    conv3 = conv.to_3x3()

    def local_rest_matrix(b: _GFBone) -> Matrix:
        t = Matrix.Translation(conv @ (b.translation * global_scale))
        q = _transform_quat_basis(_euler_to_quat_xyz(b.rotation), conv3)
        r = q.to_matrix().to_4x4()
        s = Matrix.Diagonal(Vector((b.scale.x, b.scale.y, b.scale.z, 1.0)))
        return t @ r @ s

                                                                                                          
                                                                                                
    rest_world = _compute_rest_world_mats(model, conv, global_scale, ssc=False)

                                                                 
                                                                                                
                                                                                                    
    for b in model.skeleton:
        eb = arm_data.edit_bones.new(b.name)
        bones_by_name[b.name] = eb

    for b in model.skeleton:
        if b.parent and b.parent in bones_by_name:
            bones_by_name[b.name].parent = bones_by_name[b.parent]

    for b in model.skeleton:
        eb = bones_by_name[b.name]
        mw = rest_world.get(b.name, Matrix.Identity(4))
        loc, rot, _sca = mw.decompose()
        head = loc
        rot3 = rot.to_matrix()

                                                                                                         
                                                                                                    
        length = max(0.01, 0.05 * global_scale)
        y_axis = rot3 @ Vector((0.0, 1.0, 0.0))
        if y_axis.length == 0:
            y_axis = Vector((0.0, 1.0, 0.0))
        else:
            y_axis.normalize()

        eb.head = head
        eb.tail = head + y_axis * length

                                                                     
        try:
            z_axis = rot3 @ Vector((0.0, 0.0, 1.0))
            if z_axis.length != 0:
                z_axis.normalize()
                eb.align_roll(z_axis)
        except Exception:
            pass

    bpy.ops.object.mode_set(mode="OBJECT")

                                                        
                                                                                                      
                                                                                                  
    for pb in arm_obj.pose.bones:
        try:
            pb.bone.inherit_scale = "FULL"
        except Exception:
            pass
    return arm_obj

                                                        
                                                                                                      
                                                                                                  
    for pb in arm_obj.pose.bones:
        try:
            pb.bone.inherit_scale = "FULL"
        except Exception:
            pass
    return arm_obj


def _import_model_to_blender(
    ctx: bpy.types.Context,
    model: _GFModel,
    textures: List[_GFTexture],
    motions: List[_GFMotion],
    shaders: List[_GFShader],
    *,
    import_textures: bool,
    import_animations: bool,
    import_material_animations: bool,
    import_visibility_animations: bool,
    global_scale: float,
    axis_forward: str,
    axis_up: str,
) -> None:
    conv = axis_conversion(
        from_forward=axis_forward, from_up=axis_up, to_forward="-Y", to_up="Z"
    ).to_4x4()
    conv3 = conv.to_3x3()

    coll = bpy.data.collections.new(f"GFModel_{model.name}")
    ctx.scene.collection.children.link(coll)

    images: Dict[str, bpy.types.Image] = {}
    if import_textures:
        for t in textures:
            images[t.name] = _make_image(t)

    shader_by_name: Dict[str, _GFShader] = {s.name: s for s in shaders}

    mats_by_name: Dict[str, bpy.types.Material] = {}
    for m in model.materials:
        mats_by_name[m.name] = _make_material(m, images, shader_by_name)

    arm_obj = _build_armature(ctx, model, conv, global_scale, coll)
                                                                                
    try:
        arm_obj["gfmodel_source_path"] = str(
            ctx.scene.get("gfmodel_last_import_path", "")
        )
        arm_obj["gfmodel_axis_forward"] = str(axis_forward)
        arm_obj["gfmodel_axis_up"] = str(axis_up)
        arm_obj["gfmodel_global_scale"] = float(global_scale)
    except Exception:
        pass

    mesh_objs_by_sm_name: Dict[str, bpy.types.Object] = {}
    for submesh_index, sm in enumerate(model.submeshes):
        positions, normals, uvs, colors, weights = _read_vertices(sm)
        if not positions or not sm.indices:
            continue

        verts = [conv @ (p * global_scale) for p in positions]
        tris = _triangulate(sm.indices, sm.primitive_mode)
        if not tris:
            continue

                                                                                        
                                                                                     
        base_name = f"{model.name}_{sm.name}"
        unique_name = base_name
        if unique_name in bpy.data.meshes:
            unique_name = f"{base_name}_{int(sm.mesh_index)}_{int(sm.face_index)}"
        if unique_name in bpy.data.meshes:
            unique_name = f"{base_name}_{int(submesh_index)}"

        mesh = bpy.data.meshes.new(unique_name)
        obj_name = mesh.name
        if obj_name in bpy.data.objects:
            obj_name = f"{obj_name}_obj"
        if obj_name in bpy.data.objects:
            obj_name = f"{obj_name}_{int(submesh_index)}"
        obj = bpy.data.objects.new(obj_name, mesh)
        coll.objects.link(obj)
        mesh_objs_by_sm_name[sm.name] = obj
                                                                                                      
        try:
            obj["gfmodel_model_name"] = str(model.name)
            obj["gfmodel_submesh_index"] = int(submesh_index)
            obj["gfmodel_mesh_index"] = int(sm.mesh_index)
            obj["gfmodel_face_index"] = int(sm.face_index)
            obj["gfmodel_material_name"] = str(sm.name)
            obj["gfmodel_mesh_name"] = str(sm.mesh_name)
                                                                                   
            idx_len = int(getattr(sm, "index_data_len", 0) or 0)
            elem = int(getattr(sm, "index_elem_size", 0) or 0)
            if elem not in (1, 2):
                                                                     
                elem = 2
            obj["gfmodel_index_data_len"] = int(idx_len)
            obj["gfmodel_index_elem_size"] = int(elem)
            obj["gfmodel_index_capacity"] = int(idx_len // elem) if idx_len > 0 else 0
            obj["gfmodel_index_count_file"] = int(len(sm.indices))
                                                                                          
            stride = int(getattr(sm, "vertex_stride", 0) or 0)
            vtx_len = int(len(getattr(sm, "raw_buffer", b"") or b""))
            obj["gfmodel_vertex_stride"] = int(stride)
            obj["gfmodel_vertex_data_len"] = int(vtx_len)
            obj["gfmodel_vertex_capacity"] = int(vtx_len // stride) if stride > 0 else 0
            obj["gfmodel_vertex_count_file"] = int(getattr(sm, "vertex_count", 0) or 0)
        except Exception:
            pass

        mesh.from_pydata([tuple(v) for v in verts], [], [list(t) for t in tris])
        mesh.validate(verbose=False)
        mesh.update()

        if normals:
            mesh.normals_split_custom_set_from_vertices(
                [tuple((conv.to_3x3() @ n).normalized()) for n in normals]
            )
            if hasattr(mesh, "use_auto_smooth"):
                mesh.use_auto_smooth = True

        uv_layer = mesh.uv_layers.new(name="UVMap")
        if uv_layer:
            for poly in mesh.polygons:
                for li in poly.loop_indices:
                    vi = mesh.loops[li].vertex_index
                    uv = uvs[vi] if vi < len(uvs) else Vector((0.0, 0.0))
                    uv_layer.data[li].uv = (uv.x, uv.y)

                                                  
                                                                                     
        try:
            col_attr = mesh.color_attributes.get("Col")
            if col_attr is None:
                col_attr = mesh.color_attributes.new(
                    name="Col", domain="CORNER", type="FLOAT_COLOR"
                )
            if col_attr is not None:
                for poly in mesh.polygons:
                    for li in poly.loop_indices:
                        vi = mesh.loops[li].vertex_index
                        c = (
                            colors[vi]
                            if vi < len(colors)
                            else Vector((1.0, 1.0, 1.0, 1.0))
                        )
                        col_attr.data[li].color = (
                            float(max(0.0, min(1.0, c.x))),
                            float(max(0.0, min(1.0, c.y))),
                            float(max(0.0, min(1.0, c.z))),
                            float(max(0.0, min(1.0, c.w))),
                        )
        except Exception:
            pass

        mat = mats_by_name.get(sm.name)
        if mat:
            mesh.materials.append(mat)
            for poly in mesh.polygons:
                poly.material_index = 0

                  
        obj.parent = arm_obj
        mod = obj.modifiers.new(name="Armature", type="ARMATURE")
        mod.object = arm_obj

        for bone in model.skeleton:
            obj.vertex_groups.get(bone.name) or obj.vertex_groups.new(name=bone.name)

        palette = sm.bone_indices
        for vi, wlist in enumerate(weights):
            for bi, bw in wlist:
                if not palette or bi < 0 or bi >= len(palette):
                    continue
                skel_idx = palette[bi]
                if skel_idx < 0 or skel_idx >= len(model.skeleton):
                    continue
                bone_name = model.skeleton[skel_idx].name
                vg = obj.vertex_groups.get(bone_name)
                if vg is not None:
                    vg.add([vi], float(bw), "REPLACE")

    created_actions: Dict[int, bpy.types.Action] = {}
    if import_animations and motions and model.skeleton:
        rest_by_name = {b.name: b for b in model.skeleton}
        bone_by_name = {b.name: b for b in model.skeleton}

        depth_cache: Dict[str, int] = {}

        def bone_depth(name: str) -> int:
            if name in depth_cache:
                return depth_cache[name]
            b = bone_by_name.get(name)
            if b is None or not b.parent:
                depth_cache[name] = 0
                return 0
            depth_cache[name] = 1 + bone_depth(b.parent)
            return depth_cache[name]

        bone_order = sorted([b.name for b in model.skeleton], key=bone_depth)

        ctx.view_layer.objects.active = arm_obj
        arm_obj.animation_data_create()

                                                     
        rest_abs_by_name: Dict[str, Matrix] = {}
        rest_rel_by_name: Dict[str, Matrix] = {}
        for pb in arm_obj.pose.bones:
            rest_abs_by_name[pb.name] = pb.bone.matrix_local.copy()
        for name in bone_order:
            rb = rest_by_name.get(name)
            rest_abs = rest_abs_by_name.get(name)
            if rb is None or rest_abs is None:
                continue
            if rb.parent and rb.parent in rest_abs_by_name:
                parent_rest_abs = rest_abs_by_name[rb.parent]
                rest_rel_by_name[name] = parent_rest_abs.inverted() @ rest_abs
            else:
                rest_rel_by_name[name] = rest_abs

                                                                                                    
        _gf_runtime_cache_armature(
            arm_obj,
            model=model,
            motions=motions,
            conv=conv,
            conv3=conv3,
            global_scale=global_scale,
            bone_order=bone_order,
            rest_by_name=rest_by_name,
            rest_rel_by_name=rest_rel_by_name,
        )
        try:
            arm_obj.gfmodel_runtime_motion_index = 0
        except Exception:
            pass

        for mot in motions:
            if mot.frames_count <= 0:
                print(f"[GFModel] Skip motion {mot.index}: frames_count=0")
                continue
            if not mot.bones:
                                                                              
                print(
                    f"[GFModel] Skip skeletal action for motion {mot.index}: bones=0 (uv={len(mot.uv_transforms)})"
                )
                continue

            action = bpy.data.actions.new(name=f"{model.name}_Motion_{mot.index}")
            arm_obj.animation_data.action = action
            created_actions[mot.index] = action
            arm_obj.rotation_mode = "QUATERNION"
            arm_obj.scale = (1.0, 1.0, 1.0)

            for pb in arm_obj.pose.bones:
                pb.rotation_mode = "QUATERNION"

            bt_by_name = {bt.name: bt for bt in mot.bones}

            for frame in range(mot.frames_count):
                bpy.context.scene.frame_set(frame)
                pose_mats: Dict[str, Matrix] = {}
                for name in bone_order:
                    pb = arm_obj.pose.bones.get(name)
                    rb = rest_by_name.get(name)
                    if pb is None or rb is None:
                        continue
                    bt = bt_by_name.get(name)

                    sx = _mot_eval(bt.sx, frame, rb.scale.x) if bt else rb.scale.x
                    sy = _mot_eval(bt.sy, frame, rb.scale.y) if bt else rb.scale.y
                    sz = _mot_eval(bt.sz, frame, rb.scale.z) if bt else rb.scale.z

                    tx = (
                        _mot_eval(bt.tx, frame, rb.translation.x)
                        if bt
                        else rb.translation.x
                    )
                    ty = (
                        _mot_eval(bt.ty, frame, rb.translation.y)
                        if bt
                        else rb.translation.y
                    )
                    tz = (
                        _mot_eval(bt.tz, frame, rb.translation.z)
                        if bt
                        else rb.translation.z
                    )

                    rx = _mot_eval(bt.rx, frame, rb.rotation.x) if bt else rb.rotation.x
                    ry = _mot_eval(bt.ry, frame, rb.rotation.y) if bt else rb.rotation.y
                    rz = _mot_eval(bt.rz, frame, rb.rotation.z) if bt else rb.rotation.z

                    if bt and bt.is_axis_angle:
                        axis = Vector((rx, ry, rz))
                        angle = axis.length * 2.0
                        if angle > 0:
                            q_cur = Quaternion(axis.normalized(), angle)
                        else:
                            q_cur = Quaternion((1.0, 0.0, 0.0, 0.0))
                    else:
                        q_cur = _euler_to_quat_xyz(Vector((rx, ry, rz)))

                    q_cur_t = _transform_quat_basis(q_cur, conv3)
                    t_anim = Matrix.Translation(
                        conv @ (Vector((tx, ty, tz)) * global_scale)
                    )
                    r_anim = q_cur_t.to_matrix().to_4x4()
                    s_anim = Matrix.Diagonal(Vector((sx, sy, sz, 1.0)))
                    local_mat = t_anim @ r_anim @ s_anim

                    parent_name = rb.parent
                    if parent_name and parent_name in pose_mats:
                        if _bone_uses_ssc(int(rb.flags)):
                                                                            
                                                                                                                                                 
                            prb = rest_by_name.get(parent_name)
                            pbt = bt_by_name.get(parent_name)
                            psx = (
                                _mot_eval(pbt.sx, frame, prb.scale.x)
                                if (pbt and prb)
                                else (prb.scale.x if prb else 1.0)
                            )
                            psy = (
                                _mot_eval(pbt.sy, frame, prb.scale.y)
                                if (pbt and prb)
                                else (prb.scale.y if prb else 1.0)
                            )
                            psz = (
                                _mot_eval(pbt.sz, frame, prb.scale.z)
                                if (pbt and prb)
                                else (prb.scale.z if prb else 1.0)
                            )

                            inv_psx = 1.0 / psx if psx != 0 else 0.0
                            inv_psy = 1.0 / psy if psy != 0 else 0.0
                            inv_psz = 1.0 / psz if psz != 0 else 0.0
                            inv_s_parent = Matrix.Diagonal(
                                Vector((inv_psx, inv_psy, inv_psz, 1.0))
                            )

                            t_scaled = Vector((tx * psx, ty * psy, tz * psz))
                            t_anim = Matrix.Translation(
                                conv @ (t_scaled * global_scale)
                            )
                            local_mat_ssc = t_anim @ r_anim @ s_anim

                            pose_mat = (
                                pose_mats[parent_name] @ inv_s_parent @ local_mat_ssc
                            )
                        else:
                            pose_mat = pose_mats[parent_name] @ local_mat
                    else:
                        pose_mat = local_mat

                    pose_mats[name] = pose_mat

                                                                                           
                                                                      
                                                                      
                                                          
                    if (
                        rb.parent
                        and rb.parent in pose_mats
                        and rb.parent in rest_abs_by_name
                    ):
                        pose_local = pose_mats[rb.parent].inverted() @ pose_mat
                    else:
                        pose_local = pose_mat
                    rest_local = rest_rel_by_name.get(name)
                    if rest_local is None:
                        rest_local = pb.bone.matrix_local.copy()
                    pb.matrix_basis = rest_local.inverted() @ pose_local
                    pb.keyframe_insert(data_path="location", frame=frame)
                    pb.keyframe_insert(data_path="rotation_quaternion", frame=frame)
                    pb.keyframe_insert(data_path="scale", frame=frame)

                                                                      
                dbg_enabled = bool(
                    getattr(ctx.scene, "gfmodel_debug_animations", False)
                )
                dbg_motion = int(getattr(ctx.scene, "gfmodel_debug_motion", -1))
                if (
                    dbg_enabled
                    and (dbg_motion < 0 or dbg_motion == mot.index)
                    and frame < 8
                ):
                    dbg_name = getattr(ctx.scene, "gfmodel_debug_bone", "Waist")
                    m = pose_mats.get(dbg_name)
                    if m is not None:
                        loc, rot, sca = m.decompose()
                        print(
                            f"[GFModel][AnimDebug] mot={mot.index} frame={frame} bone={dbg_name} "
                            f"loc=({loc.x:.4f},{loc.y:.4f},{loc.z:.4f}) "
                            f"rot=({rot.w:.4f},{rot.x:.4f},{rot.y:.4f},{rot.z:.4f}) "
                            f"sca=({sca.x:.4f},{sca.y:.4f},{sca.z:.4f})"
                        )

                                                                 
        try:
            arm_obj.animation_data.action = None
        except Exception:
            pass

                                              
        try:
            bpy.context.scene.frame_set(0)
        except Exception:
            pass
        for pb in arm_obj.pose.bones:
            pb.location = (0.0, 0.0, 0.0)
            pb.rotation_mode = "QUATERNION"
            pb.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
            pb.scale = (1.0, 1.0, 1.0)
        try:
            bpy.context.view_layer.update()
        except Exception:
            pass
    if import_material_animations and motions:
        for mot in motions:
            if not mot.uv_transforms or mot.frames_count <= 0:
                continue
                                                                                          
            mats_in_motion = {uv.name for uv in mot.uv_transforms}
            for mat_name in mats_in_motion:
                mat = mats_by_name.get(mat_name)
                if mat is None or mat.node_tree is None:
                    continue
                nt = mat.node_tree
                nt.animation_data_create()
                action = bpy.data.actions.new(
                    name=f"{model.name}_UV_{mot.index}_{mat.name}"
                )
                nt.animation_data.action = action
                                                                                                      
                mapping_defaults: Dict[
                    str,
                    Tuple[
                        Tuple[float, float, float],
                        Tuple[float, float, float],
                        Tuple[float, float, float],
                    ],
                ] = {}
                for uv in (u for u in mot.uv_transforms if u.name == mat_name):
                    mapping = mat.node_tree.nodes.get(f"GF_MAPPING_{uv.unit_index}")
                    if mapping is None:
                        continue
                    loc = tuple(mapping.inputs["Location"].default_value)
                    rotv = tuple(mapping.inputs["Rotation"].default_value)
                    scv = tuple(mapping.inputs["Scale"].default_value)
                    mapping_defaults[mapping.name] = (loc, rotv, scv)
                for frame in range(mot.frames_count):
                    bpy.context.scene.frame_set(frame)
                    for uv in (u for u in mot.uv_transforms if u.name == mat_name):
                        mapping = mat.node_tree.nodes.get(f"GF_MAPPING_{uv.unit_index}")
                        if mapping is None:
                            continue

                        sx = _mot_eval(uv.sx, frame, 1.0)
                        sy = _mot_eval(uv.sy, frame, 1.0)
                        rot = _mot_eval(uv.rot, frame, 0.0)
                        tx = _mot_eval(uv.tx, frame, 0.0)
                        ty = _mot_eval(uv.ty, frame, 0.0)

                        loc = mapping.inputs["Location"].default_value
                        loc[0] = tx
                        loc[1] = ty
                        mapping.inputs["Location"].default_value = loc

                        rvec = mapping.inputs["Rotation"].default_value
                        rvec[2] = rot
                        mapping.inputs["Rotation"].default_value = rvec

                        sc = mapping.inputs["Scale"].default_value
                        sc[0] = sx
                        sc[1] = sy
                        mapping.inputs["Scale"].default_value = sc

                        mapping.inputs["Location"].keyframe_insert(
                            data_path="default_value", frame=frame
                        )
                        mapping.inputs["Rotation"].keyframe_insert(
                            data_path="default_value", frame=frame
                        )
                        mapping.inputs["Scale"].keyframe_insert(
                            data_path="default_value", frame=frame
                        )

                                                                                                                 
                mat["gfmodel_has_uv_anims"] = True
                mat["gfmodel_uv_action"] = action.name
                nt.animation_data.action = None
                                           
                for node_name, (loc, rotv, scv) in mapping_defaults.items():
                    node = mat.node_tree.nodes.get(node_name)
                    if node is None:
                        continue
                    node.inputs["Location"].default_value = loc                            
                    node.inputs["Rotation"].default_value = rotv                            
                    node.inputs["Scale"].default_value = scv                            

                                                                         
        _apply_uv_anim_enable(ctx.scene)

    if import_visibility_animations and motions:
                                                                                   
        for mot in motions:
            if not mot.visibility_tracks or mot.frames_count <= 0:
                continue
            for track in mot.visibility_tracks:
                obj = (
                    mesh_objs_by_sm_name.get(track.name)
                    or bpy.data.objects.get(f"{model.name}_{track.name}")
                    or bpy.data.objects.get(track.name)
                )
                if obj is None:
                    continue
                obj.animation_data_create()
                action = bpy.data.actions.new(
                    name=f"{model.name}_VIS_{mot.index}_{obj.name}"
                )
                obj.animation_data.action = action
                hide_default = (bool(obj.hide_viewport), bool(obj.hide_render))

                for frame in range(mot.frames_count + 1):
                    bpy.context.scene.frame_set(frame)
                    visible = (
                        bool(track.values[frame])
                        if frame < len(track.values)
                        else bool(track.values[-1])
                    )
                    obj.hide_viewport = not visible
                    obj.hide_render = not visible
                    obj.keyframe_insert(data_path="hide_viewport", frame=frame)
                    obj.keyframe_insert(data_path="hide_render", frame=frame)

                obj["gfmodel_has_vis_anims"] = True
                obj["gfmodel_vis_action"] = action.name
                obj.animation_data.action = None
                obj.hide_viewport, obj.hide_render = hide_default

        _apply_visibility_anim_enable(ctx.scene)


def _import_gfmodel_bytes(
    context: bpy.types.Context,
    data: bytes,
    *,
    source_path: str,
    import_textures: bool,
    import_animations: bool,
    import_material_animations: bool = True,
    import_visibility_animations: bool = True,
    global_scale: float = 1.0,
    axis_forward: str = "-Z",
    axis_up: str = "Y",
) -> bool:
                                                                                     
                                                                                
    try:
        context.scene["gfmodel_last_import_source"] = str(source_path)
        context.scene["gfmodel_last_import_breadcrumb"] = str(source_path)
    except Exception:
        pass

                                                                                         
                                                                                        
                                
    source_path_real = str(source_path)
    try:
        if not os.path.isfile(source_path_real):
            import hashlib
            import tempfile

            h = hashlib.md5(data).hexdigest()[:12]
            tmp_root = ""
            try:
                tmp_root = str(getattr(bpy.app, "tempdir", "") or "").strip()
            except Exception:
                tmp_root = ""
            if not tmp_root:
                tmp_root = tempfile.gettempdir()
            base = os.path.join(tmp_root, "gfmodel_imports")
            os.makedirs(base, exist_ok=True)
            source_path_real = os.path.join(base, f"import_{h}.bin")
            if (not os.path.exists(source_path_real)) or (
                os.path.getsize(source_path_real) != len(data)
            ):
                with open(source_path_real, "wb") as f:
                    f.write(data)
    except Exception:
                                                                    
        source_path_real = str(source_path)

    models, textures, motions, shaders = _load_any(data)
    return _import_gfmodel_loaded(
        context,
        models=models,
        textures=textures,
        motions=motions,
        shaders=shaders,
        source_path=source_path_real,
        import_textures=import_textures,
        import_animations=import_animations,
        import_material_animations=import_material_animations,
        import_visibility_animations=import_visibility_animations,
        global_scale=global_scale,
        axis_forward=axis_forward,
        axis_up=axis_up,
    )


def _import_gfmodel_loaded(
    context: bpy.types.Context,
    *,
    models: List[_GFModel],
    textures: List[_GFTexture],
    motions: List[_GFMotion],
    shaders: List[_GFShader],
    source_path: str,
    import_textures: bool,
    import_animations: bool,
    import_material_animations: bool = True,
    import_visibility_animations: bool = True,
    global_scale: float = 1.0,
    axis_forward: str = "-Z",
    axis_up: str = "Y",
) -> bool:
                                                            
    try:
        sp = str(source_path)
        if os.path.isfile(sp):
            context.scene["gfmodel_last_import_path"] = sp
                                                                              
            bc = str(context.scene.get("gfmodel_last_import_breadcrumb", "")).strip()
            if not bc:
                bc = sp
            context.scene["gfmodel_last_import_source"] = bc
            context.scene["gfmodel_last_import_breadcrumb"] = bc
        else:
            context.scene["gfmodel_last_import_source"] = sp
            context.scene["gfmodel_last_import_breadcrumb"] = sp
        context.scene["gfmodel_last_axis_forward"] = str(axis_forward)
        context.scene["gfmodel_last_axis_up"] = str(axis_up)
        context.scene["gfmodel_last_global_scale"] = float(global_scale)
    except Exception:
        pass

                                                                                         
    for idx, mot in enumerate(motions):
        mot.index = idx

    print(
        f"[GFModel] Loaded: models={len(models)} textures={len(textures)} motions={len(motions)} shaders={len(shaders)}"
    )
    if motions:
        for mot in motions[:10]:
            print(
                f"[GFModel] Motion {mot.index}: frames={mot.frames_count} bones={len(mot.bones)} uv={len(mot.uv_transforms)} vis={len(mot.visibility_tracks)}"
            )

    if not models:
        return False

    for i, model in enumerate(models):
        _import_model_to_blender(
            context,
            model,
            textures,
            motions if i == 0 else [],
            shaders,
            import_textures=bool(import_textures),
            import_animations=bool(import_animations) and i == 0,
            import_material_animations=bool(import_material_animations) and i == 0,
            import_visibility_animations=bool(import_visibility_animations) and i == 0,
            global_scale=float(global_scale),
            axis_forward=str(axis_forward),
            axis_up=str(axis_up),
        )
                                                                           
    _apply_uv_anim_enable(context.scene)
    return True


class IMPORT_SCENE_OT_gfmodel(bpy.types.Operator, ImportHelper):
    bl_idname = "import_scene.gfmodel"
    bl_label = "Import GFModel (GFL2)"
    bl_options = {"UNDO"}

    filename_ext = ""
                                                                                                
    filter_glob: StringProperty(default="*", options={"HIDDEN"})

    import_textures: BoolProperty(name="Import Textures", default=True)
    import_animations: BoolProperty(name="Import Skeletal Animations", default=True)
    import_material_animations: BoolProperty(
        name="Import Material (UV) Animations",
        default=True,
        description="Bake UV/material animations into muted NLA strips on materials (toggle in GFModel panel)",
    )
    import_visibility_animations: BoolProperty(
        name="Import Visibility Animations",
        default=True,
        description="Bake mesh visibility tracks into muted per-object actions (toggle in GFModel panel)",
    )
    debug_animations: BoolProperty(
        name="Debug Animations (Console)",
        default=False,
        description="Print extra info about bone transforms during import",
    )
    debug_bone: StringProperty(
        name="Debug Bone",
        default="Waist",
        description="Bone name to print (when Debug Animations is enabled)",
    )
    debug_motion: bpy.props.IntProperty(                              
        name="Debug Motion Index",
        default=-1,
        description="If >= 0, only print debug for this motion index",
    )
    set_active_action: BoolProperty(
        name="Set Active Action",
        default=False,
        description="Assign a selected skeletal action to the armature after import",
    )
    active_action_index: bpy.props.IntProperty(                              
        name="Active Motion Index",
        default=0,
        min=0,
        description="Motion index to set active when Set Active Action is enabled",
    )
    global_scale: FloatProperty(name="Scale", default=1.0, min=0.0001, max=1000.0)

    axis_forward: EnumProperty(
        name="Forward",
        items=[(a, a, "") for a in ("X", "Y", "Z", "-X", "-Y", "-Z")],
        default="-Z",
    )
    axis_up: EnumProperty(
        name="Up",
        items=[(a, a, "") for a in ("X", "Y", "Z", "-X", "-Y", "-Z")],
        default="Y",
    )

    def execute(self, context: bpy.types.Context):
                                                                                    
        prev_dbg = bool(getattr(context.scene, "gfmodel_debug_animations", False))
        prev_dbg_bone = str(getattr(context.scene, "gfmodel_debug_bone", "Waist"))
        prev_dbg_motion = int(getattr(context.scene, "gfmodel_debug_motion", -1))
        if self.debug_animations:
            context.scene.gfmodel_debug_animations = True
            context.scene.gfmodel_debug_bone = self.debug_bone
            context.scene.gfmodel_debug_motion = self.debug_motion
        with open(self.filepath, "rb") as f:
            data = f.read()
        ok = _import_gfmodel_bytes(
            context,
            data,
            source_path=str(self.filepath),
            import_textures=bool(self.import_textures),
            import_animations=bool(self.import_animations),
            import_material_animations=bool(self.import_material_animations),
            import_visibility_animations=bool(self.import_visibility_animations),
            global_scale=float(self.global_scale),
            axis_forward=str(self.axis_forward),
            axis_up=str(self.axis_up),
        )
        if not ok:
            self.report({"ERROR"}, "No GFModel content found in file")
            return {"CANCELLED"}
        if self.debug_animations:
            context.scene.gfmodel_debug_animations = prev_dbg
            context.scene.gfmodel_debug_bone = prev_dbg_bone
            context.scene.gfmodel_debug_motion = prev_dbg_motion
        return {"FINISHED"}


def menu_func_import(self, context):
    self.layout.operator(
        IMPORT_SCENE_OT_gfmodel.bl_idname, text="GFModel (GFL2) (.bin/CP/CM)"
    )


_CLASSES = (IMPORT_SCENE_OT_gfmodel,)


def register() -> None:
    for c in _CLASSES:
        bpy.utils.register_class(c)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)


def unregister() -> None:
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
    for c in reversed(_CLASSES):
        bpy.utils.unregister_class(c)
