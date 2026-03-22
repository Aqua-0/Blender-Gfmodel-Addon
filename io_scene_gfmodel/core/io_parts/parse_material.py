
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ..math_compat import Vector
from ..pica import (
    _decode_rgba_u32,
    _decode_texenv_stage_from_regs,
    _parse_pica_vec_float24,
    _pica_read_commands,
)
from ..types import (
    _GFLUT,
    _GFMaterial,
    _GFTextureUnit,
)
from .reader import (
    _BinReader,
    _gf_read_hash_name,
    _gf_read_vec2,
    _gf_read_vec3,
    _gf_read_vec4,
    _gf_skip_padding16,
    _read_gf_section,
)

def _parse_gf_material(data: bytes, offset: int) -> Tuple[_GFMaterial, int]:
    r = _BinReader(data, offset)
    _sect_magic, sect_len = _read_gf_section(r)
    start = r.tell
    sect_total_len = 0x10 + int(sect_len)
    raw_blob = data[offset : offset + sect_total_len]

    mat_name = _gf_read_hash_name(r)
    shader_name = _gf_read_hash_name(r)
    vtx_shader = _gf_read_hash_name(r)
    frag_shader = _gf_read_hash_name(r)

    lut_hashes = (r.u32(), r.u32(), r.u32())
    r.skip(4)

    bump_texture = r.s8()
    const_assignments = [r.u8() for _ in range(6)]
    r.skip(1)


    colors_rgba = [tuple(r.u8() for _ in range(4)) for _ in range(12)]
    edge_type = int(r.u32())
    id_edge_enable = int(r.u32())
    edge_id = int(r.u32())
    projection_type = int(r.u32())
    rim_pow = float(r.f32())
    rim_scale = float(r.f32())
    phong_pow = float(r.f32())
    phong_scale = float(r.f32())
    id_edge_offset_enable = int(r.u32())
    edge_map_alpha_mask = int(r.u32())
    bake_ops = [int(r.u32()) for _ in range(9)]
    vertex_shader_type = int(r.u32())
    shader_params = tuple(float(r.f32()) for _ in range(4))

    units_count = r.u32()
    tex_names: List[Optional[str]] = [None, None, None]
    tex_units: List[_GFTextureUnit] = []
    for unit in range(units_count):
        tex_name = _gf_read_hash_name(r)
        if unit < 3:
            tex_names[unit] = tex_name
        unit_index = r.u8()
        mapping_type = r.u8()
        scale = _gf_read_vec2(r)
        rotation = r.f32()
        translation = _gf_read_vec2(r)
        sampler_words = [r.u32() for _ in range(5)]
        tex_units.append(
            _GFTextureUnit(
                name=tex_name,
                unit_index=unit_index,
                mapping_type=mapping_type,
                scale=scale,
                rotation=rotation,
                translation=translation,
                sampler_words=sampler_words,
            )
        )

    _gf_skip_padding16(r)

    commands_len = r.u32()
    render_priority = r.u32()
    unk_render = r.u32()
    render_layer = r.u32()
    header_hashes = (r.u32(), r.u32(), r.u32(), r.u32())
    commands_u32 = [r.u32() for _ in range(commands_len // 4)]

    alpha_test_enabled = False
    alpha_test_ref = 0.0
    alpha_test_func = 1
    blend_func: Optional[dict] = None
    blend_color_rgba: Optional[Tuple[int, int, int, int]] = None
    stencil_test: Optional[dict] = None
    stencil_op: Optional[dict] = None
    depth_test_enabled: Optional[bool] = None
    depth_test_func: Optional[int] = None
    depth_write: Optional[bool] = None
    color_write_mask: Optional[Tuple[bool, bool, bool, bool]] = None
    face_culling: Optional[int] = None
    regs: Dict[int, int] = {}

    for reg, params in _pica_read_commands(commands_u32):
        p = params[0]
        regs[reg] = p
        if reg == 0x0040:
            face_culling = int(p & 3)
        elif reg == 0x0104:
            alpha_test_enabled = (p & 1) != 0
            ref = (p >> 8) & 0xFF
            alpha_test_ref = float(ref) / 255.0
            alpha_test_func = (p >> 4) & 7
        elif reg == 0x0101:
            blend_func = {
                "color_eq": (p >> 0) & 7,
                "alpha_eq": (p >> 8) & 7,
                "color_src": (p >> 16) & 0xF,
                "color_dst": (p >> 20) & 0xF,
                "alpha_src": (p >> 24) & 0xF,
                "alpha_dst": (p >> 28) & 0xF,
            }
        elif reg == 0x0103:
            blend_color_rgba = (
                int((p >> 0) & 0xFF),
                int((p >> 8) & 0xFF),
                int((p >> 16) & 0xFF),
                int((p >> 24) & 0xFF),
            )
        elif reg == 0x0105:
            stencil_test = {
                "enabled": (p & 1) != 0,
                "func": int((p >> 4) & 7),
                "buffer_mask": int((p >> 8) & 0xFF),
                "ref": int((p >> 16) & 0xFF),
                "mask": int((p >> 24) & 0xFF),
            }
        elif reg == 0x0106:
            stencil_op = {
                "fail": int((p >> 0) & 7),
                "zfail": int((p >> 4) & 7),
                "zpass": int((p >> 8) & 7),
            }
        elif reg == 0x0107:
            depth_test_enabled = (p & 1) != 0
            depth_test_func = int((p >> 4) & 7)
            color_write_mask = (
                (p & 0x0100) != 0,
                (p & 0x0200) != 0,
                (p & 0x0400) != 0,
                (p & 0x0800) != 0,
            )
            depth_write = (p & 0x1000) != 0

    end = start + sect_len
    r.seek(end)
    return _GFMaterial(
        name=mat_name,
        shader_name=shader_name,
        vtx_shader=vtx_shader,
        frag_shader=frag_shader,
        lut_hashes=lut_hashes,
        tex0=tex_names[0],
        tex1=tex_names[1],
        tex2=tex_names[2],
        tex_units=tex_units,
        bump_texture=bump_texture,
        edge_type=edge_type,
        id_edge_enable=id_edge_enable,
        edge_id=edge_id,
        projection_type=projection_type,
        rim_pow=rim_pow,
        rim_scale=rim_scale,
        phong_pow=phong_pow,
        phong_scale=phong_scale,
        id_edge_offset_enable=id_edge_offset_enable,
        edge_map_alpha_mask=edge_map_alpha_mask,
        bake_ops=bake_ops,
        vertex_shader_type=vertex_shader_type,
        shader_params=shader_params,
        const_assignments=const_assignments,
        colors_rgba=colors_rgba,
        alpha_test_enabled=alpha_test_enabled,
        alpha_test_ref=alpha_test_ref,
        alpha_test_func=alpha_test_func,
        blend_func=blend_func,
        blend_color_rgba=blend_color_rgba,
        stencil_test=stencil_test,
        stencil_op=stencil_op,
        depth_test_enabled=depth_test_enabled,
        depth_test_func=depth_test_func,
        depth_write=depth_write,
        color_write_mask=color_write_mask,
        face_culling=face_culling,
        render_priority=int(render_priority),
        render_layer=int(render_layer),
        header_hashes=header_hashes,
        unk_render=int(unk_render),
        pica_commands=commands_u32,
        pica_regs=regs,
        raw_blob=raw_blob,
    ), r.tell
