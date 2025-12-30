"""Binary/container parsing for GFModel-related formats.

This module handles:
- CP/CM (2-byte) container wrappers
- Optional LZ11 decompression
- GFModel pack/model/texture/material/shader parsing
- GFMotion parsing (skeletal + UV/material tracks)

No Blender registration logic lives here.
"""

from __future__ import annotations

import struct
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .math_compat import Vector
from .motion import _parse_gf_motion
from .pica import (
    _decode_rgba_u32,
    _decode_texenv_stage_from_regs,
    _parse_pica_vec_float24,
    _pica_read_commands,
)
from .types import (
    _GFLUT,
    _GFBone,
    _GFMaterial,
    _GFModel,
    _GFMotion,
    _GFShader,
    _GFSubMesh,
    _GFTexture,
    _GFTextureUnit,
    _PICAAttribute,
    _PICAFixedAttribute,
)


class _BinReader:
    __slots__ = ("_b", "_o")

    def __init__(self, data: bytes, offset: int = 0):
        self._b = memoryview(data)
        self._o = offset

    @property
    def tell(self) -> int:
        return self._o

    def seek(self, offset: int) -> None:
        if offset < 0 or offset > len(self._b):
            raise ValueError("seek out of range")
        self._o = offset

    def skip(self, size: int) -> None:
        self.seek(self._o + size)

    def read(self, size: int) -> bytes:
        o = self._o
        n = o + size
        if n > len(self._b):
            raise EOFError("read past end")
        self._o = n
        return self._b[o:n].tobytes()

    def u8(self) -> int:
        v = self._b[self._o]
        self._o += 1
        return int(v)

    def s8(self) -> int:
        v = struct.unpack_from("<b", self._b, self._o)[0]
        self._o += 1
        return int(v)

    def u16(self) -> int:
        v = struct.unpack_from("<H", self._b, self._o)[0]
        self._o += 2
        return int(v)

    def s16(self) -> int:
        v = struct.unpack_from("<h", self._b, self._o)[0]
        self._o += 2
        return int(v)

    def u32(self) -> int:
        v = struct.unpack_from("<I", self._b, self._o)[0]
        self._o += 4
        return int(v)

    def s32(self) -> int:
        v = struct.unpack_from("<i", self._b, self._o)[0]
        self._o += 4
        return int(v)

    def f32(self) -> float:
        v = struct.unpack_from("<f", self._b, self._o)[0]
        self._o += 4
        return float(v)

    def padded_string(self, length: int) -> str:
        raw = self.read(length)
        end = raw.find(b"\0")
        if end >= 0:
            raw = raw[:end]
        try:
            return raw.decode("ascii", "replace")
        except Exception:
            return ""

    def byte_len_string(self) -> str:
        return self.padded_string(self.u8())

    def int_len_string(self) -> str:
        return self.padded_string(self.s32())

    def align(self, boundary: int) -> None:
        mask = boundary - 1
        if (self._o & mask) != 0:
            self._o += boundary - (self._o & mask)


def _lzss_ninty_decompress(data: bytes) -> bytes:
    if len(data) < 4:
        return data
    hdr = struct.unpack_from("<I", data, 0)[0]
    if (hdr & 0xFF) != 0x11:
        return data
    decoded_len = hdr >> 8
    inp = memoryview(data)[4:]
    in_off = 0
    out = bytearray(decoded_len)
    out_off = 0

    mask = 0
    header = 0
    while out_off < decoded_len:
        mask >>= 1
        if mask == 0:
            header = inp[in_off]
            in_off += 1
            mask = 0x80

        if (header & mask) == 0:
            out[out_off] = inp[in_off]
            out_off += 1
            in_off += 1
            continue

        byte1 = int(inp[in_off])
        in_off += 1
        top = byte1 >> 4

        if top == 0:
            byte2 = int(inp[in_off])
            byte3 = int(inp[in_off + 1])
            in_off += 2
            position = ((byte2 & 0xF) << 8) | byte3
            length = (((byte1 & 0xF) << 4) | (byte2 >> 4)) + 0x11
        elif top == 1:
            byte2 = int(inp[in_off])
            byte3 = int(inp[in_off + 1])
            byte4 = int(inp[in_off + 2])
            in_off += 3
            position = ((byte3 & 0xF) << 8) | byte4
            length = (((byte1 & 0xF) << 12) | (byte2 << 4) | (byte3 >> 4)) + 0x111
        else:
            byte2 = int(inp[in_off])
            in_off += 1
            position = ((byte1 & 0xF) << 8) | byte2
            length = (byte1 >> 4) + 1

        position += 1
        for _ in range(length):
            out[out_off] = out[out_off - position]
            out_off += 1
            if out_off >= decoded_len:
                break

    return bytes(out)


def _looks_like_lz11(data: bytes) -> bool:
                                                                               
                                                                                                  
    if len(data) < 4 or data[0] != 0x11:
        return False
    decoded_len = data[1] | (data[2] << 8) | (data[3] << 16)
    if decoded_len <= 0:
        return False
                                                                            
    if decoded_len <= len(data):
        return False
    return True


def _gf_skip_padding16(r: _BinReader) -> None:
    r.align(0x10)


def _gf_read_hash_name(r: _BinReader) -> str:
    _ = r.u32()        
    return r.byte_len_string()


def _gf_read_vec2(r: _BinReader) -> Vector:
    return Vector((r.f32(), r.f32()))


def _gf_read_vec3(r: _BinReader) -> Vector:
    return Vector((r.f32(), r.f32(), r.f32()))


def _gf_read_vec4(r: _BinReader) -> Vector:
    return Vector((r.f32(), r.f32(), r.f32(), r.f32()))


def _read_gf_section(r: _BinReader) -> Tuple[str, int]:
    magic = r.padded_string(8)
    length = r.u32()
    _ = r.u32()
    return magic, length


def _parse_gf_texture(data: bytes) -> _GFTexture:
    r = _BinReader(data)
    _magic = r.u32()              
    _count = r.u32()
    _sect_magic, _sect_len = _read_gf_section(r)
    tex_len = r.u32()
    r.skip(0x0C)
    name = r.padded_string(0x40)
    width = r.u16()
    height = r.u16()
    fmt = r.u16()
    _mipmap = r.u16()
    r.skip(0x10)
    raw = r.read(tex_len)
    return _GFTexture(name=name, width=width, height=height, fmt=fmt, raw=raw)


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


def _parse_gf_shader(data: bytes, offset: int) -> Tuple[_GFShader, int]:
    r = _BinReader(data, offset)
    _magic = r.u32()
    _count = r.u32()
    _gf_skip_padding16(r)
    _sect_magic, sect_len = _read_gf_section(r)
    start = r.tell

    name = r.padded_string(0x40)
    r.skip(4)        
    r.skip(4)         
    _gf_skip_padding16(r)

    cmd_len = r.u32()
    r.skip(4)                  
    r.skip(4)                 
    r.skip(4)           
    r.padded_string(0x40)            

    cmd_words = [r.u32() for _ in range(cmd_len // 4)]

    regs: Dict[int, int] = {}
    for reg, params in _pica_read_commands(cmd_words):
        regs[reg] = params[0]

    stages = [_decode_texenv_stage_from_regs(i, regs) for i in range(6)]
    texenv_buf = regs.get(0x00FD)
    texenv_update = regs.get(0x00E0)

    end = start + sect_len
    r.seek(end)
    return _GFShader(
        name=name,
        texenv_stages=stages,
        texenv_buffer_color=texenv_buf,
        texenv_update_buffer=texenv_update,
        pica_commands=cmd_words,
        pica_regs=regs,
    ), r.tell


def _parse_gf_bone(data: bytes, offset: int) -> Tuple[_GFBone, int]:
    r = _BinReader(data, offset)
    name = r.byte_len_string()
    parent = r.byte_len_string()
    flags = r.u8()
    scale = _gf_read_vec3(r)
    rotation = _gf_read_vec3(r)
    translation = _gf_read_vec3(r)
    return _GFBone(
        name=name,
        parent=parent,
        flags=flags,
        scale=scale,
        rotation=rotation,
        translation=translation,
    ), r.tell


def _parse_gf_mesh(data: bytes, offset: int) -> Tuple[List[_GFSubMesh], int]:
    r = _BinReader(data, offset)
    mesh_section_off = int(offset)
    mesh_section_len_off = int(offset) + 8
    _sect_magic, sect_len = _read_gf_section(r)
    start = r.tell

    _name_hash = r.u32()
    mesh_name = r.padded_string(0x40)
    mesh_is_blend_shape = r.u32() != 0
    mesh_bbox_min = _gf_read_vec4(r)
    mesh_bbox_max = _gf_read_vec4(r)

    mesh_face_count = int(r.u32())
    mesh_weight_max = int(r.u32())

                                                                                               
                                                                                              
                                                                       
    mesh_hdr_end = start + 0x80
    sect_end = start + int(sect_len)
    if r.tell < mesh_hdr_end and mesh_hdr_end <= sect_end:
        r.seek(mesh_hdr_end)

    cmd_lists: List[List[int]] = []
    cmd_list_u32_offs: List[int] = []
    cmd_list_lens_u32: List[int] = []
    while True:
        commands_len = r.u32()
        cmd_index = r.u32()
        commands_count = r.u32()
        _pad = r.u32()
        cmd_list_u32_offs.append(int(r.tell))
        cmd_list_lens_u32.append(int(commands_len // 4))
                                                                                         
                                                                               
        cmd_u32 = [int(r.u32()) for _ in range(int(commands_len // 4))]
        cmd_lists.append(cmd_u32)
        if cmd_index >= commands_count - 1:
            break

    sub_infos: List[
        Tuple[str, int, List[int], int, int, int, int, int, int, int, int]
    ] = []
    for face_index in range(mesh_face_count):
        _sm_hash = r.u32()
        sm_name = r.int_len_string()
        bone_count = r.u8()
        bone_indices = [r.u8() for _ in range(0x1F)]
        vtx_count_off = int(r.tell)
        vtx_count = r.s32()
        idx_count_off = int(r.tell)
        idx_count = r.s32()
        vtx_len_off = int(r.tell)
        vtx_len = r.s32()
        idx_len_off = int(r.tell)
        idx_len = r.s32()
        sub_infos.append(
            (
                sm_name,
                bone_count,
                bone_indices,
                vtx_count,
                idx_count,
                vtx_len,
                idx_len,
                int(idx_count_off),
                int(vtx_count_off),
                int(vtx_len_off),
                int(idx_len_off),
            )
        )

    scales = (1.0 / 127.0, 1.0 / 255.0, 1.0 / 32767.0, 1.0)

    submeshes: List[_GFSubMesh] = []
                                                                                             
    mesh_index = -1
    for face_index, info in enumerate(sub_infos):
        (
            sm_name,
            bone_count,
            bone_indices,
            _vcount,
            _icount,
            vtx_len,
            idx_len,
            idx_count_off,
            vtx_count_off,
            vtx_len_off,
            idx_len_off,
        ) = info
        vcount, icount = int(_vcount), int(_icount)

        enable_cmds = cmd_lists[face_index * 3 + 0]
        disable_cmds = cmd_lists[face_index * 3 + 1]
        index_cmds = cmd_lists[face_index * 3 + 2]
        index_cmds_off = 0
        index_cmds_len_u32 = 0
        try:
            ci = int(face_index) * 3 + 2
            index_cmds_off = int(cmd_list_u32_offs[ci])
            index_cmds_len_u32 = int(cmd_list_lens_u32[ci])
        except Exception:
            index_cmds_off = 0
            index_cmds_len_u32 = 0

        fixed_words: List[Tuple[int, int, int]] = [(0, 0, 0)] * 12
        buffer_formats = 0
        buffer_attributes = 0
        buffer_perm = 0
        attrs_total = 0
        attrs_count = 0
        fixed_index = 0
        vertex_stride = 0

        for reg, params in _pica_read_commands(enable_cmds):
            p = params[0]
            if reg == 0x0201:                            
                buffer_formats |= p << 0
            elif reg == 0x0202:        
                buffer_formats |= p << 32
            elif reg == 0x0204:                         
                buffer_attributes |= p
            elif reg == 0x0205:                         
                buffer_attributes |= (p & 0xFFFF) << 32
                vertex_stride = (p >> 16) & 0xFF
                attrs_count = (p >> 28) & 0xF
            elif reg == 0x0232:                     
                fixed_index = int(p)
            elif reg == 0x0233:
                w0, w1, w2 = fixed_words[fixed_index]
                fixed_words[fixed_index] = (p, w1, w2)
            elif reg == 0x0234:
                w0, w1, w2 = fixed_words[fixed_index]
                fixed_words[fixed_index] = (w0, p, w2)
            elif reg == 0x0235:
                w0, w1, w2 = fixed_words[fixed_index]
                fixed_words[fixed_index] = (w0, w1, p)
            elif reg == 0x0242:                
                attrs_total = int(p) + 1
            elif reg == 0x02BB:
                buffer_perm |= p << 0
            elif reg == 0x02BC:
                buffer_perm |= p << 32

        attributes: List[_PICAAttribute] = []
        fixed_attributes: List[_PICAFixedAttribute] = []

        for idx in range(attrs_total):
            if ((buffer_formats >> (48 + idx)) & 1) != 0:
                name = (buffer_perm >> (idx * 4)) & 0xF
                scale = scales[1] if name in (3, 8) else 1.0
                w0, w1, w2 = fixed_words[idx]
                x, y, z, w = _parse_pica_vec_float24(w0, w1, w2)
                fixed_attributes.append(
                    _PICAFixedAttribute(
                        name=int(name),
                        x=x * scale,
                        y=y * scale,
                        z=z * scale,
                        w=w * scale,
                    )
                )
            else:
                perm_idx = (buffer_attributes >> (idx * 4)) & 0xF
                attr_name = (buffer_perm >> (perm_idx * 4)) & 0xF
                attr_fmt = (buffer_formats >> (perm_idx * 4)) & 0xF
                fmt = int(attr_fmt & 3)
                elements = int((attr_fmt >> 2) + 1)
                scale = scales[fmt]
                if int(attr_name) == 7:                               
                    scale = 1.0
                attributes.append(
                    _PICAAttribute(
                        name=int(attr_name), fmt=fmt, elements=elements, scale=scale
                    )
                )

        primitive_mode = 0
        buffer_address = 0
        buffer_count = 0
        for reg, params in _pica_read_commands(index_cmds):
            p = params[0]
            if reg == 0x0227:
                buffer_address = p
            elif reg == 0x0228:
                buffer_count = p
            elif reg == 0x025E:
                primitive_mode = (p >> 8) & 0xF

        raw_buffer_off = int(r.tell)
        raw_buffer = r.read(vtx_len)
        idx_start = r.tell

        indices: List[int] = []
        idx16 = (buffer_address >> 31) != 0
        elem_size = 2 if idx16 else 1
        for _ in range(buffer_count):
            if idx16:
                indices.append(r.u16())
            else:
                indices.append(r.u8())
        consumed = int(buffer_count) * int(elem_size)
        pad_len = int(idx_len) - int(consumed)
        if pad_len < 0:
            pad_len = 0
        index_pad = r.read(pad_len) if pad_len else b""
                                                                      
        try:
            r.seek(idx_start + idx_len)
        except Exception:
            pass

        submeshes.append(
            _GFSubMesh(
                name=sm_name,
                mesh_index=int(mesh_index),
                face_index=int(face_index),
                mesh_name=mesh_name,
                mesh_bbox_min=mesh_bbox_min,
                mesh_bbox_max=mesh_bbox_max,
                mesh_is_blend_shape=mesh_is_blend_shape,
                mesh_face_count=mesh_face_count,
                mesh_weight_max=mesh_weight_max,
                bone_indices_count=bone_count,
                bone_indices=bone_indices[:bone_count],
                vertex_count=vcount,
                index_count=icount,
                vertex_stride=vertex_stride,
                primitive_mode=int(primitive_mode),
                indices=indices,
                raw_buffer=raw_buffer,
                raw_buffer_off=int(raw_buffer_off),
                attributes=attributes,
                fixed_attributes=fixed_attributes,
                enable_cmds=[int(x) for x in enable_cmds],
                disable_cmds=[int(x) for x in disable_cmds],
                index_cmds=[int(x) for x in index_cmds],
                index_data_len=int(idx_len),
                index_pad_bytes=index_pad,
                index_buffer_off=int(idx_start),
                index_elem_size=int(elem_size),
                index_count_off=int(idx_count_off),
                index_cmds_off=int(index_cmds_off),
                index_cmds_len_u32=int(index_cmds_len_u32),
                vertex_count_off=int(vtx_count_off),
                vertex_data_len_off=int(vtx_len_off),
                index_data_len_off=int(idx_len_off),
                mesh_section_off=int(mesh_section_off),
                mesh_section_len_off=int(mesh_section_len_off),
            )
        )

    end = start + sect_len
    r.seek(end)
    return submeshes, r.tell


def _parse_gf_model(data: bytes, offset: int, name: str) -> Tuple[_GFModel, int]:
    r = _BinReader(data, offset)
    _magic = r.u32()              
    _sections = r.u32()
    _gf_skip_padding16(r)

    _model_sect_magic, _model_sect_len = _read_gf_section(r)

    def read_hash_table() -> List[str]:
        count = r.u32()
        out: List[str] = []
        for _ in range(count):
            _ = r.u32()
            out.append(r.padded_string(0x40))
        return out

    shader_names = read_hash_table()
    texture_names = read_hash_table()
    material_names = read_hash_table()
    mesh_names = read_hash_table()

    bbox_min = _gf_read_vec4(r)
    bbox_max = _gf_read_vec4(r)
    transform_rows = tuple(tuple(float(r.f32()) for _ in range(4)) for _ in range(4))

    unk_len = r.u32()
    unk_off = r.u32()
    r.skip(8)
    unk_start = r.tell + unk_off
    unk_end = unk_start + unk_len
    unknown_blob = b""
    if (
        0 <= unk_start <= len(data)
        and 0 <= unk_end <= len(data)
        and unk_end >= unk_start
    ):
        unknown_blob = data[unk_start:unk_end]
    r.skip(unk_off + unk_len)

    bones_count = r.s32()
    r.skip(0x0C)
    skeleton: List[_GFBone] = []
    for _ in range(bones_count):
        bone, new_off = _parse_gf_bone(data, r.tell)
        skeleton.append(bone)
        r.seek(new_off)

    _gf_skip_padding16(r)

    luts_count = r.s32()
    lut_len = r.s32()
    _gf_skip_padding16(r)
    luts: List[_GFLUT] = []
    for _ in range(int(luts_count)):
        texture_hash = r.u32()
        r.skip(0x0C)
        command_bytes = r.read(int(lut_len))
        command_words: List[int] = []
        if (len(command_bytes) % 4) == 0:
            command_words = list(
                struct.unpack_from(
                    "<" + "I" * (len(command_bytes) // 4), command_bytes, 0
                )
            )
        luts.append(
            _GFLUT(
                texture_hash=int(texture_hash),
                command_bytes=command_bytes,
                command_words=command_words,
            )
        )

    materials: List[_GFMaterial] = []
    for _ in range(len(material_names)):
        mat, new_off = _parse_gf_material(data, r.tell)
        materials.append(mat)
        r.seek(new_off)

    submeshes_all: List[_GFSubMesh] = []
    for mesh_index in range(len(mesh_names)):
        sms, new_off = _parse_gf_mesh(data, r.tell)
                                                                  
        for sm in sms:
            try:
                sm.mesh_index = int(mesh_index)
            except Exception:
                pass
        submeshes_all.extend(sms)
        r.seek(new_off)

    return _GFModel(
        name=name,
        shader_names=shader_names,
        texture_names=texture_names,
        material_names=material_names,
        mesh_names=mesh_names,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        transform_rows=transform_rows,
        unknown_blob=unknown_blob,
        unknown_off=int(unk_off),
        luts=luts,
        skeleton=skeleton,
        materials=materials,
        submeshes=submeshes_all,
    ), r.tell


def _parse_gf_model_pack(
    data: bytes,
) -> Tuple[List[_GFModel], List[_GFTexture], List[_GFShader]]:
                                                                                                   
    models: List[_GFModel] = []
    textures: List[_GFTexture] = []
    shaders: List[_GFShader] = []

    if len(data) < 4 + 20:
        return models, textures, shaders

    r = _BinReader(data)
    _magic = r.u32()              
    counts = [r.u32() for _ in range(5)]
    pointers_addr = r.tell

    total_ptrs = sum(counts)
    if pointers_addr + total_ptrs * 4 > len(data):
        return models, textures, shaders

    position = 0                                                                 
    sect_ptr_base = pointers_addr

    for sect in range(5):
        sect_count = counts[sect]
        for entry in range(sect_count):
            ptr_off = struct.unpack_from("<I", data, sect_ptr_base + entry * 4)[0]
            if ptr_off == 0 or ptr_off >= len(data):
                continue

                                                                 
            if ptr_off + 1 > len(data):
                continue
            name_len = data[ptr_off]
            header_end = ptr_off + 1 + name_len + 4
            if header_end > len(data):
                continue

            name_bytes = data[ptr_off + 1 : ptr_off + 1 + name_len]
            try:
                name = name_bytes.decode("ascii", "replace")
            except Exception:
                name = ""

            address = struct.unpack_from("<I", data, ptr_off + 1 + name_len)[0]
            if address >= len(data):
                continue

            try:
                if sect == 0:
                    model, _ = _parse_gf_model(data, position + address, name)
                    models.append(model)
                elif sect == 1:
                    textures.append(_parse_gf_texture(data[position + address :]))
                elif sect == 4:
                    sh, _ = _parse_gf_shader(data, position + address)
                    shaders.append(sh)
            except Exception:
                continue

        sect_ptr_base += sect_count * 4

    return models, textures, shaders


def _parse_pkmn_container(data: bytes) -> Tuple[str, List[bytes]]:
    r = _BinReader(data)
    magic = r.read(2).decode("ascii", "replace")
    count = r.u16()
    offsets = [struct.unpack_from("<I", data, 4 + i * 4)[0] for i in range(count + 1)]
    files: List[bytes] = []
    for i in range(count):
        start, end = offsets[i], offsets[i + 1]
        files.append(data[start:end])
    return magic, files


def _load_any(
    data: bytes,
) -> Tuple[List[_GFModel], List[_GFTexture], List[_GFMotion], List[_GFShader]]:
    models: List[_GFModel] = []
    textures: List[_GFTexture] = []
    motions: List[_GFMotion] = []
    shaders: List[_GFShader] = []

                                     
    if len(data) >= 4 and 65 <= data[0] <= 90 and 65 <= data[1] <= 90:
        magic, entries = _parse_pkmn_container(data)
        if magic == "CP" and len(entries) >= 2:
            ent = entries[1]
            if _looks_like_lz11(ent):
                ent = _lzss_ninty_decompress(ent)
            return _load_any(ent)
        for ent in entries:
            if _looks_like_lz11(ent):
                ent = _lzss_ninty_decompress(ent)
            m, t, a, s = _load_any(ent)
            models.extend(m)
            textures.extend(t)
            motions.extend(a)
            shaders.extend(s)
        return models, textures, motions, shaders

    if len(data) < 4:
        return models, textures, motions, shaders

    u0 = struct.unpack_from("<I", data, 0)[0]
                                                       
                                                                                                            
                                                                                                               
                                                                                                  
     
                                                                                                
    if u0 == 0x15052616 and len(data) >= 16:
        try:
            section_count = struct.unpack_from("<I", data, 4)[0]
            fake = struct.pack("<II", 0x15122117, int(section_count)) + data[8:]
            model, _ = _parse_gf_model(fake, 0, "Model")
            models.append(model)
            return models, textures, motions, shaders
        except Exception:
            pass
    if u0 == 0x14110400 and len(data) >= 24:
        try:
            section_count = struct.unpack_from("<I", data, 4)[0]
            fake = struct.pack("<II", 0x15041213, int(section_count)) + data[8:]
            textures.append(_parse_gf_texture(fake))
            return models, textures, motions, shaders
        except Exception:
            pass
    if u0 == 0x00050000 and len(data) >= 32:
        try:
            fake = struct.pack("<I", 0x00060000) + data[4:]
            motions.append(_parse_gf_motion(fake, index=len(motions)))
            return models, textures, motions, shaders
        except Exception:
            pass

    if u0 == 0x00010000:
        m, t, s = _parse_gf_model_pack(data)
        models.extend(m)
        textures.extend(t)
        shaders.extend(s)
        return models, textures, motions, shaders
    if u0 == 0x15041213:
                                                                                         
                                                                                            
                                                                  
        try:
            sect8 = ""
            if len(data) >= 0x10:
                sect8 = data[0x08:0x10].split(b"\0", 1)[0].decode("ascii", "replace")
            sect10 = ""
            if len(data) >= 0x18:
                sect10 = data[0x10:0x18].split(b"\0", 1)[0].decode("ascii", "replace")

            if sect8 == "texture":
                textures.append(_parse_gf_texture(data))
            elif sect10 == "shader":
                sh, _ = _parse_gf_shader(data, 0)
                shaders.append(sh)
            else:
                                                           
                try:
                    textures.append(_parse_gf_texture(data))
                except Exception:
                    sh, _ = _parse_gf_shader(data, 0)
                    shaders.append(sh)
        except Exception:
            pass
        return models, textures, motions, shaders
    if u0 == 0x15122117:
        model, _ = _parse_gf_model(data, 0, "Model")
        models.append(model)
        return models, textures, motions, shaders
    if u0 == 0x00060000:
                                                                                              
        motions.append(_parse_gf_motion(data, index=len(motions)))
        return models, textures, motions, shaders

                                                       
    anims_count = u0
    if 0 < anims_count < 4096 and len(data) >= 4 + anims_count * 4:
        r = _BinReader(data)
        _ = r.u32()
        offsets = [r.u32() for _ in range(anims_count)]
        for i, off in enumerate(offsets):
            if off == 0:
                continue
            if off >= len(data):
                continue
            sub = data[4 + off :]
            if len(sub) >= 4 and struct.unpack_from("<I", sub, 0)[0] == 0x00060000:
                motions.append(_parse_gf_motion(sub, index=i))
        if motions:
            return models, textures, motions, shaders

    return models, textures, motions, shaders
