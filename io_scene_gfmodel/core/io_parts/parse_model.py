
from __future__ import annotations

import struct
from typing import Dict, List, Optional, Tuple

from ..pica import _parse_pica_vec_float24, _pica_read_commands

from ..types import (
    _GFBone,
    _GFMaterial,
    _GFModel,
    _GFShader,
    _GFSubMesh,
    _GFTexture,
    _GFLUT,
    _PICAAttribute,
    _PICAFixedAttribute,
)
from .parse_material import _parse_gf_material
from .parse_shader import _parse_gf_shader
from .parse_texture import _parse_gf_texture
from .reader import (
    _BinReader,
    _gf_read_hash_name,
    _gf_read_vec3,
    _gf_read_vec4,
    _gf_skip_padding16,
    _read_gf_section,
)

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
