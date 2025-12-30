"""Binary writer helpers for GFModel export (v1 scaffold-based).

This module intentionally contains no Blender registration logic.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .pica import _pica_read_commands
from .types import (
    _GFLUT,
    _GFBone,
    _GFMaterial,
    _GFModel,
    _GFShader,
    _GFSubMesh,
    _GFTexture,
)


def _gfnv1_32(data: bytes) -> int:
    """GameFreak's FNV1 variant used by GFHashName/GFHashName2 in CTR-era assets.

    Matches SPICA's `GFNV1`:
    - prime: 16777619 (0x01000193)
    - initial hash: prime (not the standard FNV offset basis)
    - step: h = (h * prime) ^ b
    """
    prime = 0x01000193
    h = prime
    for b in data:
        h = (h * prime) & 0xFFFFFFFF
        h ^= int(b)
    return int(h)


class _BinWriter:
    __slots__ = ("_b",)

    def __init__(self) -> None:
        self._b = bytearray()

    @property
    def tell(self) -> int:
        return len(self._b)

    def align(self, boundary: int, pad: int = 0) -> None:
        mask = boundary - 1
        while (len(self._b) & mask) != 0:
            self._b.append(pad & 0xFF)

    def pad_to_next_boundary(
        self, boundary: int, *, pad: int = 0, minimum: int = 0
    ) -> None:
        """Pad with `pad` bytes up to the next `boundary`.

        If already aligned and `minimum`>0, pads exactly `minimum` bytes.
        """
        if boundary <= 0:
            return
        cur = len(self._b)
        mod = cur % int(boundary)
        need = (int(boundary) - mod) if mod else 0
        if need == 0 and int(minimum) > 0:
            need = int(minimum)
        if need > 0:
            self._b += bytes([pad & 0xFF]) * int(need)

    def u8(self, v: int) -> None:
        self._b.append(int(v) & 0xFF)

    def u16(self, v: int) -> None:
        self._b += struct.pack("<H", int(v) & 0xFFFF)

    def u32(self, v: int) -> None:
        self._b += struct.pack("<I", int(v) & 0xFFFFFFFF)

    def s32(self, v: int) -> None:
        self._b += struct.pack("<i", int(v))

    def f32(self, v: float) -> None:
        self._b += struct.pack("<f", float(v))

    def bytes(self, b: bytes) -> None:
        self._b += b

    def padded_string(self, s: str, size: int) -> None:
        raw = (s or "").encode("ascii", "replace")
        raw = raw[:size]
        self._b += raw
        if len(raw) < size:
            self._b += b"\x00" * (size - len(raw))

    def byte_len_string(self, s: str) -> None:
        raw = (s or "").encode("ascii", "replace")
        if len(raw) > 0xFF:
            raw = raw[:0xFF]
        self.u8(len(raw))
        self.bytes(raw)

    def int_len_string(self, s: str) -> None:
        raw = (s or "").encode("ascii", "replace")
        self.s32(len(raw))
        self.bytes(raw)

    def int_len_string_padded4(self, s: str, pad: int = 0) -> None:
        raw = (s or "").encode("ascii", "replace")
        plen = (len(raw) + 3) & ~3
        self.s32(int(plen))
        self.bytes(raw)
        if plen > len(raw):
            self.bytes(bytes([pad & 0xFF]) * (plen - len(raw)))

    def reserve_u32(self) -> int:
        o = self.tell
        self.u32(0)
        return o

    def patch_u32(self, offset: int, v: int) -> None:
        struct.pack_into("<I", self._b, int(offset), int(v) & 0xFFFFFFFF)

    def finish(self) -> bytes:
        return bytes(self._b)


def _write_gf_section_header(w: _BinWriter, magic8: bytes, length: int) -> None:
    if len(magic8) != 8:
        raise ValueError("section magic must be 8 bytes")
    w.bytes(magic8)
    w.u32(int(length))
    w.u32(0xFFFFFFFF)


def _pica_patch_first_param(cmds: List[int], reg: int, new_param: int) -> bool:
    """Patch the first write to `reg` in-place, best-effort."""
    i = 0
    while i + 1 < len(cmds):
        param0_i = i
        cmd_i = i + 1
        cmd = int(cmds[cmd_i])
        i += 2
        base_reg = cmd & 0xFFFF
        extra = (cmd >> 20) & 0x7FF
        consecutive = (cmd >> 31) != 0
        if consecutive:
            for j in range(extra + 1):
                if int(base_reg + j) == int(reg):
                    cmds[param0_i] = int(new_param) & 0xFFFFFFFF
                    return True
                if j < extra:
                    param0_i = i
                    i += 1
        else:
            if int(base_reg) == int(reg):
                cmds[param0_i] = int(new_param) & 0xFFFFFFFF
                return True
            i += int(extra)
        if (i & 1) != 0:
            i += 1
    return False


def _pica_patch_all_params(cmds: List[int], reg: int, new_param: int) -> int:
    """Patch all writes to `reg` in-place, best-effort.

    Some meshes write the same register multiple times (e.g. reasserting state before the draw).
    For counts/config derived from exported buffers, patching only the first write can leave later
    writes stale and cause in-game issues.
    """
    patched = 0
    i = 0
    while i + 1 < len(cmds):
        param0_i = i
        cmd_i = i + 1
        cmd = int(cmds[cmd_i])
        i += 2
        base_reg = cmd & 0xFFFF
        extra = (cmd >> 20) & 0x7FF
        consecutive = (cmd >> 31) != 0
        if consecutive:
            for j in range(extra + 1):
                if int(base_reg + j) == int(reg):
                    cmds[param0_i] = int(new_param) & 0xFFFFFFFF
                    patched += 1
                if j < extra:
                    param0_i = i
                    i += 1
        else:
            if int(base_reg) == int(reg):
                cmds[param0_i] = int(new_param) & 0xFFFFFFFF
                patched += 1
            i += int(extra)
        if (i & 1) != 0:
            i += 1
    return int(patched)


def _pica_get_first_param(cmds: List[int], reg: int) -> Optional[int]:
    """Return the first param written to `reg`, if present (best-effort)."""
    i = 0
    while i + 1 < len(cmds):
        param0_i = i
        cmd_i = i + 1
        cmd = int(cmds[cmd_i])
        i += 2
        base_reg = cmd & 0xFFFF
        extra = (cmd >> 20) & 0x7FF
        consecutive = (cmd >> 31) != 0
        if consecutive:
            for j in range(extra + 1):
                if int(base_reg + j) == int(reg):
                    return int(cmds[param0_i]) & 0xFFFFFFFF
                if j < extra:
                    param0_i = i
                    i += 1
        else:
            if int(base_reg) == int(reg):
                return int(cmds[param0_i]) & 0xFFFFFFFF
            i += int(extra)
        if (i & 1) != 0:
            i += 1
    return None


def _pica_write_cmd_stream_raw(w: _BinWriter, cmds: List[int]) -> None:
    """Write a raw packed PICA command word stream.

    Retail GF model meshes store the command buffer as a flat u32 array where the 8-byte
    padding words used by PICA packets are included in `cmdLength` and are expected by
    parsers like SPICA (and our importer’s `_pica_read_commands`).
    """
    for v in cmds:
        w.u32(int(v) & 0xFFFFFFFF)


def build_gf_texture_rgba8(
    name: str, width: int, height: int, raw_rgba: bytes
) -> _GFTexture:
    """Creates an in-memory GFTexture using GF fmt=0x4 (RGBA8).

    Note: raw encoding/swizzle is handled by the caller; this struct is just the parsed representation.
    """
    return _GFTexture(
        name=name, width=int(width), height=int(height), fmt=0x4, raw=raw_rgba
    )


def encode_pica_rgba8_swizzled_abgr(raw_rgba: bytes, width: int, height: int) -> bytes:
    """Encode linear RGBA8 (top-left origin) into PICA RGBA8 swizzled ABGR bytes.

    This is the inverse of `_pica_decode_to_bgra` for fmt=RGBA8 for the swizzle+channel parts
    (not including the external vertical flip used by the importer).
    """
    if width % 8 != 0 or height % 8 != 0:
        raise ValueError("RGBA8 swizzle requires width/height multiples of 8")
    if len(raw_rgba) != width * height * 4:
        raise ValueError("raw_rgba size mismatch")

                                                                                         
                                                                                  
                                              
                                                                                             
                                           
    out = bytearray(width * height * 4)

                                     
    from .pica import _SWIZZLE_LUT                

    i_off = 0
    for ty in range(0, height, 8):
        for tx in range(0, width, 8):
            for px in range(64):
                x = _SWIZZLE_LUT[px] & 7
                y = (_SWIZZLE_LUT[px] - x) >> 3
                sx = tx + x
                sy = height - 1 - (ty + y)
                si = (sy * width + sx) * 4
                r = raw_rgba[si + 0]
                g = raw_rgba[si + 1]
                b = raw_rgba[si + 2]
                a = raw_rgba[si + 3]
                                                     
                out[i_off + 0] = a
                out[i_off + 1] = r
                out[i_off + 2] = g
                out[i_off + 3] = b
                i_off += 4
    return bytes(out)


def write_gf_texture_blob(tex: _GFTexture) -> bytes:
    w = _BinWriter()
    w.u32(0x15041213)
    w.u32(1)
                                          
    sect_start = w.tell
    _write_gf_section_header(w, b"texture\x00", 0)
    payload_start = w.tell

    w.u32(len(tex.raw))
    w.bytes(b"\x00" * 0x0C)
    w.padded_string(tex.name, 0x40)
    w.u16(int(tex.width))
    w.u16(int(tex.height))
    w.u16(int(tex.fmt))
    w.u16(0)                                      
    w.bytes(b"\x00" * 0x10)
    w.bytes(tex.raw)

    payload_len = w.tell - payload_start
    struct.pack_into("<I", w._b, sect_start + 8, int(payload_len))                              
    return w.finish()


def write_gf_material_blob(mat: _GFMaterial) -> bytes:
    w = _BinWriter()
    sect_start = w.tell
    _write_gf_section_header(w, b"material", 0)
    payload_start = w.tell

    def write_hash_name(s: str) -> None:
        w.u32(_gfnv1_32((s or "").encode("ascii", "replace")))
        w.byte_len_string(s)

    write_hash_name(mat.name)
    write_hash_name(mat.shader_name)
    write_hash_name(mat.vtx_shader)
    write_hash_name(mat.frag_shader)

    w.u32(int(mat.lut_hashes[0]))
    w.u32(int(mat.lut_hashes[1]))
    w.u32(int(mat.lut_hashes[2]))
    w.u32(0)

    w.u8(int(mat.bump_texture) & 0xFF)
    for c in mat.const_assignments:
        w.u8(int(c) & 0xFF)
    w.u8(0)

    for rgba in mat.colors_rgba:
        for c in rgba:
            w.u8(int(c) & 0xFF)

    w.u32(int(mat.edge_type))
    w.u32(int(mat.id_edge_enable))
    w.u32(int(mat.edge_id))
    w.u32(int(mat.projection_type))
    w.f32(float(mat.rim_pow))
    w.f32(float(mat.rim_scale))
    w.f32(float(mat.phong_pow))
    w.f32(float(mat.phong_scale))
    w.u32(int(mat.id_edge_offset_enable))
    w.u32(int(mat.edge_map_alpha_mask))
    for x in mat.bake_ops:
        w.u32(int(x))
    w.u32(int(mat.vertex_shader_type))
    for f in mat.shader_params:
        w.f32(float(f))

    w.u32(len(mat.tex_units))
    for tu in mat.tex_units:
        write_hash_name(tu.name)
        w.u8(int(tu.unit_index) & 0xFF)
        w.u8(int(tu.mapping_type) & 0xFF)
        w.f32(float(tu.scale.x))
        w.f32(float(tu.scale.y))
        w.f32(float(tu.rotation))
        w.f32(float(tu.translation.x))
        w.f32(float(tu.translation.y))
        for sw in tu.sampler_words:
            w.u32(int(sw))

    w.align(0x10, 0)

    commands_len = len(mat.pica_commands) * 4
    w.u32(int(commands_len))
    w.u32(int(mat.render_priority))
    w.u32(int(mat.unk_render))
    w.u32(int(mat.render_layer))
    for h in mat.header_hashes:
        w.u32(int(h))
    for cw in mat.pica_commands:
        w.u32(int(cw))

                                                        
    w.align(0x10, 0)
    payload_len = w.tell - payload_start
    struct.pack_into("<I", w._b, sect_start + 8, int(payload_len))                              
    return w.finish()


def write_gf_mesh_blob(mesh_name: str, faces: List[_GFSubMesh]) -> bytes:
    if not faces:
        raise ValueError("mesh has no faces")

                                                          
    hdr = faces[0]
                                                                                            
                                                                                           
    idx16_by_face: List[bool] = []
    for sm in faces:
        old = _pica_get_first_param([int(x) for x in sm.index_cmds], 0x0227) or 0
        idx16 = ((int(old) >> 31) & 1) != 0 or any(int(ii) > 0xFF for ii in sm.indices)
        idx16_by_face.append(bool(idx16))

    w = _BinWriter()
    sect_start = w.tell
    _write_gf_section_header(w, b"mesh\x00\x00\x00\x00", 0)
    payload_start = w.tell

    w.u32(_gfnv1_32(mesh_name.encode("ascii", "replace")))
    w.padded_string(mesh_name, 0x40)
    w.u32(1 if hdr.mesh_is_blend_shape else 0)
    for v in (hdr.mesh_bbox_min, hdr.mesh_bbox_max):
        w.f32(float(v.x))
        w.f32(float(v.y))
        w.f32(float(v.z))
        w.f32(float(v.w))
    w.u32(int(hdr.mesh_face_count))
    w.u32(int(hdr.mesh_weight_max))

                                                                     
    pad_len = (0x80 - ((w.tell - payload_start) % 0x80)) % 0x80
    if pad_len:
        w.bytes(b"\xff" * pad_len)

                                
    total_lists = int(len(faces) * 3)
    for i, sm in enumerate(faces):
        lists = (sm.enable_cmds, sm.disable_cmds, sm.index_cmds)
        for li, cmd_words in enumerate(lists):
            cmds = [int(x) for x in cmd_words]
                                                                 
            if li == 2:
                                                                                        
                old = _pica_get_first_param(cmds, 0x0227) or 0
                idx16 = bool(idx16_by_face[int(i)])
                new = (int(old) & 0x7FFFFFFF) | (0x80000000 if idx16 else 0)
                _pica_patch_all_params(cmds, 0x0227, int(new))
                _pica_patch_all_params(cmds, 0x0228, int(len(sm.indices)))
            w.u32(len(cmds) * 4)
            w.u32(i * 3 + li)
            w.u32(total_lists)
            w.u32(0)
            _pica_write_cmd_stream_raw(w, cmds)

               
    for sm_i, sm in enumerate(faces):
        w.u32(_gfnv1_32(sm.name.encode("ascii", "replace")))
                                                                                       
                                                                                         
        w.int_len_string_padded4(sm.name, pad=0)
        w.u8(int(sm.bone_indices_count) & 0xFF)
        bi = list(sm.bone_indices)[: int(sm.bone_indices_count)]
        bi += [0] * (0x1F - len(bi))
        for b in bi[:0x1F]:
            w.u8(int(b) & 0xFF)
        w.s32(int(sm.vertex_count))
        w.s32(int(sm.index_count))
        w.s32(int(len(sm.raw_buffer)))
                                                             
        idx16 = bool(idx16_by_face[int(sm_i)])
        elem_size = 2 if idx16 else 1
        idx_bytes = int(len(sm.indices) * elem_size)
        desired = int(getattr(sm, "index_data_len", 0) or 0)
        if desired < idx_bytes:
            desired = idx_bytes
        w.s32(int(desired))

                                   
    for sm_i, sm in enumerate(faces):
        w.bytes(sm.raw_buffer)
        idx16 = bool(idx16_by_face[int(sm_i)])
        elem_size = 2 if idx16 else 1
        if idx16:
            for ii in sm.indices:
                w.u16(int(ii))
        else:
            for ii in sm.indices:
                w.u8(int(ii))
                                                                                             
        desired = int(getattr(sm, "index_data_len", 0) or 0)
        if desired > 0:
            written = int(len(sm.indices) * elem_size)
            pad_len = int(desired) - int(written)
            if pad_len > 0:
                pad = getattr(sm, "index_pad_bytes", b"") or b""
                if len(pad) == pad_len:
                    w.bytes(pad)
                else:
                    w.bytes(b"\x00" * pad_len)

                                                        
                                                                                       
                                                                     
    cur_payload = int(w.tell - payload_start)
    mod = cur_payload & 0xF
    pad_len = (0x10 - mod) if mod else 0x10
    w.bytes(b"\x00" * int(pad_len))
    payload_len = w.tell - payload_start
    struct.pack_into("<I", w._b, sect_start + 8, int(payload_len))                              
    return w.finish()


def write_gf_model_blob(model: _GFModel, *, meshes_by_index: Dict[int, bytes]) -> bytes:
    w = _BinWriter()
    w.u32(0x15122117)
                                                                                  
                                                        
    w.u32(1 + int(len(model.materials)) + int(len(meshes_by_index)))
    w.align(0x10, 0)

    sect_start = w.tell
    _write_gf_section_header(w, b"gfmodel\x00", 0)
    payload_start = w.tell

    def write_hash_table(names: List[str]) -> None:
        w.u32(len(names))
        for n in names:
            w.u32(_gfnv1_32(n.encode("ascii", "replace")))
            w.padded_string(n, 0x40)

    write_hash_table(model.shader_names)
    write_hash_table(model.texture_names)
    write_hash_table(model.material_names)
    write_hash_table(model.mesh_names)

    for v in (model.bbox_min, model.bbox_max):
        w.f32(float(v.x))
        w.f32(float(v.y))
        w.f32(float(v.z))
        w.f32(float(v.w))
    for row in model.transform_rows:
        for f in row:
            w.f32(float(f))

                                                                                              
    unk = model.unknown_blob or b""
    w.u32(len(unk))
    unk_off = int(getattr(model, "unknown_off", 0) or 0)
    w.u32(unk_off)
    w.u32(0)
    w.u32(0)
    if unk_off:
        w.bytes(b"\x00" * unk_off)
    w.bytes(unk)

    w.s32(len(model.skeleton))
    w.bytes(b"\x00" * 0x0C)
    for b in model.skeleton:
        w.byte_len_string(b.name)
        w.byte_len_string(b.parent or "")
        w.u8(int(b.flags) & 0xFF)
        for v in (b.scale, b.rotation, b.translation):
            w.f32(float(v.x))
            w.f32(float(v.y))
            w.f32(float(v.z))

    w.align(0x10, 0)

    luts = model.luts or []
    lut_len = (
        int(luts[0].command_bytes and len(luts[0].command_bytes) or 0) if luts else 0
    )
    w.s32(len(luts))
    w.s32(lut_len)
    w.align(0x10, 0)
    for lut in luts:
        w.u32(int(lut.texture_hash))
        w.bytes(b"\x00" * 0x0C)
        cb = lut.command_bytes
        if len(cb) != lut_len:
            raise ValueError("LUT length mismatch")
        w.bytes(cb)

                                                                                    
                                                                                           
    w.align(0x10, 0)

                                                                                             
    payload_len = w.tell - payload_start
    struct.pack_into("<I", w._b, sect_start + 8, int(payload_len))                              

    for mat in model.materials:
        if getattr(mat, "raw_blob", None):
            w.bytes(mat.raw_blob)                          
        else:
            w.bytes(write_gf_material_blob(mat))

                                                                      
    for mesh_idx in range(len(model.mesh_names)):
        blob = meshes_by_index.get(int(mesh_idx))
        if blob is None:
            raise ValueError(f"missing mesh blob for mesh_index={mesh_idx}")
        w.bytes(blob)

    return w.finish()


@dataclass
class _GFPackEntry:
    name: str
    blob: bytes


def parse_gf_model_pack_entries(
    data: bytes,
) -> Tuple[List[List[_GFPackEntry]], List[int]]:
    """Parse raw GFModelPack entries for all 5 sections (including unknown sections)."""
    if len(data) < 4 + 5 * 4:
        raise ValueError("data too small for GFModelPack")
    pos = 0
    magic = struct.unpack_from("<I", data, 0)[0]
    if magic != 0x00010000:
        raise ValueError("not a GFModelPack")
    counts = [struct.unpack_from("<I", data, 4 + i * 4)[0] for i in range(5)]
    pointers_addr = 4 + 5 * 4

    entries: List[Tuple[int, int, str, int]] = []                                   
    base = pointers_addr
    for sect in range(5):
        sect_count = int(counts[sect])
        for entry_index in range(sect_count):
            ptr = struct.unpack_from("<I", data, base + entry_index * 4)[0]
            if ptr == 0 or ptr >= len(data):
                continue
            name_len = data[ptr]
            name = data[ptr + 1 : ptr + 1 + name_len].decode("ascii", "replace")
            addr = struct.unpack_from("<I", data, ptr + 1 + name_len)[0]
            if addr >= len(data):
                continue
            entries.append((int(sect), int(entry_index), name, int(addr)))
        base += sect_count * 4

    entries_sorted = sorted(entries, key=lambda t: t[3])
    addr_to_end: Dict[int, int] = {}
    for i, (_sect, _ei, _name, addr) in enumerate(entries_sorted):
        end = len(data)
        if i + 1 < len(entries_sorted):
            end = int(entries_sorted[i + 1][3])
        addr_to_end[int(addr)] = int(end)

    sections: List[List[_GFPackEntry]] = [[] for _ in range(5)]
    for sect, _ei, name, addr in entries:
        end = addr_to_end.get(int(addr), len(data))
        sections[int(sect)].append(_GFPackEntry(name=name, blob=data[addr:end]))

    return sections, counts


def write_gf_model_pack(
    *,
    models: List[_GFPackEntry],
    textures: List[_GFPackEntry],
    unknown2: List[_GFPackEntry],
    unknown3: List[_GFPackEntry],
    shaders: List[_GFPackEntry],
) -> bytes:
    sections = [models, textures, unknown2, unknown3, shaders]
    counts = [len(s) for s in sections]

    w = _BinWriter()
    w.u32(0x00010000)
    for c in counts:
        w.u32(int(c))

                             
    ptr_tables_off = w.tell
    ptr_offsets: List[int] = []
    for sect in range(5):
        for _ in range(counts[sect]):
            ptr_offsets.append(w.reserve_u32())

                                                                                       
    addr_patch_offsets: List[int] = []
    for sect in range(5):
        for entry in sections[sect]:
            ptr_patch_off = ptr_offsets.pop(0)
            w.patch_u32(ptr_patch_off, w.tell)
            w.byte_len_string(entry.name)
            addr_patch_offsets.append(w.reserve_u32())

                                                                  
    for sect in range(5):
        for entry in sections[sect]:
                                                                                           
                                                                                               
                                                                                         
            w.align(0x80, 0)
            addr = w.tell
            if not addr_patch_offsets:
                raise ValueError("internal error: address patch underrun")
            w.patch_u32(addr_patch_offsets.pop(0), addr)
            w.bytes(entry.blob)

    if addr_patch_offsets:
        raise ValueError("internal error: address patch list not fully consumed")

    return w.finish()
