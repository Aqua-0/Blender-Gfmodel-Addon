"""GFMotion parsing helpers.

Split out of `gfmodel_io.py` to keep modules smaller and focused.
This module intentionally has no Blender registration code.
"""

from __future__ import annotations

import struct
from typing import List, Tuple

from .math_compat import Vector
from .types import (
    __GFMotUVTransform,
    __GFMotVisibilityTrack,
    _GFMotBoneTransform,
    _GFMotion,
    _GFMotKeyFrame,
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

    def u16(self) -> int:
        v = struct.unpack_from("<H", self._b, self._o)[0]
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
        return raw.decode("ascii", "replace")

    def byte_len_string(self) -> str:
        return self.padded_string(self.u8())

    def align(self, boundary: int) -> None:
        mask = boundary - 1
        if (self._o & mask) != 0:
            self._o += boundary - (self._o & mask)


def _gf_read_vec3(r: _BinReader) -> Vector:
    return Vector((r.f32(), r.f32(), r.f32()))


def _parse_gf_motion(data: bytes, index: int) -> _GFMotion:
    r = _BinReader(data)
    _magic = r.u32()              
    section_count = r.u32()
    sections: List[Tuple[int, int, int]] = []
    for _ in range(section_count):
        sect_name = r.u32()
        length = r.u32()
        addr = r.u32()
        sections.append((sect_name, length, addr))

    r.seek(sections[0][2])
    frames_count = r.u32()
    is_looping = (r.u16() & 1) != 0
    is_blended = (r.u16() & 1) != 0
    anim_region_min = _gf_read_vec3(r)
    anim_region_max = _gf_read_vec3(r)
    anim_hash = int(r.u32())

    bones: List[_GFMotBoneTransform] = []
    uv_transforms: List[__GFMotUVTransform] = []
    visibility_tracks: List[__GFMotVisibilityTrack] = []
    unknown_sections: List[Tuple[int, int, int, bytes]] = []

    for sect_name, length, addr in sections[1:]:
        if sect_name == 1:                
            r.seek(addr)
            bone_names_count = r.s32()
            bone_names_len = r.u32()
            names_start = r.tell
            bone_names = [r.byte_len_string() for _ in range(bone_names_count)]
            r.seek(names_start + bone_names_len)
            for bn in bone_names:
                bone = _parse_gf_mot_bone_transform(data, r.tell, bn, frames_count)
                bones.append(bone)
                r.seek(getattr(bone, "_end_offset"))
        elif sect_name == 3:                
            uv_transforms.extend(_parse_gf_material_mot(data, addr, frames_count))
        elif sect_name == 6:                  
            r.seek(addr)
            mesh_names_count = r.s32()
            mesh_names_len = r.u32()
            names_start = r.tell
            mesh_names = [r.byte_len_string() for _ in range(mesh_names_count)]
            r.seek(names_start + mesh_names_len)

            for name in mesh_names:
                values: List[bool] = []
                cur = 0
                for i in range(int(frames_count) + 1):
                    bit = i & 7
                    if bit == 0:
                        cur = r.u8()
                    values.append((cur & (1 << bit)) != 0)
                visibility_tracks.append(
                    __GFMotVisibilityTrack(name=name, values=values)
                )
        else:
            raw = data[addr : addr + length] if 0 <= addr <= len(data) else b""
            unknown_sections.append((int(sect_name), int(length), int(addr), raw))

    return _GFMotion(
        index=index,
        frames_count=frames_count,
        is_looping=is_looping,
        is_blended=is_blended,
        anim_region_min=anim_region_min,
        anim_region_max=anim_region_max,
        anim_hash=anim_hash,
        bones=bones,
        uv_transforms=uv_transforms,
        visibility_tracks=visibility_tracks,
        unknown_sections=unknown_sections,
    )


def _parse_gf_material_mot(
    data: bytes, offset: int, frames_count: int
) -> List[__GFMotUVTransform]:
    r = _BinReader(data, offset)
    material_count = r.s32()
    names_len = r.u32()
    units = [r.u32() for _ in range(material_count)]
    names_start = r.tell
    names = [r.byte_len_string() for _ in range(material_count)]
    r.seek(names_start + names_len)

    out: List[__GFMotUVTransform] = []
    for mi, name in enumerate(names):
        for _u in range(units[mi]):
            unit_index = r.u32()
            flags = r.u32()
            _length = r.u32()
            f = flags
            sx = _mot_keyframes_set_list(r, f, frames_count)
            f >>= 3
            sy = _mot_keyframes_set_list(r, f, frames_count)
            f >>= 3
            rot = _mot_keyframes_set_list(r, f, frames_count)
            f >>= 3
            tx = _mot_keyframes_set_list(r, f, frames_count)
            f >>= 3
            ty = _mot_keyframes_set_list(r, f, frames_count)
            out.append(
                __GFMotUVTransform(
                    name=name,
                    unit_index=int(unit_index),
                    sx=sx,
                    sy=sy,
                    rot=rot,
                    tx=tx,
                    ty=ty,
                )
            )
    return out


def _mot_keyframes_set_list(
    r: _BinReader, flags: int, frames_count: int
) -> List[_GFMotKeyFrame]:
    mode = flags & 7
    if mode == 3:
        return [_GFMotKeyFrame(frame=0, value=r.f32(), slope=0.0)]
    if mode not in (4, 5):
        return []

    kf_count = r.u32()
    if frames_count > 0xFF:
        frames = [r.u16() for _ in range(kf_count)]
    else:
        frames = [r.u8() for _ in range(kf_count)]

    r.align(4)

    out: List[_GFMotKeyFrame] = []
    if (flags & 1) != 0:
        for i in range(kf_count):
            out.append(
                _GFMotKeyFrame(frame=int(frames[i]), value=r.f32(), slope=r.f32())
            )
        return out

    value_scale = r.f32()
    value_off = r.f32()
    slope_scale = r.f32()
    slope_off = r.f32()
    for i in range(kf_count):
        v = (r.u16() / 65535.0) * value_scale + value_off
        s = (r.u16() / 65535.0) * slope_scale + slope_off
        out.append(_GFMotKeyFrame(frame=int(frames[i]), value=float(v), slope=float(s)))
    return out


def _parse_gf_mot_bone_transform(
    data: bytes, offset: int, name: str, frames_count: int
) -> _GFMotBoneTransform:
    r = _BinReader(data, offset)
    flags = r.u32()
    _length = r.u32()
    is_axis_angle = (flags >> 31) == 0

    lists: List[List[_GFMotKeyFrame]] = []
    f = flags
    for _ in range(9):
        lists.append(_mot_keyframes_set_list(r, f, frames_count))
        f >>= 3

    bone = _GFMotBoneTransform(
        name=name,
        is_axis_angle=is_axis_angle,
        sx=lists[0],
        sy=lists[1],
        sz=lists[2],
        rx=lists[3],
        ry=lists[4],
        rz=lists[5],
        tx=lists[6],
        ty=lists[7],
        tz=lists[8],
    )
    setattr(bone, "_end_offset", r.tell)
    return bone


def _herp(
    lhs: float, rhs: float, ls: float, rs: float, diff: float, weight: float
) -> float:
    res = lhs + (lhs - rhs) * (2 * weight - 3) * weight * weight
    res += (diff * (weight - 1)) * (ls * (weight - 1) + rs * weight)
    return res


def _mot_eval(kfs: List[_GFMotKeyFrame], frame: int, default: float) -> float:
    if not kfs:
        return default
    if len(kfs) == 1:
        return kfs[0].value
    lhs = None
    rhs = None
    for kf in kfs:
        if kf.frame <= frame:
            lhs = kf
        if kf.frame >= frame:
            rhs = kf
            break
    if lhs is None:
        lhs = kfs[0]
    if rhs is None:
        rhs = kfs[-1]
    if lhs.frame == rhs.frame:
        return lhs.value
    diff = float(frame - lhs.frame)
    w = diff / float(rhs.frame - lhs.frame)
    return _herp(lhs.value, rhs.value, lhs.slope, rhs.slope, diff, w)
