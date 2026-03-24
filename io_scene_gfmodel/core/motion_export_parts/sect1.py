

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Dict, List, Tuple

from .parse import _Sect, _align_up, _parse_section_table

class _RW:
    __slots__ = ("b", "o")

    def __init__(self, buf: bytearray, off: int):
        self.b = buf
        self.o = int(off)

    @property
    def tell(self) -> int:
        return int(self.o)

    def seek(self, off: int) -> None:
        self.o = int(off)

    def u8(self) -> int:
        v = self.b[self.o]
        self.o += 1
        return int(v)

    def u16(self) -> int:
        v = struct.unpack_from('<H', self.b, self.o)[0]
        self.o += 2
        return int(v)

    def u32(self) -> int:
        v = struct.unpack_from('<I', self.b, self.o)[0]
        self.o += 4
        return int(v)

    def s32(self) -> int:
        v = struct.unpack_from('<i', self.b, self.o)[0]
        self.o += 4
        return int(v)

    def f32(self) -> float:
        v = struct.unpack_from('<f', self.b, self.o)[0]
        self.o += 4
        return float(v)

    def write_f32_at(self, off: int, v: float) -> None:
        struct.pack_into('<f', self.b, int(off), float(v))

    def write_u16_at(self, off: int, v: int) -> None:
        struct.pack_into('<H', self.b, int(off), int(v) & 0xFFFF)

    def padded_string(self, length: int) -> str:
        raw = bytes(self.b[self.o : self.o + int(length)])
        self.o += int(length)
        end = raw.find(b'\0')
        if end >= 0:
            raw = raw[:end]
        return raw.decode('ascii', 'replace')

    def byte_len_string(self) -> str:
        return self.padded_string(self.u8())

    def align(self, boundary: int) -> None:
        m = int(boundary) - 1
        if (self.o & m) != 0:
            self.o += int(boundary) - (self.o & m)


def rewrite_gfmot_section1_values(
    *,
    src_bytes: bytes,
    frames_count: int,
    bones: Sequence[_GFMotBoneTransform],
) -> bytes:

    src = bytes(src_bytes)
    magic, sects = _parse_section_table(src)
    if magic != _GFMOT_MAGIC:
        raise ValueError(f'not a gfmot (magic=0x{magic:08X})')

    sec1 = next((s for s in sects if int(s.name) == 1), None)
    if sec1 is None:
        raise ValueError('gfmot missing section 1')

    out = bytearray(src)
    r = _RW(out, int(sec1.addr))

    bone_names_count = r.s32()
    bone_names_len = r.u32()
    names_start = r.tell
    names = [r.byte_len_string() for _ in range(int(bone_names_count))]
    r.seek(int(names_start) + int(bone_names_len))


    want_n = int(frames_count)
    by_name = {str(b.name): b for b in bones}

    def dense(kfs: Sequence[_GFMotKeyFrame]) -> List[float]:
        arr: List[float | None] = [None] * int(want_n)
        for kf in kfs:
            fi = int(getattr(kf, 'frame', 0) or 0)
            if 0 <= fi < want_n:
                arr[fi] = float(getattr(kf, 'value', 0.0) or 0.0)
        out: List[float] = [0.0] * int(want_n)
        last = 0.0
        for i in range(int(want_n)):
            v = arr[i]
            if v is None:
                out[i] = float(last)
            else:
                out[i] = float(v)
                last = float(v)
        return out

    dense_by_name: dict[str, dict[str, List[float]]] = {}
    for nm, bt in by_name.items():
        dense_by_name[nm] = {
            'sx': dense(bt.sx),
            'sy': dense(bt.sy),
            'sz': dense(bt.sz),
            'rx': dense(bt.rx),
            'ry': dense(bt.ry),
            'rz': dense(bt.rz),
            'tx': dense(bt.tx),
            'ty': dense(bt.ty),
            'tz': dense(bt.tz),
        }

    chan_keys = ['sx','sy','sz','rx','ry','rz','tx','ty','tz']

    for bn in names:
        bt_vals = dense_by_name.get(str(bn))

        flags_off = r.tell
        flags = r.u32()
        length = r.u32()
        bone_end = int(flags_off) + 8 + int(length)

        f = int(flags)
        for ci, ck in enumerate(chan_keys):
            mode = int(f & 7)
            if mode == 3:
                val_off = r.tell
                _old = r.f32()
                if bt_vals is not None and want_n > 0:
                    r.write_f32_at(val_off, float(bt_vals[ck][0]))
            elif mode in (4, 5):
                kf_count = r.u32()
                if want_n > 0xFF:
                    frames = [r.u16() for _ in range(int(kf_count))]
                else:
                    frames = [r.u8() for _ in range(int(kf_count))]
                r.align(4)

                if (f & 1) != 0:

                    for ki in range(int(kf_count)):
                        v_off = r.tell
                        _v = r.f32()
                        _s = r.f32()
                        if bt_vals is not None:
                            fr = int(frames[ki])
                            if 0 <= fr < want_n:
                                r.write_f32_at(v_off, float(bt_vals[ck][fr]))
                else:

                    value_scale = float(r.f32())
                    value_off = float(r.f32())
                    _slope_scale = float(r.f32())
                    _slope_off = float(r.f32())
                    for ki in range(int(kf_count)):
                        v_u16_off = r.tell
                        v_u16 = r.u16()
                        _s_u16 = r.u16()
                        if bt_vals is not None and value_scale != 0.0:
                            fr = int(frames[ki])
                            if 0 <= fr < want_n:
                                v = float(bt_vals[ck][fr])
                                t = (v - value_off) / value_scale
                                u = int(round(t * 65535.0))
                                if u < 0:
                                    u = 0
                                elif u > 65535:
                                    u = 65535
                                r.write_u16_at(v_u16_off, u)
            else:

                pass

            f >>= 3


        if r.tell != bone_end:
            r.seek(bone_end)

    return bytes(out)


class _Sect1BoneLayout:
    name: str
    flags: int
    is_axis_angle: bool
    channel_modes: List[int]
    channel_bytes: List[bytes]


def _parse_section1_layout(src_bytes: bytes, *, frames_count: int) -> tuple[bytes, List[_Sect1BoneLayout]]:
    src = bytes(src_bytes)
    magic, sects = _parse_section_table(src)
    if magic != _GFMOT_MAGIC:
        raise ValueError(f'not a gfmot (magic=0x{magic:08X})')
    sec1 = next((s for s in sects if int(s.name) == 1), None)
    if sec1 is None:
        raise ValueError('gfmot missing section 1')

    start = int(sec1.addr)
    end = int(sec1.addr) + int(sec1.length)
    if start < 0 or end < start or end > len(src):
        raise ValueError('section 1 out of range')

    b = bytearray(src[start:end])
    r = _RW(b, 0)

    bone_count = r.s32()
    names_len = r.u32()
    names_start = r.tell
    names = [r.byte_len_string() for _ in range(int(bone_count))]
    r.seek(int(names_start) + int(names_len))
    prefix = bytes(b[0 : int(r.tell)])

    layouts: List[_Sect1BoneLayout] = []

    for bn in names:
        flags = r.u32()
        length = r.u32()
        bone_payload_start = int(r.tell)
        bone_payload_end = bone_payload_start + int(length)
        if bone_payload_end < bone_payload_start or bone_payload_end > len(b):
            raise ValueError('bone payload out of range')

        channel_modes: List[int] = []
        channel_bytes: List[bytes] = []

        f = int(flags)
        for _ci in range(9):
            mode = int(f & 7)
            channel_modes.append(int(mode))
            chan_start = int(r.tell)
            if mode == 3:
                _ = r.f32()
            elif mode in (4, 5):
                kf_count = r.u32()
                if int(frames_count) > 0xFF:
                    for _ in range(int(kf_count)):
                        _ = r.u16()
                else:
                    for _ in range(int(kf_count)):
                        _ = r.u8()
                r.align(4)

                float_keys = (int(flags) & 1) != 0
                if float_keys:
                    for _ in range(int(kf_count)):
                        _ = r.f32(); _ = r.f32()
                else:
                    _ = r.f32(); _ = r.f32(); _ = r.f32(); _ = r.f32()
                    for _ in range(int(kf_count)):
                        _ = r.u16(); _ = r.u16()
            else:

                pass

            chan_end = int(r.tell)
            channel_bytes.append(bytes(b[chan_start:chan_end]))
            f >>= 3


        r.seek(int(bone_payload_end))

        layouts.append(
            _Sect1BoneLayout(
                name=str(bn),
                flags=int(flags) & 0xFFFFFFFF,
                is_axis_angle=((int(flags) >> 31) == 0),
                channel_modes=channel_modes,
                channel_bytes=channel_bytes,
            )
        )

    return prefix, layouts
