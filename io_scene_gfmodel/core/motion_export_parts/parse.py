

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Dict, List, Tuple

class _Sect:
    name: int
    length: int
    addr: int


def _align_up(x: int, boundary: int) -> int:
    m = int(boundary) - 1
    return (int(x) + m) & ~m


def _read_u32(data: bytes, off: int) -> int:
    return int(struct.unpack_from('<I', data, off)[0])


def _read_u16(data: bytes, off: int) -> int:
    return int(struct.unpack_from('<H', data, off)[0])


def _parse_section_table(data: bytes) -> Tuple[int, List[_Sect]]:
    if len(data) < 8:
        raise ValueError('gfmot too small')
    magic = _read_u32(data, 0)
    sect_count = _read_u32(data, 4)
    if sect_count <= 0 or sect_count > 0x1000:
        raise ValueError(f'invalid section count: {sect_count}')
    need = 8 + sect_count * 12
    if len(data) < need:
        raise ValueError('gfmot section table truncated')
    sects: List[_Sect] = []
    o = 8
    for _ in range(int(sect_count)):
        name = _read_u32(data, o + 0)
        length = _read_u32(data, o + 4)
        addr = _read_u32(data, o + 8)
        sects.append(_Sect(name=int(name), length=int(length), addr=int(addr)))
        o += 12
    return int(magic), sects


def _parse_frames_count(data: bytes, sect0: _Sect) -> int:
    if sect0.addr < 0 or sect0.addr + 4 > len(data):
        raise ValueError('section0 address out of range')
    return _read_u32(data, int(sect0.addr))
