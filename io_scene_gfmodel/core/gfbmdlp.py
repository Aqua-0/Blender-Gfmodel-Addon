"""Parser for BinaryModelPacker's packed output (.gfbmdlp).

This is *not* the same as the in-game GFModelPack (0x00010000) used by many extracted
0.bin files. BinaryModelPacker produces a file that begins with 0x00010000 but then
stores a flat list of whole compiled files (.gfbmdl/.btex/.gfbvsh/.gfbfsh/.gfbgsh/.gfbmot)
with a filename+offset table and 0x80-aligned file payloads.

Reference: ModelPacker/Packer/BinaryModelPacker/Program.cs (PackModel/PackFiles).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class GfbmdlpEntry:
    index: int
    name: str
    data_off: int
    data_end: int
                                                                             
    trimmed_end: Optional[int] = None


def _read_7bit_u32(data: bytes, off: int) -> Tuple[int, int]:
    """Read C# BinaryWriter/BinaryReader 7-bit encoded int."""
    value = 0
    shift = 0
    for _ in range(5):
        if off >= len(data):
            raise ValueError("7bit int out of range")
        b = data[off]
        off += 1
        value |= (b & 0x7F) << shift
        if (b & 0x80) == 0:
            return int(value), int(off)
        shift += 7
    raise ValueError("invalid 7bit int (too long)")


def _align_up(x: int, a: int) -> int:
    return (int(x) + (int(a) - 1)) & ~(int(a) - 1)


def _try_guess_trimmed_end(blob: bytes) -> Optional[int]:
    """Best-effort file length detection for common toolchain-wrapped binaries."""
    if len(blob) < 8:
        return None
    u0 = struct.unpack_from("<I", blob, 0)[0]

                                                                                 
    if u0 == 0x15052616 and len(blob) >= 0x20:
        section_count = struct.unpack_from("<I", blob, 4)[0]
        cur = 0x10
        for _ in range(int(section_count)):
            if cur + 16 > len(blob):
                return None
                              
            sect_len = struct.unpack_from("<I", blob, cur + 8)[0]
            cur = _align_up(cur + 16 + int(sect_len), 0x10)
        return int(cur)

                                                                             
    if u0 == 0x14110400 and len(blob) >= 0x18:
        section_count = struct.unpack_from("<I", blob, 4)[0]
        if int(section_count) <= 0:
            return None
        if 0x08 + 16 > len(blob):
            return None
        sect_len = struct.unpack_from("<I", blob, 0x08 + 8)[0]
        return int(0x08 + 16 + int(sect_len))

                                                              
    if u0 == 0x00050000 and len(blob) >= 0x0C:
        section_count = struct.unpack_from("<I", blob, 4)[0]
        table_off = 0x08
        table_len = int(section_count) * 12
        if table_off + table_len > len(blob):
            return None
        max_end = 0
        for i in range(int(section_count)):
            _name, length, addr = struct.unpack_from("<III", blob, table_off + i * 12)
            max_end = max(int(max_end), int(addr) + int(length))
        return int(max_end)

    return None


def parse_gfbmdlp(data: bytes) -> List[GfbmdlpEntry]:
    """Parse a BinaryModelPacker .gfbmdlp blob and return entries.

    The file begins with:
      u32 magic/version (0x00010000)
      u32 model_count
      u32 texture_count
      u32 vsh_count
      u32 gsh_count
      u32 fsh_count

    Then a table of u32 offsets (count = sum of all files packed) pointing to each
    filename+dataOffset record:
      string name (C# BinaryWriter 7-bit length + UTF-8 bytes)
      u32 data_off (absolute, 0x80-aligned)

    File payloads are stored at data_off, also 0x80-aligned, with zero padding between.
    """
    if len(data) < 0x18:
        raise ValueError("too small for gfbmdlp header")
    magic = struct.unpack_from("<I", data, 0)[0]
    if int(magic) != 0x00010000:
        raise ValueError("not a gfbmdlp (magic != 0x00010000)")

    model_count, tex_count, vsh_count, gsh_count, fsh_count = struct.unpack_from(
        "<IIIII", data, 4
    )
    file_count = (
        int(model_count)
        + int(tex_count)
        + int(vsh_count)
        + int(gsh_count)
        + int(fsh_count)
    )
    if file_count <= 0:
        return []
    offs_base = 0x18
    offs_end = offs_base + file_count * 4
    if offs_end > len(data):
        raise ValueError("offset table out of range")
    name_ptrs = [
        struct.unpack_from("<I", data, offs_base + i * 4)[0] for i in range(file_count)
    ]

    entries_tmp: List[Tuple[int, str, int]] = []
    for i, ptr in enumerate(name_ptrs):
        if ptr <= 0 or ptr >= len(data):
            raise ValueError(f"name ptr out of range: {ptr} (i={i})")
        slen, p = _read_7bit_u32(data, int(ptr))
        if slen < 0 or p + slen > len(data):
            raise ValueError("string out of range")
        name = data[p : p + slen].decode("utf-8", "replace")
        p2 = p + slen
        if p2 + 4 > len(data):
            raise ValueError("missing data offset")
        data_off = struct.unpack_from("<I", data, p2)[0]
        entries_tmp.append((int(i), str(name), int(data_off)))

                                            
    sorted_by_off = sorted(entries_tmp, key=lambda t: int(t[2]))
    data_ends: dict[int, int] = {}
    for (i, _n, off), nxt in zip(
        sorted_by_off, sorted_by_off[1:] + [(None, "", len(data))]
    ):                           
        end = int(nxt[2]) if nxt[0] is not None else int(nxt[2])
        data_ends[int(i)] = int(end)

    out: List[GfbmdlpEntry] = []
    for i, name, off in entries_tmp:
        end = int(data_ends[int(i)])
        if not (0 <= off <= end <= len(data)):
            raise ValueError("invalid data range")
        blob = data[off:end]
        trimmed = _try_guess_trimmed_end(blob)
        trimmed_end = (
            (off + int(trimmed))
            if trimmed is not None and int(trimmed) <= len(blob)
            else None
        )
        out.append(
            GfbmdlpEntry(
                index=int(i),
                name=str(name),
                data_off=int(off),
                data_end=int(end),
                trimmed_end=trimmed_end,
            )
        )
    return sorted(out, key=lambda e: int(e.index))
