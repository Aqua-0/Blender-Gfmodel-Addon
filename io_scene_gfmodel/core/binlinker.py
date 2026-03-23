
from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import List, Optional, Tuple


def _u16(b: bytes, o: int) -> int:
    return int(struct.unpack_from("<H", b, o)[0])


def _u32(b: bytes, o: int) -> int:
    return int(struct.unpack_from("<I", b, o)[0])


def _align_up(v: int, align: int) -> int:
    a = int(align)
    if a <= 1:
        return int(v)
    return (int(v) + (a - 1)) & ~(a - 1)


@dataclass(frozen=True)
class BinLinker:
    signature: int
    file_count: int
    offsets: Tuple[int, ...]                           

    def span(self, index: int) -> Tuple[int, int]:
        i = int(index)
        if i < 0 or i >= int(self.file_count):
            raise IndexError("binlinker index out of range")
        a = int(self.offsets[i])
        b = int(self.offsets[i + 1])
        if b < a:
            raise ValueError("binlinker offset table invalid")
        return a, b

    def extract(self, blob: bytes, index: int) -> bytes:
        a, b = self.span(index)
        if a < 0 or b < a or b > len(blob):
            raise ValueError("binlinker slice out of range")
        return bytes(blob[a:b])


def looks_like_binlinker(data: bytes) -> bool:
    if len(data) < 8:
        return False
    file_count = _u16(data, 2)
    if file_count <= 0 or file_count > 0x4000:
        return False
    table_len = 4 + (int(file_count) + 1) * 4
    if table_len > len(data):
        return False
                                                                                        
                                                                                     
                                                                          
    first_off = _u32(data, 4)
    if first_off < table_len:
        return False
    prev = 0
    for i in range(int(file_count) + 1):
        off = _u32(data, 4 + i * 4)
        if off < prev or off > len(data):
            return False
        prev = off
    return True


def parse_binlinker(data: bytes) -> BinLinker:
    if not looks_like_binlinker(data):
        raise ValueError("not a BinLinker (header/offsets invalid)")
    sig = _u16(data, 0)
    file_count = _u16(data, 2)
    offsets: List[int] = []
    for i in range(int(file_count) + 1):
        offsets.append(_u32(data, 4 + i * 4))
    return BinLinker(signature=int(sig), file_count=int(file_count), offsets=tuple(offsets))


def rebuild_with_replaced_data(
    data: bytes,
    *,
    index: int,
    replacement: bytes,
    align_blobs: Optional[int] = None,
) -> bytes:
    bl = parse_binlinker(data)
    idx = int(index)
    if idx < 0 or idx >= int(bl.file_count):
        raise IndexError("binlinker replace index out of range")

    file_count = int(bl.file_count)
    offsets_count = file_count + 1
    header_end = 4 + offsets_count * 4

    first_off = int(bl.offsets[0]) if bl.offsets else int(header_end)
    is_padded = first_off >= header_end and first_off != header_end and (first_off % 0x80 == 0)
    align = int(align_blobs or (0x80 if is_padded else 1))

    out = bytearray()
    out += struct.pack("<HH", int(bl.signature) & 0xFFFF, int(file_count) & 0xFFFF)
    out += b"\x00" * (offsets_count * 4)

    cursor = _align_up(len(out), align) if align > 1 else len(out)
    if cursor > len(out):
        out += b"\x00" * (cursor - len(out))

    new_offsets: List[int] = [0] * offsets_count
    new_offsets[0] = int(cursor)

    for i in range(file_count):
        blob = bytes(replacement) if i == idx else bl.extract(data, i)
        out += blob
        if align > 1:
            cursor = _align_up(len(out), align)
            if cursor > len(out):
                out += b"\x00" * (cursor - len(out))
        new_offsets[i + 1] = int(len(out))

    for i, off in enumerate(new_offsets):
        struct.pack_into("<I", out, 4 + i * 4, int(off) & 0xFFFFFFFF)
    return bytes(out)
