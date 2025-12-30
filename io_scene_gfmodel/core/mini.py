"""pk3DS-style 'Mini' container parsing.

Mini format (see `Spica/pk3DS/pk3DS.Core/CTR/mini.cs`):
- `char[2] ident` (ASCII)
- `u16 count` (little-endian)
- `u32[count] offsets`
- `u32 length` (file size, also offsets[count])
- data blobs live at `offsets[i]..offsets[i+1]`
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import List


def _u16le(b: memoryview, o: int) -> int:
    return int(struct.unpack_from("<H", b, o)[0])


def _u32le(b: memoryview, o: int) -> int:
    return int(struct.unpack_from("<I", b, o)[0])


@dataclass(frozen=True)
class _Mini:
    ident: str
    offsets: List[int]
    size: int

    @property
    def count(self) -> int:
        return max(0, len(self.offsets) - 1)

    def extract(self, blob: bytes, index: int) -> bytes:
        if index < 0 or index >= self.count:
            raise IndexError("mini index out of range")
        start = int(self.offsets[index])
        end = int(self.offsets[index + 1])
        if start < 0 or end < start or end > int(self.size) or end > len(blob):
            raise ValueError("mini slice out of range")
        return blob[start:end]


def parse_mini(blob: bytes) -> _Mini:
    b = memoryview(blob)
    if len(b) < 8:
        raise ValueError("file too small for mini")
    ident = b[0:2].tobytes().decode("ascii", "replace")
    count = _u16le(b, 2)
    if count <= 0 or count > 0x4000:
        raise ValueError("invalid mini count")
    header_size = 4 + 4 + int(count) * 4
    if header_size > len(b):
        raise ValueError("mini header out of range")
    offsets: List[int] = []
    for i in range(int(count)):
        offsets.append(_u32le(b, 4 + i * 4))
    length = _u32le(b, 4 + int(count) * 4)
    if int(length) <= 0:
        raise ValueError("mini length invalid")
                                                                                                            
    if int(length) > len(b):
        raise ValueError("mini length mismatch")
    offsets.append(int(length))
                                                               
    prev = 0
    for off in offsets:
        if off < prev or off > int(length):
            raise ValueError("mini offsets invalid")
        prev = off
    return _Mini(ident=ident, offsets=offsets, size=int(length))


def _align_up(x: int, a: int) -> int:
    a = int(a)
    if a <= 1:
        return int(x)
    return (int(x) + (a - 1)) & ~(a - 1)


def patch_mini(
    blob: bytes,
    *,
    index: int,
    replacement: bytes,
    align: int = 0x80,
) -> bytes:
    """Patch a single mini subfile.

    If the replacement fits in the existing stored segment (`len(replacement) <= len(old_seg)`),
    this performs an in-place patch and preserves the original header/offset table bytes exactly.
    Otherwise, it rebuilds the mini using `align` (padding with zeros).
    """
    mini = parse_mini(blob)
    if index < 0 or index >= mini.count:
        raise IndexError("mini index out of range")

    rep = bytes(replacement)

                                             
    start = int(mini.offsets[index])
    end = int(mini.offsets[index + 1])
    if start < 0 or end < start or end > len(blob):
        raise ValueError("mini slice out of range")
    old_seg = blob[start:end]
    if rep == old_seg:
        return bytes(blob)
    if len(rep) <= len(old_seg):
        out = bytearray(blob)
        out[start : start + len(rep)] = rep
        return bytes(out)

                                                                 
                                                                      
    segments: List[bytes] = []
    for i in range(mini.count):
        s = int(mini.offsets[i])
        e = int(mini.offsets[i + 1])
        segments.append(blob[s:e])
    stored_len = int(_align_up(len(rep), int(align)))
    segments[index] = rep + (b"\x00" * (stored_len - len(rep)))

                                                                         
    count = mini.count
    header_end = 4 + count * 4 + 4
    first_off = int(_align_up(header_end, int(align)))

    offsets: List[int] = []
    cur = first_off
    for seg in segments:
        cur = int(_align_up(cur, int(align)))
        offsets.append(cur)
        cur += len(seg)
    length = int(cur)

    out = bytearray()
    ident_b = (mini.ident or "\x00\x00").encode("ascii", "replace")[:2]
    if len(ident_b) < 2:
        ident_b = (ident_b + b"\x00\x00")[:2]
    out += ident_b
    out += struct.pack("<H", int(count))
    for o in offsets:
        out += struct.pack("<I", int(o))
    out += struct.pack("<I", int(length))

    if len(out) < first_off:
        out += b"\x00" * (first_off - len(out))

    for i, seg in enumerate(segments):
        start = offsets[i]
        if len(out) < start:
            out += b"\x00" * (start - len(out))
        out += seg

    if len(out) != length:
                                                      
        if len(out) < length:
            out += b"\x00" * (length - len(out))
        else:
            out = out[:length]

    return bytes(out)
