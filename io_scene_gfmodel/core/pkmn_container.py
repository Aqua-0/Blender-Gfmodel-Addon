"""Pokemon-style 2-byte containers (CP/CM/etc).

Format (see `docs/gfmodel_binary_format.md`):
- `char[2] magic` (ASCII)
- `u16 count`
- `u32[count+1] offsets` (relative to start of container)
- entry i is `data[offsets[i]..offsets[i+1]]`
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
class _PkmnContainer:
    magic2: str
    offsets: List[int]

    @property
    def count(self) -> int:
        return max(0, len(self.offsets) - 1)

    def extract(self, blob: bytes, index: int) -> bytes:
        if index < 0 or index >= self.count:
            raise IndexError("container index out of range")
        start = int(self.offsets[index])
        end = int(self.offsets[index + 1])
        if start < 0 or end < start or end > len(blob):
            raise ValueError("container slice out of range")
        return blob[start:end]


def parse_container(blob: bytes) -> _PkmnContainer:
    b = memoryview(blob)
    if len(b) < 8:
        raise ValueError("file too small for container")
    magic2 = b[0:2].tobytes().decode("ascii", "replace")
    count = _u16le(b, 0x02)
    if count <= 0 or count > 0x4000:
        raise ValueError("invalid container count")
    table_bytes = 4 + (int(count) + 1) * 4
    if table_bytes > len(b):
        raise ValueError("container table out of range")
    offsets = [_u32le(b, 0x04 + i * 4) for i in range(int(count) + 1)]
    prev = 0
    for off in offsets:
        if off < prev or off > len(b):
            raise ValueError("container offsets invalid")
        prev = off
    return _PkmnContainer(magic2=magic2, offsets=[int(x) for x in offsets])


def _align_up(x: int, a: int) -> int:
    a = int(a)
    if a <= 1:
        return int(x)
    return (int(x) + (a - 1)) & ~(a - 1)


def patch_container(
    blob: bytes,
    *,
    index: int,
    replacement: bytes,
    align: int = 0x80,
    pad_byte: int = 0x00,
) -> bytes:
    """Patch a single container entry.

    If the replacement fits in the existing stored segment (`len(replacement) <= len(old_seg)`),
    this performs an in-place patch and preserves the original header/offset table bytes exactly.
    Otherwise, it rebuilds the container using `align` and `pad_byte`.
    """
    cont = parse_container(blob)
    if index < 0 or index >= cont.count:
        raise IndexError("container index out of range")

    rep = bytes(replacement)

                                                                                                    
    start = int(cont.offsets[index])
    end = int(cont.offsets[index + 1])
    if start < 0 or end < start or end > len(blob):
        raise ValueError("container slice out of range")
    old_seg = blob[start:end]
    if rep == old_seg:
        return bytes(blob)
    if len(rep) <= len(old_seg):
        out = bytearray(blob)
        out[start : start + len(rep)] = rep
        return bytes(out)

                                                                 
    segments: List[bytes] = []
    for i in range(cont.count):
        s = int(cont.offsets[i])
        e = int(cont.offsets[i + 1])
        segments.append(blob[s:e])
    stored_len = int(_align_up(len(rep), int(align)))
    segments[index] = rep + bytes((int(pad_byte) & 0xFF,)) * (stored_len - len(rep))

    count = cont.count
    header_end = 4 + (count + 1) * 4
    first_off = int(_align_up(header_end, int(align)))

    offsets: List[int] = []
    cur = first_off
    for seg in segments:
        cur = int(_align_up(cur, int(align)))
        offsets.append(cur)
        cur += len(seg)
    offsets.append(int(cur))

    out = bytearray()
    magic_b = (cont.magic2 or "\x00\x00").encode("ascii", "replace")[:2]
    if len(magic_b) < 2:
        magic_b = (magic_b + b"\x00\x00")[:2]
    out += magic_b
    out += struct.pack("<H", int(count))
    for o in offsets:
        out += struct.pack("<I", int(o))

    if len(out) < first_off:
        out += bytes((int(pad_byte) & 0xFF,)) * (first_off - len(out))

    for i, seg in enumerate(segments):
        start = int(offsets[i])
        if len(out) < start:
            out += bytes((int(pad_byte) & 0xFF,)) * (start - len(out))
        out += seg

    final_len = int(offsets[-1])
    if len(out) != final_len:
        if len(out) < final_len:
            out += bytes((int(pad_byte) & 0xFF,)) * (final_len - len(out))
        else:
            out = out[:final_len]

    return bytes(out)
