

from __future__ import annotations

import struct
from typing import Tuple

from ..lz11 import decompress as _lz11_decompress
from ..lz11 import looks_like_lz11 as _lz11_looks_like
from ..math_compat import Vector

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

    return _lz11_decompress(bytes(data))


def _looks_like_lz11(data: bytes) -> bool:

    return bool(_lz11_looks_like(bytes(data)))


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
