
from __future__ import annotations

from ..types import _GFTexture
from .reader import _BinReader, _read_gf_section

def _parse_gf_texture(data: bytes) -> _GFTexture:
    r = _BinReader(data)
    _magic = r.u32()
    _count = r.u32()
    _sect_magic, _sect_len = _read_gf_section(r)
    tex_len = r.u32()
    r.skip(0x0C)
    name = r.padded_string(0x40)
    width = r.u16()
    height = r.u16()
    fmt = r.u16()
    _mipmap = r.u16()
    r.skip(0x10)
    raw = r.read(tex_len)
    return _GFTexture(name=name, width=width, height=height, fmt=fmt, raw=raw)
