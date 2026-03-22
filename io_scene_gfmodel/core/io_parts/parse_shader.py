
from __future__ import annotations

from typing import Tuple, Dict
from typing import Dict

from ..pica import _decode_texenv_stage_from_regs, _pica_read_commands

from ..types import _GFShader
from .reader import _BinReader, _read_gf_section, _gf_read_hash_name, _gf_skip_padding16

def _parse_gf_shader(data: bytes, offset: int) -> Tuple[_GFShader, int]:
    r = _BinReader(data, offset)
    _magic = r.u32()
    _count = r.u32()
    _gf_skip_padding16(r)
    _sect_magic, sect_len = _read_gf_section(r)
    start = r.tell

    name = r.padded_string(0x40)
    r.skip(4)
    r.skip(4)
    _gf_skip_padding16(r)

    cmd_len = r.u32()
    r.skip(4)
    r.skip(4)
    r.skip(4)
    r.padded_string(0x40)

    cmd_words = [r.u32() for _ in range(cmd_len // 4)]

    regs: Dict[int, int] = {}
    for reg, params in _pica_read_commands(cmd_words):
        regs[reg] = params[0]

    stages = [_decode_texenv_stage_from_regs(i, regs) for i in range(6)]
    texenv_buf = regs.get(0x00FD)
    texenv_update = regs.get(0x00E0)

    end = start + sect_len
    r.seek(end)
    return _GFShader(
        name=name,
        texenv_stages=stages,
        texenv_buffer_color=texenv_buf,
        texenv_update_buffer=texenv_update,
        pica_commands=cmd_words,
        pica_regs=regs,
    ), r.tell
