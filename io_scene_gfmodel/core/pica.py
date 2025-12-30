"""PICA200 / GPU helper routines.

Includes:
- ETC1/ETC1A4 decompression
- GF texture decode -> BGRA / RGBA float helpers
- PICA command stream decoding helpers
- TexEnv register decoding helpers

No Blender registration logic lives here.
"""

from __future__ import annotations

import struct
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .types import _GFShaderTexEnvStage


def _swap64(x: int) -> int:
    return int.from_bytes(x.to_bytes(8, "little"), "big")


def _etc1_decompress(input_data: bytes, width: int, height: int, alpha: bool) -> bytes:
    xt = (0, 4, 0, 4)
    yt = (0, 0, 4, 4)

    def saturate(v: int) -> int:
        if v < 0:
            return 0
        if v > 255:
            return 255
        return v

    etc_lut = (
        (2, 8, -2, -8),
        (5, 17, -5, -17),
        (9, 29, -9, -29),
        (13, 42, -13, -42),
        (18, 60, -18, -60),
        (24, 80, -24, -80),
        (33, 106, -33, -106),
        (47, 183, -47, -183),
    )

    def etc1_pixel(
        r: int, g: int, b: int, x: int, y: int, block: int, table: int
    ) -> Tuple[int, int, int]:
        index = x * 4 + y
        msb = (block << 1) & 0xFFFFFFFF
        if index < 8:
            lut_idx = ((block >> (index + 24)) & 1) + ((msb >> (index + 8)) & 2)
        else:
            lut_idx = ((block >> (index + 8)) & 1) + ((msb >> (index - 8)) & 2)
        pixel = etc_lut[table][lut_idx]
        return (
            saturate(r + pixel),
            saturate(g + pixel),
            saturate(b + pixel),
        )

    def etc1_tile(block: int) -> bytes:
        block_low = (block >> 32) & 0xFFFFFFFF
        block_high = block & 0xFFFFFFFF

        flip = (block_high & 0x01000000) != 0
        diff = (block_high & 0x02000000) != 0

        def sign_extend_3(v: int) -> int:
            v &= 0x7
            return v - 8 if (v & 0x4) else v

        if diff:
            b1 = (block_high & 0x0000F8) >> 0
            g1 = (block_high & 0x00F800) >> 8
            r1 = (block_high & 0xF80000) >> 16

            b2 = (int(b1 >> 3) + sign_extend_3(block_high & 0x000007)) & 0x1F
            g2 = (int(g1 >> 3) + sign_extend_3((block_high & 0x000700) >> 8)) & 0x1F
            r2 = (int(r1 >> 3) + sign_extend_3((block_high & 0x070000) >> 16)) & 0x1F

            b1 = int(b1 | (b1 >> 5))
            g1 = int(g1 | (g1 >> 5))
            r1 = int(r1 | (r1 >> 5))

            b2 = ((b2 << 3) | (b2 >> 2)) & 0xFF
            g2 = ((g2 << 3) | (g2 >> 2)) & 0xFF
            r2 = ((r2 << 3) | (r2 >> 2)) & 0xFF
        else:
            b1 = (block_high & 0x0000F0) >> 0
            g1 = (block_high & 0x00F000) >> 8
            r1 = (block_high & 0xF00000) >> 16

            b2 = (block_high & 0x00000F) << 4
            g2 = (block_high & 0x000F00) >> 4
            r2 = (block_high & 0x0F0000) >> 12

            b1 = int(b1 | (b1 >> 4))
            g1 = int(g1 | (g1 >> 4))
            r1 = int(r1 | (r1 >> 4))

            b2 = int(b2 | (b2 >> 4))
            g2 = int(g2 | (g2 >> 4))
            r2 = int(r2 | (r2 >> 4))

        table1 = (block_high >> 29) & 7
        table2 = (block_high >> 26) & 7

        out = bytearray(4 * 4 * 4)                        

        if not flip:
            for y in range(4):
                for x in range(2):
                    c1 = etc1_pixel(r1, g1, b1, x + 0, y, block_low, table1)
                    c2 = etc1_pixel(r2, g2, b2, x + 2, y, block_low, table2)
                    o1 = (y * 4 + x) * 4
                    o2 = (y * 4 + x + 2) * 4
                    out[o1 + 0], out[o1 + 1], out[o1 + 2] = c1[2], c1[1], c1[0]
                    out[o2 + 0], out[o2 + 1], out[o2 + 2] = c2[2], c2[1], c2[0]
        else:
            for y in range(2):
                for x in range(4):
                    c1 = etc1_pixel(r1, g1, b1, x, y + 0, block_low, table1)
                    c2 = etc1_pixel(r2, g2, b2, x, y + 2, block_low, table2)
                    o1 = (y * 4 + x) * 4
                    o2 = ((y + 2) * 4 + x) * 4
                    out[o1 + 0], out[o1 + 1], out[o1 + 2] = c1[2], c1[1], c1[0]
                    out[o2 + 0], out[o2 + 1], out[o2 + 2] = c2[2], c2[1], c2[0]

        return bytes(out)

    out = bytearray(width * height * 4)        
    inp = memoryview(input_data)
    in_off = 0

    for ty in range(0, height, 8):
        for tx in range(0, width, 8):
            for t in range(4):
                alpha_block = 0xFFFFFFFFFFFFFFFF
                if alpha:
                    alpha_block = struct.unpack_from("<Q", inp, in_off)[0]
                    in_off += 8
                color_block = struct.unpack_from("<Q", inp, in_off)[0]
                in_off += 8
                color_block = _swap64(color_block)
                tile = bytearray(etc1_tile(color_block))
                tile_off = 0
                for py in range(yt[t], yt[t] + 4):
                    for px in range(xt[t], xt[t] + 4):
                        o = ((height - 1 - (ty + py)) * width + (tx + px)) * 4
                        out[o : o + 3] = tile[tile_off : tile_off + 3]
                        alpha_shift = ((px & 3) * 4 + (py & 3)) << 2
                        a = (alpha_block >> alpha_shift) & 0xF
                        out[o + 3] = (a << 4) | a
                        tile_off += 4
    return bytes(out)


_SWIZZLE_LUT = (
    0,
    1,
    8,
    9,
    2,
    3,
    10,
    11,
    16,
    17,
    24,
    25,
    18,
    19,
    26,
    27,
    4,
    5,
    12,
    13,
    6,
    7,
    14,
    15,
    20,
    21,
    28,
    29,
    22,
    23,
    30,
    31,
    32,
    33,
    40,
    41,
    34,
    35,
    42,
    43,
    48,
    49,
    56,
    57,
    50,
    51,
    58,
    59,
    36,
    37,
    44,
    45,
    38,
    39,
    46,
    47,
    52,
    53,
    60,
    61,
    54,
    55,
    62,
    63,
)


def _pica_decode_to_bgra(raw: bytes, width: int, height: int, fmt: int) -> bytes:
                                       
                                                                                                            
                                                                                                    
    gf_to_pica = {
        0x2: 3,          
        0x3: 1,        
        0x4: 0,         
        0x16: 4,         
        0x17: 2,            
        0x23: 5,       
        0x24: 6,         
        0x25: 7,      
        0x26: 8,      
        0x27: 9,       
        0x28: 10,      
        0x29: 11,      
        0x2A: 12,        
        0x2B: 13,          
    }
                                                                                  
                                                                                           
    if int(fmt) in gf_to_pica:
        fmt = int(gf_to_pica[int(fmt)])
    elif int(fmt) not in range(14):
        return bytes([255, 0, 255, 255]) * (width * height)
    if fmt == 12:
        return _etc1_decompress(raw, width, height, alpha=False)
    if fmt == 13:
        return _etc1_decompress(raw, width, height, alpha=True)

    fmt_bpp = (32, 24, 16, 16, 16, 16, 16, 8, 8, 8, 4, 4, 4, 8)
    inc = fmt_bpp[fmt] // 8
    if inc == 0:
        inc = 1

    out = bytearray(width * height * 4)        
    inp = memoryview(raw)
    i_off = 0

    def get_u16(off: int) -> int:
        return inp[off] | (inp[off + 1] << 8)

    def decode_rgba5551(val: int) -> Tuple[int, int, int, int]:
        r = ((val >> 1) & 0x1F) << 3
        g = ((val >> 6) & 0x1F) << 3
        b = ((val >> 11) & 0x1F) << 3
        a = (val & 1) * 0xFF
        return (b | (b >> 5), g | (g >> 5), r | (r >> 5), a)

    def decode_rgb565(val: int) -> Tuple[int, int, int, int]:
        r = ((val >> 0) & 0x1F) << 3
        g = ((val >> 5) & 0x3F) << 2
        b = ((val >> 11) & 0x1F) << 3
        return (b | (b >> 5), g | (g >> 6), r | (r >> 5), 0xFF)

    def decode_rgba4(val: int) -> Tuple[int, int, int, int]:
        r = (val >> 4) & 0xF
        g = (val >> 8) & 0xF
        b = (val >> 12) & 0xF
        a = val & 0xF
        return ((b << 4) | b, (g << 4) | g, (r << 4) | r, (a << 4) | a)

    for ty in range(0, height, 8):
        for tx in range(0, width, 8):
            for px in range(64):
                x = _SWIZZLE_LUT[px] & 7
                y = (_SWIZZLE_LUT[px] - x) >> 3
                o_off = (tx + x + ((height - 1 - (ty + y)) * width)) * 4
                if fmt == 0:                          
                    out[o_off + 0] = inp[i_off + 3]
                    out[o_off + 1] = inp[i_off + 2]
                    out[o_off + 2] = inp[i_off + 1]
                    out[o_off + 3] = inp[i_off + 0]
                elif fmt == 1:        
                    out[o_off + 0] = inp[i_off + 2]
                    out[o_off + 1] = inp[i_off + 1]
                    out[o_off + 2] = inp[i_off + 0]
                    out[o_off + 3] = 0xFF
                elif fmt == 2:
                    b, g, r_, a = decode_rgba5551(get_u16(i_off))
                    out[o_off : o_off + 4] = bytes((b, g, r_, a))
                elif fmt == 3:
                    b, g, r_, a = decode_rgb565(get_u16(i_off))
                    out[o_off : o_off + 4] = bytes((b, g, r_, a))
                elif fmt == 4:
                    b, g, r_, a = decode_rgba4(get_u16(i_off))
                    out[o_off : o_off + 4] = bytes((b, g, r_, a))
                elif fmt == 5:       
                    l = int(inp[i_off + 1])
                    a = int(inp[i_off + 0])
                    out[o_off : o_off + 4] = bytes((l, l, l, a))
                elif fmt == 6:         
                    out[o_off + 0] = inp[i_off + 1]
                    out[o_off + 1] = inp[i_off + 0]
                    out[o_off + 2] = 0
                    out[o_off + 3] = 0xFF
                elif fmt == 7:      
                    l = int(inp[i_off])
                    out[o_off : o_off + 4] = bytes((l, l, l, 0xFF))
                elif fmt == 8:      
                    a = int(inp[i_off])
                    out[o_off : o_off + 4] = bytes((0xFF, 0xFF, 0xFF, a))
                elif fmt == 9:       
                    v = int(inp[i_off])
                    l = ((v >> 4) | (v & 0xF0)) & 0xFF
                    a = ((v << 4) | (v & 0x0F)) & 0xFF
                    out[o_off : o_off + 4] = bytes((l, l, l, a))
                elif fmt == 10:      
                    v = int(inp[i_off >> 1])
                    l4 = (v >> ((i_off & 1) << 2)) & 0xF
                    l = (l4 << 4) | l4
                    out[o_off : o_off + 4] = bytes((l, l, l, 0xFF))
                elif fmt == 11:      
                    v = int(inp[i_off >> 1])
                    a4 = (v >> ((i_off & 1) << 2)) & 0xF
                    a = (a4 << 4) | a4
                    out[o_off : o_off + 4] = bytes((0xFF, 0xFF, 0xFF, a))
                else:
                    out[o_off : o_off + 4] = b"\xff\x00\xff\xff"
                i_off += inc

    return bytes(out)


def _pica_decode_bitmap_to_bgra(raw: bytes, width: int, height: int, fmt: int) -> bytes:
                                                                         
    buf = _pica_decode_to_bgra(raw, width, height, fmt)
    stride = width * 4
    out = bytearray(len(buf))
    for y in range(height):
        i_off = stride * y
        o_off = stride * (height - 1 - y)
        for _x in range(width):
            out[o_off + 0] = buf[i_off + 2]
            out[o_off + 1] = buf[i_off + 1]
            out[o_off + 2] = buf[i_off + 0]
            out[o_off + 3] = buf[i_off + 3]
            i_off += 4
            o_off += 4
    return bytes(out)


def _flip_bgra_y(bgra: bytes, width: int, height: int) -> bytes:
    stride = width * 4
    out = bytearray(len(bgra))
    for y in range(height):
        src = y * stride
        dst = (height - 1 - y) * stride
        out[dst : dst + stride] = bgra[src : src + stride]
    return bytes(out)


def _bgra_to_rgba_floats(bgra: bytes) -> List[float]:
    inv = 1.0 / 255.0
    out: List[float] = [0.0] * (len(bgra))
    for i in range(0, len(bgra), 4):
        b = bgra[i + 0]
        g = bgra[i + 1]
        r = bgra[i + 2]
        a = bgra[i + 3]
        out[i + 0] = r * inv
        out[i + 1] = g * inv
        out[i + 2] = b * inv
        out[i + 3] = a * inv
    return out


def _decode_rgba_u32(param: int) -> Tuple[float, float, float, float]:
    r = (param >> 0) & 0xFF
    g = (param >> 8) & 0xFF
    b = (param >> 16) & 0xFF
    a = (param >> 24) & 0xFF
    return (r / 255.0, g / 255.0, b / 255.0, a / 255.0)


_TEXENV_SOURCE_NAMES = {
    0: "PrimaryColor",
    1: "FragmentPrimaryColor",
    2: "FragmentSecondaryColor",
    3: "Texture0",
    4: "Texture1",
    5: "Texture2",
    6: "Texture3",
    13: "PreviousBuffer",
    14: "Constant",
    15: "Previous",
}

_TEXENV_COLOR_OP_NAMES = {
    0: "Color",
    1: "OneMinusColor",
    2: "Alpha",
    3: "OneMinusAlpha",
    4: "Red",
    5: "OneMinusRed",
    8: "Green",
    9: "OneMinusGreen",
    12: "Blue",
    13: "OneMinusBlue",
}

_TEXENV_ALPHA_OP_NAMES = {
    0: "Alpha",
    1: "OneMinusAlpha",
    2: "Red",
    3: "OneMinusRed",
    4: "Green",
    5: "OneMinusGreen",
    6: "Blue",
    7: "OneMinusBlue",
}

_TEXENV_COMBINER_NAMES = {
    0: "Replace",
    1: "Modulate",
    2: "Add",
    3: "AddSigned",
    4: "Interpolate",
    5: "Subtract",
    6: "DotProduct3Rgb",
    7: "DotProduct3Rgba",
    8: "MultAdd",
    9: "AddMult",
}

_TEXENV_SCALE_NAMES = {
    0: "One",
    1: "Two",
    2: "Four",
}


def _decode_texenv_source(p: int) -> Dict[str, object]:
    c0 = (p >> 0) & 0xF
    c1 = (p >> 4) & 0xF
    c2 = (p >> 8) & 0xF
    a0 = (p >> 16) & 0xF
    a1 = (p >> 20) & 0xF
    a2 = (p >> 24) & 0xF
    return {
        "raw": f"0x{p:08X}",
        "color": [
            {"id": c0, "name": _TEXENV_SOURCE_NAMES.get(c0, f"Unknown({c0})")},
            {"id": c1, "name": _TEXENV_SOURCE_NAMES.get(c1, f"Unknown({c1})")},
            {"id": c2, "name": _TEXENV_SOURCE_NAMES.get(c2, f"Unknown({c2})")},
        ],
        "alpha": [
            {"id": a0, "name": _TEXENV_SOURCE_NAMES.get(a0, f"Unknown({a0})")},
            {"id": a1, "name": _TEXENV_SOURCE_NAMES.get(a1, f"Unknown({a1})")},
            {"id": a2, "name": _TEXENV_SOURCE_NAMES.get(a2, f"Unknown({a2})")},
        ],
    }


def _decode_texenv_operand(p: int) -> Dict[str, object]:
    c0 = (p >> 0) & 0xF
    c1 = (p >> 4) & 0xF
    c2 = (p >> 8) & 0xF
    a0 = (p >> 12) & 0x7
    a1 = (p >> 16) & 0x7
    a2 = (p >> 20) & 0x7
    return {
        "raw": f"0x{p:08X}",
        "color": [
            {"id": c0, "name": _TEXENV_COLOR_OP_NAMES.get(c0, f"Unknown({c0})")},
            {"id": c1, "name": _TEXENV_COLOR_OP_NAMES.get(c1, f"Unknown({c1})")},
            {"id": c2, "name": _TEXENV_COLOR_OP_NAMES.get(c2, f"Unknown({c2})")},
        ],
        "alpha": [
            {"id": a0, "name": _TEXENV_ALPHA_OP_NAMES.get(a0, f"Unknown({a0})")},
            {"id": a1, "name": _TEXENV_ALPHA_OP_NAMES.get(a1, f"Unknown({a1})")},
            {"id": a2, "name": _TEXENV_ALPHA_OP_NAMES.get(a2, f"Unknown({a2})")},
        ],
    }


def _decode_texenv_combiner(p: int) -> Dict[str, object]:
    c = (p >> 0) & 0xF
    a = (p >> 16) & 0xF
    return {
        "raw": f"0x{p:08X}",
        "color": {"id": c, "name": _TEXENV_COMBINER_NAMES.get(c, f"Unknown({c})")},
        "alpha": {"id": a, "name": _TEXENV_COMBINER_NAMES.get(a, f"Unknown({a})")},
    }


def _decode_texenv_scale(p: int) -> Dict[str, object]:
    c = (p >> 0) & 0x3
    a = (p >> 16) & 0x3
    return {
        "raw": f"0x{p:08X}",
        "color": {"id": c, "name": _TEXENV_SCALE_NAMES.get(c, f"Unknown({c})")},
        "alpha": {"id": a, "name": _TEXENV_SCALE_NAMES.get(a, f"Unknown({a})")},
    }


def _decode_texenv_update_buffer(p: int) -> Dict[int, Dict[str, bool]]:
                                                    
    return {
        1: {
            "update_color_buffer": (p & 0x0100) != 0,
            "update_alpha_buffer": (p & 0x1000) != 0,
        },
        2: {
            "update_color_buffer": (p & 0x0200) != 0,
            "update_alpha_buffer": (p & 0x2000) != 0,
        },
        3: {
            "update_color_buffer": (p & 0x0400) != 0,
            "update_alpha_buffer": (p & 0x4000) != 0,
        },
        4: {
            "update_color_buffer": (p & 0x0800) != 0,
            "update_alpha_buffer": (p & 0x8000) != 0,
        },
    }


def _decode_texenv_stage_from_regs(
    stage: int, regs: Dict[int, int]
) -> _GFShaderTexEnvStage:
    base = 0x00C0 + stage * 8
    return _GFShaderTexEnvStage(
        stage=stage,
        source=regs.get(base + 0),
        operand=regs.get(base + 1),
        combiner=regs.get(base + 2),
        color=regs.get(base + 3),
        scale=regs.get(base + 4),
    )


def _pica_float24_to_float(word24: int) -> float:
    word24 &= 0xFFFFFF
    if (word24 & 0x7FFFFF) != 0:
        mantissa = word24 & 0xFFFF
        exponent = ((word24 >> 16) & 0x7F) + 64
        sign = (word24 >> 23) & 1
        u = (mantissa << 7) | (exponent << 23) | (sign << 31)
    else:
        u = (word24 & 0x800000) << 8
    return struct.unpack("<f", struct.pack("<I", u))[0]


def _parse_pica_vec_float24(
    w0: int, w1: int, w2: int
) -> Tuple[float, float, float, float]:
    x = _pica_float24_to_float(w2 & 0xFFFFFF)
    y = _pica_float24_to_float(((w2 >> 24) | ((w1 & 0xFFFF) << 8)) & 0xFFFFFF)
    z = _pica_float24_to_float(((w1 >> 16) | ((w0 & 0xFF) << 16)) & 0xFFFFFF)
    w = _pica_float24_to_float((w0 >> 8) & 0xFFFFFF)
    return x, y, z, w


def _pica_read_commands(cmds: Sequence[int]) -> Iterable[Tuple[int, List[int]]]:
    i = 0
    while i + 1 < len(cmds):
        param0 = cmds[i]
        cmd = cmds[i + 1]
        i += 2
        reg = cmd & 0xFFFF
        _mask = (cmd >> 16) & 0xF
        extra = (cmd >> 20) & 0x7FF
        consecutive = (cmd >> 31) != 0
        if consecutive:
            for j in range(extra + 1):
                yield (reg + j, [param0])
                if j < extra:
                    if i >= len(cmds):
                        return
                    param0 = cmds[i]
                    i += 1
        else:
            params = [param0]
            for _ in range(extra):
                if i >= len(cmds):
                    return
                params.append(cmds[i])
                i += 1
            yield (reg, params)
        if (i & 1) != 0:
            i += 1
