"""PICA texture swizzle + encoding helpers.

This module is intentionally small and pure-Python (no Blender imports).
It complements `core/pica.py` which currently contains decoding.
"""

from __future__ import annotations

from typing import Dict, Tuple

from .pica import _SWIZZLE_LUT

                                                                             
_GF_FMT_TO_PICA: Dict[int, int] = {
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

                                                           
_PICA_TO_GF_FMT: Dict[int, int] = {
    0: 0x4,         
    1: 0x3,        
    2: 0x17,            
    3: 0x2,          
    4: 0x16,         
    5: 0x23,       
    6: 0x24,         
    7: 0x25,      
    8: 0x26,      
    9: 0x27,       
    10: 0x28,      
    11: 0x29,      
    12: 0x2A,        
    13: 0x2B,          
}


def gf_fmt_from_override_name(name: str) -> int:
    n = str(name or "").strip().upper()
    mapping = {
        "RGBA8": 0x4,
        "RGB8": 0x3,
        "RGBA5551": 0x17,
        "RGB565": 0x2,
        "RGBA4": 0x16,
        "LA8": 0x23,
        "HILO8": 0x24,
        "L8": 0x25,
        "A8": 0x26,
        "LA4": 0x27,
        "L4": 0x28,
        "A4": 0x29,
        "ETC1": 0x2A,
        "ETC1A4": 0x2B,
    }
    out = mapping.get(n)
    if out is None:
        raise ValueError(f"Unknown texture format override: {name!r}")
    return int(out)


def _clamp_u8(x: int) -> int:
    return 0 if x < 0 else (255 if x > 255 else int(x))


def _luma_u8(r: int, g: int, b: int) -> int:
                                            
    return _clamp_u8((77 * int(r) + 150 * int(g) + 29 * int(b) + 128) >> 8)


def _pack_rgba5551(r: int, g: int, b: int, a: int) -> int:
    r5 = int(r) >> 3
    g5 = int(g) >> 3
    b5 = int(b) >> 3
    a1 = 1 if int(a) >= 128 else 0
    return int((r5 << 1) | (g5 << 6) | (b5 << 11) | a1) & 0xFFFF


def _pack_rgb565(r: int, g: int, b: int) -> int:
    r5 = int(r) >> 3
    g6 = int(g) >> 2
    b5 = int(b) >> 3
    return int(r5 | (g6 << 5) | (b5 << 11)) & 0xFFFF


def _pack_rgba4(r: int, g: int, b: int, a: int) -> int:
    r4 = int(r) >> 4
    g4 = int(g) >> 4
    b4 = int(b) >> 4
    a4 = int(a) >> 4
    return int((r4 << 4) | (g4 << 8) | (b4 << 12) | a4) & 0xFFFF


def normalize_gf_fmt(fmt: int) -> Tuple[int, int]:
    """Return (gf_fmt, pica_fmt_index)."""
    if int(fmt) in _GF_FMT_TO_PICA:
        pica = int(_GF_FMT_TO_PICA[int(fmt)])
        return int(fmt), int(pica)
    if 0 <= int(fmt) <= 13:
        pica = int(fmt)
        gf = int(_PICA_TO_GF_FMT.get(pica, 0x4))
        return int(gf), int(pica)
    raise ValueError(f"Unsupported/unknown GFTexture fmt: 0x{int(fmt):X}")


def encode_pica_swizzled_from_rgba(
    raw_rgba: bytes,
    *,
    width: int,
    height: int,
    gf_fmt: int,
) -> bytes:
    """Encode linear RGBA8 (top-left origin) into GFTexture swizzled raw bytes for `gf_fmt`.

    Note: like decode, this applies the internal vertical flip (`sy = height-1-(ty+y)`).
    """
    w = int(width)
    h = int(height)
    if w <= 0 or h <= 0:
        raise ValueError("Invalid texture size")
    if (w % 8) != 0 or (h % 8) != 0:
        raise ValueError("PICA tiled textures require width/height multiples of 8")
    if len(raw_rgba) != w * h * 4:
        raise ValueError("raw_rgba size mismatch")

    gf_fmt, pica = normalize_gf_fmt(int(gf_fmt))

    if pica in (12, 13):
        return _encode_etc1_like(
            raw_rgba,
            width=w,
            height=h,
            with_alpha=(pica == 13),
        )

                                                        
    fmt_bpp = {
        0: 32,         
        1: 24,        
        2: 16,            
        3: 16,          
        4: 16,         
        5: 16,       
        6: 16,         
        7: 8,      
        8: 8,      
        9: 8,       
        10: 4,      
        11: 4,      
    }
    bpp = int(fmt_bpp.get(int(pica), 0))
    if bpp <= 0:
        raise ValueError(f"Unsupported PICA texture format index: {pica}")

    if bpp == 4:
        out = bytearray((w * h + 1) // 2)
    else:
        out = bytearray((w * h * bpp) // 8)

    def get_rgba_at(sx: int, sy: int) -> Tuple[int, int, int, int]:
        si = (int(sy) * w + int(sx)) * 4
        return (
            int(raw_rgba[si + 0]),
            int(raw_rgba[si + 1]),
            int(raw_rgba[si + 2]),
            int(raw_rgba[si + 3]),
        )

    i_off = 0
    for ty in range(0, h, 8):
        for tx in range(0, w, 8):
            for px in range(64):
                x = int(_SWIZZLE_LUT[px] & 7)
                y = int((_SWIZZLE_LUT[px] - x) >> 3)
                sx = int(tx + x)
                sy = int(h - 1 - (ty + y))
                r, g, b, a = get_rgba_at(sx, sy)

                if pica == 0:                          
                    out[i_off + 0] = int(a) & 0xFF
                    out[i_off + 1] = int(r) & 0xFF
                    out[i_off + 2] = int(g) & 0xFF
                    out[i_off + 3] = int(b) & 0xFF
                    i_off += 4
                elif pica == 1:        
                    out[i_off + 0] = int(r) & 0xFF
                    out[i_off + 1] = int(g) & 0xFF
                    out[i_off + 2] = int(b) & 0xFF
                    i_off += 3
                elif pica == 2:            
                    v = _pack_rgba5551(r, g, b, a)
                    out[i_off + 0] = v & 0xFF
                    out[i_off + 1] = (v >> 8) & 0xFF
                    i_off += 2
                elif pica == 3:          
                    v = _pack_rgb565(r, g, b)
                    out[i_off + 0] = v & 0xFF
                    out[i_off + 1] = (v >> 8) & 0xFF
                    i_off += 2
                elif pica == 4:         
                    v = _pack_rgba4(r, g, b, a)
                    out[i_off + 0] = v & 0xFF
                    out[i_off + 1] = (v >> 8) & 0xFF
                    i_off += 2
                elif pica == 5:              
                    l = _luma_u8(r, g, b)
                    out[i_off + 0] = int(a) & 0xFF
                    out[i_off + 1] = int(l) & 0xFF
                    i_off += 2
                elif pica == 6:                                     
                                                                                        
                    out[i_off + 0] = int(g) & 0xFF
                    out[i_off + 1] = int(b) & 0xFF
                    i_off += 2
                elif pica == 7:      
                    out[i_off] = _luma_u8(r, g, b) & 0xFF
                    i_off += 1
                elif pica == 8:      
                    out[i_off] = int(a) & 0xFF
                    i_off += 1
                elif pica == 9:                                    
                    l4 = (_luma_u8(r, g, b) >> 4) & 0xF
                    a4 = (int(a) >> 4) & 0xF
                    out[i_off] = int((l4 << 4) | a4) & 0xFF
                    i_off += 1
                elif pica == 10:      
                    l4 = (_luma_u8(r, g, b) >> 4) & 0xF
                    bi = i_off >> 1
                    if (i_off & 1) == 0:
                        out[bi] = int((out[bi] & 0xF0) | l4) & 0xFF
                    else:
                        out[bi] = int((out[bi] & 0x0F) | (l4 << 4)) & 0xFF
                    i_off += 1
                elif pica == 11:      
                    a4 = (int(a) >> 4) & 0xF
                    bi = i_off >> 1
                    if (i_off & 1) == 0:
                        out[bi] = int((out[bi] & 0xF0) | a4) & 0xFF
                    else:
                        out[bi] = int((out[bi] & 0x0F) | (a4 << 4)) & 0xFF
                    i_off += 1
                else:
                    raise ValueError(f"Unsupported PICA texture format index: {pica}")

    return bytes(out)


def _encode_etc1_like(
    raw_rgba: bytes, *, width: int, height: int, with_alpha: bool
) -> bytes:
    """Encode ETC1 / ETC1A4 in the same 8x8 tiled order expected by `_etc1_decompress`."""
    w = int(width)
    h = int(height)
    if (w % 8) != 0 or (h % 8) != 0:
        raise ValueError("ETC textures require width/height multiples of 8")
    if len(raw_rgba) != w * h * 4:
        raise ValueError("raw_rgba size mismatch")

                                                
    xt = (0, 4, 0, 4)
    yt = (0, 0, 4, 4)

    out = bytearray((w * h) if with_alpha else (w * h // 2))

    def get_rgba_at(sx: int, sy: int) -> Tuple[int, int, int, int]:
                                                
        sy = (h - 1 - int(sy)) & 0x7FFFFFFF
        si = (int(sy) * w + int(sx)) * 4
                                                                                             
                                                                                             
                                               
        r = int(raw_rgba[si + 0])
        g = int(raw_rgba[si + 1])
        b = int(raw_rgba[si + 2])
        a = int(raw_rgba[si + 3])
        return (b, g, r, a)

    o = 0
    for ty in range(0, h, 8):
        for tx in range(0, w, 8):
            for bi in range(4):
                bx = int(tx + xt[bi])
                by = int(ty + yt[bi])
                                                                                            
                                                                                
                block_rgba: list[Tuple[int, int, int, int]] = [(0, 0, 0, 0)] * 16
                for px in range(4):
                    for py in range(4):
                        idx = int(px) * 4 + int(py)
                        block_rgba[idx] = get_rgba_at(bx + px, by + py)

                if with_alpha:
                    ab = _encode_etc1a4_alpha_block(block_rgba)
                    out[o : o + 8] = ab.to_bytes(8, "little")
                    o += 8
                cb = _encode_etc1_color_block(block_rgba)
                                                                        
                out[o : o + 8] = cb.to_bytes(8, "big")
                o += 8
    return bytes(out)


def _quant4(x: int) -> int:
    return 0 if x < 0 else (15 if x > 15 else int(x))


def _quant5(x: int) -> int:
    return 0 if x < 0 else (31 if x > 31 else int(x))


def _avg_rgb(pixels: list[Tuple[int, int, int, int]]) -> Tuple[int, int, int]:
    if not pixels:
        return (0, 0, 0)
    r = sum(int(p[0]) for p in pixels) / len(pixels)
    g = sum(int(p[1]) for p in pixels) / len(pixels)
    b = sum(int(p[2]) for p in pixels) / len(pixels)
    return (int(round(r)), int(round(g)), int(round(b)))


_ETC_LUT = (
    (2, 8, -2, -8),
    (5, 17, -5, -17),
    (9, 29, -9, -29),
    (13, 42, -13, -42),
    (18, 60, -18, -60),
    (24, 80, -24, -80),
    (33, 106, -33, -106),
    (47, 183, -47, -183),
)


def _etc_apply(base: Tuple[int, int, int], mod: int) -> Tuple[int, int, int]:
    r = _clamp_u8(int(base[0]) + int(mod))
    g = _clamp_u8(int(base[1]) + int(mod))
    b = _clamp_u8(int(base[2]) + int(mod))
    return (r, g, b)


def _etc_err(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> int:
    dr = int(a[0]) - int(b[0])
    dg = int(a[1]) - int(b[1])
    db = int(a[2]) - int(b[2])
    return dr * dr + dg * dg + db * db


def _encode_etc1_color_block(pixels: list[Tuple[int, int, int, int]]) -> int:
    """Encode a single 4x4 block as ETC1 (individual mode only, best-effort)."""
    if len(pixels) != 16:
        raise ValueError("ETC1 block must have 16 pixels")

                                                                            
    def part(flip: int) -> Tuple[list[int], list[int]]:
        a_idx: list[int] = []
        b_idx: list[int] = []
        for x in range(4):
            for y in range(4):
                idx = x * 4 + y                                 
                if flip == 0:
                    (a_idx if x < 2 else b_idx).append(idx)
                else:
                    (a_idx if y < 2 else b_idx).append(idx)
        return a_idx, b_idx

    best_total = 1 << 60
    best = None

    for flip in (0, 1):
        a_idx, b_idx = part(flip)
        a_pix = [pixels[i] for i in a_idx]
        b_pix = [pixels[i] for i in b_idx]
        ar, ag, ab = _avg_rgb(a_pix)
        br, bg, bb = _avg_rgb(b_pix)

                                                       
        r1n = _quant4(int(round(ar / 17.0)))
        g1n = _quant4(int(round(ag / 17.0)))
        b1n = _quant4(int(round(ab / 17.0)))
        r2n = _quant4(int(round(br / 17.0)))
        g2n = _quant4(int(round(bg / 17.0)))
        b2n = _quant4(int(round(bb / 17.0)))
        c1 = (r1n * 17, g1n * 17, b1n * 17)
        c2 = (r2n * 17, g2n * 17, b2n * 17)

        for t1 in range(8):
            for t2 in range(8):
                selectors = [0] * 16
                total = 0
                for idx in range(16):
                    base = c1 if idx in a_idx else c2
                    table = t1 if idx in a_idx else t2
                    want = (
                        int(pixels[idx][0]),
                        int(pixels[idx][1]),
                        int(pixels[idx][2]),
                    )
                    best_i = 0
                    best_e = 1 << 60
                    for si, mod in enumerate(_ETC_LUT[table]):
                        got = _etc_apply(base, mod)
                        e = _etc_err(want, got)
                        if e < best_e:
                            best_e = e
                            best_i = si
                    selectors[idx] = int(best_i) & 3
                    total += int(best_e)
                    if total >= best_total:
                        break
                if total >= best_total:
                    continue

                best_total = int(total)
                best = (flip, r1n, g1n, b1n, r2n, g2n, b2n, t1, t2, selectors)

    if best is None:
        best = (0, 0, 0, 0, 0, 0, 0, 0, 0, [0] * 16)

    flip, r1n, g1n, b1n, r2n, g2n, b2n, t1, t2, selectors = best

                                                                                         
    block_low = 0
    for x in range(4):
        for y in range(4):
            idx = x * 4 + y
            s = int(selectors[idx]) & 3
            bit0 = s & 1
            bit1 = (s >> 1) & 1
            if idx < 8:
                if bit0:
                    block_low |= 1 << (idx + 24)
                if bit1:
                    block_low |= 1 << (idx + 8)
            else:
                if bit0:
                    block_low |= 1 << (idx + 8)
                if bit1:
                    block_low |= 1 << (idx - 8)
    block_low &= 0xFFFFFFFF

                                                              
    block_high = 0
    block_high |= (int(t1) & 7) << 29
    block_high |= (int(t2) & 7) << 26
    block_high |= 0 << 25          
    block_high |= (1 if int(flip) else 0) << 24
    block_high |= (int(r1n) & 0xF) << 20
    block_high |= (int(r2n) & 0xF) << 16
    block_high |= (int(g1n) & 0xF) << 12
    block_high |= (int(g2n) & 0xF) << 8
    block_high |= (int(b1n) & 0xF) << 4
    block_high |= (int(b2n) & 0xF) << 0
    block_high &= 0xFFFFFFFF

    return ((int(block_low) & 0xFFFFFFFF) << 32) | int(block_high)


def _encode_etc1a4_alpha_block(pixels: list[Tuple[int, int, int, int]]) -> int:
    """Encode ETC1A4 alpha (4 bits per pixel) for a 4x4 block (matches decoder nibble order)."""
    if len(pixels) != 16:
        raise ValueError("ETC1A4 alpha block must have 16 pixels")
    alpha_block = 0
    for x in range(4):
        for y in range(4):
            idx = x * 4 + y                                       
            a8 = int(pixels[idx][3])
            a4 = _quant4(int(round(a8 / 17.0)))
            shift = idx << 2
            alpha_block |= (int(a4) & 0xF) << shift
    return int(alpha_block) & 0xFFFFFFFFFFFFFFFF
