

from __future__ import annotations

import math
import struct
from typing import List, Tuple

def _slope_list(values: Sequence[float]) -> List[float]:
    n = len(values)
    if n <= 1:
        return [0.0] * n
    out: List[float] = [0.0] * n
    for i in range(n):
        if i == 0:
            out[i] = float(values[1] - values[0])
        elif i == n - 1:
            out[i] = float(values[n - 1] - values[n - 2])
        else:
            out[i] = float(values[i + 1] - values[i - 1]) * 0.5
    return out


def _encode_key_list(
    frames_count: int,
    values: Sequence[float],
    *,
    force_keys: bool = True,
    eps: float = 1e-6,
) -> Tuple[int, bytes]:

    n = int(frames_count)
    if n <= 0:

        return 0, b''

    if not values:
        if force_keys:
            return 3, struct.pack('<f', 0.0)
        return 0, b''

    if len(values) != n:
        raise ValueError(f'values length mismatch: expected={n} got={len(values)}')

    v0 = float(values[0])
    if all(abs(float(v) - v0) <= float(eps) for v in values):
        return 3, struct.pack('<f', v0)


    kf_count = n
    out = bytearray()
    out += struct.pack('<I', int(kf_count))


    if n > 0xFF:
        for fr in range(kf_count):
            out += struct.pack('<H', int(fr))
    else:
        out += bytes([int(fr) & 0xFF for fr in range(kf_count)])


    while (len(out) & 3) != 0:
        out += b'\x00'


    for i in range(kf_count):
        out += struct.pack('<ff', float(values[i]), 0.0)

    return 5, bytes(out)


def _compute_slopes(frames: Sequence[int], values: Sequence[float]) -> List[float]:
    n = len(frames)
    if n <= 1:
        return [0.0] * n
    out: List[float] = [0.0] * n
    for i in range(n):
        if i == 0:
            df = float(int(frames[1]) - int(frames[0]))
            out[i] = float(values[1] - values[0]) / df if df != 0.0 else 0.0
        elif i == n - 1:
            df = float(int(frames[n - 1]) - int(frames[n - 2]))
            out[i] = float(values[n - 1] - values[n - 2]) / df if df != 0.0 else 0.0
        else:
            df = float(int(frames[i + 1]) - int(frames[i - 1]))
            out[i] = float(values[i + 1] - values[i - 1]) / df if df != 0.0 else 0.0
    return out


def _encode_key_list_sparse_float(
    *,
    frames_count: int,
    frames: Sequence[int],
    values: Sequence[float],
    slopes_override: Sequence[float] | None = None,
) -> bytes:
    kf_count = int(len(frames))
    if kf_count <= 0:
        return b''
    if len(values) != kf_count:
        raise ValueError('values/frames length mismatch')

    out = bytearray()
    out += struct.pack('<I', int(kf_count))
    if int(frames_count) > 0xFF:
        for fr in frames:
            out += struct.pack('<H', int(fr) & 0xFFFF)
    else:
        out += bytes([int(fr) & 0xFF for fr in frames])

    while (len(out) & 3) != 0:
        out += b'\x00'

    slopes = list(slopes_override) if slopes_override is not None else _compute_slopes([int(x) for x in frames], [float(x) for x in values])
    for v, s in zip(values, slopes):
        out += struct.pack('<ff', float(v), float(s))
    return bytes(out)


def _encode_key_list_sparse_quant(
    *,
    frames_count: int,
    frames: Sequence[int],
    values: Sequence[float],
    slopes: Sequence[float],
) -> bytes:
    kf_count = int(len(frames))
    if kf_count <= 0:
        return b''
    if len(values) != kf_count or len(slopes) != kf_count:
        raise ValueError('values/frames length mismatch')

    vmin = float(min(values))
    vmax = float(max(values))
    if vmax == vmin:
        vscale = 1.0
    else:
        vscale = float(vmax - vmin)
    voff = float(vmin)

    smin = float(min(slopes))
    smax = float(max(slopes))
    if smax == smin:
        sscale = 1.0
    else:
        sscale = float(smax - smin)
    soff = float(smin)

    out = bytearray()
    out += struct.pack('<I', int(kf_count))
    if int(frames_count) > 0xFF:
        for fr in frames:
            out += struct.pack('<H', int(fr) & 0xFFFF)
    else:
        out += bytes([int(fr) & 0xFF for fr in frames])

    while (len(out) & 3) != 0:
        out += b'\x00'

    out += struct.pack('<ffff', float(vscale), float(voff), float(sscale), float(soff))

    def enc(v: float, scale: float, off: float) -> int:
        if float(scale) == 0.0:
            return 0
        t = (float(v) - float(off)) / float(scale)
        u = int(round(t * 65535.0))
        if u < 0:
            return 0
        if u > 65535:
            return 65535
        return int(u)

    for v, s in zip(values, slopes):
        out += struct.pack('<HH', enc(float(v), vscale, voff), enc(float(s), sscale, soff))

    return bytes(out)
