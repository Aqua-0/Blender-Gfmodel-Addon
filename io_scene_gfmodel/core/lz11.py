

from __future__ import annotations

import struct
import zlib
from collections import OrderedDict


_MAX_DECOMPRESSED_BYTES = 256 * 1024 * 1024

_DECOMP_CACHE_MAX_ENTRIES = 8
_DECOMP_CACHE_MAX_DECODED = 32 * 1024 * 1024
_DECOMP_CACHE: 'OrderedDict[tuple[int,int], bytes]' = OrderedDict()


def looks_like_lz11(data: bytes) -> bool:

    if len(data) < 4 or data[0] != 0x11:
        return False
    decoded_len = data[1] | (data[2] << 8) | (data[3] << 16)
    if decoded_len <= 0:
        return False
    if decoded_len > int(_MAX_DECOMPRESSED_BYTES):
        return False

    return decoded_len > len(data)


def decompress(data: bytes) -> bytes:
    if len(data) < 4:
        return data
    hdr = struct.unpack_from("<I", data, 0)[0]
    if (hdr & 0xFF) != 0x11:
        return data
    decoded_len = hdr >> 8
    if decoded_len <= 0 or decoded_len > int(_MAX_DECOMPRESSED_BYTES):
        return data
    key = None
    try:
        key = (len(data), int(zlib.crc32(data)) & 0xFFFFFFFF)
    except Exception:
        key = None
    if key is not None:
        cached = _DECOMP_CACHE.get(key)
        if cached is not None:
            try:
                _DECOMP_CACHE.move_to_end(key)
            except Exception:
                pass
            return cached
    inp = memoryview(data)[4:]
    in_off = 0
    out = bytearray(decoded_len)
    out_off = 0

    mask = 0
    header = 0
    while out_off < decoded_len:
        mask >>= 1
        if mask == 0:
            header = int(inp[in_off])
            in_off += 1
            mask = 0x80

        if (header & mask) == 0:
            out[out_off] = int(inp[in_off])
            out_off += 1
            in_off += 1
            continue

        byte1 = int(inp[in_off])
        in_off += 1
        top = byte1 >> 4

        if top == 0:
            byte2 = int(inp[in_off])
            byte3 = int(inp[in_off + 1])
            in_off += 2
            position = ((byte2 & 0xF) << 8) | byte3
            length = (((byte1 & 0xF) << 4) | (byte2 >> 4)) + 0x11
        elif top == 1:
            byte2 = int(inp[in_off])
            byte3 = int(inp[in_off + 1])
            byte4 = int(inp[in_off + 2])
            in_off += 3
            position = ((byte3 & 0xF) << 8) | byte4
            length = (((byte1 & 0xF) << 12) | (byte2 << 4) | (byte3 >> 4)) + 0x111
        else:
            byte2 = int(inp[in_off])
            in_off += 1
            position = ((byte1 & 0xF) << 8) | byte2
            length = (byte1 >> 4) + 1

        position += 1
        if position <= 0 or position > out_off:
            return bytes(out[:out_off]) if out_off < decoded_len else bytes(out)

        remaining = int(length)
        while remaining > 0 and out_off < decoded_len:
            chunk = position if remaining > position else remaining
            end = out_off + chunk
            if end > decoded_len:
                end = decoded_len
                chunk = end - out_off
            out[out_off:end] = out[out_off - position : out_off - position + chunk]
            out_off = end
            remaining -= chunk

    res = bytes(out)
    if key is not None and int(decoded_len) <= int(_DECOMP_CACHE_MAX_DECODED):
        _DECOMP_CACHE[key] = res
        try:
            _DECOMP_CACHE.move_to_end(key)
        except Exception:
            pass
        while len(_DECOMP_CACHE) > int(_DECOMP_CACHE_MAX_ENTRIES):
            try:
                _DECOMP_CACHE.popitem(last=False)
            except Exception:
                break
    return res


def compress(data: bytes, *, force_literal_prefix: int = 8) -> bytes:

    raw = bytes(data)
    n = len(raw)
    if n <= 0:
        return b"\x11\x00\x00\x00"
    if n > 0x00FFFFFF:
        raise ValueError("LZ11 only supports up to 24-bit decompressed length")



    max_candidates = 64
    window = 0x1000

    def key3(i: int) -> int:
        return (raw[i] << 16) | (raw[i + 1] << 8) | raw[i + 2]

    buckets: dict[int, list[int]] = {}

    out = bytearray()
    out += bytes((0x11, n & 0xFF, (n >> 8) & 0xFF, (n >> 16) & 0xFF))

    i = 0

    def add_pos(pos: int) -> None:
        if pos + 2 >= n:
            return
        k = key3(pos)
        lst = buckets.get(k)
        if lst is None:
            buckets[k] = [pos]
            return
        lst.append(pos)
        if len(lst) > max_candidates:
            del lst[0 : len(lst) - max_candidates]


    while i < min(n, int(force_literal_prefix)):
        add_pos(i)
        i += 1


    i = 0

    def find_match(pos: int) -> tuple[int, int]:

        if pos + 2 >= n:
            return 0, 0
        k = key3(pos)
        cand = buckets.get(k)
        if not cand:
            return 0, 0
        best_len = 0
        best_disp = 0
        min_pos = max(0, pos - window)

        for p in reversed(cand):
            if p < min_pos:
                break
            disp = pos - p
            if disp <= 0 or disp > window:
                continue

            max_len = min(n - pos, 0x111 + 0xFFFF)

            if best_len >= 3 and best_len >= max_len:
                break
            l = 0

            while l < max_len and raw[p + l] == raw[pos + l]:
                l += 1
            if l > best_len:
                best_len = l
                best_disp = disp
                if best_len >= 0x111 + 0x100:
                    break
        if best_len < 3:
            return 0, 0
        return best_len, best_disp


    while i < n:
        flag_off = len(out)
        out.append(0)
        flags = 0
        tokens = bytearray()

        for bit in range(8):
            if i >= n:
                break


            if i < int(force_literal_prefix):
                tokens.append(raw[i])
                add_pos(i)
                i += 1
                continue

            best_len, best_disp = find_match(i)
            if best_len == 0:
                tokens.append(raw[i])
                add_pos(i)
                i += 1
                continue


            flags |= 1 << (7 - bit)
            disp = best_disp - 1
            length = best_len

            if length <= 0x10:

                b1 = ((length - 1) << 4) | ((disp >> 8) & 0xF)
                b2 = disp & 0xFF
                tokens += bytes((b1 & 0xFF, b2 & 0xFF))
            elif length <= 0x110:



                l = length - 0x11
                b1 = (l >> 4) & 0xF
                b2 = ((l & 0xF) << 4) | ((disp >> 8) & 0xF)
                b3 = disp & 0xFF
                tokens += bytes((b1 & 0xFF, b2 & 0xFF, b3 & 0xFF))
            else:

                l = length - 0x111
                b1 = 0x10 | ((l >> 12) & 0xF)
                b2 = (l >> 4) & 0xFF
                b3 = ((l & 0xF) << 4) | ((disp >> 8) & 0xF)
                b4 = disp & 0xFF
                tokens += bytes((b1 & 0xFF, b2 & 0xFF, b3 & 0xFF, b4 & 0xFF))


            for k in range(length):
                add_pos(i + k)
            i += length

        out[flag_off] = flags & 0xFF
        out += tokens

    return bytes(out)
