
from __future__ import annotations

import struct
from typing import Dict, List, Tuple

from .parse import _align_up, _parse_section_table

def build_bone_section(
    *,
    frames_count: int,
    bones: Sequence[_GFMotBoneTransform],
    eps: float = 1e-6,
) -> bytes:


    name_bytes: List[bytes] = []
    for b in bones:
        nm = (b.name or '').encode('ascii', 'replace')
        if len(nm) > 0xFF:
            nm = nm[:0xFF]
        name_bytes.append(struct.pack('<B', len(nm)) + nm)

    names_blob = b''.join(name_bytes)

    names_blob_padded = names_blob
    if (len(names_blob_padded) & 3) != 0:
        names_blob_padded += b"\x00" * (4 - (len(names_blob_padded) & 3))

    out = bytearray()
    out += struct.pack('<iI', int(len(bones)), int(len(names_blob_padded)))
    out += names_blob_padded


    for b in bones:

        chan_values = [
            [kf.value for kf in b.sx],
            [kf.value for kf in b.sy],
            [kf.value for kf in b.sz],
            [kf.value for kf in b.rx],
            [kf.value for kf in b.ry],
            [kf.value for kf in b.rz],
            [kf.value for kf in b.tx],
            [kf.value for kf in b.ty],
            [kf.value for kf in b.tz],
        ]

        modes: List[int] = []
        payloads: List[bytes] = []
        for vals in chan_values:
            mode, payload = _encode_key_list(
                frames_count,
                vals,
                force_keys=True,
                eps=float(eps),
            )
            modes.append(int(mode))
            payloads.append(bytes(payload))

        flags = 0
        for i, mode in enumerate(modes):
            flags |= (int(mode) & 7) << (3 * i)
        if not bool(getattr(b, 'is_axis_angle', True)):
            flags |= 0x80000000


        bone_payload = bytearray()
        for payload in payloads:
            bone_payload += payload

        out += struct.pack('<II', int(flags) & 0xFFFFFFFF, int(len(bone_payload)))
        out += bone_payload

    return bytes(out)


def rebuild_gfmot_with_replaced_bones(
    *,
    src_bytes: bytes,
    new_bone_section: bytes,
) -> bytes:
    src = bytes(src_bytes)
    magic, sects = _parse_section_table(src)
    if magic != _GFMOT_MAGIC:
        raise ValueError(f'not a gfmot (magic=0x{magic:08X})')
    if not sects:
        raise ValueError('gfmot has no sections')


    payloads: List[Tuple[int, bytes]] = []
    for i, s in enumerate(sects):
        if s.addr < 0 or s.addr + s.length > len(src):
            raise ValueError('section slice out of range')
        if int(s.name) == 1:
            payloads.append((int(s.name), bytes(new_bone_section)))
        else:
            payloads.append((int(s.name), src[int(s.addr) : int(s.addr + s.length)]))


    out = bytearray()
    out += struct.pack('<II', _GFMOT_MAGIC, int(len(payloads)))
    table_off = len(out)
    out += b'\x00' * (12 * int(len(payloads)))


    sect_meta: List[Tuple[int, int, int]] = []
    for name, payload in payloads:
        out_off = _align_up(len(out), 4)
        if out_off != len(out):
            out += b'\x00' * (out_off - len(out))
        addr = int(out_off)
        out += bytes(payload)
        length = int(len(payload))
        sect_meta.append((int(name), int(length), int(addr)))


    for i, (name, length, addr) in enumerate(sect_meta):
        struct.pack_into('<III', out, table_off + 12 * i, int(name), int(length), int(addr))

    return bytes(out)
