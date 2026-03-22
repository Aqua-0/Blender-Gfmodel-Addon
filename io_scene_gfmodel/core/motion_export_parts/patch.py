
from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .encode import (
    _encode_key_list,
    _encode_key_list_sparse_float,
    _encode_key_list_sparse_quant,
)
from .parse import _Sect, _align_up, _parse_frames_count, _parse_section_table
from .rebuild import rebuild_gfmot_with_replaced_bones
from .sect1 import _parse_section1_layout, rewrite_gfmot_section1_values

def patch_gfmot_with_rekey_changed_frames(
    *,
    src_gfmot_bytes: bytes,
    src_motion: _GFMotion,
    new_bones: Sequence[_GFMotBoneTransform],
    frames_count: int,
    keyed_frames_by_bone: dict[str, dict[str, set[int]]] | None = None,
    rot_tol: float = 1e-2,
    pos_tol: float = 1e-4,
    scale_tol: float = 1e-4,
) -> bytes:
    fc = int(frames_count)
    if fc <= 0:
        return bytes(src_gfmot_bytes)

    def _encode_u16(v: float, scale: float, off: float) -> int:
        if float(scale) == 0.0:
            return 0
        t = (float(v) - float(off)) / float(scale)
        u = int(round(t * 65535.0))
        if u < 0:
            return 0
        if u > 65535:
            return 65535
        return int(u)

    def _parse_float_channel(chan: bytes, *, frames_count: int) -> tuple[list[int], list[float], list[float]]:
        b = memoryview(chan)
        if len(b) < 4:
            return [], [], []
        kf_count = int(struct.unpack_from('<I', b, 0)[0])
        o = 4
        frames: list[int] = []
        if int(frames_count) > 0xFF:
            need = o + kf_count * 2
            if need > len(b):
                return [], [], []
            for i in range(kf_count):
                frames.append(int(struct.unpack_from('<H', b, o + i * 2)[0]))
            o = need
        else:
            need = o + kf_count
            if need > len(b):
                return [], [], []
            frames = [int(x) for x in b[o:need].tolist()]
            o = need
        if (o & 3) != 0:
            o += 4 - (o & 3)
        need = o + kf_count * 8
        if need > len(b):
            return [], [], []
        vals: list[float] = []
        slopes: list[float] = []
        for i in range(kf_count):
            v, s = struct.unpack_from('<ff', b, o + i * 8)
            vals.append(float(v))
            slopes.append(float(s))
        return frames, vals, slopes

    def _parse_quant_channel(
        chan: bytes, *, frames_count: int
    ) -> tuple[list[int], float, float, float, float, list[int], list[int]]:
        b = memoryview(chan)
        if len(b) < 4:
            return [], 1.0, 0.0, 1.0, 0.0, [], []
        kf_count = int(struct.unpack_from('<I', b, 0)[0])
        o = 4
        frames: list[int] = []
        if int(frames_count) > 0xFF:
            need = o + kf_count * 2
            if need > len(b):
                return [], 1.0, 0.0, 1.0, 0.0, [], []
            for i in range(kf_count):
                frames.append(int(struct.unpack_from('<H', b, o + i * 2)[0]))
            o = need
        else:
            need = o + kf_count
            if need > len(b):
                return [], 1.0, 0.0, 1.0, 0.0, [], []
            frames = [int(x) for x in b[o:need].tolist()]
            o = need
        if (o & 3) != 0:
            o += 4 - (o & 3)
        if o + 16 > len(b):
            return [], 1.0, 0.0, 1.0, 0.0, [], []
        vscale, voff, sscale, soff = struct.unpack_from('<ffff', b, o)
        o += 16
        need = o + kf_count * 4
        if need > len(b):
            return [], float(vscale), float(voff), float(sscale), float(soff), [], []
        vu16: list[int] = []
        su16: list[int] = []
        for i in range(kf_count):
            v_u, s_u = struct.unpack_from('<HH', b, o + i * 4)
            vu16.append(int(v_u))
            su16.append(int(s_u))
        return frames, float(vscale), float(voff), float(sscale), float(soff), vu16, su16

    def _build_float_channel(
        *, frames_count: int, frames: list[int], values: list[float], slopes: list[float]
    ) -> bytes:
        kf_count = int(len(frames))
        out = bytearray()
        out += struct.pack('<I', int(kf_count))
        if int(frames_count) > 0xFF:
            for fr in frames:
                out += struct.pack('<H', int(fr) & 0xFFFF)
        else:
            out += bytes([int(fr) & 0xFF for fr in frames])
        while (len(out) & 3) != 0:
            out += b'\x00'
        for v, s in zip(values, slopes):
            out += struct.pack('<ff', float(v), float(s))
        return bytes(out)

    def _build_quant_channel(
        *,
        frames_count: int,
        frames: list[int],
        vscale: float,
        voff: float,
        sscale: float,
        soff: float,
        vu16: list[int],
        su16: list[int],
    ) -> bytes:
        kf_count = int(len(frames))
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
        for v_u, s_u in zip(vu16, su16):
            out += struct.pack('<HH', int(v_u) & 0xFFFF, int(s_u) & 0xFFFF)
        return bytes(out)


    base = rewrite_gfmot_section1_values(
        src_bytes=bytes(src_gfmot_bytes),
        frames_count=int(fc),
        bones=list(new_bones),
    )

    prefix, layouts = _parse_section1_layout(base, frames_count=int(fc))

    new_by_name = {str(b.name): b for b in list(new_bones)}
    src_by_name = {str(b.name): b for b in list(getattr(src_motion, 'bones', []) or [])}


    chan_keys = ['sx','sy','sz','rx','ry','rz','tx','ty','tz']

    def _dense_val(bt: _GFMotBoneTransform, ch: str, frame: int) -> float:
        kfs = getattr(bt, ch, [])
        if not kfs:
            return 0.0
        fi = int(frame)
        if 0 <= fi < len(kfs) and int(getattr(kfs[fi], 'frame', fi)) == fi:
            return float(getattr(kfs[fi], 'value', 0.0) or 0.0)
        return float(_mot_eval(list(kfs), int(frame), float(getattr(kfs[0], 'value', 0.0) or 0.0)))

    def _orig_key_frames(bt: _GFMotBoneTransform, ch: str) -> List[int]:
        kfs = getattr(bt, ch, [])
        if not kfs:
            return []
        out = sorted({int(getattr(kf, 'frame', 0) or 0) for kf in kfs})
        return [x for x in out if 0 <= x <= fc]

    any_rekey = False
    out_bone_blocks: List[bytes] = []

    for lay in layouts:
        name = str(lay.name)
        src_bt = src_by_name.get(name)
        new_bt = new_by_name.get(name)

        if src_bt is None or new_bt is None:
            payload = b''.join(lay.channel_bytes)
            out_bone_blocks.append(struct.pack('<II', int(lay.flags), int(len(payload))) + payload)
            continue

        rekey_frames_by_ci: dict[int, List[int]] = {}


        for ci, ch in enumerate(chan_keys):
            mode = int(lay.channel_modes[ci]) if ci < len(lay.channel_modes) else 0
            if mode not in (3, 4, 5):
                continue

            orig_frames = set(_orig_key_frames(src_bt, ch))


            consider_frames: list[int] | None = None
            if keyed_frames_by_bone is not None:
                bf = keyed_frames_by_bone.get(name) or {}
                if ch in ('sx','sy','sz'):
                    consider_frames = sorted(int(x) for x in (bf.get('scale') or set()) if 0 <= int(x) <= fc)
                elif ch in ('tx','ty','tz'):
                    consider_frames = sorted(int(x) for x in (bf.get('pos') or set()) if 0 <= int(x) <= fc)
                else:
                    consider_frames = sorted(int(x) for x in (bf.get('rot') or set()) if 0 <= int(x) <= fc)


            changed: List[int] = []
            frame_iter = consider_frames if consider_frames is not None else range(int(fc))
            for fr in frame_iter:
                if int(fr) == int(fc):
                    continue
                ov = float(_mot_eval(list(getattr(src_bt, ch, []) or []), int(fr), float(_dense_val(src_bt, ch, 0))))
                nv = float(_dense_val(new_bt, ch, int(fr)))
                tol = float(scale_tol if ch in ('sx','sy','sz') else pos_tol if ch in ('tx','ty','tz') else rot_tol)
                if abs(nv - ov) > tol:

                    if int(fr) not in orig_frames:
                        changed.append(int(fr))

            if changed:
                any_rekey = True

                frames = set(int(x) for x in orig_frames)
                frames.update(int(x) for x in changed)
                frames.add(0)

                if fc in orig_frames or (max(frames) < fc):
                    frames.add(int(fc))
                rekey_frames_by_ci[int(ci)] = sorted(frames)

        if not rekey_frames_by_ci:

            payload = b''.join(lay.channel_bytes)
            out_bone_blocks.append(struct.pack('<II', int(lay.flags), int(len(payload))) + payload)
            continue


        flags = int(lay.flags) & 0xFFFFFFFF
        chan_out: List[bytes] = []
        for ci, ch in enumerate(chan_keys):
            if ci in rekey_frames_by_ci:
                frs = list(rekey_frames_by_ci[ci])
                float_keys = (int(lay.flags) & 1) != 0


                if float_keys:
                    old_frames, old_vals, old_slopes = _parse_float_channel(
                        lay.channel_bytes[ci] if ci < len(lay.channel_bytes) else b'',
                        frames_count=int(fc),
                    )
                    old_by_fr = {int(fr): (float(v), float(s)) for fr, v, s in zip(old_frames, old_vals, old_slopes)}
                    out_vals: List[float] = []
                    out_slopes: List[float] = []
                    for fr in frs:
                        if int(fr) in old_by_fr:
                            v, s = old_by_fr[int(fr)]
                            out_vals.append(float(v))
                            out_slopes.append(float(s))
                            continue
                        if int(fr) == int(fc):
                            v = float(_dense_val(new_bt, ch, 0 if bool(getattr(src_motion, 'is_looping', False)) else int(fc) - 1))
                        else:
                            v = float(_dense_val(new_bt, ch, int(fr)))
                        out_vals.append(float(v))
                        out_slopes.append(0.0)
                    chan_bytes = _build_float_channel(
                        frames_count=int(fc),
                        frames=frs,
                        values=out_vals,
                        slopes=out_slopes,
                    )
                else:
                    old_frames, vscale, voff, sscale, soff, old_vu16, old_su16 = _parse_quant_channel(
                        lay.channel_bytes[ci] if ci < len(lay.channel_bytes) else b'',
                        frames_count=int(fc),
                    )
                    old_by_fr = {int(fr): (int(vu), int(su)) for fr, vu, su in zip(old_frames, old_vu16, old_su16)}
                    out_vu16: List[int] = []
                    out_su16: List[int] = []
                    for fr in frs:
                        if int(fr) in old_by_fr:
                            vu, su = old_by_fr[int(fr)]
                            out_vu16.append(int(vu))
                            out_su16.append(int(su))
                            continue
                        if int(fr) == int(fc):
                            v = float(_dense_val(new_bt, ch, 0 if bool(getattr(src_motion, 'is_looping', False)) else int(fc) - 1))
                        else:
                            v = float(_dense_val(new_bt, ch, int(fr)))

                        s = 0.0
                        out_vu16.append(_encode_u16(float(v), float(vscale), float(voff)))
                        out_su16.append(_encode_u16(float(s), float(sscale), float(soff)))
                    chan_bytes = _build_quant_channel(
                        frames_count=int(fc),
                        frames=frs,
                        vscale=float(vscale),
                        voff=float(voff),
                        sscale=float(sscale),
                        soff=float(soff),
                        vu16=out_vu16,
                        su16=out_su16,
                    )
                chan_out.append(chan_bytes)


            else:
                raw = lay.channel_bytes[ci] if ci < len(lay.channel_bytes) else b''
                chan_out.append(bytes(raw))

        payload = b''.join(chan_out)
        out_bone_blocks.append(struct.pack('<II', int(flags), int(len(payload))) + payload)

    if not any_rekey:
        return bytes(base)

    new_sec1 = bytes(prefix) + b''.join(out_bone_blocks)
    return rebuild_gfmot_with_replaced_bones(src_bytes=bytes(base), new_bone_section=new_sec1)
