
from __future__ import annotations

import copy
import struct
from typing import Dict, List, Optional, Tuple
import bmesh
import bpy
from mathutils import Matrix, Vector
from ....core.gfpack import parse_gf_model_pack
from ....core.gfpack import write_gf_model_pack as write_gf_model_pack_low
from ..grow_buffers_rewrite import _rewrite_model_blob_grow_buffers_tris

from ..textures_patch import _find_tex_image_for_unit

def _rgba8_bytes_from_image(
    img: bpy.types.Image,
    *,
    width: int,
    height: int,
    allow_scale: bool,
) -> bytes:
    img.pixels[0]                 

    w0, h0 = int(img.size[0]), int(img.size[1])
    if w0 <= 0 or h0 <= 0:
        raise ValueError(f"Invalid image size for {img.name!r}: {w0}x{h0}")

    w = int(width)
    h = int(height)
    if not allow_scale:
        if w0 != w or h0 != h:
            raise ValueError(
                f"Image size mismatch for {img.name!r}: image={w0}x{h0} expected={w}x{h}"
            )
        px_f = img.pixels[: w * h * 4]
    else:
        img2 = img.copy()
        try:
            img2.scale(w, h)
            px_f = img2.pixels[: w * h * 4]
        finally:
            bpy.data.images.remove(img2)

    return bytes(int(max(0, min(255, round(float(c) * 255.0)))) for c in px_f)


def _collect_texture_overrides_by_slot(model: "_GFModel") -> Dict[str, bpy.types.Image]:
    overrides: Dict[str, bpy.types.Image] = {}
    for mat_def in getattr(model, "materials", []) or []:
        mat = bpy.data.materials.get(str(getattr(mat_def, "name", "") or ""))
        if mat is None or getattr(mat, "node_tree", None) is None:
            continue
        for tu in getattr(mat_def, "tex_units", []) or []:
            try:
                unit_index = int(getattr(tu, "unit_index", -1))
            except Exception:
                unit_index = -1
            if unit_index < 0:
                continue
            img = _find_tex_image_for_unit(mat, unit_index)
            if img is None:
                continue
            slot_name = str(getattr(tu, "name", "") or "").strip()
            if not slot_name:
                continue
            overrides.setdefault(slot_name, img)
    return overrides


def _patch_pack_textures_rgba8(
    pack_src: bytes,
    *,
    overrides: Dict[str, bpy.types.Image],
    texture_mode: str,
    texture_max_size: int,
) -> Tuple[bytes, int]:
    if texture_mode not in ("RGBA8", "RGBA8_SAME_SIZE", "RGBA8_ORIGINAL_SIZE"):
        return pack_src, 0

    pack = parse_gf_model_pack(pack_src)
    if int(pack.counts[1]) <= 0:
        return pack_src, 0

    max_size = int(texture_max_size)
    if max_size % 8 != 0:
        max_size = max(8, (max_size // 8) * 8)

    replacements: Dict[Tuple[int, int], bytes] = {}
    changed = 0
    for i in range(int(pack.counts[1])):
        e = pack.get(1, int(i))
        if e is None:
            continue
        tex = _parse_gf_texture(e.blob)
        img = overrides.get(tex.name) or bpy.data.images.get(tex.name)
        if img is None:
            continue

        if texture_mode in ("RGBA8_SAME_SIZE", "RGBA8_ORIGINAL_SIZE"):
            w = int(tex.width)
            h = int(tex.height)
            allow_scale = texture_mode == "RGBA8_ORIGINAL_SIZE"
        else:
            w0, h0 = int(img.size[0]), int(img.size[1])
            w = int(min(w0, max_size))
            h = int(min(h0, max_size))
            w = max(8, (w // 8) * 8)
            h = max(8, (h // 8) * 8)
            allow_scale = True

        if w % 8 != 0 or h % 8 != 0:
            raise ValueError(
                f"Texture size must be multiple of 8 for tiled formats: {tex.name!r} ({w}x{h})"
            )

        px = _rgba8_bytes_from_image(img, width=w, height=h, allow_scale=allow_scale)
        raw = encode_pica_rgba8_swizzled_abgr(px, w, h)
        out_tex = _GFTexture(name=tex.name, width=w, height=h, fmt=0x4, raw=raw)
        replacements[(1, int(i))] = write_gf_texture_blob(out_tex)
        changed += 1

    if changed == 0:
        return pack_src, 0
    out = write_gf_model_pack_low(pack, replacements=replacements, align_blobs=0x80)
    return bytes(out), int(changed)
