"""Texture override and patch helpers."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import bpy

from ...core.export import write_gf_texture_blob
from ...core.gfpack import parse_gf_model_pack
from ...core.gfpack import write_gf_model_pack as write_gf_model_pack_low
from ...core.io import _parse_gf_texture
from ...core.texture_encode import (
    encode_pica_swizzled_from_rgba,
    gf_fmt_from_override_name,
)
from ...core.types import _GFTexture


def _rgba8_bytes_from_image(
    img: bpy.types.Image,
    *,
    width: int,
    height: int,
    allow_scale: bool,
) -> bytes:
    """Return linear RGBA8 bytes from a Blender image."""
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


def _encode_texture_from_image(
    *,
    tex_name: str,
    img: bpy.types.Image,
    original_width: int,
    original_height: int,
    original_fmt: int,
    texture_mode: str,
    texture_override_format: str,
    texture_max_size: int,
) -> _GFTexture:
    """Encode a Blender image into a GFTexture override according to UI settings."""
    mode = str(texture_mode or "KEEP").strip()
    if mode == "KEEP":
        raise ValueError("internal error: encode called with KEEP")

    max_size = int(texture_max_size)
    if max_size % 8 != 0:
        max_size = max(8, (max_size // 8) * 8)

    if mode in (
        "RGBA8_SAME_SIZE",
        "RGBA8_ORIGINAL_SIZE",
        "ORIGINAL_FORMAT",
        "OVERRIDE_FORMAT",
    ):
        w = int(original_width)
        h = int(original_height)
                                                                                       
                                                                                                  
        allow_scale = mode in (
            "RGBA8_ORIGINAL_SIZE",
            "ORIGINAL_FORMAT",
            "OVERRIDE_FORMAT",
        )
    else:
        w0, h0 = int(img.size[0]), int(img.size[1])
        w = int(min(w0, max_size))
        h = int(min(h0, max_size))
        w = max(8, (w // 8) * 8)
        h = max(8, (h // 8) * 8)
        allow_scale = True

    if w % 8 != 0 or h % 8 != 0:
        raise ValueError(
            f"Texture size must be multiple of 8 for tiled formats: {tex_name!r} ({w}x{h})"
        )

    px = _rgba8_bytes_from_image(img, width=w, height=h, allow_scale=allow_scale)

    if mode == "ORIGINAL_FORMAT":
        out_fmt = int(original_fmt)
    elif mode == "OVERRIDE_FORMAT":
        out_fmt = gf_fmt_from_override_name(str(texture_override_format))
    else:
                      
        out_fmt = 0x4

    raw = encode_pica_swizzled_from_rgba(px, width=w, height=h, gf_fmt=int(out_fmt))
    return _GFTexture(
        name=str(tex_name),
        width=int(w),
        height=int(h),
        fmt=int(out_fmt),
        raw=raw,
    )


def _collect_texture_overrides_by_slot(model: "_GFModel") -> Dict[str, bpy.types.Image]:
    """Map existing GF texture names -> Blender images used by materials."""
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
    texture_override_format: str = "RGBA8",
    texture_max_size: int,
) -> Tuple[bytes, int]:
    """Overwrite existing texture slots, without adding new slots."""
    if texture_mode not in (
        "RGBA8",
        "RGBA8_SAME_SIZE",
        "RGBA8_ORIGINAL_SIZE",
        "ORIGINAL_FORMAT",
        "OVERRIDE_FORMAT",
    ):
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

        out_tex = _encode_texture_from_image(
            tex_name=str(tex.name),
            img=img,
            original_width=int(tex.width),
            original_height=int(tex.height),
            original_fmt=int(tex.fmt),
            texture_mode=str(texture_mode),
            texture_override_format=str(texture_override_format),
            texture_max_size=int(max_size),
        )
        replacements[(1, int(i))] = write_gf_texture_blob(out_tex)
        changed += 1

    if changed == 0:
        return pack_src, 0
    out = write_gf_model_pack_low(pack, replacements=replacements, align_blobs=0x80)
    return bytes(out), int(changed)


def _find_tex_image_for_unit(
    mat: bpy.types.Material, unit_index: int
) -> Optional[bpy.types.Image]:
    nt = mat.node_tree
    if nt is None:
        return None
    mapping = nt.nodes.get(f"GF_MAPPING_{unit_index}")
    if mapping is None:
        return None
                                     
    for link in nt.links:
        if link.from_node == mapping and link.to_socket.name == "Vector":
            if link.to_node.type == "TEX_IMAGE":
                return getattr(link.to_node, "image", None)
                              
    for n in nt.nodes:
        if n.type == "TEX_IMAGE" and getattr(n, "image", None) is not None:
            return n.image
    return None
