
from __future__ import annotations

import os
from typing import List, Optional

import bpy
from bpy.props import (
    BoolProperty,
    CollectionProperty,
    EnumProperty,
    IntProperty,
    StringProperty,
)

from ...core.garc import parse_garc_file, rewrite_garc_file, rewrite_garc_file_inplace_atomic
from ...core.io import _load_any
from ...core.lz11 import compress, decompress, looks_like_lz11
from ...core.mini import parse_mini, patch_mini
from ...core.patch_plan import PatchPlan
from ...core.pkmn_container import parse_container, patch_container
from ..archive_patch_context import (
    make_archive_entry_plan_and_payload,
    set_scene_patch_plan,
)
from ..importer import (
    _import_gfmodel_bytes,
    _import_gfmodel_bytes_with_a094_group,
    _import_gfmodel_bytes_with_extras,
    _import_gfmodel_loaded,
)
from .usum_names import (
    a094_search_blob_for_entry as _a094_search_blob_for_entry,
    a094_species_label_for_entry as _a094_species_label_for_entry,
)


_GARC_FILE_CACHE: dict[str, object] = {}

def _looks_like_a094_context(context: bpy.types.Context) -> bool:
    try:
        ap = str(getattr(context.scene, 'gfmodel_archive_path', '') or '')
    except Exception:
        ap = ''
    return _looks_like_a094_archive_path(ap)


def _looks_like_a094_context(context: bpy.types.Context) -> bool:
    try:
        ap = str(getattr(context.scene, 'gfmodel_archive_path', '') or '')
    except Exception:
        ap = ''
    return _looks_like_a094_archive_path(ap)


def _guess_magic4(b: bytes) -> str:
    if len(b) < 4:
        return ""
    s = b[:4]
    try:
        return s.decode("ascii")
    except Exception:
        return "".join(chr(x) if 32 <= x < 127 else "." for x in s)


def _guess_ident2(b: bytes) -> str:
    if len(b) < 2:
        return ""
    s = b[:2]
    try:
        txt = s.decode("ascii")
    except Exception:
        return ""
    if all(32 <= ord(c) < 127 for c in txt):
        return txt
    return ""


def _maybe_lz11(data: bytes) -> bytes:
    try:
        if looks_like_lz11(data):
            return decompress(data)
    except Exception:
        pass
    return data


def _size_kb(length: int) -> str:
    return f"{(float(length) / 1024.0):.1f} KB"


def _parse_search_int(s: str) -> Optional[int]:
    t = (s or "").strip()
    if not t:
        return None
    try:
        return int(t, 0)
    except Exception:
        return None


def _resolve_out_path(
    archive_path: str, out_path: str, *, suffix: str = ".patched"
) -> str:
    p = (out_path or "").strip()
    if not p:
        return str(archive_path) + str(suffix)


    try:
        if p.endswith(("/", "\\")) or os.path.isdir(p):
            base = os.path.basename(str(archive_path).rstrip("\\/"))
            return os.path.join(p.rstrip("\\/"), base + str(suffix))
    except Exception:
        pass
    return p


def _selected_archive_entry_index(context: bpy.types.Context) -> int:
    entries = getattr(context.scene, "gfmodel_archive_entries", None)
    if entries is None or len(entries) == 0:
        raise ValueError("Scan the archive first (so the entry list is populated)")
    sel = int(getattr(context.scene, "gfmodel_archive_selected", 0))
    if sel < 0 or sel >= len(entries):
        raise ValueError(f"Selection out of range: {sel}")
    return int(getattr(entries[sel], "index", sel))


def _classify_import_type(data: bytes) -> str:
    if not data:
        return "OTHER"
    if data[:1] == b"\x11":
        return "LZ11"
    ident2 = _guess_ident2(data[:2])
    if ident2 in ("CP", "CM"):
        return "CONTAINER"
    if len(data) >= 4:
        m = int.from_bytes(data[:4], "little", signed=False)
        if m == 0x00060000:
            return "MOTION"
        if m in (0x00010000, 0x15122117):
            return "MODEL"
        if m == 0x15041213:
            return "TEXTURE"

    try:
        from ...core.binlinker import looks_like_binlinker

        if looks_like_binlinker(data):
            return "BINLINKER"
    except Exception:
        pass
    try:
        parse_mini(data)
        return "MINI"
    except Exception:
        return "OTHER"


def _breadcrumb(context: bpy.types.Context) -> str:
    path = str(getattr(context.scene, "gfmodel_archive_path", "")).strip()
    base = os.path.basename(path) if path else "(no archive)"
    parts = [base]

    entry_i = None
    if path:
        try:
            entry_i = _selected_archive_entry_index(context)
        except Exception:
            entry_i = int(getattr(context.scene, "gfmodel_archive_selected", 0))
        parts.append(f"GARC[{int(entry_i)}]")


    mini_i = int(getattr(context.scene, "gfmodel_mini_selected", 0))
    mini_ident = str(context.scene.get("gfmodel_archive_mini_ident", "")).strip()
    if mini_ident:
        parts.append(f"Mini {mini_ident}[{mini_i}]")

    cmagic = str(context.scene.get("gfmodel_container_magic", "")).strip()
    if cmagic:
        csel = int(getattr(context.scene, "gfmodel_container_selected", 0))
        parts.append(f"{cmagic}[{csel}]")

    cmagic2 = str(context.scene.get("gfmodel_container2_magic", "")).strip()
    if cmagic2:
        csel2 = int(getattr(context.scene, "gfmodel_container2_selected", 0))
        parts.append(f"{cmagic2}[{csel2}]")

    return " \u2192 ".join(parts)


def _garc_cached(path: str):
    g = _GARC_FILE_CACHE.get(path)
    if g is not None:
        return g
    g = parse_garc_file(path)
    _GARC_FILE_CACHE[path] = g
    return g


def _looks_like_a094_archive_path(path: str) -> bool:
    p = str(path or "").replace("\\", "/").lower()
    return "/a/0/9/4" in p


def _gather_a094_group_extras(
    archive_path: str, entry_i: int
) -> tuple[int, list[dict]]:
    garc = _garc_cached(str(archive_path))
    n = len(getattr(garc, "entries", []) or [])
    ei = int(entry_i)
    group_start = int(ei - (ei % 9))
    out: list[dict] = []
    for mid in range(group_start, group_start + 9):
        if mid == ei:
            continue
        if mid < 0 or mid >= n:
            continue
        try:
            b = bytes(garc.read_primary_bytes(int(mid)))
            pre_steps = []
            if b[:1] == b"\x11":
                pre_steps = [{"op": "lz11"}]
            payload = _maybe_lz11(b)
            bit = int(getattr(garc.entries[int(mid)], "bit", 0) or 0)
            out.append(
                {
                    "archive_path": str(archive_path),
                    "entry_index": int(mid),
                    "bit": int(bit),
                    "payload": bytes(payload),
                    "pre_steps": list(pre_steps),
                }
            )
        except Exception:
            continue
    return int(group_start), out


