
from __future__ import annotations

import csv
import os
from typing import Optional

import bpy

_USUM_A094_PACK_MAP: Optional[dict[int, dict[str, str]]] = None


def _addon_data_file(rel_path: str) -> str:
    here = os.path.dirname(__file__)
    return os.path.normpath(os.path.join(here, "..", "..", "data", rel_path))


def load_usum_a094_pack_map() -> dict[int, dict[str, str]]:
    global _USUM_A094_PACK_MAP
    if _USUM_A094_PACK_MAP is not None:
        return _USUM_A094_PACK_MAP

    mp: dict[int, dict[str, str]] = {}
    csv_path = _addon_data_file("usum_a094_pack_map.csv")
    try:
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                try:
                    pack_index = int(str(row.get("pack_index", "")).strip() or "0")
                except Exception:
                    continue
                mp[pack_index] = {k: str(v or "") for k, v in row.items()}
    except Exception:
        mp = {}

    _USUM_A094_PACK_MAP = mp
    return mp


def a094_pack_for_entry(entry_index: int) -> tuple[int, int]:
    idx = int(entry_index)
    if idx < 0:
        return 0, 0
    return idx // 9, idx % 9


def a094_species_label_for_entry(context: bpy.types.Context, entry_index: int) -> str:
    pack_index, member = a094_pack_for_entry(int(entry_index))
    info = load_usum_a094_pack_map().get(int(pack_index))

    name = str(info.get("name", "") or "").strip() if info else ""
    natdex = str(info.get("natdex", "") or "").strip() if info else ""

    base = ""
    if name and natdex and natdex.isdigit():
        base = f"{name} #{int(natdex):03d}"
    elif name:
        base = name
    elif natdex:
        base = f"Dex {natdex}"

    if base:
        return f"{base} m{int(member)}"
    return f"pack{int(pack_index):04d} m{int(member)}"


def a094_search_blob_for_entry(entry_index: int) -> str:
    pack_index, member = a094_pack_for_entry(int(entry_index))
    info = load_usum_a094_pack_map().get(int(pack_index))

    name = str(info.get("name", "") or "") if info else ""
    natdex = str(info.get("natdex", "") or "") if info else ""
    entry_start = str(info.get("entry_start", "") or "") if info else ""
    entry_end = str(info.get("entry_end", "") or "") if info else ""

    parts = [
        f"pack{int(pack_index):04d}",
        f"m{int(member)}",
        f"entry{int(entry_index):04d}",
        natdex,
        name,
        entry_start,
        entry_end,
    ]


    if info:
        for k in (
            "model",
            "texture",
            "shiny_texture",
            "greyscale_texture",
            "battle_animations",
            "refresh_animations",
            "idle,_walk_and_run_animations",
            "lip_animations",
        ):
            if k in info and info[k]:
                parts.append(str(info[k]))

    return " ".join([p for p in parts if p]).lower()
