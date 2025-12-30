"""Helpers to build and persist a generalized archive patch plan (Blender-side)."""

from __future__ import annotations

from typing import List, Optional, Tuple

import bpy

from ..core.lz11 import decompress, looks_like_lz11
from ..core.mini import parse_mini
from ..core.patch_plan import PatchPlan, find_exportable_blob, steps_to_breadcrumb
from ..core.pkmn_container import parse_container


def set_scene_patch_plan(context: bpy.types.Context, plan: PatchPlan) -> None:
    try:
        context.scene["gfmodel_patch_plan_json"] = plan.to_json()
        context.scene["gfmodel_patch_source_level"] = "plan_v1"
        context.scene["gfmodel_patch_source_archive"] = str(plan.archive_path)
        context.scene["gfmodel_patch_source_entry"] = int(plan.entry_index)
        context.scene["gfmodel_patch_source_breadcrumb"] = str(plan.breadcrumb)
    except Exception:
        pass


def make_archive_entry_plan_and_payload(
    *,
    archive_path: str,
    entry_index: int,
    entry_bytes: bytes,
    pre_steps: Optional[List[dict]] = None,
) -> Tuple[PatchPlan, bytes]:
    """Build a plan rooted at a GARC entry, and return (plan, exportable_payload_bytes).

    `pre_steps` can be used to anchor the search within a user-selected sub-blob
    (e.g. mini index / container index), before locating an exportable GFModel blob.
    """
    pre = list(pre_steps or [])
    sel = bytes(entry_bytes)
    for st in pre:
        op = str(st.get("op", "")).strip()
        if op == "lz11":
            if not looks_like_lz11(sel):
                raise ValueError("expected LZ11 wrapper, but data was not LZ11")
            sel = decompress(sel)
        elif op == "mini":
            m = parse_mini(sel)
            sel = m.extract(sel, int(st.get("index", -1)))
        elif op == "container":
            c = parse_container(sel)
            want = str(st.get("magic", "")).strip()
            if want and want != c.magic2:
                raise ValueError(
                    f"container magic mismatch: expected={want} got={c.magic2}"
                )
            sel = c.extract(sel, int(st.get("index", -1)))
        else:
            raise ValueError(f"unknown pre-step: {op!r}")

    payload, extra = find_exportable_blob(sel)
    steps = pre + list(extra)
    bc = f"{archive_path}#{int(entry_index)}"
    st = steps_to_breadcrumb(steps)
    if st:
        bc = f"{bc}/{st}"
    plan = PatchPlan(
        version=1,
        archive_path=str(archive_path),
        entry_index=int(entry_index),
        bit=0,
        steps=[dict(x) for x in steps],
        breadcrumb=str(bc),
    )
    return plan, bytes(payload)
