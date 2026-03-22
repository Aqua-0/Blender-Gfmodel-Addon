
from __future__ import annotations

import os
from typing import Optional

import bpy

from ...core.archive_patch import patch_entry_leaf_bytes
from ...core.garc import (
    parse_garc_file,
    rewrite_garc_file,
    rewrite_garc_file_inplace_atomic,
)
from ...core.lz11 import decompress, looks_like_lz11
from ...core.mini import parse_mini
from ...core.patch_plan import (
    extract_via_steps,
    patch_plan_from_json,
    patch_via_steps,
)
from ...core.pkmn_container import parse_container


def _resolve_out_archive(archive_path: str, out_archive: str) -> str:
    out_archive = str(out_archive or "").strip()
    if not out_archive:
        out_archive = str(archive_path) + ".patched"
    try:
        if out_archive.endswith(("/", "\\")) or os.path.isdir(out_archive):
            base = os.path.basename(str(archive_path).rstrip("\\/"))
            out_archive = os.path.join(out_archive.rstrip("\\/"), base + ".patched")
    except Exception:
        pass
    return str(out_archive)


def _resolve_target_collection(
    context: bpy.types.Context,
) -> Optional[bpy.types.Collection]:
    obj = getattr(context, "active_object", None)
    if obj is not None:
        try:
            name = str(obj.get("gfmodel_import_collection", "") or "").strip()
            if name:
                coll = bpy.data.collections.get(name)
                if coll is not None:
                    return coll
        except Exception:
            pass
    try:
        name = str(context.scene.get("gfmodel_last_import_collection", "") or "").strip()
        if name:
            coll = bpy.data.collections.get(name)
            if coll is not None:
                return coll
    except Exception:
        pass
    return None


def _resolve_patch_plan_json(context: bpy.types.Context) -> str:
    coll = _resolve_target_collection(context)
    if coll is not None:
        try:
            pj = str(coll.get("gfmodel_patch_plan_json", "") or "").strip()
            if pj:
                return pj
        except Exception:
            pass
    return str(context.scene.get("gfmodel_patch_plan_json", "") or "").strip()


def _resolve_last_import_compare(context: bpy.types.Context) -> tuple[str, str, str]:
    coll = _resolve_target_collection(context)
    if coll is not None:
        try:
            last_bc = str(coll.get("gfmodel_last_import_breadcrumb", "") or "").strip()
            last_src = str(coll.get("gfmodel_last_import_source", "") or "").strip()
            compare = last_bc or last_src
            return last_bc, last_src, compare
        except Exception:
            pass
    last_bc = str(context.scene.get("gfmodel_last_import_breadcrumb", "") or "").strip()
    last_src = str(context.scene.get("gfmodel_last_import_source", "") or "").strip()
    compare = last_bc or last_src
    return last_bc, last_src, compare


def _sanity_check_breadcrumb(context: bpy.types.Context, breadcrumb: str) -> None:
    bc = str(breadcrumb or "").strip()
    if not bc:
        return
    last_bc, last_src, compare = _resolve_last_import_compare(context)
    if compare and bc != compare:
        raise ValueError(
            "Patch target does not match last imported source.\n"
            f"  last_import_breadcrumb={last_bc}\n"
            f"  last_import_source={last_src}\n"
            f"  patch_target={bc}\n"
            "Re-import the model from GFModel Archive, then patch immediately."
        )


def _payload_matches_expected(payload: bytes, expected: bytes) -> bool:
    p = bytes(payload)
    e = bytes(expected)
    if p == e:
        return True
    if len(p) < len(e):
        return False
    if p[: len(e)] != e:
        return False
    tail = p[len(e) :]
    return not tail or all(b == 0 for b in tail)


def _patch_via_legacy_container2(context: bpy.types.Context, out_bytes: bytes) -> str:
    archive_path = str(context.scene.get("gfmodel_patch_source_archive", "")).strip()
    entry_i = int(context.scene.get("gfmodel_patch_source_entry", -1))
    mini_i = int(context.scene.get("gfmodel_patch_source_mini", -1))
    cont_i = int(context.scene.get("gfmodel_patch_source_container", -1))
    nest_i = int(context.scene.get("gfmodel_patch_source_nested", -1))
    if not archive_path or entry_i < 0 or mini_i < 0 or cont_i < 0 or nest_i < 0:
        raise ValueError("Patch context is incomplete; re-import from GFModel Archive")

    bc = str(context.scene.get("gfmodel_patch_source_breadcrumb", "")).strip()
    _sanity_check_breadcrumb(context, bc)

    out_archive = _resolve_out_archive(
        archive_path,
        str(getattr(context.scene, "gfmodel_archive_patch_output_path", "")),
    )
    inplace = bool(getattr(context.scene, "gfmodel_archive_patch_inplace", False))
    backup = bool(getattr(context.scene, "gfmodel_archive_patch_backup", True))

    garc = parse_garc_file(archive_path)
    entry_bytes = garc.read_primary_bytes(int(entry_i))
    new_entry = patch_entry_leaf_bytes(
        entry_bytes,
        mini_index=int(mini_i),
        container_index=int(cont_i),
        nested_index=int(nest_i),
        replacement_leaf_bytes=out_bytes,
    )

    if inplace:
        if not backup:
            raise ValueError("In-place patch requires Backup enabled")
        rewrite_garc_file_inplace_atomic(
            archive_path,
            replacements={(int(entry_i), 0): new_entry},
            make_backup=True,
        )
        out_archive = archive_path
    else:
        if os.path.abspath(out_archive) == os.path.abspath(archive_path):
            raise ValueError(
                "Archive Out path matches source; enable In-Place (with Backup) or choose a different Out file"
            )
        rewrite_garc_file(
            archive_path,
            out_archive,
            replacements={(int(entry_i), 0): new_entry},
        )


    garc2 = parse_garc_file(out_archive)
    entry2 = garc2.read_primary_bytes(int(entry_i))
    entry2_dec = decompress(entry2) if looks_like_lz11(entry2) else entry2
    mini2 = parse_mini(entry2_dec)
    seg2 = mini2.extract(entry2_dec, int(mini_i))
    seg2_dec = decompress(seg2) if looks_like_lz11(seg2) else seg2
    cont2 = parse_container(seg2_dec)
    outer2 = cont2.extract(seg2_dec, int(cont_i))
    outer2_dec = decompress(outer2) if looks_like_lz11(outer2) else outer2
    cont3 = parse_container(outer2_dec)
    leaf2 = cont3.extract(outer2_dec, int(nest_i))
    leaf2_dec = decompress(leaf2) if looks_like_lz11(leaf2) else leaf2
    if not _payload_matches_expected(leaf2_dec, out_bytes):
        raise ValueError(
            f"Patched, but verify failed (leaf bytes differ): expected={len(out_bytes)} got={len(leaf2_dec)}"
        )

    return str(out_archive)


def patch_into_source_archive(context: bpy.types.Context, out_bytes: bytes) -> str:
    plan_json = _resolve_patch_plan_json(context)
    if not plan_json:

        level = str(context.scene.get("gfmodel_patch_source_level", "")).strip()
        if level == "container2":
            return _patch_via_legacy_container2(context, out_bytes)
        raise ValueError(
            "No patch plan stored; import a model from GFModel Archive first"
        )

    plan = patch_plan_from_json(plan_json)
    _sanity_check_breadcrumb(context, plan.breadcrumb)

    archive_path = str(plan.archive_path)
    out_archive = _resolve_out_archive(
        archive_path,
        str(getattr(context.scene, "gfmodel_archive_patch_output_path", "")),
    )
    inplace = bool(getattr(context.scene, "gfmodel_archive_patch_inplace", False))
    backup = bool(getattr(context.scene, "gfmodel_archive_patch_backup", True))

    garc = parse_garc_file(archive_path)
    entry_bytes = garc.read_primary_bytes(int(plan.entry_index))
    new_entry = patch_via_steps(entry_bytes, plan.steps, out_bytes)

    if inplace:
        if not backup:
            raise ValueError("In-place patch requires Backup enabled")
        rewrite_garc_file_inplace_atomic(
            archive_path,
            replacements={(int(plan.entry_index), int(plan.bit)): new_entry},
            make_backup=True,
        )
        out_archive = archive_path
    else:
        if os.path.abspath(out_archive) == os.path.abspath(archive_path):
            raise ValueError(
                "Archive Out path matches source; enable In-Place (with Backup) or choose a different Out file"
            )
        rewrite_garc_file(
            archive_path,
            out_archive,
            replacements={(int(plan.entry_index), int(plan.bit)): new_entry},
        )


    garc2 = parse_garc_file(out_archive)
    entry2 = garc2.read_primary_bytes(int(plan.entry_index))
    payload2 = extract_via_steps(entry2, plan.steps)
    if not _payload_matches_expected(payload2, out_bytes):
        raise ValueError(
            f"Patched, but verify failed (payload bytes differ): expected={len(out_bytes)} got={len(payload2)}"
        )

    return str(out_archive)
