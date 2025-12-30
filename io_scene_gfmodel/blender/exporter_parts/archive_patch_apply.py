"""Patch exported bytes back into the source GARC entry using a recorded patch plan."""

from __future__ import annotations

import os
from typing import Optional

import bpy

from ...core.archive_patch import patch_entry_leaf_bytes
from ...core.garc import parse_garc_file, rewrite_garc_file
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


def _sanity_check_breadcrumb(context: bpy.types.Context, breadcrumb: str) -> None:
    bc = str(breadcrumb or "").strip()
    if not bc:
        return
    last_bc = str(context.scene.get("gfmodel_last_import_breadcrumb", "")).strip()
    last_src = str(context.scene.get("gfmodel_last_import_source", "")).strip()
    compare = last_bc or last_src
    if compare and bc != compare:
        raise ValueError(
            "Patch target does not match last imported source.\n"
            f"  last_import_breadcrumb={last_bc}\n"
            f"  last_import_source={last_src}\n"
            f"  patch_target={bc}\n"
            "Re-import the model from GFModel Archive, then patch immediately."
        )


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
        bak_path = archive_path + ".bak"
        if os.path.exists(bak_path):
            raise ValueError(f"Backup already exists: {bak_path}")
        os.replace(archive_path, bak_path)
        rewrite_garc_file(
            bak_path,
            archive_path,
            replacements={(int(entry_i), 0): new_entry},
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
    if bytes(leaf2_dec) != bytes(out_bytes):
        raise ValueError(
            f"Patched, but verify failed (leaf bytes differ): expected={len(out_bytes)} got={len(leaf2_dec)}"
        )

    return str(out_archive)


def patch_into_source_archive(context: bpy.types.Context, out_bytes: bytes) -> str:
    """Patch `out_bytes` into the original GARC entry and return output archive path."""
    plan_json = str(context.scene.get("gfmodel_patch_plan_json", "")).strip()
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
        bak_path = archive_path + ".bak"
        if os.path.exists(bak_path):
            raise ValueError(f"Backup already exists: {bak_path}")
        os.replace(archive_path, bak_path)
        rewrite_garc_file(
            bak_path,
            archive_path,
            replacements={(int(plan.entry_index), int(plan.bit)): new_entry},
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
    if bytes(payload2) != bytes(out_bytes):
        raise ValueError(
            f"Patched, but verify failed (payload bytes differ): expected={len(out_bytes)} got={len(payload2)}"
        )

    return str(out_archive)
