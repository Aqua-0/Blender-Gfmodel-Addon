
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
from .utils import _garc_cached, _guess_magic4, _guess_ident2, _size_kb, _maybe_lz11, _selected_archive_entry_index, _resolve_out_path


_GARC_FILE_CACHE: dict[str, object] = {}

def _looks_like_a094_context(context: bpy.types.Context) -> bool:
    try:
        ap = str(getattr(context.scene, 'gfmodel_archive_path', '') or '')
    except Exception:
        ap = ''
    return _looks_like_a094_archive_path(ap)


class GFModel_OT_archive_patch_mini_from_file(bpy.types.Operator):
    bl_idname = "gfmodel.archive_patch_mini_from_file"
    bl_label = "GFModel: Patch Mini Selected (Rebuild)"
    bl_options = {"UNDO"}

    def execute(self, context: bpy.types.Context):
        archive_path = str(getattr(context.scene, "gfmodel_archive_path", "")).strip()
        if not archive_path:
            self.report({"ERROR"}, "Set an archive path first")
            return {"CANCELLED"}
        if not os.path.exists(archive_path) or not os.path.isfile(archive_path):
            self.report({"ERROR"}, f"File not found: {archive_path}")
            return {"CANCELLED"}

        payload_path = str(
            getattr(context.scene, "gfmodel_mini_patch_payload_path", "")
        ).strip()
        if not payload_path:
            payload_path = str(
                getattr(context.scene, "gfmodel_archive_patch_payload_path", "")
            ).strip()
        if not payload_path:
            payload_path = str(
                context.scene.get("gfmodel_last_export_path", "")
            ).strip()
        if not payload_path:
            self.report({"ERROR"}, "Set Mini Payload (or Payload / last export)")
            return {"CANCELLED"}
        if not os.path.exists(payload_path) or not os.path.isfile(payload_path):
            self.report({"ERROR"}, f"Payload file not found: {payload_path}")
            return {"CANCELLED"}

        try:
            entry_i = _selected_archive_entry_index(context)
        except Exception as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}

        mini_i = int(getattr(context.scene, "gfmodel_mini_selected", 0))
        if mini_i < 0:
            self.report({"ERROR"}, f"Mini selection out of range: {mini_i}")
            return {"CANCELLED"}

        try:
            with open(payload_path, "rb") as f:
                payload = f.read()
        except Exception as e:
            self.report({"ERROR"}, f"Failed to read payload: {e}")
            return {"CANCELLED"}

        try:
            garc = _garc_cached(archive_path)
            entry_bytes = garc.read_primary_bytes(int(entry_i))
        except Exception as e:
            self.report({"ERROR"}, f"Failed to read entry: {e}")
            return {"CANCELLED"}

        was_lz11 = looks_like_lz11(entry_bytes)
        try:
            entry_dec = decompress(entry_bytes) if was_lz11 else entry_bytes
        except Exception as e:
            self.report({"ERROR"}, f"Failed to decompress LZ11: {e}")
            return {"CANCELLED"}

        try:
            mini = parse_mini(entry_dec)
        except Exception as e:
            self.report({"ERROR"}, f"Selected entry is not a Mini container: {e}")
            return {"CANCELLED"}

        if mini_i >= mini.count:
            self.report({"ERROR"}, f"Mini selection out of range: {mini_i}")
            return {"CANCELLED"}

        try:
            entry_dec_new = patch_mini(
                entry_dec, index=int(mini_i), replacement=payload
            )
        except Exception as e:
            self.report({"ERROR"}, f"Failed to patch mini: {e}")
            return {"CANCELLED"}

        if entry_dec_new == entry_dec:
            entry_new = entry_bytes
        else:
            try:
                entry_new = compress(entry_dec_new) if was_lz11 else entry_dec_new
            except Exception as e:
                self.report({"ERROR"}, f"Failed to compress LZ11: {e}")
                return {"CANCELLED"}

        out_path = _resolve_out_path(
            archive_path,
            str(getattr(context.scene, "gfmodel_archive_patch_output_path", "")),
        )
        inplace = bool(getattr(context.scene, "gfmodel_archive_patch_inplace", False))
        make_backup = bool(getattr(context.scene, "gfmodel_archive_patch_backup", True))

        if not inplace:
            try:
                if os.path.abspath(out_path) == os.path.abspath(archive_path):
                    self.report(
                        {"ERROR"},
                        "Out path matches the source archive; enable In-Place (with Backup) or choose a different Out file",
                    )
                    return {"CANCELLED"}
            except Exception:
                pass

        try:
            if inplace:
                if not make_backup:
                    self.report({"ERROR"}, "In-place patch requires Backup enabled")
                    return {"CANCELLED"}
                rewrite_garc_file_inplace_atomic(
                    archive_path,
                    replacements={(int(entry_i), 0): entry_new},
                    make_backup=True,
                )
                out_path = archive_path
            else:
                rewrite_garc_file(
                    archive_path,
                    out_path,
                    replacements={(int(entry_i), 0): entry_new},
                )
        except Exception as e:
            self.report({"ERROR"}, f"Patch failed: {e}")
            return {"CANCELLED"}

        self.report(
            {"INFO"}, f"Patched mini[{mini_i}] into entry {entry_i}: {out_path}"
        )
        return {"FINISHED"}


class GFModel_OT_archive_verify_mini_payload(bpy.types.Operator):
    bl_idname = "gfmodel.archive_verify_mini_payload"
    bl_label = "GFModel: Verify Mini Selected vs Payload"
    bl_options = {"UNDO"}

    def execute(self, context: bpy.types.Context):
        archive_path = str(getattr(context.scene, "gfmodel_archive_path", "")).strip()
        if not archive_path:
            self.report({"ERROR"}, "Set an archive path first")
            return {"CANCELLED"}
        if not os.path.exists(archive_path) or not os.path.isfile(archive_path):
            self.report({"ERROR"}, f"File not found: {archive_path}")
            return {"CANCELLED"}

        payload_path = str(
            getattr(context.scene, "gfmodel_mini_patch_payload_path", "")
        ).strip()
        if not payload_path:
            payload_path = str(
                getattr(context.scene, "gfmodel_archive_patch_payload_path", "")
            ).strip()
        if not payload_path:
            self.report({"ERROR"}, "Set Mini Payload (or Payload)")
            return {"CANCELLED"}
        if not os.path.exists(payload_path) or not os.path.isfile(payload_path):
            self.report({"ERROR"}, f"Payload file not found: {payload_path}")
            return {"CANCELLED"}

        try:
            entry_i = _selected_archive_entry_index(context)
        except Exception as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}

        mini_i = int(getattr(context.scene, "gfmodel_mini_selected", 0))

        try:
            with open(payload_path, "rb") as f:
                payload = f.read()
        except Exception as e:
            self.report({"ERROR"}, f"Failed to read payload: {e}")
            return {"CANCELLED"}

        try:
            garc = _garc_cached(archive_path)
            entry_bytes = garc.read_primary_bytes(int(entry_i))
            entry_dec = (
                decompress(entry_bytes) if looks_like_lz11(entry_bytes) else entry_bytes
            )
            mini = parse_mini(entry_dec)
            if mini_i < 0 or mini_i >= mini.count:
                self.report({"ERROR"}, f"Mini selection out of range: {mini_i}")
                return {"CANCELLED"}
            seg = mini.extract(entry_dec, int(mini_i))
        except Exception as e:
            self.report({"ERROR"}, f"Failed to read mini: {e}")
            return {"CANCELLED"}

        if seg[: len(payload)] != payload:
            self.report(
                {"ERROR"}, "Mismatch: mini segment does not start with payload bytes"
            )
            return {"CANCELLED"}

        pad = seg[len(payload) :]
        if pad and len(set(pad)) == 1:
            self.report(
                {"INFO"},
                f"Match (prefix). Stored has {len(pad)} pad bytes = 0x{pad[0]:02X}",
            )
        else:
            self.report({"INFO"}, "Match (prefix). Stored tail is non-uniform/present.")
        return {"FINISHED"}


class GFModel_OT_archive_patch_container_from_file(bpy.types.Operator):
    bl_idname = "gfmodel.archive_patch_container_from_file"
    bl_label = "GFModel: Patch Container Selected (Rebuild)"
    bl_options = {"UNDO"}

    def execute(self, context: bpy.types.Context):
        archive_path = str(getattr(context.scene, "gfmodel_archive_path", "")).strip()
        if not archive_path:
            self.report({"ERROR"}, "Set an archive path first")
            return {"CANCELLED"}
        if not os.path.exists(archive_path) or not os.path.isfile(archive_path):
            self.report({"ERROR"}, f"File not found: {archive_path}")
            return {"CANCELLED"}

        payload_path = str(
            getattr(context.scene, "gfmodel_container_patch_payload_path", "")
        ).strip()
        if not payload_path:
            payload_path = str(
                getattr(context.scene, "gfmodel_mini_patch_payload_path", "")
            ).strip()
        if not payload_path:
            payload_path = str(
                getattr(context.scene, "gfmodel_archive_patch_payload_path", "")
            ).strip()
        if not payload_path:
            payload_path = str(
                context.scene.get("gfmodel_last_export_path", "")
            ).strip()
        if not payload_path:
            self.report(
                {"ERROR"},
                "Set Container Payload (or Mini Payload / Payload / last export)",
            )
            return {"CANCELLED"}
        if not os.path.exists(payload_path) or not os.path.isfile(payload_path):
            self.report({"ERROR"}, f"Payload file not found: {payload_path}")
            return {"CANCELLED"}

        try:
            entry_i = _selected_archive_entry_index(context)
        except Exception as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        mini_i = int(getattr(context.scene, "gfmodel_mini_selected", 0))
        cont_i = int(getattr(context.scene, "gfmodel_container_selected", 0))

        try:
            with open(payload_path, "rb") as f:
                payload = f.read()
        except Exception as e:
            self.report({"ERROR"}, f"Failed to read payload: {e}")
            return {"CANCELLED"}

        try:
            garc = _garc_cached(archive_path)
            entry_bytes = garc.read_primary_bytes(int(entry_i))
        except Exception as e:
            self.report({"ERROR"}, f"Failed to read entry: {e}")
            return {"CANCELLED"}

        entry_was_lz11 = looks_like_lz11(entry_bytes)
        try:
            entry_dec = decompress(entry_bytes) if entry_was_lz11 else entry_bytes
        except Exception as e:
            self.report({"ERROR"}, f"Failed to decompress entry LZ11: {e}")
            return {"CANCELLED"}

        try:
            mini = parse_mini(entry_dec)
        except Exception as e:
            self.report({"ERROR"}, f"Selected entry is not a Mini container: {e}")
            return {"CANCELLED"}
        if mini_i < 0 or mini_i >= mini.count:
            self.report({"ERROR"}, f"Mini selection out of range: {mini_i}")
            return {"CANCELLED"}

        mini_seg_raw = mini.extract(entry_dec, int(mini_i))
        mini_seg_was_lz11 = looks_like_lz11(mini_seg_raw)
        try:
            mini_seg_dec = (
                decompress(mini_seg_raw) if mini_seg_was_lz11 else mini_seg_raw
            )
        except Exception as e:
            self.report({"ERROR"}, f"Failed to decompress mini file LZ11: {e}")
            return {"CANCELLED"}

        try:
            cont = parse_container(mini_seg_dec)
        except Exception as e:
            self.report({"ERROR"}, f"Selected mini file is not a CP/CM container: {e}")
            return {"CANCELLED"}
        if cont_i < 0 or cont_i >= cont.count:
            self.report({"ERROR"}, f"Container selection out of range: {cont_i}")
            return {"CANCELLED"}

        try:
            cont_new = patch_container(
                mini_seg_dec, index=int(cont_i), replacement=payload
            )
        except Exception as e:
            self.report({"ERROR"}, f"Failed to patch container: {e}")
            return {"CANCELLED"}

        if cont_new == mini_seg_dec:
            mini_seg_new_raw = mini_seg_raw
        else:
            try:
                mini_seg_new_raw = compress(cont_new) if mini_seg_was_lz11 else cont_new
            except Exception as e:
                self.report({"ERROR"}, f"Failed to recompress mini file LZ11: {e}")
                return {"CANCELLED"}

        entry_dec_new = patch_mini(
            entry_dec, index=int(mini_i), replacement=mini_seg_new_raw
        )
        if entry_dec_new == entry_dec:
            entry_new = entry_bytes
        else:
            try:
                entry_new = compress(entry_dec_new) if entry_was_lz11 else entry_dec_new
            except Exception as e:
                self.report({"ERROR"}, f"Failed to recompress entry LZ11: {e}")
                return {"CANCELLED"}

        out_path = _resolve_out_path(
            archive_path,
            str(getattr(context.scene, "gfmodel_archive_patch_output_path", "")),
        )
        inplace = bool(getattr(context.scene, "gfmodel_archive_patch_inplace", False))
        make_backup = bool(getattr(context.scene, "gfmodel_archive_patch_backup", True))

        if not inplace:
            try:
                if os.path.abspath(out_path) == os.path.abspath(archive_path):
                    self.report(
                        {"ERROR"},
                        "Out path matches the source archive; enable In-Place (with Backup) or choose a different Out file",
                    )
                    return {"CANCELLED"}
            except Exception:
                pass

        try:
            if inplace:
                if not make_backup:
                    self.report({"ERROR"}, "In-place patch requires Backup enabled")
                    return {"CANCELLED"}
                rewrite_garc_file_inplace_atomic(
                    archive_path,
                    replacements={(int(entry_i), 0): entry_new},
                    make_backup=True,
                )
                out_path = archive_path
            else:
                rewrite_garc_file(
                    archive_path, out_path, replacements={(int(entry_i), 0): entry_new}
                )
        except Exception as e:
            self.report({"ERROR"}, f"Patch failed: {e}")
            return {"CANCELLED"}

        self.report(
            {"INFO"},
            f"Patched container[{cont_i}] into mini[{mini_i}] entry {entry_i}: {out_path}",
        )
        return {"FINISHED"}


class GFModel_OT_archive_verify_container_payload(bpy.types.Operator):
    bl_idname = "gfmodel.archive_verify_container_payload"
    bl_label = "GFModel: Verify Container Selected vs Payload"
    bl_options = {"UNDO"}

    def execute(self, context: bpy.types.Context):
        archive_path = str(getattr(context.scene, "gfmodel_archive_path", "")).strip()
        if not archive_path:
            self.report({"ERROR"}, "Set an archive path first")
            return {"CANCELLED"}
        if not os.path.exists(archive_path) or not os.path.isfile(archive_path):
            self.report({"ERROR"}, f"File not found: {archive_path}")
            return {"CANCELLED"}

        payload_path = str(
            getattr(context.scene, "gfmodel_container_patch_payload_path", "")
        ).strip()
        if not payload_path:
            payload_path = str(
                getattr(context.scene, "gfmodel_mini_patch_payload_path", "")
            ).strip()
        if not payload_path:
            self.report({"ERROR"}, "Set Container Payload (or Mini Payload)")
            return {"CANCELLED"}
        if not os.path.exists(payload_path) or not os.path.isfile(payload_path):
            self.report({"ERROR"}, f"Payload file not found: {payload_path}")
            return {"CANCELLED"}

        try:
            entry_i = _selected_archive_entry_index(context)
        except Exception as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}

        mini_i = int(getattr(context.scene, "gfmodel_mini_selected", 0))
        cont_i = int(getattr(context.scene, "gfmodel_container_selected", 0))

        try:
            with open(payload_path, "rb") as f:
                payload = f.read()
        except Exception as e:
            self.report({"ERROR"}, f"Failed to read payload: {e}")
            return {"CANCELLED"}

        try:
            garc = _garc_cached(archive_path)
            entry_bytes = garc.read_primary_bytes(int(entry_i))
            entry_dec = (
                decompress(entry_bytes) if looks_like_lz11(entry_bytes) else entry_bytes
            )
            mini = parse_mini(entry_dec)
            if mini_i < 0 or mini_i >= mini.count:
                self.report({"ERROR"}, f"Mini selection out of range: {mini_i}")
                return {"CANCELLED"}
            mini_seg_raw = mini.extract(entry_dec, int(mini_i))
            mini_seg_dec = (
                decompress(mini_seg_raw)
                if looks_like_lz11(mini_seg_raw)
                else mini_seg_raw
            )
            cont = parse_container(mini_seg_dec)
            if cont_i < 0 or cont_i >= cont.count:
                self.report({"ERROR"}, f"Container selection out of range: {cont_i}")
                return {"CANCELLED"}
            seg = cont.extract(mini_seg_dec, int(cont_i))
        except Exception as e:
            self.report({"ERROR"}, f"Failed to read container: {e}")
            return {"CANCELLED"}

        if seg[: len(payload)] != payload:
            self.report(
                {"ERROR"}, "Mismatch: container entry does not start with payload bytes"
            )
            return {"CANCELLED"}
        pad = seg[len(payload) :]
        if pad and len(set(pad)) == 1:
            self.report(
                {"INFO"},
                f"Match (prefix). Stored has {len(pad)} pad bytes = 0x{pad[0]:02X}",
            )
        else:
            self.report({"INFO"}, "Match (prefix). Stored tail is non-uniform/present.")
        return {"FINISHED"}


class GFModel_OT_archive_patch_container2_from_file(bpy.types.Operator):
    bl_idname = "gfmodel.archive_patch_container2_from_file"
    bl_label = "GFModel: Patch Nested Container Selected (Rebuild)"
    bl_options = {"UNDO"}

    def execute(self, context: bpy.types.Context):
        archive_path = str(getattr(context.scene, "gfmodel_archive_path", "")).strip()
        if not archive_path:
            self.report({"ERROR"}, "Set an archive path first")
            return {"CANCELLED"}
        if not os.path.exists(archive_path) or not os.path.isfile(archive_path):
            self.report({"ERROR"}, f"File not found: {archive_path}")
            return {"CANCELLED"}

        payload_path = str(
            getattr(context.scene, "gfmodel_container2_patch_payload_path", "")
        ).strip()
        if not payload_path:
            payload_path = str(
                getattr(context.scene, "gfmodel_container_patch_payload_path", "")
            ).strip()
        if not payload_path:
            payload_path = str(
                getattr(context.scene, "gfmodel_mini_patch_payload_path", "")
            ).strip()
        if not payload_path:
            payload_path = str(
                getattr(context.scene, "gfmodel_archive_patch_payload_path", "")
            ).strip()
        if not payload_path:
            payload_path = str(
                context.scene.get("gfmodel_last_export_path", "")
            ).strip()
        if not payload_path:
            self.report(
                {"ERROR"}, "Set Nested Payload (or Container/Mini/Payload/last export)"
            )
            return {"CANCELLED"}
        if not os.path.exists(payload_path) or not os.path.isfile(payload_path):
            self.report({"ERROR"}, f"Payload file not found: {payload_path}")
            return {"CANCELLED"}

        try:
            entry_i = _selected_archive_entry_index(context)
        except Exception as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}

        mini_i = int(getattr(context.scene, "gfmodel_mini_selected", 0))
        cont_i = int(getattr(context.scene, "gfmodel_container_selected", 0))
        cont2_i = int(getattr(context.scene, "gfmodel_container2_selected", 0))

        try:
            with open(payload_path, "rb") as f:
                payload = f.read()
        except Exception as e:
            self.report({"ERROR"}, f"Failed to read payload: {e}")
            return {"CANCELLED"}

        try:
            garc = _garc_cached(archive_path)
            entry_bytes = garc.read_primary_bytes(int(entry_i))
        except Exception as e:
            self.report({"ERROR"}, f"Failed to read entry: {e}")
            return {"CANCELLED"}

        entry_was_lz11 = looks_like_lz11(entry_bytes)
        try:
            entry_dec = decompress(entry_bytes) if entry_was_lz11 else entry_bytes
        except Exception as e:
            self.report({"ERROR"}, f"Failed to decompress entry LZ11: {e}")
            return {"CANCELLED"}

        try:
            mini = parse_mini(entry_dec)
        except Exception as e:
            self.report({"ERROR"}, f"Selected entry is not a Mini container: {e}")
            return {"CANCELLED"}
        if mini_i < 0 or mini_i >= mini.count:
            self.report({"ERROR"}, f"Mini selection out of range: {mini_i}")
            return {"CANCELLED"}

        mini_seg_raw = mini.extract(entry_dec, int(mini_i))
        mini_seg_was_lz11 = looks_like_lz11(mini_seg_raw)
        try:
            mini_seg_dec = (
                decompress(mini_seg_raw) if mini_seg_was_lz11 else mini_seg_raw
            )
        except Exception as e:
            self.report({"ERROR"}, f"Failed to decompress mini file LZ11: {e}")
            return {"CANCELLED"}

        try:
            cont = parse_container(mini_seg_dec)
        except Exception as e:
            self.report({"ERROR"}, f"Selected mini file is not a CP/CM container: {e}")
            return {"CANCELLED"}
        if cont_i < 0 or cont_i >= cont.count:
            self.report({"ERROR"}, f"Container selection out of range: {cont_i}")
            return {"CANCELLED"}

        outer_raw = cont.extract(mini_seg_dec, int(cont_i))
        outer_was_lz11 = looks_like_lz11(outer_raw)
        try:
            outer_dec = decompress(outer_raw) if outer_was_lz11 else outer_raw
        except Exception as e:
            self.report({"ERROR"}, f"Failed to decompress nested entry LZ11: {e}")
            return {"CANCELLED"}

        try:
            cont2 = parse_container(outer_dec)
        except Exception as e:
            self.report(
                {"ERROR"}, f"Selected container entry is not a nested CP/CM: {e}"
            )
            return {"CANCELLED"}
        if cont2_i < 0 or cont2_i >= cont2.count:
            self.report({"ERROR"}, f"Nested selection out of range: {cont2_i}")
            return {"CANCELLED"}

        try:
            cont2_new = patch_container(
                outer_dec, index=int(cont2_i), replacement=payload
            )
        except Exception as e:
            self.report({"ERROR"}, f"Failed to patch nested container: {e}")
            return {"CANCELLED"}

        if cont2_new == outer_dec:
            outer_new_raw = outer_raw
        else:
            try:
                outer_new_raw = compress(cont2_new) if outer_was_lz11 else cont2_new
            except Exception as e:
                self.report({"ERROR"}, f"Failed to recompress nested entry LZ11: {e}")
                return {"CANCELLED"}


        cont_new = patch_container(
            mini_seg_dec, index=int(cont_i), replacement=outer_new_raw
        )
        if cont_new == mini_seg_dec:
            mini_seg_new_raw = mini_seg_raw
        else:
            try:
                mini_seg_new_raw = compress(cont_new) if mini_seg_was_lz11 else cont_new
            except Exception as e:
                self.report({"ERROR"}, f"Failed to recompress mini file LZ11: {e}")
                return {"CANCELLED"}

        entry_dec_new = patch_mini(
            entry_dec, index=int(mini_i), replacement=mini_seg_new_raw
        )
        if entry_dec_new == entry_dec:
            entry_new = entry_bytes
        else:
            try:
                entry_new = compress(entry_dec_new) if entry_was_lz11 else entry_dec_new
            except Exception as e:
                self.report({"ERROR"}, f"Failed to recompress entry LZ11: {e}")
                return {"CANCELLED"}

        out_path = _resolve_out_path(
            archive_path,
            str(getattr(context.scene, "gfmodel_archive_patch_output_path", "")),
        )
        inplace = bool(getattr(context.scene, "gfmodel_archive_patch_inplace", False))
        make_backup = bool(getattr(context.scene, "gfmodel_archive_patch_backup", True))

        if not inplace:
            try:
                if os.path.abspath(out_path) == os.path.abspath(archive_path):
                    self.report(
                        {"ERROR"},
                        "Out path matches the source archive; enable In-Place (with Backup) or choose a different Out file",
                    )
                    return {"CANCELLED"}
            except Exception:
                pass

        try:
            if inplace:
                if not make_backup:
                    self.report({"ERROR"}, "In-place patch requires Backup enabled")
                    return {"CANCELLED"}
                rewrite_garc_file_inplace_atomic(
                    archive_path,
                    replacements={(int(entry_i), 0): entry_new},
                    make_backup=True,
                )
                out_path = archive_path
            else:
                rewrite_garc_file(
                    archive_path, out_path, replacements={(int(entry_i), 0): entry_new}
                )
        except Exception as e:
            self.report({"ERROR"}, f"Patch failed: {e}")
            return {"CANCELLED"}

        self.report(
            {"INFO"},
            f"Patched nested[{cont2_i}] into container[{cont_i}] mini[{mini_i}] entry {entry_i}: {out_path}",
        )
        return {"FINISHED"}


class GFModel_OT_archive_verify_container2_payload(bpy.types.Operator):
    bl_idname = "gfmodel.archive_verify_container2_payload"
    bl_label = "GFModel: Verify Nested Selected vs Payload"
    bl_options = {"UNDO"}

    def execute(self, context: bpy.types.Context):
        archive_path = str(getattr(context.scene, "gfmodel_archive_path", "")).strip()
        if not archive_path:
            self.report({"ERROR"}, "Set an archive path first")
            return {"CANCELLED"}
        if not os.path.exists(archive_path) or not os.path.isfile(archive_path):
            self.report({"ERROR"}, f"File not found: {archive_path}")
            return {"CANCELLED"}

        payload_path = str(
            getattr(context.scene, "gfmodel_container2_patch_payload_path", "")
        ).strip()
        if not payload_path:
            payload_path = str(
                getattr(context.scene, "gfmodel_container_patch_payload_path", "")
            ).strip()
        if not payload_path:
            self.report({"ERROR"}, "Set Nested Payload (or Container Payload)")
            return {"CANCELLED"}
        if not os.path.exists(payload_path) or not os.path.isfile(payload_path):
            self.report({"ERROR"}, f"Payload file not found: {payload_path}")
            return {"CANCELLED"}

        try:
            entry_i = _selected_archive_entry_index(context)
        except Exception as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}

        mini_i = int(getattr(context.scene, "gfmodel_mini_selected", 0))
        cont_i = int(getattr(context.scene, "gfmodel_container_selected", 0))
        cont2_i = int(getattr(context.scene, "gfmodel_container2_selected", 0))

        try:
            with open(payload_path, "rb") as f:
                payload = f.read()
        except Exception as e:
            self.report({"ERROR"}, f"Failed to read payload: {e}")
            return {"CANCELLED"}

        try:
            garc = _garc_cached(archive_path)
            entry_bytes = garc.read_primary_bytes(int(entry_i))
            entry_dec = (
                decompress(entry_bytes) if looks_like_lz11(entry_bytes) else entry_bytes
            )
            mini = parse_mini(entry_dec)
            if mini_i < 0 or mini_i >= mini.count:
                self.report({"ERROR"}, f"Mini selection out of range: {mini_i}")
                return {"CANCELLED"}
            mini_seg_raw = mini.extract(entry_dec, int(mini_i))
            mini_seg_dec = (
                decompress(mini_seg_raw)
                if looks_like_lz11(mini_seg_raw)
                else mini_seg_raw
            )
            cont = parse_container(mini_seg_dec)
            if cont_i < 0 or cont_i >= cont.count:
                self.report({"ERROR"}, f"Container selection out of range: {cont_i}")
                return {"CANCELLED"}
            outer_raw = cont.extract(mini_seg_dec, int(cont_i))
            outer_dec = (
                decompress(outer_raw) if looks_like_lz11(outer_raw) else outer_raw
            )
            cont2 = parse_container(outer_dec)
            if cont2_i < 0 or cont2_i >= cont2.count:
                self.report({"ERROR"}, f"Nested selection out of range: {cont2_i}")
                return {"CANCELLED"}
            seg = cont2.extract(outer_dec, int(cont2_i))
        except Exception as e:
            self.report({"ERROR"}, f"Failed to read nested container: {e}")
            return {"CANCELLED"}

        if seg[: len(payload)] != payload:
            self.report(
                {"ERROR"}, "Mismatch: nested entry does not start with payload bytes"
            )
            return {"CANCELLED"}
        pad = seg[len(payload) :]
        if pad and len(set(pad)) == 1:
            self.report(
                {"INFO"},
                f"Match (prefix). Stored has {len(pad)} pad bytes = 0x{pad[0]:02X}",
            )
        else:
            self.report({"INFO"}, "Match (prefix). Stored tail is non-uniform/present.")
        return {"FINISHED"}


