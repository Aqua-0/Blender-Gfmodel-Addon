
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
from .utils import _garc_cached, _guess_magic4, _guess_ident2, _size_kb, _maybe_lz11, _selected_archive_entry_index, _looks_like_a094_archive_path, _gather_a094_group_extras, _resolve_out_path


_GARC_FILE_CACHE: dict[str, object] = {}

def _looks_like_a094_context(context: bpy.types.Context) -> bool:
    try:
        ap = str(getattr(context.scene, 'gfmodel_archive_path', '') or '')
    except Exception:
        ap = ''
    return _looks_like_a094_archive_path(ap)


def _import_archive_payload(
    context: bpy.types.Context,
    *,
    payload: bytes,
    source_path: str,
    archive_path: str,
    entry_i: int,
    import_textures: bool,
    import_animations: bool,
) -> bool:
    payload2 = _maybe_lz11(bytes(payload))

    auto_a094 = bool(getattr(context.scene, "gfmodel_archive_auto_resolve_a094_group", False))
    if auto_a094 and _looks_like_a094_archive_path(str(archive_path)):
        group_start, extras = _gather_a094_group_extras(str(archive_path), int(entry_i))
        if extras:
            print(
                f"[GFModel] a094 auto-resolve: entry={int(entry_i)} group={group_start}..{group_start+8} extras={len(extras)}"
            )
            a094_pack = str(getattr(context.scene, "gfmodel_a094_motion_pack", "ALL"))
            a094_name = bool(getattr(context.scene, "gfmodel_a094_name_motions", True))
            return _import_gfmodel_bytes_with_a094_group(
                context,
                payload2,
                a094_group_members=extras,
                a094_motion_pack=a094_pack,
                a094_name_motions=a094_name,
                source_path=str(source_path),
                import_textures=bool(import_textures),
                import_animations=bool(import_animations),
            )

    return _import_gfmodel_bytes(
        context,
        payload2,
        source_path=str(source_path),
        import_textures=bool(import_textures),
        import_animations=bool(import_animations),
    )


class GFModel_OT_archive_scan(bpy.types.Operator):
    bl_idname = "gfmodel.archive_scan"
    bl_label = "GFModel: Scan Archive"
    bl_options = {"UNDO"}

    def execute(self, context: bpy.types.Context):
        path = str(getattr(context.scene, "gfmodel_archive_path", "")).strip()
        if not path:
            self.report({"ERROR"}, "Set an archive path first")
            return {"CANCELLED"}
        if not os.path.exists(path) or not os.path.isfile(path):
            self.report({"ERROR"}, f"File not found: {path}")
            return {"CANCELLED"}

        try:
            garc = _garc_cached(path)
        except Exception as e:
            self.report({"ERROR"}, f"Not a supported GARC/CRAG: {e}")
            return {"CANCELLED"}

        entries = context.scene.gfmodel_archive_entries
        entries.clear()
        with open(path, "rb") as f:
            for ent in garc.entries:
                start = int(ent.start)
                length = int(ent.length)
                abs_off = int(garc.data_offset) + int(start)
                f.seek(abs_off)
                head8 = f.read(8)
                head4 = head8[:4]
                magic4 = _guess_magic4(head4)
                lz_tag = ""
                if head4[:1] == b"\x11":
                    kind = "LZ11"


                    if len(head8) >= 7:
                        lz_tag = _guess_ident2(head8[5:7])
                else:
                    kind = _guess_ident2(head4) or magic4
                mini_info = ""

                it = entries.add()
                it.index = int(ent.index)
                it.flags_hex = f"0x{int(ent.flags):08X}"
                it.start_hex = f"0x{int(start):X}"
                it.length_hex = f"0x{int(length):X}"
                it.magic4 = magic4
                it.kind = kind
                it.lz_tag = lz_tag
                it.size_kb = _size_kb(int(length))
                it.mini_info = mini_info

        context.scene.gfmodel_archive_selected = min(
            int(context.scene.gfmodel_archive_selected), max(0, len(entries) - 1)
        )
        context.scene["gfmodel_archive_entries_count"] = int(len(entries))
        context.scene.gfmodel_mini_entries.clear()
        try:
            context.scene["gfmodel_archive_mini_ident"] = ""
            context.scene["gfmodel_archive_mini_count"] = 0
        except Exception:
            pass
        return {"FINISHED"}


class GFModel_OT_archive_import_entry(bpy.types.Operator):
    bl_idname = "gfmodel.archive_import_entry"
    bl_label = "GFModel: Import Selected Entry"
    bl_options = {"UNDO"}

    def execute(self, context: bpy.types.Context):
        path = str(getattr(context.scene, "gfmodel_archive_path", "")).strip()
        if not path:
            self.report({"ERROR"}, "Set an archive path first")
            return {"CANCELLED"}
        if not os.path.exists(path) or not os.path.isfile(path):
            self.report({"ERROR"}, f"File not found: {path}")
            return {"CANCELLED"}

        try:
            garc = _garc_cached(path)
        except Exception as e:
            self.report({"ERROR"}, f"Not a supported GARC/CRAG: {e}")
            return {"CANCELLED"}

        try:
            entry_i = _selected_archive_entry_index(context)
        except Exception as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        if entry_i < 0 or entry_i >= len(garc.entries):
            self.report({"ERROR"}, f"Archive entry out of range: {entry_i}")
            return {"CANCELLED"}

        try:
            entry_bytes = garc.read_primary_bytes(int(entry_i))
        except Exception as e:
            self.report({"ERROR"}, f"Failed to extract entry: {e}")
            return {"CANCELLED"}

        plan: Optional[PatchPlan] = None
        payload = _maybe_lz11(bytes(entry_bytes))
        source_path = f"{path}#{int(entry_i)}"
        try:
            plan, payload = make_archive_entry_plan_and_payload(
                archive_path=str(path),
                entry_index=int(entry_i),
                entry_bytes=bytes(entry_bytes),
                pre_steps=[],
            )
            source_path = str(plan.breadcrumb) or source_path
        except Exception:

            payload = _maybe_lz11(bytes(entry_bytes))

        _pending_key = "gfmodel_pending_patch_plan_json"
        if plan is not None:
            context.scene[_pending_key] = plan.to_json()
        try:
            ok = _import_archive_payload(
            context,
            payload=payload,
            source_path=str(source_path),
            archive_path=str(path),
            entry_i=int(entry_i),
            import_textures=True,
            import_animations=True,
        )
        finally:
            if _pending_key in context.scene:
                del context.scene[_pending_key]
        if not ok:
            self.report({"ERROR"}, "No GFModel content found in selected entry")
            return {"CANCELLED"}

        if plan is not None:
            set_scene_patch_plan(context, plan)
        return {"FINISHED"}


class GFModel_OT_archive_patch_entry_from_file(bpy.types.Operator):
    bl_idname = "gfmodel.archive_patch_entry_from_file"
    bl_label = "GFModel: Patch Selected Entry (Raw)"
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
            getattr(context.scene, "gfmodel_archive_patch_payload_path", "")
        ).strip()
        if not payload_path:
            payload_path = str(
                context.scene.get("gfmodel_last_export_path", "")
            ).strip()
        if not payload_path:
            self.report({"ERROR"}, "Set a payload path (or export a file first)")
            return {"CANCELLED"}
        if not os.path.exists(payload_path) or not os.path.isfile(payload_path):
            self.report({"ERROR"}, f"Payload file not found: {payload_path}")
            return {"CANCELLED"}

        try:
            entry_i = _selected_archive_entry_index(context)
        except Exception as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        bit = int(getattr(context.scene, "gfmodel_archive_patch_bit", 0))
        if bit < 0 or bit > 31:
            self.report({"ERROR"}, f"Bit out of range: {bit}")
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
            if os.path.exists(out_path) and os.path.isdir(out_path):
                self.report({"ERROR"}, f"Out path is a directory: {out_path}")
                return {"CANCELLED"}
        except Exception:
            pass

        try:
            with open(payload_path, "rb") as f:
                payload = f.read()
        except Exception as e:
            self.report({"ERROR"}, f"Failed to read payload: {e}")
            return {"CANCELLED"}

        try:
            if inplace:
                if not make_backup:
                    self.report({"ERROR"}, "In-place patch requires Backup enabled")
                    return {"CANCELLED"}
                rewrite_garc_file_inplace_atomic(
                    archive_path,
                    replacements={(int(entry_i), int(bit)): payload},
                    make_backup=True,
                )
                out_path = archive_path
            else:
                rewrite_garc_file(
                    archive_path,
                    out_path,
                    replacements={(int(entry_i), int(bit)): payload},
                )
        except Exception as e:
            self.report({"ERROR"}, f"Patch failed: {e}")
            return {"CANCELLED"}

        self.report({"INFO"}, f"Patched: {out_path}")
        return {"FINISHED"}


class GFModel_OT_archive_verify_entry_payload(bpy.types.Operator):
    bl_idname = "gfmodel.archive_verify_entry_payload"
    bl_label = "GFModel: Verify Selected Entry vs Payload"
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
            getattr(context.scene, "gfmodel_archive_patch_payload_path", "")
        ).strip()
        if not payload_path:
            payload_path = str(
                context.scene.get("gfmodel_last_export_path", "")
            ).strip()
        if not payload_path:
            self.report({"ERROR"}, "Set a payload path (or export a file first)")
            return {"CANCELLED"}
        if not os.path.exists(payload_path) or not os.path.isfile(payload_path):
            self.report({"ERROR"}, f"Payload file not found: {payload_path}")
            return {"CANCELLED"}

        try:
            entry_i = _selected_archive_entry_index(context)
        except Exception as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}

        try:
            with open(payload_path, "rb") as f:
                payload = f.read()
        except Exception as e:
            self.report({"ERROR"}, f"Failed to read payload: {e}")
            return {"CANCELLED"}

        try:
            garc = _garc_cached(archive_path)
            entry = garc.read_primary_bytes(int(entry_i))
        except Exception as e:
            self.report({"ERROR"}, f"Failed to read entry: {e}")
            return {"CANCELLED"}

        if entry[: len(payload)] != payload:
            self.report({"ERROR"}, "Mismatch: entry does not start with payload bytes")
            return {"CANCELLED"}

        pad = entry[len(payload) :]
        if pad and len(set(pad)) == 1:
            self.report(
                {"INFO"},
                f"Match (prefix). Stored has {len(pad)} pad bytes = 0x{pad[0]:02X}",
            )
        else:
            self.report({"INFO"}, "Match (prefix). Stored tail is non-uniform/present.")
        return {"FINISHED"}


