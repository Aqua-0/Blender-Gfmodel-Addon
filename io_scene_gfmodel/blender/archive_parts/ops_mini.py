

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
from .utils import _garc_cached, _guess_magic4, _guess_ident2, _size_kb, _maybe_lz11, _selected_archive_entry_index, _classify_import_type

from .ops_archive import _import_archive_payload

_GARC_FILE_CACHE: dict[str, object] = {}

def _looks_like_a094_context(context: bpy.types.Context) -> bool:
    try:
        ap = str(getattr(context.scene, 'gfmodel_archive_path', '') or '')
    except Exception:
        ap = ''
    return _looks_like_a094_archive_path(ap)


class GFModel_OT_archive_scan_mini(bpy.types.Operator):
    bl_idname = "gfmodel.archive_scan_mini"
    bl_label = "GFModel: Scan Mini"
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
            entry_bytes = _maybe_lz11(garc.read_primary_bytes(int(entry_i)))
            mini = parse_mini(entry_bytes)
        except Exception as e:
            self.report({"ERROR"}, f"Selected entry is not a Mini container: {e}")
            return {"CANCELLED"}

        try:
            context.scene["gfmodel_archive_mini_ident"] = str(mini.ident)
            context.scene["gfmodel_archive_mini_count"] = int(mini.count)
        except Exception:
            pass

        lst = context.scene.gfmodel_mini_entries
        lst.clear()
        for i in range(mini.count):
            sub = mini.extract(entry_bytes, i)
            sub_dec = _maybe_lz11(sub)
            it = lst.add()
            it.index = int(i)
            it.start_hex = f"0x{int(mini.offsets[i]):X}"
            it.length_hex = f"0x{len(sub):X}"
            head8 = sub[:8]
            head4 = head8[:4]
            it.magic4 = _guess_magic4(head4)
            it.size_kb = _size_kb(len(sub))
            it.lz_tag = (
                _guess_ident2(head8[5:7])
                if len(head8) >= 7 and head4[:1] == b"\x11"
                else ""
            )
            it.kind = (
                "LZ11"
                if head4[:1] == b"\x11"
                else (_guess_ident2(head4[:2]) or it.magic4)
            )
            it.category = _classify_import_type(sub_dec)
            try:
                m2 = parse_mini(sub_dec)
                it.mini_info = f"{m2.ident}({m2.count})"
            except Exception:
                it.mini_info = ""

        context.scene.gfmodel_mini_selected = min(
            int(context.scene.gfmodel_mini_selected), max(0, len(lst) - 1)
        )
        return {"FINISHED"}


class GFModel_OT_archive_import_mini(bpy.types.Operator):
    bl_idname = "gfmodel.archive_import_mini"
    bl_label = "GFModel: Import Selected Mini File"
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
            entry_bytes_raw = garc.read_primary_bytes(entry_i)
            entry_bytes = _maybe_lz11(entry_bytes_raw)
            mini = parse_mini(entry_bytes)
        except Exception as e:
            self.report({"ERROR"}, f"Selected entry is not a Mini container: {e}")
            return {"CANCELLED"}

        mini_i = int(getattr(context.scene, "gfmodel_mini_selected", 0))
        if mini_i < 0 or mini_i >= mini.count:
            self.report({"ERROR"}, f"Mini selection out of range: {mini_i}")
            return {"CANCELLED"}

        try:
            sub_bytes = mini.extract(entry_bytes, mini_i)
        except Exception as e:
            self.report({"ERROR"}, f"Failed to extract mini file: {e}")
            return {"CANCELLED"}

        plan: Optional[PatchPlan] = None
        payload = _maybe_lz11(bytes(sub_bytes))
        source_path = f"{path}#{entry_i}/mini#{mini_i}"
        try:
            pre: List[dict] = []
            cur = bytes(entry_bytes_raw)
            if looks_like_lz11(cur):
                pre.append({"op": "lz11"})
                cur = decompress(cur)

            m = parse_mini(cur)
            pre.append({"op": "mini", "index": int(mini_i), "ident": str(m.ident)})
            cur = m.extract(cur, int(mini_i))
            if looks_like_lz11(cur):
                pre.append({"op": "lz11"})
                cur = decompress(cur)
            plan, payload = make_archive_entry_plan_and_payload(
                archive_path=str(path),
                entry_index=int(entry_i),
                entry_bytes=bytes(entry_bytes_raw),
                pre_steps=pre,
            )
            source_path = str(plan.breadcrumb) or source_path
        except Exception:
            payload = _maybe_lz11(bytes(sub_bytes))

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
            self.report({"ERROR"}, "No GFModel content found in selected mini file")
            return {"CANCELLED"}

        if plan is not None:
            set_scene_patch_plan(context, plan)
        return {"FINISHED"}


def _get_selected_mini_file_bytes(
    context: bpy.types.Context,
    *,
    want_decompressed: bool,
) -> Tuple[str, int, int, bytes]:

    path = str(getattr(context.scene, "gfmodel_archive_path", "")).strip()
    if not path:
        raise ValueError("Set an archive path first")
    if not os.path.exists(path) or not os.path.isfile(path):
        raise ValueError(f"File not found: {path}")
    garc = _garc_cached(path)
    entry_i = _selected_archive_entry_index(context)
    if entry_i < 0 or entry_i >= len(garc.entries):
        raise ValueError(f"Archive entry out of range: {entry_i}")
    entry_bytes = garc.read_primary_bytes(entry_i)
    entry_bytes = _maybe_lz11(entry_bytes)
    mini = parse_mini(entry_bytes)
    mini_i = int(getattr(context.scene, "gfmodel_mini_selected", 0))
    if mini_i < 0 or mini_i >= mini.count:
        raise ValueError(f"Mini selection out of range: {mini_i}")
    sub = mini.extract(entry_bytes, mini_i)
    if want_decompressed:
        sub = _maybe_lz11(sub)
    return path, entry_i, mini_i, sub


class GFModel_OT_archive_scan_container(bpy.types.Operator):
    bl_idname = "gfmodel.archive_scan_container"
    bl_label = "GFModel: Scan Container (CP/CM)"
    bl_options = {"UNDO"}

    def execute(self, context: bpy.types.Context):
        try:
            path, entry_i, mini_i, sub = _get_selected_mini_file_bytes(
                context, want_decompressed=True
            )
        except Exception as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}

        try:
            cont = parse_container(sub)
        except Exception as e:
            self.report({"ERROR"}, f"Selected mini file is not a CP/CM container: {e}")
            return {"CANCELLED"}

        try:
            context.scene["gfmodel_container_magic"] = str(cont.magic2)
            context.scene["gfmodel_container_count"] = int(cont.count)
        except Exception:
            pass

        lst = context.scene.gfmodel_container_entries
        lst.clear()
        for i in range(cont.count):
            b = cont.extract(sub, i)
            head4 = b[:4]
            it = lst.add()
            it.index = int(i)
            it.magic4 = _guess_magic4(head4)
            it.length_hex = f"0x{len(b):X}"
            it.size_kb = _size_kb(len(b))
            it.kind = _guess_ident2(head4[:2]) or it.magic4
            try:
                m2 = parse_mini(_maybe_lz11(b))
                it.mini_info = f"{m2.ident}({m2.count})"
            except Exception:
                it.mini_info = ""

        context.scene.gfmodel_container_selected = min(
            int(getattr(context.scene, "gfmodel_container_selected", 0)),
            max(0, len(lst) - 1),
        )
        context.scene.gfmodel_container2_entries.clear()
        try:
            context.scene["gfmodel_container2_magic"] = ""
            context.scene["gfmodel_container2_count"] = 0
        except Exception:
            pass
        return {"FINISHED"}


class GFModel_OT_archive_import_container_entry(bpy.types.Operator):
    bl_idname = "gfmodel.archive_import_container_entry"
    bl_label = "GFModel: Import Container Entry"
    bl_options = {"UNDO"}

    def execute(self, context: bpy.types.Context):
        try:
            path, entry_i, mini_i, sub = _get_selected_mini_file_bytes(
                context, want_decompressed=True
            )
        except Exception as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}

        try:
            cont = parse_container(sub)
        except Exception as e:
            self.report({"ERROR"}, f"Selected mini file is not a CP/CM container: {e}")
            return {"CANCELLED"}

        sel = int(getattr(context.scene, "gfmodel_container_selected", 0))
        if sel < 0 or sel >= cont.count:
            self.report({"ERROR"}, f"Container selection out of range: {sel}")
            return {"CANCELLED"}
        try:
            b = cont.extract(sub, sel)
        except Exception as e:
            self.report({"ERROR"}, f"Failed to extract container entry: {e}")
            return {"CANCELLED"}

        plan: Optional[PatchPlan] = None
        payload = _maybe_lz11(bytes(b))
        source_path = f"{path}#{entry_i}/mini#{mini_i}/{cont.magic2}[{sel}]"
        try:
            garc = _garc_cached(path)
            entry_bytes_raw = garc.read_primary_bytes(int(entry_i))
            pre: List[dict] = []
            cur = bytes(entry_bytes_raw)
            if looks_like_lz11(cur):
                pre.append({"op": "lz11"})
                cur = decompress(cur)
            m = parse_mini(cur)
            pre.append({"op": "mini", "index": int(mini_i), "ident": str(m.ident)})
            cur = m.extract(cur, int(mini_i))
            if looks_like_lz11(cur):
                pre.append({"op": "lz11"})
                cur = decompress(cur)
            c = parse_container(cur)
            pre.append({"op": "container", "index": int(sel), "magic": str(c.magic2)})
            cur = c.extract(cur, int(sel))
            if looks_like_lz11(cur):
                pre.append({"op": "lz11"})
                cur = decompress(cur)
            plan, payload = make_archive_entry_plan_and_payload(
                archive_path=str(path),
                entry_index=int(entry_i),
                entry_bytes=bytes(entry_bytes_raw),
                pre_steps=pre,
            )
            source_path = str(plan.breadcrumb) or source_path
        except Exception:
            payload = _maybe_lz11(bytes(b))

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
            self.report(
                {"ERROR"}, "No GFModel content found in selected container entry"
            )
            return {"CANCELLED"}

        if plan is not None:
            set_scene_patch_plan(context, plan)
        return {"FINISHED"}


class GFModel_OT_archive_scan_container2(bpy.types.Operator):
    bl_idname = "gfmodel.archive_scan_container2"
    bl_label = "GFModel: Scan Nested Container"
    bl_options = {"UNDO"}

    def execute(self, context: bpy.types.Context):
        try:
            _path, _entry_i, _mini_i, sub = _get_selected_mini_file_bytes(
                context, want_decompressed=True
            )
        except Exception as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}

        try:
            cont = parse_container(sub)
        except Exception as e:
            self.report({"ERROR"}, f"Selected mini file is not a CP/CM container: {e}")
            return {"CANCELLED"}

        sel = int(getattr(context.scene, "gfmodel_container_selected", 0))
        if sel < 0 or sel >= cont.count:
            self.report({"ERROR"}, f"Container selection out of range: {sel}")
            return {"CANCELLED"}

        try:
            inner = _maybe_lz11(cont.extract(sub, sel))
            cont2 = parse_container(inner)
        except Exception as e:
            self.report(
                {"ERROR"}, f"Selected container entry is not a nested CP/CM: {e}"
            )
            return {"CANCELLED"}

        try:
            context.scene["gfmodel_container2_magic"] = str(cont2.magic2)
            context.scene["gfmodel_container2_count"] = int(cont2.count)
        except Exception:
            pass

        lst = context.scene.gfmodel_container2_entries
        lst.clear()
        for i in range(cont2.count):
            b = cont2.extract(inner, i)
            it = lst.add()
            it.index = int(i)
            it.magic4 = _guess_magic4(b[:4])
            it.length_hex = f"0x{len(b):X}"
            it.size_kb = _size_kb(len(b))
            it.kind = _guess_ident2(b[:2]) or it.magic4
            try:
                m2 = parse_mini(_maybe_lz11(b))
                it.mini_info = f"{m2.ident}({m2.count})"
            except Exception:
                it.mini_info = ""

        context.scene.gfmodel_container2_selected = min(
            int(getattr(context.scene, "gfmodel_container2_selected", 0)),
            max(0, len(lst) - 1),
        )
        return {"FINISHED"}


class GFModel_OT_archive_import_container2_entry(bpy.types.Operator):
    bl_idname = "gfmodel.archive_import_container2_entry"
    bl_label = "GFModel: Import Nested Container Entry"
    bl_options = {"UNDO"}

    def execute(self, context: bpy.types.Context):
        try:
            path, entry_i, mini_i, sub = _get_selected_mini_file_bytes(
                context, want_decompressed=True
            )
        except Exception as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}

        try:
            cont = parse_container(sub)
        except Exception as e:
            self.report({"ERROR"}, f"Selected mini file is not a CP/CM container: {e}")
            return {"CANCELLED"}

        sel = int(getattr(context.scene, "gfmodel_container_selected", 0))
        if sel < 0 or sel >= cont.count:
            self.report({"ERROR"}, f"Container selection out of range: {sel}")
            return {"CANCELLED"}

        try:
            inner = _maybe_lz11(cont.extract(sub, sel))
            cont2 = parse_container(inner)
        except Exception as e:
            self.report(
                {"ERROR"}, f"Selected container entry is not a nested CP/CM: {e}"
            )
            return {"CANCELLED"}

        sel2 = int(getattr(context.scene, "gfmodel_container2_selected", 0))
        if sel2 < 0 or sel2 >= cont2.count:
            self.report({"ERROR"}, f"Nested selection out of range: {sel2}")
            return {"CANCELLED"}

        try:
            b = cont2.extract(inner, sel2)
        except Exception as e:
            self.report({"ERROR"}, f"Failed to extract nested container entry: {e}")
            return {"CANCELLED"}

        plan: Optional[PatchPlan] = None
        payload = _maybe_lz11(bytes(b))
        source_path = f"{path}#{entry_i}/mini#{mini_i}/{cont.magic2}[{sel}]/{cont2.magic2}[{sel2}]"
        try:
            garc = _garc_cached(path)
            entry_bytes_raw = garc.read_primary_bytes(int(entry_i))
            pre: List[dict] = []
            cur = bytes(entry_bytes_raw)
            if looks_like_lz11(cur):
                pre.append({"op": "lz11"})
                cur = decompress(cur)
            m = parse_mini(cur)
            pre.append({"op": "mini", "index": int(mini_i), "ident": str(m.ident)})
            cur = m.extract(cur, int(mini_i))
            if looks_like_lz11(cur):
                pre.append({"op": "lz11"})
                cur = decompress(cur)
            c1 = parse_container(cur)
            pre.append({"op": "container", "index": int(sel), "magic": str(c1.magic2)})
            cur = c1.extract(cur, int(sel))
            if looks_like_lz11(cur):
                pre.append({"op": "lz11"})
                cur = decompress(cur)
            c2 = parse_container(cur)
            pre.append({"op": "container", "index": int(sel2), "magic": str(c2.magic2)})
            cur = c2.extract(cur, int(sel2))
            if looks_like_lz11(cur):
                pre.append({"op": "lz11"})
                cur = decompress(cur)
            plan, payload = make_archive_entry_plan_and_payload(
                archive_path=str(path),
                entry_index=int(entry_i),
                entry_bytes=bytes(entry_bytes_raw),
                pre_steps=pre,
            )
            source_path = str(plan.breadcrumb) or source_path
        except Exception:
            payload = _maybe_lz11(bytes(b))

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
            self.report({"ERROR"}, "No GFModel content found in selected nested entry")
            return {"CANCELLED"}

        if plan is not None:
            set_scene_patch_plan(context, plan)
        return {"FINISHED"}


class GFModel_OT_archive_import_container2_pair01(bpy.types.Operator):
    bl_idname = "gfmodel.archive_import_container2_pair01"
    bl_label = "GFModel: Import Nested 0+1 (Model+Motions)"
    bl_options = {"UNDO"}

    def execute(self, context: bpy.types.Context):
        try:
            path, entry_i, mini_i, sub = _get_selected_mini_file_bytes(
                context, want_decompressed=True
            )
        except Exception as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}

        try:
            cont = parse_container(sub)
        except Exception as e:
            self.report({"ERROR"}, f"Selected mini file is not a CP/CM container: {e}")
            return {"CANCELLED"}

        sel = int(getattr(context.scene, "gfmodel_container_selected", 0))
        if sel < 0 or sel >= cont.count:
            self.report({"ERROR"}, f"Container selection out of range: {sel}")
            return {"CANCELLED"}

        try:
            inner = _maybe_lz11(cont.extract(sub, sel))
            cont2 = parse_container(inner)
        except Exception as e:
            self.report(
                {"ERROR"}, f"Selected container entry is not a nested CP/CM: {e}"
            )
            return {"CANCELLED"}

        if cont2.count < 2:
            self.report(
                {"ERROR"}, f"Nested container has <2 entries (count={cont2.count})"
            )
            return {"CANCELLED"}

        b0 = _maybe_lz11(cont2.extract(inner, 0))
        b1 = _maybe_lz11(cont2.extract(inner, 1))

        models0, textures0, motions0, shaders0 = _load_any(b0)
        models1, textures1, motions1, shaders1 = _load_any(b1)

        models = models0 or models1
        if not models:
            self.report({"ERROR"}, "No models found in nested entry 0 or 1")
            return {"CANCELLED"}

        tex_by_name = {t.name: t for t in textures0}
        for t in textures1:
            if t.name not in tex_by_name:
                tex_by_name[t.name] = t
        textures = list(tex_by_name.values())

        sh_by_name = {s.name: s for s in shaders0}
        for s in shaders1:
            if s.name not in sh_by_name:
                sh_by_name[s.name] = s
        shaders = list(sh_by_name.values())

        motions = list(motions0) + [m for m in motions1]

        ok = _import_gfmodel_loaded(
            context,
            models=models,
            textures=textures,
            motions=motions,
            shaders=shaders,
            source_path=f"{path}#{entry_i}/mini#{mini_i}/{cont.magic2}[{sel}]/{cont2.magic2}[0+1]",
            import_textures=True,
            import_animations=True,
        )
        if not ok:
            self.report({"ERROR"}, "Import failed (no GFModel content)")
            return {"CANCELLED"}
        return {"FINISHED"}


