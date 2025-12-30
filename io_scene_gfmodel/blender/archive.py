"""Blender UI: browse GARC archives and import a single entry as GFModel."""

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

from ..core.garc import parse_garc_file, rewrite_garc_file
from ..core.io import _load_any
from ..core.lz11 import compress, decompress, looks_like_lz11
from ..core.mini import parse_mini, patch_mini
from ..core.patch_plan import PatchPlan
from ..core.pkmn_container import parse_container, patch_container
from .archive_patch_context import (
    make_archive_entry_plan_and_payload,
    set_scene_patch_plan,
)
from .importer import _import_gfmodel_bytes, _import_gfmodel_loaded

_GARC_FILE_CACHE: dict[str, object] = {}


class GFModelArchiveEntry(bpy.types.PropertyGroup):
    index: bpy.props.IntProperty()                            
    flags_hex: bpy.props.StringProperty()                            
    start_hex: bpy.props.StringProperty()                            
    length_hex: bpy.props.StringProperty()                            
    magic4: bpy.props.StringProperty()                            
    kind: bpy.props.StringProperty()                            
    lz_tag: bpy.props.StringProperty()                            
    size_kb: bpy.props.StringProperty()                            
    mini_info: bpy.props.StringProperty()                            


class GFModelMiniEntry(bpy.types.PropertyGroup):
    index: bpy.props.IntProperty()                            
    start_hex: bpy.props.StringProperty()                            
    length_hex: bpy.props.StringProperty()                            
    magic4: bpy.props.StringProperty()                            
    mini_info: bpy.props.StringProperty()                            
    kind: bpy.props.StringProperty()                            
    lz_tag: bpy.props.StringProperty()                            
    size_kb: bpy.props.StringProperty()                            
    category: bpy.props.StringProperty()                            


class GFModelContainerEntry(bpy.types.PropertyGroup):
    index: bpy.props.IntProperty()                            
    kind: bpy.props.StringProperty()                            
    size_kb: bpy.props.StringProperty()                            
    length_hex: bpy.props.StringProperty()                            
    magic4: bpy.props.StringProperty()                            
    mini_info: bpy.props.StringProperty()                            


class GFModel_UL_archive_entries(bpy.types.UIList):
    def draw_item(
        self,
        context: bpy.types.Context,
        layout: bpy.types.UILayout,
        data: object,
        item: object,
        icon: int,
        active_data: object,
        active_propname: str,
        index: int,
    ) -> None:
        e: GFModelArchiveEntry = item                            
        row = layout.row(align=True)
        row.label(text=f"{int(e.index):04d}")
        row.label(text=str(getattr(e, "kind", "")))
        row.label(text=str(getattr(e, "lz_tag", "")))
        row.label(text=str(getattr(e, "size_kb", "")))
        row.label(text=str(getattr(e, "length_hex", "")))
        row.label(text=str(getattr(e, "flags_hex", "")))
        mi = str(getattr(e, "mini_info", ""))
        if mi:
            row.label(text=mi)

    def filter_items(self, context: bpy.types.Context, data: object, propname: str):
        items = getattr(data, propname)
        flt_flags = []
        flt_neworder = []
        query = str(getattr(context.scene, "gfmodel_archive_search", "")).strip()
        want = _parse_search_int(query) if query else None
        for it in items:
            show = True
            if want is not None:
                show = int(getattr(it, "index", -1)) == int(want)
            elif query:
                s = str(getattr(it, "index", ""))
                show = query in s or query in f"{int(getattr(it, 'index', 0)):04d}"
            flt_flags.append(self.bitflag_filter_item if show else 0)
        return flt_flags, flt_neworder


class GFModel_UL_mini_entries(bpy.types.UIList):
    def draw_item(
        self,
        context: bpy.types.Context,
        layout: bpy.types.UILayout,
        data: object,
        item: object,
        icon: int,
        active_data: object,
        active_propname: str,
        index: int,
    ) -> None:
        e: GFModelMiniEntry = item                            
        row = layout.row(align=True)
        row.label(text=f"{int(e.index):04d}")
        row.label(text=str(getattr(e, "kind", "")))
        row.label(text=str(getattr(e, "lz_tag", "")))
        row.label(text=str(getattr(e, "size_kb", "")))
        row.label(text=str(getattr(e, "length_hex", "")))
        mi = str(getattr(e, "mini_info", ""))
        if mi:
            row.label(text=mi)

    def filter_items(self, context: bpy.types.Context, data: object, propname: str):
        items = getattr(data, propname)
        flt_flags = []
        flt_neworder = []
        want = str(getattr(context.scene, "gfmodel_mini_filter", "ALL"))
        query = str(getattr(context.scene, "gfmodel_mini_search", "")).strip()
        want_i = _parse_search_int(query) if query else None
        for it in items:
            show = True
            if want != "ALL":
                show = str(getattr(it, "category", "")) == want
            if show and want_i is not None:
                show = int(getattr(it, "index", -1)) == int(want_i)
            elif show and query:
                s = str(getattr(it, "index", ""))
                show = query in s or query in f"{int(getattr(it, 'index', 0)):04d}"
            flt_flags.append(self.bitflag_filter_item if show else 0)
        return flt_flags, flt_neworder


class GFModel_UL_container_entries(bpy.types.UIList):
    def draw_item(
        self,
        context: bpy.types.Context,
        layout: bpy.types.UILayout,
        data: object,
        item: object,
        icon: int,
        active_data: object,
        active_propname: str,
        index: int,
    ) -> None:
        e: GFModelContainerEntry = item                            
        row = layout.row(align=True)
        row.label(text=f"{int(e.index):04d}")
        row.label(text=str(getattr(e, "kind", "")))
        row.label(text=str(getattr(e, "size_kb", "")))
        row.label(text=str(getattr(e, "length_hex", "")))
        mi = str(getattr(e, "mini_info", ""))
        if mi:
            row.label(text=mi)

    def filter_items(self, context: bpy.types.Context, data: object, propname: str):
        items = getattr(data, propname)
        flt_flags = []
        flt_neworder = []
        query_prop = (
            "gfmodel_container2_search"
            if str(propname) == "gfmodel_container2_entries"
            else "gfmodel_container_search"
        )
        query = str(getattr(context.scene, query_prop, "")).strip()
        want = _parse_search_int(query) if query else None
        for it in items:
            show = True
            if want is not None:
                show = int(getattr(it, "index", -1)) == int(want)
            elif query:
                s = str(getattr(it, "index", ""))
                show = query in s or query in f"{int(getattr(it, 'index', 0)):04d}"
            flt_flags.append(self.bitflag_filter_item if show else 0)
        return flt_flags, flt_neworder


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
        payload = bytes(entry_bytes)
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
                                                           
            payload = bytes(entry_bytes)

        ok = _import_gfmodel_bytes(
            context,
            payload,
            source_path=str(source_path),
            import_textures=True,
            import_animations=True,
        )
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
                bak_path = archive_path + ".bak"
                if os.path.exists(bak_path):
                    self.report({"ERROR"}, f"Backup already exists: {bak_path}")
                    return {"CANCELLED"}
                os.replace(archive_path, bak_path)
                rewrite_garc_file(
                    bak_path,
                    archive_path,
                    replacements={(int(entry_i), int(bit)): payload},
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
                bak_path = archive_path + ".bak"
                if os.path.exists(bak_path):
                    self.report({"ERROR"}, f"Backup already exists: {bak_path}")
                    return {"CANCELLED"}
                os.replace(archive_path, bak_path)
                rewrite_garc_file(
                    bak_path,
                    archive_path,
                    replacements={(int(entry_i), 0): entry_new},
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
                bak_path = archive_path + ".bak"
                if os.path.exists(bak_path):
                    self.report({"ERROR"}, f"Backup already exists: {bak_path}")
                    return {"CANCELLED"}
                os.replace(archive_path, bak_path)
                rewrite_garc_file(
                    bak_path, archive_path, replacements={(int(entry_i), 0): entry_new}
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
                bak_path = archive_path + ".bak"
                if os.path.exists(bak_path):
                    self.report({"ERROR"}, f"Backup already exists: {bak_path}")
                    return {"CANCELLED"}
                os.replace(archive_path, bak_path)
                rewrite_garc_file(
                    bak_path, archive_path, replacements={(int(entry_i), 0): entry_new}
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
        payload = bytes(sub_bytes)
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
            payload = bytes(sub_bytes)

        ok = _import_gfmodel_bytes(
            context,
            payload,
            source_path=str(source_path),
            import_textures=True,
            import_animations=True,
        )
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
    """Return (archive_path, garc_entry_index, mini_index, mini_file_bytes)."""
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
        payload = bytes(b)
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
            payload = bytes(b)

        ok = _import_gfmodel_bytes(
            context,
            payload,
            source_path=str(source_path),
            import_textures=True,
            import_animations=True,
        )
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
        payload = bytes(b)
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
            payload = bytes(b)

        ok = _import_gfmodel_bytes(
            context,
            payload,
            source_path=str(source_path),
            import_textures=True,
            import_animations=True,
        )
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


class VIEW3D_PT_gfmodel_archive(bpy.types.Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "GFModel"
    bl_label = "GFModel Archive"

    def draw(self, context: bpy.types.Context):
        layout = self.layout
        layout.prop(context.scene, "gfmodel_archive_path", text="Archive")
        bc = _breadcrumb(context)
        if len(bc) > 96:
            layout.label(text=bc[:96])
            layout.label(text=bc[96:])
        else:
            layout.label(text=bc)
        row = layout.row(align=True)
        row.operator("gfmodel.archive_scan", text="Scan")
        row.operator("gfmodel.archive_import_entry", text="Import Selected")
        row = layout.row(align=True)
        row.prop(context.scene, "gfmodel_archive_patch_payload_path", text="Payload")
        row.operator("gfmodel.archive_patch_entry_from_file", text="Patch Selected")
        row.operator("gfmodel.archive_verify_entry_payload", text="Verify")
        row = layout.row(align=True)
        row.prop(context.scene, "gfmodel_archive_patch_output_path", text="Out")
        row.prop(context.scene, "gfmodel_archive_patch_inplace", text="In-Place")
        row.prop(context.scene, "gfmodel_archive_patch_backup", text="Backup")

        entries = getattr(context.scene, "gfmodel_archive_entries", None)
        if entries is None or len(entries) == 0:
            layout.label(text="(No entries scanned)")
            return

        layout.prop(context.scene, "gfmodel_archive_search", text="Find Entry")
        layout.template_list(
            "GFModel_UL_archive_entries",
            "",
            context.scene,
            "gfmodel_archive_entries",
            context.scene,
            "gfmodel_archive_selected",
            rows=8,
        )

        layout.separator()
        layout.label(text="Mini (pk3DS) in selected entry")
        ident = str(context.scene.get("gfmodel_archive_mini_ident", ""))
        count = int(context.scene.get("gfmodel_archive_mini_count", 0))
        if ident:
            layout.label(text=f"Ident: {ident}  Count: {count}")
        layout.prop(context.scene, "gfmodel_mini_filter", text="Folder")
        layout.prop(context.scene, "gfmodel_mini_search", text="Find Mini")
        row = layout.row(align=True)
        row.operator("gfmodel.archive_scan_mini", text="Scan Mini")
        row.operator("gfmodel.archive_import_mini", text="Import Mini Selected")
        row = layout.row(align=True)
        row.prop(context.scene, "gfmodel_mini_patch_payload_path", text="Mini Payload")
        row.operator("gfmodel.archive_patch_mini_from_file", text="Patch Mini Selected")
        row.operator("gfmodel.archive_verify_mini_payload", text="Verify")

        mini_entries = getattr(context.scene, "gfmodel_mini_entries", None)
        if mini_entries is None or len(mini_entries) == 0:
            layout.label(text="(No mini entries scanned)")
            return
        layout.template_list(
            "GFModel_UL_mini_entries",
            "",
            context.scene,
            "gfmodel_mini_entries",
            context.scene,
            "gfmodel_mini_selected",
            rows=6,
        )

        layout.separator()
        layout.label(text="CP/CM container in selected mini file")
        cmagic = str(context.scene.get("gfmodel_container_magic", ""))
        ccount = int(context.scene.get("gfmodel_container_count", 0))
        if cmagic:
            layout.label(text=f"Container: {cmagic}  Count: {ccount}")
        row = layout.row(align=True)
        row.prop(
            context.scene,
            "gfmodel_container_patch_payload_path",
            text="Container Payload",
        )
        row.operator(
            "gfmodel.archive_patch_container_from_file", text="Patch Container Selected"
        )
        row.operator("gfmodel.archive_verify_container_payload", text="Verify")
        row = layout.row(align=True)
        row.operator("gfmodel.archive_scan_container", text="Scan")
        row.operator("gfmodel.archive_import_container_entry", text="Import")

        c_entries = getattr(context.scene, "gfmodel_container_entries", None)
        if c_entries is None or len(c_entries) == 0:
            layout.label(text="(No container scanned)")
            return
        layout.prop(context.scene, "gfmodel_container_search", text="Find Container")
        layout.template_list(
            "GFModel_UL_container_entries",
            "",
            context.scene,
            "gfmodel_container_entries",
            context.scene,
            "gfmodel_container_selected",
            rows=5,
        )

        layout.separator()
        layout.label(text="Nested CP/CM (selected container entry)")
        cmagic2 = str(context.scene.get("gfmodel_container2_magic", ""))
        ccount2 = int(context.scene.get("gfmodel_container2_count", 0))
        if cmagic2:
            layout.label(text=f"Nested: {cmagic2}  Count: {ccount2}")
        row = layout.row(align=True)
        row.prop(
            context.scene,
            "gfmodel_container2_patch_payload_path",
            text="Nested Payload",
        )
        row.operator(
            "gfmodel.archive_patch_container2_from_file", text="Patch Nested Selected"
        )
        row.operator("gfmodel.archive_verify_container2_payload", text="Verify")
        row = layout.row(align=True)
        row.operator("gfmodel.archive_scan_container2", text="Scan Nested")
        row.operator("gfmodel.archive_import_container2_entry", text="Import Nested")
        row.operator("gfmodel.archive_import_container2_pair01", text="Import 0+1")
        row = layout.row(align=True)
        row.operator(
            "gfmodel.patch_current_scene_grow_buffers_tris",
            text="Patch (Grow Buffers)",
        )

        layout.separator()
        box = layout.box()
        box.label(text="Buffer Headroom (Active Mesh)")
        obj = context.active_object
        if (
            obj is None
            or obj.type != "MESH"
            or obj.get("gfmodel_submesh_index") is None
        ):
            box.label(text="Select an imported GFModel mesh object")
        else:
            mesh = obj.data
            try:
                mesh.calc_loop_triangles()
            except Exception:
                pass
            tri_count = int(len(getattr(mesh, "loop_triangles", []) or []))
            scene_indices = int(tri_count) * 3
            cap = int(obj.get("gfmodel_index_capacity", 0) or 0)
            elem = int(obj.get("gfmodel_index_elem_size", 0) or 0)
            file_idx = int(obj.get("gfmodel_index_count_file", 0) or 0)
            headroom = int(cap - scene_indices) if cap > 0 else 0

            stride = int(obj.get("gfmodel_vertex_stride", 0) or 0)
            vcap = int(obj.get("gfmodel_vertex_capacity", 0) or 0)
            vfile = int(obj.get("gfmodel_vertex_count_file", 0) or 0)
            vscene = int(len(getattr(mesh, "vertices", []) or []))
            vhead = int(vcap - vscene) if vcap > 0 else 0

            mat_name = str(obj.get("gfmodel_material_name", "") or "")
            smi = int(obj.get("gfmodel_submesh_index", -1))
            box.label(text=f"submesh={smi} {mat_name}")
            box.label(
                text=f"IDX elem={elem} cap={cap} file={file_idx} scene={scene_indices} headroom={headroom}"
            )
            box.label(
                text=f"VTX stride={stride} cap={vcap} file={vfile} scene={vscene} headroom={vhead}"
            )
            if cap > 0 and scene_indices > cap:
                box.label(text="OVER CAPACITY: topology patch will fail", icon="ERROR")
            if vcap > 0 and vscene > vcap:
                box.label(
                    text="OVER CAPACITY: verts+topo patch will fail", icon="ERROR"
                )

        c2_entries = getattr(context.scene, "gfmodel_container2_entries", None)
        if c2_entries is None or len(c2_entries) == 0:
            return
        layout.prop(context.scene, "gfmodel_container2_search", text="Find Nested")
        layout.template_list(
            "GFModel_UL_container_entries",
            "NESTED",
            context.scene,
            "gfmodel_container2_entries",
            context.scene,
            "gfmodel_container2_selected",
            rows=5,
        )


classes = (
    GFModelArchiveEntry,
    GFModelMiniEntry,
    GFModelContainerEntry,
    GFModel_UL_archive_entries,
    GFModel_UL_mini_entries,
    GFModel_UL_container_entries,
    GFModel_OT_archive_scan,
    GFModel_OT_archive_import_entry,
    GFModel_OT_archive_patch_entry_from_file,
    GFModel_OT_archive_verify_entry_payload,
    GFModel_OT_archive_patch_mini_from_file,
    GFModel_OT_archive_verify_mini_payload,
    GFModel_OT_archive_patch_container_from_file,
    GFModel_OT_archive_verify_container_payload,
    GFModel_OT_archive_patch_container2_from_file,
    GFModel_OT_archive_verify_container2_payload,
    GFModel_OT_archive_scan_mini,
    GFModel_OT_archive_import_mini,
    GFModel_OT_archive_scan_container,
    GFModel_OT_archive_import_container_entry,
    GFModel_OT_archive_scan_container2,
    GFModel_OT_archive_import_container2_entry,
    GFModel_OT_archive_import_container2_pair01,
    VIEW3D_PT_gfmodel_archive,
)


def register() -> None:
    for c in classes:
        bpy.utils.register_class(c)

    bpy.types.Scene.gfmodel_archive_path = StringProperty(                              
        name="GARC/CRAG Archive Path",
        default="",
        subtype="FILE_PATH",
    )
    bpy.types.Scene.gfmodel_archive_patch_payload_path = StringProperty(                              
        name="Payload Path",
        default="",
        subtype="FILE_PATH",
        description="Raw bytes to inject into the selected archive entry (bit 0 by default)",
    )
    bpy.types.Scene.gfmodel_mini_patch_payload_path = StringProperty(                              
        name="Mini Payload Path",
        default="",
        subtype="FILE_PATH",
        description="Raw bytes to inject into the selected Mini subfile",
    )
    bpy.types.Scene.gfmodel_container_patch_payload_path = StringProperty(                              
        name="Container Payload Path",
        default="",
        subtype="FILE_PATH",
        description="Raw bytes to inject into the selected CP/CM container entry",
    )
    bpy.types.Scene.gfmodel_container2_patch_payload_path = StringProperty(                              
        name="Nested Payload Path",
        default="",
        subtype="FILE_PATH",
        description="Raw bytes to inject into the selected nested CP/CM container entry",
    )
    bpy.types.Scene.gfmodel_archive_patch_output_path = StringProperty(                              
        name="Output Archive Path",
        default="",
        subtype="FILE_PATH",
        description="Where to write the patched archive (empty => '<archive>.patched')",
    )
    bpy.types.Scene.gfmodel_archive_patch_bit = IntProperty(                              
        name="Bit",
        default=0,
        min=0,
        max=31,
        description="Subentry bit to patch (0 is the primary payload in most GARCs)",
    )
    bpy.types.Scene.gfmodel_archive_patch_inplace = BoolProperty(                              
        name="In-Place",
        default=False,
        description="Replace the archive file directly (requires Backup)",
    )
    bpy.types.Scene.gfmodel_archive_patch_backup = BoolProperty(                              
        name="Backup",
        default=True,
        description="When patching in-place, rename the original to '<archive>.bak' first",
    )
    bpy.types.Scene.gfmodel_archive_entries = CollectionProperty(                              
        type=GFModelArchiveEntry
    )
    bpy.types.Scene.gfmodel_archive_selected = IntProperty(                              
        name="Selected Entry",
        default=0,
        min=0,
    )
    bpy.types.Scene.gfmodel_archive_search = StringProperty(                              
        name="Find Entry",
        default="",
        description="Filter the archive list by entry index (decimal or 0x... hex)",
    )
    bpy.types.Scene.gfmodel_mini_entries = CollectionProperty(                              
        type=GFModelMiniEntry
    )
    bpy.types.Scene.gfmodel_mini_selected = IntProperty(                              
        name="Selected Mini File",
        default=0,
        min=0,
    )
    bpy.types.Scene.gfmodel_mini_search = StringProperty(                              
        name="Find Mini",
        default="",
        description="Filter the mini list by subfile index (decimal or 0x... hex)",
    )
    bpy.types.Scene.gfmodel_mini_filter = EnumProperty(                              
        name="Mini Folder",
        items=[
            ("ALL", "All", ""),
            ("MODEL", "Model", ""),
            ("MOTION", "Motion", ""),
            ("TEXTURE", "Texture", ""),
            ("CONTAINER", "Container", ""),
            ("MINI", "Mini", ""),
            ("LZ11", "LZ11", ""),
            ("OTHER", "Other", ""),
        ],
        default="ALL",
    )
    bpy.types.Scene.gfmodel_container_entries = CollectionProperty(                              
        type=GFModelContainerEntry
    )
    bpy.types.Scene.gfmodel_container_selected = IntProperty(                              
        name="Selected Container Entry",
        default=0,
        min=0,
    )
    bpy.types.Scene.gfmodel_container_search = StringProperty(                              
        name="Find Container",
        default="",
        description="Filter the container list by entry index (decimal or 0x... hex)",
    )
    bpy.types.Scene.gfmodel_container2_entries = CollectionProperty(                              
        type=GFModelContainerEntry
    )
    bpy.types.Scene.gfmodel_container2_selected = IntProperty(                              
        name="Selected Nested Container Entry",
        default=0,
        min=0,
    )
    bpy.types.Scene.gfmodel_container2_search = StringProperty(                              
        name="Find Nested",
        default="",
        description="Filter the nested container list by entry index (decimal or 0x... hex)",
    )


def unregister() -> None:
    if hasattr(bpy.types.Scene, "gfmodel_mini_filter"):
        del bpy.types.Scene.gfmodel_mini_filter
    if hasattr(bpy.types.Scene, "gfmodel_container2_search"):
        del bpy.types.Scene.gfmodel_container2_search
    if hasattr(bpy.types.Scene, "gfmodel_container2_selected"):
        del bpy.types.Scene.gfmodel_container2_selected
    if hasattr(bpy.types.Scene, "gfmodel_container2_entries"):
        del bpy.types.Scene.gfmodel_container2_entries
    if hasattr(bpy.types.Scene, "gfmodel_container_search"):
        del bpy.types.Scene.gfmodel_container_search
    if hasattr(bpy.types.Scene, "gfmodel_container_selected"):
        del bpy.types.Scene.gfmodel_container_selected
    if hasattr(bpy.types.Scene, "gfmodel_container_entries"):
        del bpy.types.Scene.gfmodel_container_entries
    if hasattr(bpy.types.Scene, "gfmodel_mini_search"):
        del bpy.types.Scene.gfmodel_mini_search
    if hasattr(bpy.types.Scene, "gfmodel_mini_selected"):
        del bpy.types.Scene.gfmodel_mini_selected
    if hasattr(bpy.types.Scene, "gfmodel_mini_entries"):
        del bpy.types.Scene.gfmodel_mini_entries
    if hasattr(bpy.types.Scene, "gfmodel_archive_search"):
        del bpy.types.Scene.gfmodel_archive_search
    if hasattr(bpy.types.Scene, "gfmodel_archive_selected"):
        del bpy.types.Scene.gfmodel_archive_selected
    if hasattr(bpy.types.Scene, "gfmodel_archive_entries"):
        del bpy.types.Scene.gfmodel_archive_entries
    if hasattr(bpy.types.Scene, "gfmodel_archive_path"):
        del bpy.types.Scene.gfmodel_archive_path
    if hasattr(bpy.types.Scene, "gfmodel_archive_patch_backup"):
        del bpy.types.Scene.gfmodel_archive_patch_backup
    if hasattr(bpy.types.Scene, "gfmodel_archive_patch_inplace"):
        del bpy.types.Scene.gfmodel_archive_patch_inplace
    if hasattr(bpy.types.Scene, "gfmodel_archive_patch_bit"):
        del bpy.types.Scene.gfmodel_archive_patch_bit
    if hasattr(bpy.types.Scene, "gfmodel_archive_patch_output_path"):
        del bpy.types.Scene.gfmodel_archive_patch_output_path
    if hasattr(bpy.types.Scene, "gfmodel_archive_patch_payload_path"):
        del bpy.types.Scene.gfmodel_archive_patch_payload_path
    if hasattr(bpy.types.Scene, "gfmodel_mini_patch_payload_path"):
        del bpy.types.Scene.gfmodel_mini_patch_payload_path
    if hasattr(bpy.types.Scene, "gfmodel_container_patch_payload_path"):
        del bpy.types.Scene.gfmodel_container_patch_payload_path
    if hasattr(bpy.types.Scene, "gfmodel_container2_patch_payload_path"):
        del bpy.types.Scene.gfmodel_container2_patch_payload_path

    for c in reversed(classes):
        bpy.utils.unregister_class(c)
