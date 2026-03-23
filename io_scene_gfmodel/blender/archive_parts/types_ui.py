
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


from .utils import _looks_like_a094_archive_path, _parse_search_int

_GARC_FILE_CACHE: dict[str, object] = {}

def _looks_like_a094_context(context: bpy.types.Context) -> bool:
    try:
        ap = str(getattr(context.scene, 'gfmodel_archive_path', '') or '')
    except Exception:
        ap = ''
    return _looks_like_a094_archive_path(ap)


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
        show_names = bool(getattr(context.scene, "gfmodel_archive_show_species_names", False)) or _looks_like_a094_context(context)
        if show_names:
            row.label(text=_a094_species_label_for_entry(context, int(getattr(e, "index", 0))))
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
        query = str(getattr(context.scene, "gfmodel_archive_search", "") or "").strip()
        ql = query.lower()

        want = _parse_search_int(query) if query else None
                                        
        use_names = bool(getattr(context.scene, "gfmodel_archive_show_species_names", False)) or _looks_like_a094_context(context)

        for it in items:
            show = True
            idx = int(getattr(it, "index", -1))
            if want is not None:
                show = idx == int(want)
            elif query:
                if use_names:
                    blob = _a094_search_blob_for_entry(idx)
                    show = ql in blob
                else:
                    s = str(idx)
                    show = query in s or query in f"{idx:04d}"
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


