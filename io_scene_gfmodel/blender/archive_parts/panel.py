
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

from .utils import _breadcrumb, _looks_like_a094_archive_path

_GARC_FILE_CACHE: dict[str, object] = {}

def _looks_like_a094_context(context: bpy.types.Context) -> bool:
    try:
        ap = str(getattr(context.scene, 'gfmodel_archive_path', '') or '')
    except Exception:
        ap = ''
    return _looks_like_a094_archive_path(ap)


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
        layout.prop(context.scene, "gfmodel_archive_auto_resolve_a094_group", text="Auto a094 textures/anims")
        if bool(getattr(context.scene, 'gfmodel_archive_auto_resolve_a094_group', False)) and _looks_like_a094_archive_path(str(getattr(context.scene, 'gfmodel_archive_path', ''))):
            row = layout.row(align=True)
            row.prop(context.scene, 'gfmodel_a094_motion_pack', text='a094 motions')
            row.prop(context.scene, 'gfmodel_a094_name_motions', text='Name')
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

        row = layout.row(align=True)
        row.prop(context.scene, "gfmodel_archive_search", text="Find Entry")
        row.prop(context.scene, "gfmodel_archive_show_species_names", text="Names")
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

