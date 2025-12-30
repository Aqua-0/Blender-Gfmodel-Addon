"""Blender export/patch entrypoint for GFModel.

Implementation is split across `blender/exporter_parts/` to keep individual files manageable.
"""

from __future__ import annotations

import bpy
from bpy.props import CollectionProperty

from .exporter_parts.ops_export import EXPORT_SCENE_OT_gfmodel
from .exporter_parts.ops_grow_buffers import (
    GFModel_OT_patch_current_scene_grow_buffers_tris,
    GFModel_OT_patch_current_scene_rebuild_active_submesh,
)
from .exporter_parts.ops_patch_archive import (
    GFModel_OT_patch_current_scene_to_archive,
    GFModel_OT_patch_imported_bytes_to_archive,
)
from .exporter_parts.ops_patch_inplace import (
    GFModel_OT_patch_current_scene_indices_in_place,
    GFModel_OT_patch_current_scene_normals_in_place,
    GFModel_OT_patch_current_scene_pos_nrm_in_place,
    GFModel_OT_patch_current_scene_pos_nrm_uv0_in_place,
    GFModel_OT_patch_current_scene_pos_nrm_uv0_skin_in_place,
    GFModel_OT_patch_current_scene_positions_in_place,
    GFModel_OT_patch_current_scene_topology_tris_in_place,
    GFModel_OT_patch_current_scene_uv0_in_place,
    GFModel_OT_patch_current_scene_verts_topology_tris_in_place,
)
from .exporter_parts.props import GFModel_GrowBuffersMaterialSource


def menu_func_export(self, context):
    self.layout.operator(
        EXPORT_SCENE_OT_gfmodel.bl_idname, text="GFModel (GFL2) (.bin/CP/CM)"
    )


_CLASSES = (
    GFModel_GrowBuffersMaterialSource,
    EXPORT_SCENE_OT_gfmodel,
    GFModel_OT_patch_current_scene_to_archive,
    GFModel_OT_patch_imported_bytes_to_archive,
    GFModel_OT_patch_current_scene_positions_in_place,
    GFModel_OT_patch_current_scene_normals_in_place,
    GFModel_OT_patch_current_scene_pos_nrm_in_place,
    GFModel_OT_patch_current_scene_uv0_in_place,
    GFModel_OT_patch_current_scene_pos_nrm_uv0_in_place,
    GFModel_OT_patch_current_scene_pos_nrm_uv0_skin_in_place,
    GFModel_OT_patch_current_scene_indices_in_place,
    GFModel_OT_patch_current_scene_topology_tris_in_place,
    GFModel_OT_patch_current_scene_verts_topology_tris_in_place,
    GFModel_OT_patch_current_scene_grow_buffers_tris,
    GFModel_OT_patch_current_scene_rebuild_active_submesh,
)


def register() -> None:
    for c in _CLASSES:
        bpy.utils.register_class(c)
    bpy.types.Scene.gfmodel_grow_buffers_material_sources = CollectionProperty(
        type=GFModel_GrowBuffersMaterialSource
    )
    bpy.types.TOPBAR_MT_file_export.append(menu_func_export)


def unregister() -> None:
    try:
        bpy.types.TOPBAR_MT_file_export.remove(menu_func_export)
    except Exception:
        pass
    try:
        del bpy.types.Scene.gfmodel_grow_buffers_material_sources
    except Exception:
        pass
    for c in reversed(_CLASSES):
        bpy.utils.unregister_class(c)
