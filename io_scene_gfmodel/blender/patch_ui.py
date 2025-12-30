"""Blender UI panels for patching (N-panel).

These panels are meant to expose patch actions without requiring users to stay inside
the Archive browser UI.
"""

from __future__ import annotations

import bpy


class VIEW3D_PT_gfmodel_patching(bpy.types.Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "GFModel"
    bl_label = "Patching"

    def draw(self, context: bpy.types.Context) -> None:
        layout = self.layout
        scene = context.scene

        bc = str(scene.get("gfmodel_last_import_breadcrumb", "")).strip()
        if bc:
            layout.label(text=f"Source: {bc}")

        has_plan = bool(str(scene.get("gfmodel_patch_plan_json", "")).strip())
        has_legacy = (
            str(scene.get("gfmodel_patch_source_level", "")).strip() == "container2"
        )
        if not (has_plan or has_legacy):
            layout.label(text="(Import via GFModel Archive to enable patch-back)")

        obj = context.active_object
        if (
            obj is None
            or obj.type != "MESH"
            or not str(obj.get("gfmodel_model_name", "") or "").strip()
        ):
            layout.label(text="Select an imported GFModel mesh object")

        col = layout.column(align=True)
        col.operator(
            "gfmodel.patch_current_scene_grow_buffers_tris", text="Patch (Grow Buffers)"
        )
        col.operator(
            "gfmodel.patch_current_scene_rebuild_active_submesh",
            text="Patch (Rebuild Active Submesh)",
        )


_CLASSES = (VIEW3D_PT_gfmodel_patching,)


def register() -> None:
    for c in _CLASSES:
        bpy.utils.register_class(c)


def unregister() -> None:
    for c in reversed(_CLASSES):
        bpy.utils.unregister_class(c)
