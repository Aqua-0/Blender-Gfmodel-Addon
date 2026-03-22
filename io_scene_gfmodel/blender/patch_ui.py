
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

        coll = None
        try:
            obj = context.active_object
            if obj is not None:
                name = str(obj.get("gfmodel_import_collection", "") or "").strip()
                if name:
                    coll = bpy.data.collections.get(name)
        except Exception:
            coll = None
        if coll is None:
            try:
                name = str(scene.get("gfmodel_last_import_collection", "") or "").strip()
                if name:
                    coll = bpy.data.collections.get(name)
            except Exception:
                coll = None

        bc = ""
        if coll is not None:
            try:
                bc = str(coll.get("gfmodel_last_import_breadcrumb", "") or "").strip()
            except Exception:
                bc = ""
        if not bc:
            bc = str(scene.get("gfmodel_last_import_breadcrumb", "") or "").strip()
        if bc:
            layout.label(text=f"Source: {bc}")

        has_plan = False
        if coll is not None:
            try:
                has_plan = bool(str(coll.get("gfmodel_patch_plan_json", "") or "").strip())
            except Exception:
                has_plan = False
        if not has_plan:
            has_plan = bool(str(scene.get("gfmodel_patch_plan_json", "") or "").strip())
        has_legacy = (
            str(scene.get("gfmodel_patch_source_level", "") or "").strip() == "container2"
        )
        if not (has_plan or has_legacy):
            layout.label(text="(Import via GFModel Archive to enable patch-back)")

        obj = context.active_object
        if (
            obj is None
            or obj.type not in {"MESH", "ARMATURE"}
            or not str(obj.get("gfmodel_import_collection", "") or "").strip()
        ):
            layout.label(text="Select an imported GFModel mesh/armature")

        col = layout.column(align=True)
        col.operator(
            "gfmodel.patch_current_scene_grow_buffers_tris", text="Patch (Grow Buffers)"
        )
        col.operator(
            "gfmodel.patch_current_scene_grow_buffers_robust_autoroute",
            text="Patch (Robust + Auto Route New Meshes)",
        )
        col.operator(
            "gfmodel.patch_current_scene_rebuild_active_submesh",
            text="Patch (Rebuild Active Submesh)",
        )
        op = col.operator(
            "gfmodel.patch_current_scene_export_textures",
            text="Export Texture Data (Overwrite Slots)",
        )
        try:
            op.texture_mode = str(
                scene.get("gfmodel_last_export_texture_mode", "RGBA8_ORIGINAL_SIZE")
                or "RGBA8_ORIGINAL_SIZE"
            )
            op.texture_max_size = int(
                scene.get("gfmodel_last_export_texture_max_size", 256) or 256
            )
            op.texture_override_format = str(
                scene.get("gfmodel_last_export_texture_override_format", "RGBA8")
                or "RGBA8"
            )
        except Exception:
            pass


_CLASSES = (VIEW3D_PT_gfmodel_patching,)


def register() -> None:
    for c in _CLASSES:
        bpy.utils.register_class(c)


def unregister() -> None:
    for c in reversed(_CLASSES):
        bpy.utils.unregister_class(c)
