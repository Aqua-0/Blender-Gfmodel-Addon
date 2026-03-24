

from __future__ import annotations

import bpy
from bpy.props import BoolProperty


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




        try:
            if obj is not None and obj.type == 'MESH':
                vg_count = int(len(getattr(obj, 'vertex_groups', []) or []))
                used_groups = None
                try:
                    me = getattr(obj, 'data', None)
                    vs = getattr(me, 'vertices', None)
                    if vs is not None and len(vs) <= 250000:
                        used = set()
                        for v in vs:
                            for g in getattr(v, 'groups', []) or []:
                                if float(getattr(g, 'weight', 0.0) or 0.0) > 0.0:
                                    used.add(int(getattr(g, 'group', -1)))
                        used_groups = int(len(used))
                except Exception:
                    used_groups = None

                if used_groups is None:
                    layout.label(text=f'Submesh palette (vertex groups): {vg_count}')
                else:
                    layout.label(text=f'Submesh palette (vertex groups): {vg_count} (used: {used_groups})')

        except Exception:
            pass

        col = layout.column(align=True)
        col.prop(scene, 'gfmodel_patch_palette_append_only', text='Append bone palette (no rebuild)')
        col.prop(scene, 'gfmodel_patch_prune_unused_palette', text='Prune unused bones from palette')
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
    bpy.types.Scene.gfmodel_patch_palette_append_only = BoolProperty(
        name='Append Bone Palette (No Rebuild)',
        default=False,
        description='When using Robust+Auto-Route, preserve existing palette order and only append newly-used bones (if space allows)',
    )
    bpy.types.Scene.gfmodel_patch_prune_unused_palette = BoolProperty(
        name='Prune Unused Bones From Palette',
        default=False,
        description='When rewriting palettes, remove bones that are not referenced by any vertex weights in the patched submesh',
    )
    for c in _CLASSES:
        bpy.utils.register_class(c)


def unregister() -> None:
    try:
        del bpy.types.Scene.gfmodel_patch_palette_append_only
        del bpy.types.Scene.gfmodel_patch_prune_unused_palette
    except Exception:
        pass
    for c in reversed(_CLASSES):
        bpy.utils.unregister_class(c)
