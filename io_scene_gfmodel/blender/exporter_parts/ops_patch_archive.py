"""Archive patch operators."""

from __future__ import annotations

import json

import bpy



class GFModel_OT_patch_current_scene_to_archive(bpy.types.Operator):
    bl_idname = "gfmodel.patch_current_scene_to_archive"
    bl_label = "GFModel: Patch Current Scene To Source Archive"
    bl_options = {"UNDO"}

    def execute(self, context: bpy.types.Context):
        import os
        import tempfile
        import time

                                                                                            
        tmp_root = ""
        try:
            tmp_root = str(getattr(bpy.app, "tempdir", "") or "").strip()
        except Exception:
            tmp_root = ""
        if not tmp_root:
            tmp_root = tempfile.gettempdir()
        os.makedirs(tmp_root, exist_ok=True)
        tmp_path = os.path.join(
            tmp_root, f"gfmodel_export_{int(time.time() * 1000)}.bin"
        )

                                                                                    
        kwargs = {
            "filepath": tmp_path,
            "patch_into_source_archive": True,
            "remember_last_export_settings": False,
        }
        try:
            if "gfmodel_last_export_export_meshes" in context.scene:
                kwargs["export_meshes"] = bool(
                    context.scene.get("gfmodel_last_export_export_meshes", True)
                )
            if "gfmodel_last_export_mesh_export_mode" in context.scene:
                kwargs["mesh_export_mode"] = str(
                    context.scene.get(
                        "gfmodel_last_export_mesh_export_mode", "REBUILD_FROM_SCENE"
                    )
                )
            if "gfmodel_last_export_export_material_textures" in context.scene:
                kwargs["export_material_textures"] = bool(
                    context.scene.get(
                        "gfmodel_last_export_export_material_textures", True
                    )
                )
            if "gfmodel_last_export_material_export_mode" in context.scene:
                kwargs["material_export_mode"] = str(
                    context.scene.get(
                        "gfmodel_last_export_material_export_mode", "PRESERVE_RAW"
                    )
                )
            if "gfmodel_last_export_export_textures" in context.scene:
                kwargs["export_textures"] = bool(
                    context.scene.get("gfmodel_last_export_export_textures", False)
                )
            if "gfmodel_last_export_texture_max_size" in context.scene:
                kwargs["texture_max_size"] = int(
                    context.scene.get("gfmodel_last_export_texture_max_size", 256)
                )
            if "gfmodel_last_export_texture_mode" in context.scene:
                kwargs["texture_mode"] = str(
                    context.scene.get("gfmodel_last_export_texture_mode", "KEEP")
                )
            if "gfmodel_last_export_output_container" in context.scene:
                kwargs["output_container"] = str(
                    context.scene.get("gfmodel_last_export_output_container", "AUTO")
                )
        except Exception:
            pass

        try:
            res = bpy.ops.export_scene.gfmodel("EXEC_DEFAULT", **kwargs)
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

        if "FINISHED" not in set(res):
            self.report({"ERROR"}, "Patch failed (export operator did not finish)")
            return {"CANCELLED"}
        used_mode = str(kwargs.get("mesh_export_mode", "REBUILD_FROM_SCENE"))
        self.report(
            {"INFO"},
            f"Patched current scene into archive (mesh_export_mode={used_mode})",
        )
        return {"FINISHED"}


class GFModel_OT_patch_imported_bytes_to_archive(bpy.types.Operator):
    bl_idname = "gfmodel.patch_imported_bytes_to_archive"
    bl_label = "GFModel: Patch Imported Bytes To Source Archive (No-op)"
    bl_options = {"UNDO"}

    def execute(self, context: bpy.types.Context):
        import os
        import tempfile
        import time

        tmp_root = ""
        try:
            tmp_root = str(getattr(bpy.app, "tempdir", "") or "").strip()
        except Exception:
            tmp_root = ""
        if not tmp_root:
            tmp_root = tempfile.gettempdir()
        os.makedirs(tmp_root, exist_ok=True)
        tmp_path = os.path.join(
            tmp_root, f"gfmodel_export_{int(time.time() * 1000)}.bin"
        )

        kwargs = {
            "filepath": tmp_path,
            "patch_into_source_archive": True,
            "export_meshes": True,
            "mesh_export_mode": "PRESERVE_SOURCE_BYTES",
            "remember_last_export_settings": False,
        }

        try:
            res = bpy.ops.export_scene.gfmodel("EXEC_DEFAULT", **kwargs)
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

        if "FINISHED" not in set(res):
            self.report({"ERROR"}, "Patch failed (export operator did not finish)")
            return {"CANCELLED"}
        self.report({"INFO"}, "Patched imported bytes into archive (no-op)")
        return {"FINISHED"}
