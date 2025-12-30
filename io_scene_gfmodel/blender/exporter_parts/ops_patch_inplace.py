"""In-place patch operators."""

from __future__ import annotations

import bpy

class GFModel_OT_patch_current_scene_positions_in_place(bpy.types.Operator):
    bl_idname = "gfmodel.patch_current_scene_positions_in_place"
    bl_label = "GFModel: Patch Current Scene (Positions In-Place)"
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
            "export_meshes": True,
            "mesh_export_mode": "UPDATE_POSITIONS_IN_PLACE",
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
        self.report({"INFO"}, "Patched current scene (positions in-place)")
        return {"FINISHED"}


class GFModel_OT_patch_current_scene_normals_in_place(bpy.types.Operator):
    bl_idname = "gfmodel.patch_current_scene_normals_in_place"
    bl_label = "GFModel: Patch Current Scene (Normals In-Place)"
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
            "export_meshes": True,
            "mesh_export_mode": "UPDATE_NORMALS_IN_PLACE",
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
        self.report({"INFO"}, "Patched current scene (normals in-place)")
        return {"FINISHED"}


class GFModel_OT_patch_current_scene_pos_nrm_in_place(bpy.types.Operator):
    bl_idname = "gfmodel.patch_current_scene_pos_nrm_in_place"
    bl_label = "GFModel: Patch Current Scene (Pos+Nrm In-Place)"
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
            "export_meshes": True,
            "mesh_export_mode": "UPDATE_POS_NRM_IN_PLACE",
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
        self.report({"INFO"}, "Patched current scene (pos+normals in-place)")
        return {"FINISHED"}


class GFModel_OT_patch_current_scene_uv0_in_place(bpy.types.Operator):
    bl_idname = "gfmodel.patch_current_scene_uv0_in_place"
    bl_label = "GFModel: Patch Current Scene (UV0 In-Place)"
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
            "export_meshes": True,
            "mesh_export_mode": "UPDATE_UV0_IN_PLACE",
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
        self.report({"INFO"}, "Patched current scene (UV0 in-place)")
        return {"FINISHED"}


class GFModel_OT_patch_current_scene_pos_nrm_uv0_in_place(bpy.types.Operator):
    bl_idname = "gfmodel.patch_current_scene_pos_nrm_uv0_in_place"
    bl_label = "GFModel: Patch Current Scene (Pos+Nrm+UV0 In-Place)"
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
            "export_meshes": True,
            "mesh_export_mode": "UPDATE_POS_NRM_UV0_IN_PLACE",
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
        self.report({"INFO"}, "Patched current scene (pos+nrm+uv0 in-place)")
        return {"FINISHED"}


class GFModel_OT_patch_current_scene_pos_nrm_uv0_skin_in_place(bpy.types.Operator):
    bl_idname = "gfmodel.patch_current_scene_pos_nrm_uv0_skin_in_place"
    bl_label = "GFModel: Patch Current Scene (Pos+Nrm+UV0+Skin In-Place)"
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
            "export_meshes": True,
            "mesh_export_mode": "UPDATE_POS_NRM_UV0_SKIN_IN_PLACE",
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
        self.report({"INFO"}, "Patched current scene (pos+nrm+uv0+skin in-place)")
        return {"FINISHED"}


class GFModel_OT_patch_current_scene_indices_in_place(bpy.types.Operator):
    bl_idname = "gfmodel.patch_current_scene_indices_in_place"
    bl_label = "GFModel: Patch Current Scene (Indices In-Place)"
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
            "export_meshes": True,
            "mesh_export_mode": "UPDATE_INDICES_IN_PLACE",
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
        self.report({"INFO"}, "Patched current scene (indices in-place)")
        return {"FINISHED"}


class GFModel_OT_patch_current_scene_topology_tris_in_place(bpy.types.Operator):
    bl_idname = "gfmodel.patch_current_scene_topology_tris_in_place"
    bl_label = "GFModel: Patch Current Scene (Topology Tris In-Place)"
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
            "export_meshes": True,
            "mesh_export_mode": "UPDATE_TOPOLOGY_TRIS_IN_PLACE",
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
        self.report({"INFO"}, "Patched current scene (topology tris in-place)")
        return {"FINISHED"}


class GFModel_OT_patch_current_scene_verts_topology_tris_in_place(bpy.types.Operator):
    bl_idname = "gfmodel.patch_current_scene_verts_topology_tris_in_place"
    bl_label = "GFModel: Patch Current Scene (Verts+Topo Tris In-Place)"
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
            "export_meshes": True,
            "mesh_export_mode": "UPDATE_VERTS_TOPOLOGY_TRIS_IN_PLACE",
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
        self.report({"INFO"}, "Patched current scene (verts+topo tris in-place)")
        return {"FINISHED"}



