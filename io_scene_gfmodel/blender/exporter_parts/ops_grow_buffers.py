"""Grow Buffers and experimental rebuild operators."""

from __future__ import annotations

import json

import bpy
from bpy.props import (
    BoolProperty,
    EnumProperty,
    FloatProperty,
    IntProperty,
    StringProperty,
)


class GFModel_OT_patch_current_scene_grow_buffers_tris(bpy.types.Operator):
    bl_idname = "gfmodel.patch_current_scene_grow_buffers_tris"
    bl_label = "GFModel: Patch Current Scene (Grow Buffers, Tris)"
    bl_options = {"UNDO"}

    export_textures: BoolProperty(
        name="Export Texture Data (Overwrite Slots)",
        default=False,
        description="If enabled, overwrites existing GFModelPack texture slots (does not add new slots)",
    )
    texture_mode: EnumProperty(
        name="Texture Mode",
        items=[
            ("KEEP", "Keep Original", "Do not overwrite texture data"),
            (
                "ORIGINAL_FORMAT",
                "Original Format",
                "Encode to the original texture format+size from the file (errors if encoder not implemented for that fmt)",
            ),
            ("RGBA8", "RGBA8", "Write RGBA8 swizzled (no ETC encoding)"),
            (
                "RGBA8_SAME_SIZE",
                "RGBA8 (Same Size)",
                "Write RGBA8 swizzled; requires image size match original texture",
            ),
            (
                "RGBA8_ORIGINAL_SIZE",
                "RGBA8 (Auto-Resize to Original)",
                "Write RGBA8 swizzled; auto-resizes images to the original texture size",
            ),
            (
                "OVERRIDE_FORMAT",
                "Override Format (Original Size)",
                "Encode to a selected format using the original file's texture size",
            ),
        ],
        default="KEEP",
    )
    texture_override_format: EnumProperty(
        name="Texture Override Format",
        items=[
            ("RGBA8", "RGBA8", "PICA RGBA8 (GF fmt=0x4)"),
            ("RGB8", "RGB8", "PICA RGB8 (GF fmt=0x3)"),
            ("RGB565", "RGB565", "PICA RGB565 (GF fmt=0x2)"),
            ("RGBA4", "RGBA4", "PICA RGBA4 (GF fmt=0x16)"),
            ("RGBA5551", "RGBA5551", "PICA RGBA5551 (GF fmt=0x17)"),
            ("LA8", "LA8", "PICA LA8 (GF fmt=0x23)"),
            ("L8", "L8", "PICA L8 (GF fmt=0x25)"),
            ("A8", "A8", "PICA A8 (GF fmt=0x26)"),
            ("LA4", "LA4", "PICA LA4 (GF fmt=0x27)"),
            ("L4", "L4", "PICA L4 (GF fmt=0x28)"),
            ("A4", "A4", "PICA A4 (GF fmt=0x29)"),
            ("ETC1", "ETC1", "PICA ETC1 (GF fmt=0x2A)"),
            ("ETC1A4", "ETC1A4", "PICA ETC1A4 (GF fmt=0x2B)"),
        ],
        default="RGBA8",
    )
    texture_max_size: IntProperty(
        name="Texture Max Size",
        default=256,
        min=8,
        description="Only used in Texture Mode=RGBA8 (downscales to fit; rounded to multiple of 8)",
    )
    routing_strategy: EnumProperty(
        name="Routing Strategy",
        items=[
            (
                "MOST_SPECIFIC",
                "Most Specific Palette",
                "Prefer the smallest matching bone palette",
            ),
            (
                "ORIGINAL_ORDER",
                "Original Slot Order",
                "Use the first matching submesh slot in file order",
            ),
            ("BALANCE", "Balance", "Spread triangles across matching slots"),
        ],
        default="MOST_SPECIFIC",
    )
    weight_cutoff: FloatProperty(
        name="Weight Cutoff",
        default=0.01,
        min=0.0,
        max=1.0,
        description="Ignore weights below this value when routing/clamping triangles to palettes (helps remove stray tiny influences)",
    )
    uv_strategy: EnumProperty(
        name="UV Strategy",
        items=[
            (
                "DUPLICATE",
                "Duplicate Verts (Accurate)",
                "Preserve UV seams by duplicating vertices where needed (GPU-style; most accurate)",
            ),
            (
                "SMEAR",
                "Smear (Low Verts)",
                "Force one UV per 3D vertex (collapses seams; can smear textures; reduces vertex duplication)",
            ),
            (
                "STITCH_TRANSLATE",
                "Stitch Translate (Try)",
                "Try to remove seams by translating UV islands to align across seams (translation-only heuristic)",
            ),
        ],
        default="DUPLICATE",
        description="Controls how UV seams are handled when rebuilding vertex/index buffers",
    )
    rebuild_mode: EnumProperty(
        name="Rebuild Mode",
        items=[
            (
                "CLAMP_ROUTE",
                "Clamp/Route (No Rebuild)",
                "Keep original submesh structure; route/clamp triangles to existing bone palettes",
            ),
            (
                "REBUILD_PALETTE",
                "Rebuild Palette Only",
                "Rebuild each target slot's bone palette from its triangles (max 31); does not split/add face records",
            ),
            (
                "REBUILD_SPLIT",
                "Rebuild + Split",
                "Rebuild palettes and split/add face records as needed (may change submesh structure)",
            ),
        ],
        default="CLAMP_ROUTE",
        description="Controls whether Grow Buffers can rebuild palettes and/or split/add face records",
    )
    clamp_conflict_mode: EnumProperty(
        name="Clamp/Route Conflicts",
        items=[
            (
                "CLAMP_BY_WEIGHT",
                "Clamp (By Weight)",
                "When a triangle spans palettes, pick the best slot by weight and trim weights to that palette (can cause cracks)",
            ),
            (
                "CLAMP_BY_NEIGHBORS",
                "Clamp (Prefer Neighbors)",
                "When a triangle spans palettes, prefer the same slot as adjacent triangles to reduce visible cracks, then trim weights",
            ),
            (
                "DROP_CONFLICTS",
                "Drop Conflicts",
                "Drop triangles that don't fit any single existing palette (avoids swaying cracks, but creates holes)",
            ),
        ],
        default="CLAMP_BY_WEIGHT",
        description="What to do when a triangle uses bones that don't co-occur in any existing submesh palette (CLAMP_ROUTE only)",
    )

    def invoke(self, context: bpy.types.Context, event):                          
        try:
            self.export_textures = bool(
                context.scene.get("gfmodel_last_export_export_textures", False)
            )
            self.texture_mode = str(
                context.scene.get("gfmodel_last_export_texture_mode", "KEEP")
            )
            self.texture_max_size = int(
                context.scene.get("gfmodel_last_export_texture_max_size", 256)
            )
            self.texture_override_format = str(
                context.scene.get(
                    "gfmodel_last_export_texture_override_format", "RGBA8"
                )
            )
        except Exception:
            pass
                                                                                
        try:
            ao = context.active_object
            model_name = ""
            if ao is not None and ao.type == "MESH":
                model_name = str(ao.get("gfmodel_model_name", "") or "")
            if not model_name:
                raise ValueError(
                    "Select an imported GFModel mesh object before running Grow Buffers"
                )
            mats: List[str] = []
            seen = set()
            for o in bpy.data.objects:
                if o.type != "MESH":
                    continue
                if str(o.get("gfmodel_model_name", "") or "") != model_name:
                    continue
                mn = str(o.get("gfmodel_material_name", "") or "").strip()
                if not mn or mn in seen:
                    continue
                mats.append(mn)
                seen.add(mn)
            mats = sorted(mats)

            stored = str(
                context.scene.get("gfmodel_grow_buffers_material_sources_json", "")
                or ""
            ).strip()
            stored_map: Dict[str, str] = {}
            if stored:
                try:
                    v = json.loads(stored)
                    if isinstance(v, dict):
                        stored_map = {str(k): str(v[k]) for k in v.keys()}
                except Exception:
                    stored_map = {}

            coll = getattr(context.scene, "gfmodel_grow_buffers_material_sources", None)
            if coll is not None:
                coll.clear()
                for mn in mats:
                    it = coll.add()
                    it.material_name = str(mn)
                    on = stored_map.get(str(mn), "")
                    if on:
                        it.source_object = bpy.data.objects.get(str(on))
        except Exception:
                                                                                      
            pass
        try:
            self.routing_strategy = str(
                context.scene.get(
                    "gfmodel_grow_buffers_routing_strategy", "MOST_SPECIFIC"
                )
            )
        except Exception:
            pass
        try:
            self.weight_cutoff = float(
                context.scene.get("gfmodel_grow_buffers_weight_cutoff", 0.01)
            )
        except Exception:
            pass
        try:
            self.uv_strategy = str(
                context.scene.get("gfmodel_grow_buffers_uv_strategy", "DUPLICATE")
            )
        except Exception:
            pass
        try:
            mode = str(
                context.scene.get("gfmodel_grow_buffers_rebuild_mode", "") or ""
            ).strip()
            if mode:
                self.rebuild_mode = mode
            else:
                legacy = bool(
                    context.scene.get("gfmodel_grow_buffers_rebuild_submeshes", False)
                )
                self.rebuild_mode = "REBUILD_SPLIT" if legacy else "CLAMP_ROUTE"
        except Exception:
            pass
        try:
            self.clamp_conflict_mode = str(
                context.scene.get(
                    "gfmodel_grow_buffers_clamp_conflict_mode", "CLAMP_BY_WEIGHT"
                )
            )
        except Exception:
            pass
        return context.window_manager.invoke_props_dialog(self, width=420)

    def draw(self, context: bpy.types.Context):                          
        layout = self.layout
        layout.prop(self, "export_textures")
        col = layout.column()
        col.enabled = bool(self.export_textures)
        col.prop(self, "texture_mode")
        if str(self.texture_mode) == "OVERRIDE_FORMAT":
            col.prop(self, "texture_override_format")
        if str(self.texture_mode) == "RGBA8":
            col.prop(self, "texture_max_size")
        layout.separator()
        layout.prop(self, "routing_strategy")
        layout.prop(self, "weight_cutoff")
        layout.prop(self, "uv_strategy")
        layout.prop(self, "rebuild_mode")
        if str(self.rebuild_mode) == "CLAMP_ROUTE":
            layout.prop(self, "clamp_conflict_mode")
        coll = getattr(context.scene, "gfmodel_grow_buffers_material_sources", None)
        if coll:
            layout.label(text="Replacement Sources (optional; per material):")
            box = layout.box()
            for it in coll:
                row = box.row(align=True)
                row.label(text=str(it.material_name))
                row.prop(it, "source_object", text="")

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
            "mesh_export_mode": "GROW_BUFFERS_TRIS",
            "export_textures": bool(self.export_textures),
            "texture_mode": str(self.texture_mode),
            "texture_override_format": str(self.texture_override_format),
            "texture_max_size": int(self.texture_max_size),
            "grow_buffers_routing_strategy": str(self.routing_strategy),
            "grow_buffers_weight_cutoff": float(self.weight_cutoff),
            "grow_buffers_uv_strategy": str(self.uv_strategy),
            "grow_buffers_rebuild_mode": str(self.rebuild_mode),
            "grow_buffers_clamp_conflict_mode": str(self.clamp_conflict_mode),
        }
        mat_map: Dict[str, str] = {}
        coll = getattr(context.scene, "gfmodel_grow_buffers_material_sources", None)
        if coll:
            for it in coll:
                if not it.material_name:
                    continue
                obj = it.source_object
                if obj is None or getattr(obj, "type", "") != "MESH":
                    continue
                mat_map[str(it.material_name)] = str(obj.name)
        if mat_map:
            kwargs["grow_buffers_material_sources_json"] = json.dumps(mat_map)
            try:
                context.scene["gfmodel_grow_buffers_material_sources_json"] = (
                    json.dumps(mat_map)
                )
            except Exception:
                pass
                                                        
        try:
            context.scene["gfmodel_grow_buffers_routing_strategy"] = str(
                self.routing_strategy
            )
        except Exception:
            pass
        try:
            context.scene["gfmodel_grow_buffers_weight_cutoff"] = float(
                self.weight_cutoff
            )
        except Exception:
            pass
        try:
            context.scene["gfmodel_grow_buffers_uv_strategy"] = str(self.uv_strategy)
        except Exception:
            pass
        try:
            context.scene["gfmodel_grow_buffers_rebuild_mode"] = str(self.rebuild_mode)
        except Exception:
            pass
        try:
            context.scene["gfmodel_grow_buffers_clamp_conflict_mode"] = str(
                self.clamp_conflict_mode
            )
        except Exception:
            pass
        try:
            context.scene["gfmodel_last_export_texture_override_format"] = str(
                self.texture_override_format
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
        self.report({"INFO"}, "Patched current scene (grow buffers, triangles)")
        return {"FINISHED"}


class GFModel_OT_patch_current_scene_rebuild_active_submesh(bpy.types.Operator):
    bl_idname = "gfmodel.patch_current_scene_rebuild_active_submesh"
    bl_label = "GFModel: Patch Current Scene (Rebuild Active Submesh)"
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
            "mesh_export_mode": "REBUILD_ACTIVE_SUBMESH",
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
        self.report({"INFO"}, "Patched current scene (rebuild active submesh)")
        return {"FINISHED"}
