

from __future__ import annotations

import json
import os
import struct
from typing import Dict, List, Optional, Sequence, Tuple

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, StringProperty
from bpy_extras.io_utils import ImportHelper, axis_conversion
from mathutils import Matrix, Quaternion, Vector

from ...core.io import _load_any
from ...core.patch_plan import PatchPlan, steps_to_breadcrumb
from ...core.pica import (
    _bgra_to_rgba_floats,
    _decode_rgba_u32,
    _decode_texenv_update_buffer,
    _flip_bgra_y,
    _pica_decode_bitmap_to_bgra,
)
from ...core.types import (
    _GFMaterial,
    _GFModel,
    _GFMotion,
    _GFShader,
    _GFSubMesh,
    _GFTexture,
)
from ..anim import (
    _apply_uv_anim_enable,
    _apply_visibility_anim_enable,
    _compute_rest_world_mats,
    _euler_to_quat_xyz,
    _gf_runtime_cache_armature,
    _mot_eval,
    _transform_quat_basis,
)


class IMPORT_SCENE_OT_gfmodel(bpy.types.Operator, ImportHelper):
    bl_idname = "import_scene.gfmodel"
    bl_label = "Import GFModel (GFL2)"
    bl_options = {"UNDO"}

    filename_ext = ""

    filter_glob: StringProperty(default="*", options={"HIDDEN"})

    import_textures: BoolProperty(name="Import Textures", default=True)
    import_animations: BoolProperty(name="Import Skeletal Animations", default=True)
    import_material_animations: BoolProperty(
        name="Import Material (UV) Animations",
        default=True,
        description="Bake UV/material animations into muted NLA strips on materials (toggle in GFModel panel)",
    )
    import_visibility_animations: BoolProperty(
        name="Import Visibility Animations",
        default=True,
        description="Bake mesh visibility tracks into muted per-object actions (toggle in GFModel panel)",
    )
    debug_animations: BoolProperty(
        name="Debug Animations (Console)",
        default=False,
        description="Print extra info about bone transforms during import",
    )
    debug_bone: StringProperty(
        name="Debug Bone",
        default="Waist",
        description="Bone name to print (when Debug Animations is enabled)",
    )
    debug_motion: bpy.props.IntProperty(
        name="Debug Motion Index",
        default=-1,
        description="If >= 0, only print debug for this motion index",
    )
    set_active_action: BoolProperty(
        name="Set Active Action",
        default=False,
        description="Assign a selected skeletal action to the armature after import",
    )
    active_action_index: bpy.props.IntProperty(
        name="Active Motion Index",
        default=0,
        min=0,
        description="Motion index to set active when Set Active Action is enabled",
    )
    global_scale: FloatProperty(name="Scale", default=1.0, min=0.0001, max=1000.0)

    axis_forward: EnumProperty(
        name="Forward",
        items=[(a, a, "") for a in ("X", "Y", "Z", "-X", "-Y", "-Z")],
        default="-Z",
    )
    axis_up: EnumProperty(
        name="Up",
        items=[(a, a, "") for a in ("X", "Y", "Z", "-X", "-Y", "-Z")],
        default="Y",
    )

    def execute(self, context: bpy.types.Context):

        prev_dbg = bool(getattr(context.scene, "gfmodel_debug_animations", False))
        prev_dbg_bone = str(getattr(context.scene, "gfmodel_debug_bone", "Waist"))
        prev_dbg_motion = int(getattr(context.scene, "gfmodel_debug_motion", -1))
        if self.debug_animations:
            context.scene.gfmodel_debug_animations = True
            context.scene.gfmodel_debug_bone = self.debug_bone
            context.scene.gfmodel_debug_motion = self.debug_motion
        with open(self.filepath, "rb") as f:
            data = f.read()
        ok = _import_gfmodel_bytes(
            context,
            data,
            source_path=str(self.filepath),
            import_textures=bool(self.import_textures),
            import_animations=bool(self.import_animations),
            import_material_animations=bool(self.import_material_animations),
            import_visibility_animations=bool(self.import_visibility_animations),
            global_scale=float(self.global_scale),
            axis_forward=str(self.axis_forward),
            axis_up=str(self.axis_up),
        )
        if not ok:
            self.report({"ERROR"}, "No GFModel content found in file")
            return {"CANCELLED"}
        if self.debug_animations:
            context.scene.gfmodel_debug_animations = prev_dbg
            context.scene.gfmodel_debug_bone = prev_dbg_bone
            context.scene.gfmodel_debug_motion = prev_dbg_motion
        return {"FINISHED"}


def menu_func_import(self, context):
    self.layout.operator(
        IMPORT_SCENE_OT_gfmodel.bl_idname, text="GFModel (GFL2) (.bin/CP/CM)"
    )


_CLASSES = (IMPORT_SCENE_OT_gfmodel,)


def register() -> None:
    for c in _CLASSES:
        bpy.utils.register_class(c)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)


def unregister() -> None:
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
    for c in reversed(_CLASSES):
        bpy.utils.unregister_class(c)
