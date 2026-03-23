
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


def _build_armature(
    ctx: bpy.types.Context,
    model: _GFModel,
    conv: Matrix,
    global_scale: float,
    collection: bpy.types.Collection,
) -> bpy.types.Object:
    arm_data = bpy.data.armatures.new(f"{model.name}_Armature")
    arm_obj = bpy.data.objects.new(arm_data.name, arm_data)
    collection.objects.link(arm_obj)

    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode="EDIT")

    bones_by_name: Dict[str, bpy.types.EditBone] = {}
    conv3 = conv.to_3x3()

    def local_rest_matrix(b: _GFBone) -> Matrix:
        t = Matrix.Translation(conv @ (b.translation * global_scale))
        q = _transform_quat_basis(_euler_to_quat_xyz(b.rotation), conv3)
        r = q.to_matrix().to_4x4()
        s = Matrix.Diagonal(Vector((b.scale.x, b.scale.y, b.scale.z, 1.0)))
        return t @ r @ s

    rest_world = _compute_rest_world_mats(model, conv, global_scale, ssc=False)

    for b in model.skeleton:
        eb = arm_data.edit_bones.new(b.name)
        bones_by_name[b.name] = eb

    for b in model.skeleton:
        if b.parent and b.parent in bones_by_name:
            bones_by_name[b.name].parent = bones_by_name[b.parent]

    for b in model.skeleton:
        eb = bones_by_name[b.name]
        mw = rest_world.get(b.name, Matrix.Identity(4))
        loc, rot, _sca = mw.decompose()
        head = loc
        rot3 = rot.to_matrix()

        length = max(0.01, 0.05 * global_scale)
        y_axis = rot3 @ Vector((0.0, 1.0, 0.0))
        if y_axis.length == 0:
            y_axis = Vector((0.0, 1.0, 0.0))
        else:
            y_axis.normalize()

        eb.head = head
        eb.tail = head + y_axis * length

        try:
            z_axis = rot3 @ Vector((0.0, 0.0, 1.0))
            if z_axis.length != 0:
                z_axis.normalize()
                eb.align_roll(z_axis)
        except Exception:
            pass

    bpy.ops.object.mode_set(mode="OBJECT")

    for pb in arm_obj.pose.bones:
        try:
            pb.bone.inherit_scale = "FULL"
        except Exception:
            pass

                                                                 
    try:
        for b in model.skeleton:
            db = arm_data.bones.get(b.name)
            if db is None:
                continue
            db["gfmodel_flags"] = int(getattr(b, "flags", 0) or 0)
            db["gfmodel_rest_scale"] = (float(b.scale.x), float(b.scale.y), float(b.scale.z))
            db["gfmodel_rest_rot"] = (float(b.rotation.x), float(b.rotation.y), float(b.rotation.z))
            db["gfmodel_rest_trans"] = (float(b.translation.x), float(b.translation.y), float(b.translation.z))
    except Exception:
        pass

    return arm_obj

    for pb in arm_obj.pose.bones:
        try:
            pb.bone.inherit_scale = "FULL"
        except Exception:
            pass
    return arm_obj


