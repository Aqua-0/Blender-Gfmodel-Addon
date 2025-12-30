"""Debug/diff dump operators for the GFModel Blender add-on."""

from __future__ import annotations

import json
import math
import os
from typing import Dict, List

import bpy
from bpy.props import BoolProperty, StringProperty
from bpy_extras.io_utils import ExportHelper
from mathutils import Matrix

from ..core.io import _load_any
from .anim import _compute_rest_world_mats, _eval_motion_pose_world_mats


def _mat4_to_list(m: Matrix) -> List[List[float]]:
    return [[float(m[r][c]) for c in range(4)] for r in range(4)]


class GFModel_OT_dump_action_json(bpy.types.Operator, ExportHelper):
    bl_idname = "gfmodel.dump_action_json"
    bl_label = "GFModel: Dump Action JSON"
    bl_options = {"UNDO"}

    filename_ext = ".json"
    filter_glob: StringProperty(default="*.json", options={"HIDDEN"})

    use_scene_range: BoolProperty(
        name="Use Scene Frame Range",
        default=False,
        description="Dump frames from Scene Start/End instead of the action's frame range",
    )
    frame_start: bpy.props.IntProperty(                              
        name="Start Frame",
        default=0,
        min=-100000,
    )
    frame_end: bpy.props.IntProperty(                              
        name="End Frame",
        default=0,
        min=-100000,
    )
    action_name: StringProperty(
        name="Action Name (optional)",
        default="",
        description="If empty, uses the active action on the armature",
    )

    def execute(self, context: bpy.types.Context):
        import os

        arm_obj = context.object
        if arm_obj is None or arm_obj.type != "ARMATURE":
            arm_obj = next(
                (o for o in context.selected_objects if o.type == "ARMATURE"), None
            )
        if arm_obj is None or arm_obj.type != "ARMATURE":
            self.report({"ERROR"}, "Select an Armature object to dump")
            return {"CANCELLED"}

        arm_obj.animation_data_create()
        prev_action = arm_obj.animation_data.action

        if self.action_name.strip():
            action = bpy.data.actions.get(self.action_name.strip())
            if action is None:
                self.report({"ERROR"}, f"Action not found: {self.action_name.strip()}")
                return {"CANCELLED"}
        else:
            action = prev_action
            if action is None:
                self.report({"ERROR"}, "No active action on selected armature")
                return {"CANCELLED"}

        prev_frame = context.scene.frame_current
        try:
            if self.use_scene_range:
                start = int(context.scene.frame_start)
                end = int(context.scene.frame_end)
            else:
                a0, a1 = action.frame_range
                start = int(math.floor(a0))
                end = int(math.ceil(a1))
                if self.frame_end != 0 or self.frame_start != 0:
                    start = int(self.frame_start)
                    end = int(self.frame_end)

            if end < start:
                self.report({"ERROR"}, "End Frame must be >= Start Frame")
                return {"CANCELLED"}

            arm_obj.animation_data.action = action

            bones = []
            for b in arm_obj.data.bones:
                bones.append(
                    {
                        "name": b.name,
                        "parent": b.parent.name if b.parent else None,
                        "matrix_local": _mat4_to_list(b.matrix_local),
                    }
                )

            frames = []
            for frame in range(start, end + 1):
                context.scene.frame_set(frame)
                try:
                    context.view_layer.update()
                except Exception:
                    pass
                frame_entry = {"frame": int(frame), "bones": {}}
                for pb in arm_obj.pose.bones:
                    rest = pb.bone.matrix_local
                    pose = pb.matrix
                    delta = rest.inverted() @ pose
                    frame_entry["bones"][pb.name] = {
                        "matrix": _mat4_to_list(pose),
                        "delta_matrix": _mat4_to_list(delta),
                        "location": [float(x) for x in pb.location],
                        "rotation_quaternion": [
                            float(pb.rotation_quaternion.w),
                            float(pb.rotation_quaternion.x),
                            float(pb.rotation_quaternion.y),
                            float(pb.rotation_quaternion.z),
                        ],
                        "scale": [float(x) for x in pb.scale],
                    }
                frames.append(frame_entry)

            out = {
                "blender": bpy.app.version_string,
                "scene_fps": float(context.scene.render.fps),
                "armature_object": arm_obj.name,
                "action": action.name,
                "frame_start": int(start),
                "frame_end": int(end),
                "armature_matrix_world": _mat4_to_list(arm_obj.matrix_world),
                "bones": bones,
                "frames": frames,
            }

            os.makedirs(os.path.dirname(self.filepath) or ".", exist_ok=True)
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2)
        finally:
            try:
                arm_obj.animation_data.action = prev_action
            except Exception:
                pass
            try:
                context.scene.frame_set(prev_frame)
            except Exception:
                pass
            try:
                context.view_layer.update()
            except Exception:
                pass

        self.report({"INFO"}, f"Wrote {self.filepath}")
        return {"FINISHED"}


class GFModel_OT_dump_eval_json(bpy.types.Operator, ExportHelper):
    bl_idname = "gfmodel.dump_eval_json"
    bl_label = "GFModel: Dump Eval JSON"
    bl_options = {"UNDO"}

    filename_ext = ".json"
    filter_glob: StringProperty(default="*.json", options={"HIDDEN"})

    motion_index: bpy.props.IntProperty(                              
        name="Motion Index",
        default=0,
        min=0,
    )
    frame_start: bpy.props.IntProperty(                              
        name="Start Frame",
        default=0,
        min=0,
    )
    frame_end: bpy.props.IntProperty(                              
        name="End Frame (0=auto)",
        default=0,
        min=0,
    )
    to_blender_basis: BoolProperty(
        name="Convert To Blender Basis",
        default=False,
        description="If enabled, applies the same axis conversion as the importer (useful for viewing); for SPICA diffing keep this off",
    )

    def execute(self, context: bpy.types.Context):
        import os

        path = str(context.scene.get("gfmodel_last_import_path", ""))
        if not path:
            self.report({"ERROR"}, "No last import path stored; import a GFModel first")
            return {"CANCELLED"}
        try:
            with open(path, "rb") as f:
                data = f.read()
        except Exception as e:
            self.report({"ERROR"}, f"Failed to read: {path} ({e})")
            return {"CANCELLED"}

        axis_forward = str(context.scene.get("gfmodel_last_axis_forward", "-Z"))
        axis_up = str(context.scene.get("gfmodel_last_axis_up", "Y"))
        global_scale = float(context.scene.get("gfmodel_last_global_scale", 1.0))

        models, _textures, motions, _shaders = _load_any(data)
        if not models:
            self.report({"ERROR"}, "No models found in file")
            return {"CANCELLED"}
        if not motions:
            self.report({"ERROR"}, "No motions found in file")
            return {"CANCELLED"}
        mi = int(self.motion_index)
        if mi < 0 or mi >= len(motions):
            self.report(
                {"ERROR"}, f"Motion index out of range: {mi} (motions={len(motions)})"
            )
            return {"CANCELLED"}

        model = models[0]
        mot = motions[mi]

        if self.to_blender_basis:
            conv = axis_conversion(
                from_forward=axis_forward, from_up=axis_up, to_forward="-Y", to_up="Z"
            ).to_4x4()
        else:
            conv = Matrix.Identity(4)

                                                                                                                
        rest_world = _compute_rest_world_mats(model, conv, global_scale, ssc=False)

        end_default = max(0, int(mot.frames_count) - 1)
        start = int(self.frame_start)
        end = int(self.frame_end) if int(self.frame_end) > 0 else end_default
        if end < start:
            self.report({"ERROR"}, "End Frame must be >= Start Frame")
            return {"CANCELLED"}
        end = min(end, end_default)

        bones = []
        for b in model.skeleton:
            mw = rest_world.get(b.name, Matrix.Identity(4))
            bones.append(
                {
                    "name": b.name,
                    "parent": b.parent if b.parent else None,
                    "bind_matrix": _mat4_to_list(mw),
                }
            )

        frames = []
        for frame in range(start, end + 1):
            pose_world = _eval_motion_pose_world_mats(
                model, mot, conv, global_scale, frame
            )
            entry = {"frame": int(frame), "bones": {}}
            for b in model.skeleton:
                mw = rest_world.get(b.name, Matrix.Identity(4))
                pw = pose_world.get(b.name, Matrix.Identity(4))
                dm = mw.inverted() @ pw
                entry["bones"][b.name] = {
                    "matrix": _mat4_to_list(pw),
                    "delta_matrix": _mat4_to_list(dm),
                }
            frames.append(entry)

        out = {
            "type": "gfmodel_eval",
            "source_path": path,
            "model": model.name,
            "motion_index": int(mi),
            "frames_count": int(mot.frames_count),
            "frame_start": int(start),
            "frame_end": int(end),
            "motion": {
                "is_looping": bool(mot.is_looping),
                "is_blended": bool(mot.is_blended),
                "anim_region_min": [
                    float(mot.anim_region_min.x),
                    float(mot.anim_region_min.y),
                    float(mot.anim_region_min.z),
                ],
                "anim_region_max": [
                    float(mot.anim_region_max.x),
                    float(mot.anim_region_max.y),
                    float(mot.anim_region_max.z),
                ],
                "anim_hash": int(mot.anim_hash),
                "bones": int(len(mot.bones)),
                "uv_tracks": int(len(mot.uv_transforms)),
                "visibility_tracks": int(len(mot.visibility_tracks)),
                "unknown_sections": [
                    {"id": int(sid), "length": int(length), "addr": int(addr)}
                    for (sid, length, addr, _raw) in mot.unknown_sections
                ],
            },
            "axis_forward": axis_forward,
            "axis_up": axis_up,
            "to_blender_basis": bool(self.to_blender_basis),
            "global_scale": float(global_scale),
            "bones": bones,
            "frames": frames,
            "visibility": [
                {
                    "name": t.name,
                    "values": [
                        bool(t.values[i])
                        for i in range(
                            int(start),
                            min(int(end) + 2, len(t.values)),
                        )
                    ],
                }
                for t in mot.visibility_tracks
            ],
        }

        os.makedirs(os.path.dirname(self.filepath) or ".", exist_ok=True)
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        self.report({"INFO"}, f"Wrote {self.filepath}")
        return {"FINISHED"}


class GFModel_OT_dump_vertices_json(bpy.types.Operator, ExportHelper):
    bl_idname = "gfmodel.dump_vertices_json"
    bl_label = "GFModel: Dump Vertices JSON"
    bl_options = {"UNDO"}

    filename_ext = ".json"
    filter_glob: StringProperty(default="*.json", options={"HIDDEN"})

    model_index: bpy.props.IntProperty(                              
        name="Model Index",
        default=0,
        min=0,
    )
    submesh_index: bpy.props.IntProperty(                              
        name="Submesh Index (-1=all)",
        default=-1,
        min=-1,
    )
    max_vertices: bpy.props.IntProperty(                              
        name="Max Vertices (0=all)",
        default=512,
        min=0,
        description="Limits output size; set 0 to dump all vertices",
    )
    only_used_by_indices: BoolProperty(
        name="Only Vertices Used By Indices",
        default=True,
        description="If enabled, dumps only vertices referenced by the submesh index buffer",
    )

    def execute(self, context: bpy.types.Context):
        import os

        path = str(context.scene.get("gfmodel_last_import_path", ""))
        if not path:
            self.report({"ERROR"}, "No last import path stored; import a GFModel first")
            return {"CANCELLED"}
        try:
            with open(path, "rb") as f:
                data = f.read()
        except Exception as e:
            self.report({"ERROR"}, f"Failed to read: {path} ({e})")
            return {"CANCELLED"}

        models, _textures, _motions, _shaders = _load_any(data)
        if not models:
            self.report({"ERROR"}, "No models found in file")
            return {"CANCELLED"}

        mi = int(self.model_index)
        if mi < 0 or mi >= len(models):
            self.report(
                {"ERROR"}, f"Model index out of range: {mi} (models={len(models)})"
            )
            return {"CANCELLED"}

        model = models[mi]
        skel_names = [b.name for b in model.skeleton]

        def dump_submesh(sm: _GFSubMesh, sm_i: int) -> dict:
            _pos, _nrm, _uv0, _col, weights = _read_vertices(sm)

            palette = list(sm.bone_indices or [])

            used_set = None
            if self.only_used_by_indices and sm.indices:
                used_set = set(int(i) for i in sm.indices if int(i) >= 0)

            out_verts = []
            limit = int(self.max_vertices)
            dumped = 0

            for vi, wlist in enumerate(weights):
                if used_set is not None and vi not in used_set:
                    continue
                raw = [{"i": int(bi), "w": float(bw)} for (bi, bw) in wlist]
                resolved = []
                for bi, bw in wlist:
                    pal_idx = int(bi)
                    skel_idx = palette[pal_idx] if 0 <= pal_idx < len(palette) else None
                    bone_name = (
                        skel_names[int(skel_idx)]
                        if isinstance(skel_idx, int)
                        and 0 <= int(skel_idx) < len(skel_names)
                        else None
                    )
                    resolved.append(
                        {
                            "palette_index": pal_idx,
                            "weight": float(bw),
                            "skeleton_index": int(skel_idx)
                            if isinstance(skel_idx, int)
                            else None,
                            "bone_name": bone_name,
                        }
                    )
                out_verts.append(
                    {
                        "vertex_index": int(vi),
                        "raw": raw,
                        "resolved": resolved,
                        "weights_sum": float(sum(float(bw) for _bi, bw in wlist)),
                    }
                )
                dumped += 1
                if limit > 0 and dumped >= limit:
                    break

            return {
                "submesh_index": int(sm_i),
                "name": sm.name,
                "vertex_stride": int(sm.vertex_stride),
                "primitive_mode": int(sm.primitive_mode),
                "indices_count": int(len(sm.indices or [])),
                "palette_count": int(sm.bone_indices_count),
                "palette": palette[: int(sm.bone_indices_count)],
                "vertices_dumped": int(len(out_verts)),
                "only_used_by_indices": bool(self.only_used_by_indices),
                "max_vertices": int(self.max_vertices),
                "vertices": out_verts,
            }

        sms = model.submeshes
        smi = int(self.submesh_index)
        if smi == -1:
            submeshes_out = [dump_submesh(sm, i) for i, sm in enumerate(sms)]
        else:
            if smi < 0 or smi >= len(sms):
                self.report(
                    {"ERROR"},
                    f"Submesh index out of range: {smi} (submeshes={len(sms)})",
                )
                return {"CANCELLED"}
            submeshes_out = [dump_submesh(sms[smi], smi)]

        out = {
            "type": "gfmodel_vertices",
            "source_path": path,
            "model_index": int(mi),
            "model": model.name,
            "skeleton_bones": skel_names,
            "submeshes": submeshes_out,
        }

        os.makedirs(os.path.dirname(self.filepath) or ".", exist_ok=True)
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        self.report({"INFO"}, f"Wrote {self.filepath}")
        return {"FINISHED"}


class GFModel_OT_dump_rest_compare_json(bpy.types.Operator, ExportHelper):
    bl_idname = "gfmodel.dump_rest_compare_json"
    bl_label = "GFModel: Dump Rest Compare JSON"
    bl_options = {"UNDO"}

    filename_ext = ".json"
    filter_glob: StringProperty(default="*.json", options={"HIDDEN"})

    to_blender_basis: BoolProperty(
        name="Convert To Blender Basis",
        default=True,
        description="If enabled, uses the same axis conversion as the importer",
    )

    def execute(self, context: bpy.types.Context):
        import os

        arm_obj = context.object
        if arm_obj is None or arm_obj.type != "ARMATURE":
            arm_obj = next(
                (o for o in context.selected_objects if o.type == "ARMATURE"), None
            )
        if arm_obj is None or arm_obj.type != "ARMATURE":
            self.report({"ERROR"}, "Select an Armature object to dump")
            return {"CANCELLED"}

        path = str(context.scene.get("gfmodel_last_import_path", ""))
        if not path:
            self.report({"ERROR"}, "No last import path stored; import a GFModel first")
            return {"CANCELLED"}
        try:
            with open(path, "rb") as f:
                data = f.read()
        except Exception as e:
            self.report({"ERROR"}, f"Failed to read: {path} ({e})")
            return {"CANCELLED"}

        axis_forward = str(context.scene.get("gfmodel_last_axis_forward", "-Z"))
        axis_up = str(context.scene.get("gfmodel_last_axis_up", "Y"))
        global_scale = float(context.scene.get("gfmodel_last_global_scale", 1.0))

        models, _textures, _motions, _shaders = _load_any(data)
        if not models:
            self.report({"ERROR"}, "No models found in file")
            return {"CANCELLED"}
        model = models[0]

        if self.to_blender_basis:
            conv = axis_conversion(
                from_forward=axis_forward, from_up=axis_up, to_forward="-Y", to_up="Z"
            ).to_4x4()
        else:
            conv = Matrix.Identity(4)

        gf_rest_world = _compute_rest_world_mats(model, conv, global_scale, ssc=False)

                                                           
        bl_rest_world: Dict[str, Matrix] = {}
        for pb in arm_obj.pose.bones:
            bl_rest_world[pb.name] = pb.bone.matrix_local.copy()

        rows = []
        worst = {"bone": None, "max_abs": 0.0}
        for b in model.skeleton:
            name = b.name
            g = gf_rest_world.get(name)
            bl = bl_rest_world.get(name)
            if g is None or bl is None:
                rows.append(
                    {
                        "name": name,
                        "present_gf": g is not None,
                        "present_blender": bl is not None,
                    }
                )
                continue
            d = bl - g
            max_abs = max(abs(float(d[r][c])) for r in range(4) for c in range(4))
            if max_abs > float(worst["max_abs"]):
                worst = {"bone": name, "max_abs": float(max_abs)}
            rows.append(
                {
                    "name": name,
                    "gf": _mat4_to_list(g),
                    "blender": _mat4_to_list(bl),
                    "diff": _mat4_to_list(d),
                    "max_abs": float(max_abs),
                }
            )

        out = {
            "type": "gfmodel_rest_compare",
            "source_path": path,
            "armature_object": arm_obj.name,
            "axis_forward": axis_forward,
            "axis_up": axis_up,
            "to_blender_basis": bool(self.to_blender_basis),
            "global_scale": float(global_scale),
            "worst": worst,
            "bones": rows,
        }

        os.makedirs(os.path.dirname(self.filepath) or ".", exist_ok=True)
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        self.report({"INFO"}, f"Wrote {self.filepath}")
        return {"FINISHED"}


_CLASSES = (
    GFModel_OT_dump_action_json,
    GFModel_OT_dump_eval_json,
    GFModel_OT_dump_vertices_json,
    GFModel_OT_dump_rest_compare_json,
)


def register() -> None:
    for c in _CLASSES:
        bpy.utils.register_class(c)


def unregister() -> None:
    for c in reversed(_CLASSES):
        bpy.utils.unregister_class(c)
