
from __future__ import annotations

import os
import math
from typing import Dict, List, Optional, Tuple

import bpy
from bpy_extras.io_utils import axis_conversion
from mathutils import Matrix, Quaternion, Vector

from ..core.garc import parse_garc_file, rewrite_garc_file, rewrite_garc_file_inplace_atomic
from ..core.motion import _parse_gf_motion, _mot_eval
from ..core.motion_export import patch_gfmot_with_rekey_changed_frames
from ..core.patch_plan import PatchPlan, extract_via_steps, patch_plan_from_json, patch_via_steps
from ..core.types import _GFMotBoneTransform, _GFMotKeyFrame


def _collect_action_keyed_frames(action: bpy.types.Action, *, frames_count: int) -> Dict[str, Dict[str, set[int]]]:
    out: Dict[str, Dict[str, set[int]]] = {}
    fc = int(frames_count)

    try:
        fcurves = list(getattr(action, 'fcurves', []) or [])
    except Exception:
        fcurves = []

    for fcu in fcurves:
        try:
            dp = str(getattr(fcu, 'data_path', '') or '')
        except Exception:
            continue
        if not dp.startswith('pose.bones[\"'):
            continue

        a = dp.find('[\"')
        b = dp.find('\"]', a + 2)
        if a < 0 or b < 0:
            continue
        bone_name = dp[a + 2 : b]
        rest = dp[b + 2 :]
        if rest.startswith('.'):
            rest = rest[1:]

        if rest.startswith('location'):
            kind = 'pos'
        elif rest.startswith('scale'):
            kind = 'scale'
        elif rest.startswith('rotation_'):
            kind = 'rot'
        else:
            continue

        try:
            kps = list(getattr(fcu, 'keyframe_points', []) or [])
        except Exception:
            kps = []
        if not kps:
            continue

        d = out.setdefault(str(bone_name), {'pos': set(), 'rot': set(), 'scale': set()})
        s = d[kind]
        for kp in kps:
            try:
                fr = int(round(float(kp.co[0])))
            except Exception:
                continue
            if 0 <= fr <= fc:
                s.add(int(fr))

    return out


def _resolve_out_archive(archive_path: str, out_archive: str) -> str:
    out_archive = str(out_archive or "").strip()
    if not out_archive:
        out_archive = str(archive_path) + ".patched"
    try:
        if out_archive.endswith(("/", "\\")) or os.path.isdir(out_archive):
            base = os.path.basename(str(archive_path).rstrip("\\/"))
            out_archive = os.path.join(out_archive.rstrip("\\/"), base + ".patched")
    except Exception:
        pass
    return str(out_archive)


def _payload_matches_expected(payload: bytes, expected: bytes) -> bool:
    p = bytes(payload)
    e = bytes(expected)
    if p == e:
        return True
    if len(p) < len(e):
        return False
    if p[: len(e)] != e:
        return False
    tail = p[len(e) :]
    return not tail or all(b == 0 for b in tail)


def _bone_uses_ssc(flags: int) -> bool:
    return (int(flags) & 0x02) != 0


def _transform_quat_basis(q: Quaternion, conv3: Matrix) -> Quaternion:
    m = conv3 @ q.to_matrix() @ conv3.inverted()
    return m.to_quaternion()


def _quat_to_log_axis_angle(q: Quaternion) -> Vector:
    try:
        qn = Quaternion(q)
        qn.normalize()
    except Exception:
        return Vector((0.0, 0.0, 0.0))

    w = max(-1.0, min(1.0, float(qn.w)))
    half = float(math.acos(w))
    s = float(max(0.0, 1.0 - w * w)) ** 0.5
    if s < 1e-8 or half == 0.0:
        return Vector((0.0, 0.0, 0.0))

    v = Vector((float(qn.x), float(qn.y), float(qn.z)))
    try:
        v /= float(s)
    except Exception:
        return Vector((0.0, 0.0, 0.0))
    return v * float(half)


def _quat_to_log_near(q: Quaternion, ref: Vector) -> Vector:
    try:
        ref_v = Vector(ref)
    except Exception:
        ref_v = Vector((0.0, 0.0, 0.0))


    cands: list[Vector] = []
    try:
        qn = Quaternion(q)
        qn.normalize()
        cands.append(_quat_to_log_axis_angle(qn))
        cands.append(_quat_to_log_axis_angle(-qn))
    except Exception:
        return Vector((0.0, 0.0, 0.0))

    best = cands[0]
    best_d = (best - ref_v).length
    for v in list(cands):
        axis = Vector(v)
        if axis.length != 0.0:
            axis.normalize()
        else:
            axis = Vector((1.0, 0.0, 0.0))
        for k in (-1, 0, 1):
            vv = Vector(v) + axis * (2.0 * math.pi * float(k))
            d = (vv - ref_v).length
            if d < best_d:
                best_d = d
                best = vv
    return Vector((float(best.x), float(best.y), float(best.z)))
def _sample_action_to_motion_bones(
    *,
    ctx: bpy.types.Context,
    arm_obj: bpy.types.Object,
    action: bpy.types.Action,
    src_motion_bones: List[_GFMotBoneTransform],
    frames_count: int,
    axis_forward: str,
    axis_up: str,
    global_scale: float,
) -> List[_GFMotBoneTransform]:
    scene = ctx.scene

    conv = axis_conversion(
        from_forward=str(axis_forward), from_up=str(axis_up), to_forward="-Y", to_up="Z"
    ).to_4x4()
    conv3 = conv.to_3x3()
    conv_inv = conv.inverted()


    bone_names = [b.name for b in arm_obj.data.bones]
    parent_by_name = {b.name: (b.parent.name if b.parent else "") for b in arm_obj.data.bones}

    depth_cache: Dict[str, int] = {}

    def depth(name: str) -> int:
        if name in depth_cache:
            return depth_cache[name]
        p = parent_by_name.get(name, "")
        if not p:
            depth_cache[name] = 0
            return 0
        depth_cache[name] = 1 + depth(p)
        return depth_cache[name]

    bone_order = sorted(bone_names, key=depth)


    rest_abs_by_name: Dict[str, Matrix] = {}
    rest_rel_by_name: Dict[str, Matrix] = {}
    for b in arm_obj.data.bones:
        rest_abs_by_name[b.name] = b.matrix_local.copy()
    for name in bone_order:
        rest_abs = rest_abs_by_name.get(name)
        if rest_abs is None:
            continue
        p = parent_by_name.get(name, "")
        if p and p in rest_abs_by_name:
            rest_rel_by_name[name] = rest_abs_by_name[p].inverted() @ rest_abs
        else:
            rest_rel_by_name[name] = rest_abs

    flags_by_name: Dict[str, int] = {}
    for b in arm_obj.data.bones:
        try:
            flags_by_name[b.name] = int(b.get("gfmodel_flags", 0) or 0)
        except Exception:
            flags_by_name[b.name] = 0


    arm_obj.animation_data_create()
    prev_action = arm_obj.animation_data.action
    prev_frame = int(scene.frame_current)

    try:
        arm_obj.animation_data.action = action


        vals: Dict[str, Dict[str, List[float]]] = {}
        for bt in src_motion_bones:
            vals[bt.name] = {
                "sx": [],
                "sy": [],
                "sz": [],
                "rx": [],
                "ry": [],
                "rz": [],
                "tx": [],
                "ty": [],
                "tz": [],
            }


        prev_q_by_name: Dict[str, Quaternion] = {}
        for fr in range(int(frames_count)):
            scene.frame_set(int(fr))

            sca_gf_by_name: Dict[str, Vector] = {}

            for name in bone_order:
                if name not in vals:

                    continue

                pb = arm_obj.pose.bones.get(name)
                if pb is None:
                    continue

                rest_rel = rest_rel_by_name.get(name)
                if rest_rel is None:
                    continue

                pose_local = rest_rel @ pb.matrix_basis

                use_ssc = False
                p = parent_by_name.get(name, "")
                if p:
                    use_ssc = _bone_uses_ssc(flags_by_name.get(name, 0))

                if use_ssc and p and p in sca_gf_by_name:
                    ps = sca_gf_by_name[p]
                    psx, psy, psz = float(ps.x), float(ps.y), float(ps.z)
                    sp = Matrix.Diagonal(Vector((psx, psy, psz, 1.0)))
                    local_mat = sp @ pose_local
                else:
                    psx = psy = psz = 1.0
                    local_mat = pose_local

                loc_b, rot_b, sca_b = local_mat.decompose()


                sca_gf = Vector((float(sca_b.x), float(sca_b.y), float(sca_b.z)))
                sca_gf_by_name[name] = sca_gf


                q_gf = _transform_quat_basis(Quaternion(rot_b), conv3.inverted())


                try:
                    src_bt = next((x for x in src_motion_bones if x.name == name), None)
                    if src_bt is not None and bool(getattr(src_bt, 'is_axis_angle', True)):
                        ox = float(_mot_eval(getattr(src_bt, 'rx', []), int(fr), 0.0))
                        oy = float(_mot_eval(getattr(src_bt, 'ry', []), int(fr), 0.0))
                        oz = float(_mot_eval(getattr(src_bt, 'rz', []), int(fr), 0.0))
                        v0 = Vector((ox, oy, oz))
                        ang = float(v0.length) * 2.0
                        if ang > 0.0:
                            ax = Vector(v0)
                            if ax.length != 0.0:
                                ax.normalize()
                                q0 = Quaternion(ax, float(ang))
                            else:
                                q0 = Quaternion((1.0, 0.0, 0.0, 0.0))
                        else:
                            q0 = Quaternion((1.0, 0.0, 0.0, 0.0))
                        if float(q_gf.dot(q0)) < 0.0:
                            q_gf = -q_gf
                except Exception:
                    pass
                try:
                    prevq = prev_q_by_name.get(name)
                    if prevq is not None and float(q_gf.dot(prevq)) < 0.0:
                        q_gf = -q_gf
                    prev_q_by_name[name] = Quaternion(q_gf)
                except Exception:
                    pass


                loc4 = conv_inv @ Vector((float(loc_b.x), float(loc_b.y), float(loc_b.z), 1.0))
                t_scaled = Vector((float(loc4.x), float(loc4.y), float(loc4.z)))
                if float(global_scale) != 0.0:
                    t_scaled /= float(global_scale)

                if use_ssc and (psx != 0.0 and psy != 0.0 and psz != 0.0):
                    tx = float(t_scaled.x) / float(psx)
                    ty = float(t_scaled.y) / float(psy)
                    tz = float(t_scaled.z) / float(psz)
                else:
                    tx, ty, tz = float(t_scaled.x), float(t_scaled.y), float(t_scaled.z)


                src_bt = next((x for x in src_motion_bones if x.name == name), None)
                is_axis = bool(getattr(src_bt, "is_axis_angle", True)) if src_bt else True
                if is_axis:
                    ref_v = Vector((0.0, 0.0, 0.0))
                    try:
                        if src_bt is not None:
                            ox = float(_mot_eval(getattr(src_bt, 'rx', []), int(fr), 0.0))
                            oy = float(_mot_eval(getattr(src_bt, 'ry', []), int(fr), 0.0))
                            oz = float(_mot_eval(getattr(src_bt, 'rz', []), int(fr), 0.0))
                            ref_v = Vector((ox, oy, oz))
                    except Exception:
                        ref_v = Vector((0.0, 0.0, 0.0))
                    v = _quat_to_log_near(q_gf, ref_v)
                    rx, ry, rz = float(v.x), float(v.y), float(v.z)
                else:
                    e = q_gf.to_euler("XYZ")
                    rx, ry, rz = float(e.x), float(e.y), float(e.z)

                vdst = vals.get(name)
                if vdst is None:
                    continue
                vdst["sx"].append(float(sca_gf.x))
                vdst["sy"].append(float(sca_gf.y))
                vdst["sz"].append(float(sca_gf.z))
                vdst["rx"].append(float(rx))
                vdst["ry"].append(float(ry))
                vdst["rz"].append(float(rz))
                vdst["tx"].append(float(tx))
                vdst["ty"].append(float(ty))
                vdst["tz"].append(float(tz))

        out: List[_GFMotBoneTransform] = []
        for src_bt in src_motion_bones:
            v = vals.get(src_bt.name)
            if v is None:

                out.append(src_bt)
                continue
            n = int(frames_count)

            def mk(vals_list: List[float]) -> List[_GFMotKeyFrame]:
                if len(vals_list) != n:

                    return []
                return [
                    _GFMotKeyFrame(frame=i, value=float(vals_list[i]), slope=0.0)
                    for i in range(n)
                ]

            out.append(
                _GFMotBoneTransform(
                    name=str(src_bt.name),
                    is_axis_angle=bool(getattr(src_bt, "is_axis_angle", True)),
                    sx=mk(v["sx"]),
                    sy=mk(v["sy"]),
                    sz=mk(v["sz"]),
                    rx=mk(v["rx"]),
                    ry=mk(v["ry"]),
                    rz=mk(v["rz"]),
                    tx=mk(v["tx"]),
                    ty=mk(v["ty"]),
                    tz=mk(v["tz"]),
                )
            )

        return out
    finally:
        try:
            arm_obj.animation_data.action = prev_action
        except Exception:
            pass
        try:
            scene.frame_set(prev_frame)
        except Exception:
            pass


class GFModel_OT_patch_selected_action_to_source_motion(bpy.types.Operator):
    bl_idname = "gfmodel.patch_selected_action_to_source_motion"
    bl_label = "GFModel: Patch Selected Action to Source Motion"
    bl_options = {"UNDO"}

    def execute(self, context: bpy.types.Context):
        obj = context.object
        if obj is None or obj.type != "ARMATURE":
            self.report({"ERROR"}, "Select an imported GFModel armature")
            return {"CANCELLED"}
        if not obj.animation_data or not obj.animation_data.action:
            self.report({"ERROR"}, "No active Action on the selected armature")
            return {"CANCELLED"}

        action = obj.animation_data.action
        plan_json = str(action.get("gfmodel_motion_patch_plan_json", "") or "").strip()
        if not plan_json:
            self.report({"ERROR"}, "This Action has no recorded source motion; re-import from GFModel Archive")
            return {"CANCELLED"}

        try:
            plan = patch_plan_from_json(plan_json)
        except Exception as e:
            self.report({"ERROR"}, f"Invalid motion patch plan: {e}")
            return {"CANCELLED"}

        if not os.path.isfile(str(plan.archive_path)):
            self.report({"ERROR"}, f"Archive not found: {plan.archive_path}")
            return {"CANCELLED"}

        axis_forward = str(obj.get("gfmodel_axis_forward", "-Z") or "-Z")
        axis_up = str(obj.get("gfmodel_axis_up", "Y") or "Y")
        try:
            global_scale = float(obj.get("gfmodel_global_scale", 1.0) or 1.0)
        except Exception:
            global_scale = 1.0

        try:
            garc = parse_garc_file(str(plan.archive_path))
            entry_bytes = garc.read_primary_bytes(int(plan.entry_index))
        except Exception as e:
            self.report({"ERROR"}, f"Failed to read archive entry: {e}")
            return {"CANCELLED"}

        try:
            src_motion_bytes = extract_via_steps(entry_bytes, plan.steps)
        except Exception as e:
            self.report({"ERROR"}, f"Failed to extract source motion bytes: {e}")
            return {"CANCELLED"}

        try:
            src_motion = _parse_gf_motion(bytes(src_motion_bytes), index=0)
        except Exception as e:
            self.report({"ERROR"}, f"Failed to parse source motion: {e}")
            return {"CANCELLED"}

        frames_count = int(getattr(src_motion, "frames_count", 0) or 0)
        if frames_count <= 0:
            self.report({"ERROR"}, "Source motion has 0 frames")
            return {"CANCELLED"}

        if not getattr(src_motion, "bones", None):
            self.report({"ERROR"}, "Source motion has no skeletal tracks")
            return {"CANCELLED"}

        prev_runtime = None
        try:
            prev_runtime = bool(getattr(obj, 'gfmodel_runtime_enabled', False))
        except Exception:
            prev_runtime = None
        if prev_runtime:
            try:
                obj.gfmodel_runtime_enabled = False
            except Exception:
                pass

        keyed = _collect_action_keyed_frames(action, frames_count=int(frames_count))
        if not keyed:
            keyed = None

        try:
            new_bones = _sample_action_to_motion_bones(
                ctx=context,
                arm_obj=obj,
                action=action,
                src_motion_bones=list(src_motion.bones),
                frames_count=int(frames_count),
                axis_forward=axis_forward,
                axis_up=axis_up,
                global_scale=float(global_scale),
            )
        except Exception as e:
            self.report({"ERROR"}, f"Failed to sample action: {e}")
            return {"CANCELLED"}
        finally:
            if prev_runtime:
                try:
                    obj.gfmodel_runtime_enabled = True
                except Exception:
                    pass

        try:
            new_motion_bytes = patch_gfmot_with_rekey_changed_frames(
                src_gfmot_bytes=bytes(src_motion_bytes),
                src_motion=src_motion,
                new_bones=new_bones,
                frames_count=int(frames_count),
                keyed_frames_by_bone=keyed,
            )
        except Exception as e:
            self.report({"ERROR"}, f"Failed to rebuild motion bytes: {e}")
            return {"CANCELLED"}

        try:
            new_entry = patch_via_steps(bytes(entry_bytes), plan.steps, bytes(new_motion_bytes))
        except Exception as e:
            self.report({"ERROR"}, f"Failed to patch entry bytes: {e}")
            return {"CANCELLED"}

        out_archive = _resolve_out_archive(
            str(plan.archive_path),
            str(getattr(context.scene, "gfmodel_archive_patch_output_path", "")),
        )
        inplace = bool(getattr(context.scene, "gfmodel_archive_patch_inplace", False))
        backup = bool(getattr(context.scene, "gfmodel_archive_patch_backup", True))

        try:
            if inplace:
                if not backup:
                    raise ValueError('In-place patch requires Backup enabled')
                rewrite_garc_file_inplace_atomic(
                    str(plan.archive_path),
                    replacements={(int(plan.entry_index), int(plan.bit)): bytes(new_entry)},
                    make_backup=True,
                )
                out_archive = str(plan.archive_path)
            else:
                if os.path.abspath(out_archive) == os.path.abspath(str(plan.archive_path)):
                    raise ValueError(
                        'Archive Out path matches source; enable In-Place (with Backup) or choose a different Out file'
                    )
                rewrite_garc_file(
                    str(plan.archive_path),
                    str(out_archive),
                    replacements={(int(plan.entry_index), int(plan.bit)): bytes(new_entry)},
                )
        except Exception as e:
            self.report({"ERROR"}, f"Write failed: {e}")
            return {"CANCELLED"}


        try:
            garc2 = parse_garc_file(str(out_archive))
            entry2 = garc2.read_primary_bytes(int(plan.entry_index))
            motion2 = extract_via_steps(entry2, plan.steps)
            if not _payload_matches_expected(motion2, bytes(new_motion_bytes)):
                raise ValueError(
                    f'verify failed (motion bytes differ): expected={len(new_motion_bytes)} got={len(motion2)}'
                )
        except Exception as e:
            self.report({"ERROR"}, f"Patched, but verify failed: {e}")
            return {"CANCELLED"}

        self.report({"INFO"}, f"Patched motion into: {out_archive}")
        return {"FINISHED"}


_CLASSES = (GFModel_OT_patch_selected_action_to_source_motion,)


def register() -> None:
    for c in _CLASSES:
        bpy.utils.register_class(c)


def unregister() -> None:
    for c in reversed(_CLASSES):
        bpy.utils.unregister_class(c)
