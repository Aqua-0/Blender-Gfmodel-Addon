"""Animation/runtime evaluation and UI state (Blender module).

Contains:
- Motion evaluation helpers
- Runtime (frame-change) evaluator for SSC-accurate playback
- UV animation enable toggle handler
- GFModel N-panel UI

No import/export operator classes live here.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import bpy
from bpy.app.handlers import persistent
from bpy_extras.io_utils import axis_conversion
from mathutils import Matrix, Quaternion, Vector

from ..core.io import _load_any
from ..core.types import _GFModel, _GFMotion, _GFMotKeyFrame


def _herp(
    lhs: float, rhs: float, ls: float, rs: float, diff: float, weight: float
) -> float:
    res = lhs + (lhs - rhs) * (2 * weight - 3) * weight * weight
    res += (diff * (weight - 1)) * (ls * (weight - 1) + rs * weight)
    return res


def _mot_eval(kfs: List[_GFMotKeyFrame], frame: int, default: float) -> float:
    if not kfs:
        return default
    if len(kfs) == 1:
        return kfs[0].value
    lhs = None
    rhs = None
    for kf in kfs:
        if kf.frame <= frame:
            lhs = kf
        if kf.frame >= frame:
            rhs = kf
            break
    if lhs is None:
        lhs = kfs[0]
    if rhs is None:
        rhs = kfs[-1]
    if lhs.frame == rhs.frame:
        return lhs.value
    diff = float(frame - lhs.frame)
    w = diff / float(rhs.frame - lhs.frame)
    return _herp(lhs.value, rhs.value, lhs.slope, rhs.slope, diff, w)


def _bone_uses_ssc(flags: int) -> bool:
                                                                           
                                                                                 
    return (int(flags) & 0x02) != 0


def _euler_to_quat_xyz(e: Vector) -> Quaternion:
                                    
    qx = Quaternion((1.0, 0.0, 0.0), e.x)
    qy = Quaternion((0.0, 1.0, 0.0), e.y)
    qz = Quaternion((0.0, 0.0, 1.0), e.z)
    return qz @ qy @ qx


def _transform_quat_basis(q: Quaternion, conv3: Matrix) -> Quaternion:
                                                                          
    m = conv3 @ q.to_matrix() @ conv3.inverted()
    return m.to_quaternion()


_GF_RUNTIME_CACHE: Dict[int, Dict[str, object]] = {}


def _gf_runtime_cache_armature(
    arm_obj: bpy.types.Object,
    *,
    model: _GFModel,
    motions: List[_GFMotion],
    conv: Matrix,
    conv3: Matrix,
    global_scale: float,
    bone_order: List[str],
    rest_by_name: Dict[str, _GFBone],
    rest_rel_by_name: Dict[str, Matrix],
) -> None:
    _GF_RUNTIME_CACHE[int(arm_obj.as_pointer())] = {
        "model": model,
        "motions": motions,
        "conv": conv,
        "conv3": conv3,
        "global_scale": float(global_scale),
        "bone_order": bone_order,
        "rest_by_name": rest_by_name,
        "rest_rel_by_name": rest_rel_by_name,
        "last_frame": None,
        "last_motion": None,
    }
    try:
        arm_obj["gfmodel_has_runtime_data"] = True
        arm_obj["gfmodel_motions_count"] = int(len(motions))
    except Exception:
        pass


def _gf_runtime_apply_pose(
    ctx: bpy.types.Context,
    arm_obj: bpy.types.Object,
    frame: int,
    *,
    motion_index: int,
) -> None:
    cache = _GF_RUNTIME_CACHE.get(int(arm_obj.as_pointer()))
    if cache is None:
        cache = _gf_runtime_ensure_cache(ctx, arm_obj)
    if not cache:
        return

    model = cache["model"]
    motions = cache["motions"]
    conv = cache["conv"]
    conv3 = cache["conv3"]
    global_scale = cache["global_scale"]
    bone_order = cache["bone_order"]
    rest_by_name = cache["rest_by_name"]
    rest_rel_by_name = cache["rest_rel_by_name"]

    if not isinstance(motions, list) or not motions:
        return
    if motion_index < 0 or motion_index >= len(motions):
        return

    mot = motions[motion_index]
    if not getattr(mot, "bones", None):
        return
    if int(mot.frames_count) <= 0:
        return

    loop = bool(getattr(arm_obj, "gfmodel_runtime_loop", True))
    if loop:
        fr = int(frame) % int(mot.frames_count)
    else:
        fr = max(0, min(int(frame), int(mot.frames_count) - 1))

    if cache.get("last_frame") == fr and cache.get("last_motion") == motion_index:
        return

    bt_by_name = {bt.name: bt for bt in mot.bones}
    pose_world: Dict[str, Matrix] = {}

    def pose_for(name: str) -> Matrix:
        if name in pose_world:
            return pose_world[name]
        rb = rest_by_name.get(name)
        if rb is None:
            pose_world[name] = Matrix.Identity(4)
            return pose_world[name]
        bt = bt_by_name.get(name)

        sx = _mot_eval(bt.sx, fr, rb.scale.x) if bt else rb.scale.x
        sy = _mot_eval(bt.sy, fr, rb.scale.y) if bt else rb.scale.y
        sz = _mot_eval(bt.sz, fr, rb.scale.z) if bt else rb.scale.z

        tx = _mot_eval(bt.tx, fr, rb.translation.x) if bt else rb.translation.x
        ty = _mot_eval(bt.ty, fr, rb.translation.y) if bt else rb.translation.y
        tz = _mot_eval(bt.tz, fr, rb.translation.z) if bt else rb.translation.z

        rx = _mot_eval(bt.rx, fr, rb.rotation.x) if bt else rb.rotation.x
        ry = _mot_eval(bt.ry, fr, rb.rotation.y) if bt else rb.rotation.y
        rz = _mot_eval(bt.rz, fr, rb.rotation.z) if bt else rb.rotation.z

        if bt and bt.is_axis_angle:
            axis = Vector((rx, ry, rz))
            angle = axis.length * 2.0
            if angle > 0:
                q_cur = Quaternion(axis.normalized(), angle)
            else:
                q_cur = Quaternion((1.0, 0.0, 0.0, 0.0))
        else:
            q_cur = _euler_to_quat_xyz(Vector((rx, ry, rz)))

        q_cur_t = _transform_quat_basis(q_cur, conv3)
        r_anim = q_cur_t.to_matrix().to_4x4()
        s_anim = Matrix.Diagonal(Vector((sx, sy, sz, 1.0)))

        if rb.parent and rb.parent in rest_by_name:
            prb = rest_by_name[rb.parent]
            pbt = bt_by_name.get(rb.parent)
            psx = _mot_eval(pbt.sx, fr, prb.scale.x) if pbt else prb.scale.x
            psy = _mot_eval(pbt.sy, fr, prb.scale.y) if pbt else prb.scale.y
            psz = _mot_eval(pbt.sz, fr, prb.scale.z) if pbt else prb.scale.z

            inv_psx = 1.0 / psx if psx != 0 else 0.0
            inv_psy = 1.0 / psy if psy != 0 else 0.0
            inv_psz = 1.0 / psz if psz != 0 else 0.0
            inv_s_parent = Matrix.Diagonal(Vector((inv_psx, inv_psy, inv_psz, 1.0)))

            t_scaled = Vector((tx * psx, ty * psy, tz * psz))
            t_anim = Matrix.Translation(conv @ (t_scaled * global_scale))
            pose = pose_for(rb.parent) @ inv_s_parent @ (t_anim @ r_anim @ s_anim)
        else:
            t_anim = Matrix.Translation(conv @ (Vector((tx, ty, tz)) * global_scale))
            pose = t_anim @ r_anim @ s_anim

        pose_world[name] = pose
        return pose

    for b in model.skeleton:
        pose_for(b.name)

    for name in bone_order:
        pb = arm_obj.pose.bones.get(name)
        rb = rest_by_name.get(name)
        if pb is None or rb is None:
            continue
        pose_mat = pose_world.get(name)
        if pose_mat is None:
            continue
        if rb.parent and rb.parent in pose_world:
            pose_local = pose_world[rb.parent].inverted() @ pose_mat
        else:
            pose_local = pose_mat
        rest_local = rest_rel_by_name.get(name)
        if rest_local is None:
            rest_local = pb.bone.matrix_local.copy()
        try:
            pb.matrix_basis = rest_local.inverted() @ pose_local
        except Exception:
            pass

    cache["last_frame"] = fr
    cache["last_motion"] = motion_index

    if bool(getattr(ctx.scene, "gfmodel_debug_animations", False)):
        dbg_motion = int(getattr(ctx.scene, "gfmodel_debug_motion", -1))
        if dbg_motion < 0 or dbg_motion == motion_index:
            dbg_name = getattr(ctx.scene, "gfmodel_debug_bone", "Waist")
            m = pose_world.get(dbg_name)
            if m is not None:
                loc, rot, sca = m.decompose()
                print(
                    f"[GFModel][AnimDebug][Runtime] mot={motion_index} frame={fr} bone={dbg_name} "
                    f"loc=({loc.x:.4f},{loc.y:.4f},{loc.z:.4f}) "
                    f"rot=({rot.w:.4f},{rot.x:.4f},{rot.y:.4f},{rot.z:.4f}) "
                    f"sca=({sca.x:.4f},{sca.y:.4f},{sca.z:.4f})"
                )


def _gf_runtime_ensure_cache(
    ctx: bpy.types.Context, arm_obj: bpy.types.Object
) -> Optional[Dict[str, object]]:
                                                                                                   
    ptr = int(arm_obj.as_pointer())
    existing = _GF_RUNTIME_CACHE.get(ptr)
    if existing:
        return existing
    path = arm_obj.get("gfmodel_source_path")
    if not isinstance(path, str) or not path:
        return None
    try:
        with open(path, "rb") as f:
            data = f.read()
    except Exception:
        return None

    axis_forward = arm_obj.get("gfmodel_axis_forward", "-Z")
    axis_up = arm_obj.get("gfmodel_axis_up", "Y")
    try:
        global_scale = float(arm_obj.get("gfmodel_global_scale", 1.0))
    except Exception:
        global_scale = 1.0

    conv = axis_conversion(
        from_forward=str(axis_forward),
        from_up=str(axis_up),
        to_forward="-Y",
        to_up="Z",
    ).to_4x4()
    conv3 = conv.to_3x3()

    models, _textures, motions, _shaders = _load_any(data)
    if not models or not motions:
        return None
    model = models[0]
    rest_by_name = {b.name: b for b in model.skeleton}
    bone_by_name = rest_by_name

    depth_cache: Dict[str, int] = {}

    def bone_depth(name: str) -> int:
        if name in depth_cache:
            return depth_cache[name]
        b = bone_by_name.get(name)
        if b is None or not b.parent:
            depth_cache[name] = 0
            return 0
        depth_cache[name] = 1 + bone_depth(b.parent)
        return depth_cache[name]

    bone_order = sorted([b.name for b in model.skeleton], key=bone_depth)

                                                                                   
    rest_abs_by_name: Dict[str, Matrix] = {}
    rest_rel_by_name: Dict[str, Matrix] = {}
    for pb in arm_obj.pose.bones:
        rest_abs_by_name[pb.name] = pb.bone.matrix_local.copy()
    for name in bone_order:
        rb = rest_by_name.get(name)
        rest_abs = rest_abs_by_name.get(name)
        if rb is None or rest_abs is None:
            continue
        if rb.parent and rb.parent in rest_abs_by_name:
            parent_rest_abs = rest_abs_by_name[rb.parent]
            rest_rel_by_name[name] = parent_rest_abs.inverted() @ rest_abs
        else:
            rest_rel_by_name[name] = rest_abs

    _gf_runtime_cache_armature(
        arm_obj,
        model=model,
        motions=motions,
        conv=conv,
        conv3=conv3,
        global_scale=global_scale,
        bone_order=bone_order,
        rest_by_name=rest_by_name,
        rest_rel_by_name=rest_rel_by_name,
    )
    if bool(getattr(ctx.scene, "gfmodel_debug_animations", False)):
        print(f"[GFModel] Rebuilt runtime cache for {arm_obj.name} from {path}")
    return _GF_RUNTIME_CACHE.get(ptr)


@persistent
def _gfmodel_frame_change_post(scene: bpy.types.Scene, _depsgraph=None) -> None:
    ctx = bpy.context
    for obj in scene.objects:
        if obj.type != "ARMATURE":
            continue
        if not bool(getattr(obj, "gfmodel_runtime_enabled", False)):
            continue
        mot_i = int(getattr(obj, "gfmodel_runtime_motion_index", 0))
        _gf_runtime_apply_pose(ctx, obj, int(scene.frame_current), motion_index=mot_i)


class VIEW3D_PT_gfmodel(bpy.types.Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "GFModel"
    bl_label = "GFModel"

    def draw(self, context: bpy.types.Context):
        layout = self.layout
        enabled = bool(getattr(context.scene, "gfmodel_uv_anims_enabled", False))
        layout.prop(
            context.scene, "gfmodel_uv_anims_enabled", text="Enable UV Animations"
        )
        layout.prop(
            context.scene,
            "gfmodel_vis_anims_enabled",
            text="Enable Visibility Animations",
        )
        obj = context.object
        if (
            obj is not None
            and obj.type == "ARMATURE"
            and obj.get("gfmodel_has_runtime_data")
        ):
            box = layout.box()
            box.label(text="Runtime (SSC-accurate)")
            box.prop(obj, "gfmodel_runtime_enabled", text="Enable Runtime Motion")
            row = box.row(align=True)
            row.prop(obj, "gfmodel_runtime_motion_index", text="Motion")
            row.prop(obj, "gfmodel_runtime_loop", text="Loop")
        layout.operator("gfmodel.dump_action_json", text="Dump Action JSON")
        layout.operator("gfmodel.dump_eval_json", text="Dump Eval JSON")
        layout.operator("gfmodel.dump_vertices_json", text="Dump Vertices JSON")
        layout.operator("gfmodel.dump_rest_compare_json", text="Dump Rest Compare JSON")
        col = layout.column(align=True)
        col.prop(context.scene, "gfmodel_debug_animations", text="Debug Animations")
        if getattr(context.scene, "gfmodel_debug_animations", False):
            col.prop(context.scene, "gfmodel_debug_bone", text="Debug Bone")
            col.prop(context.scene, "gfmodel_debug_motion", text="Debug Motion")
        layout.label(
            text="(Imported UV strips are muted by default)"
            if not enabled
            else "(UV strips unmuted)"
        )


def register() -> None:
    bpy.utils.register_class(VIEW3D_PT_gfmodel)
    bpy.types.Scene.gfmodel_uv_anims_enabled = bpy.props.BoolProperty(
        name="Enable UV Animations",
        default=False,
        update=_uv_toggle_update,
    )
    bpy.types.Scene.gfmodel_vis_anims_enabled = bpy.props.BoolProperty(
        name="Enable Visibility Animations",
        default=False,
        update=_vis_toggle_update,
    )
    bpy.types.Scene.gfmodel_debug_animations = bpy.props.BoolProperty(
        name="GFModel Debug Animations",
        default=False,
    )
    bpy.types.Scene.gfmodel_debug_bone = bpy.props.StringProperty(
        name="GFModel Debug Bone",
        default="Waist",
    )
    bpy.types.Scene.gfmodel_debug_motion = bpy.props.IntProperty(
        name="GFModel Debug Motion Index",
        default=-1,
    )
    bpy.types.Object.gfmodel_runtime_enabled = bpy.props.BoolProperty(
        name="Enable Runtime Motion",
        default=False,
        description="Apply motions by directly setting pose matrices each frame (more accurate than baked keyframes)",
        update=_runtime_toggle_update,
    )
    bpy.types.Object.gfmodel_runtime_motion_index = bpy.props.IntProperty(
        name="Runtime Motion Index",
        default=0,
        min=0,
        update=_runtime_motion_update,
    )
    bpy.types.Object.gfmodel_runtime_loop = bpy.props.BoolProperty(
        name="Loop Runtime Motion",
        default=True,
    )
    if _gfmodel_frame_change_post not in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.append(_gfmodel_frame_change_post)


def unregister() -> None:
    if hasattr(bpy.types.Scene, "gfmodel_uv_anims_enabled"):
        del bpy.types.Scene.gfmodel_uv_anims_enabled
    if hasattr(bpy.types.Scene, "gfmodel_vis_anims_enabled"):
        del bpy.types.Scene.gfmodel_vis_anims_enabled
    if hasattr(bpy.types.Scene, "gfmodel_debug_animations"):
        del bpy.types.Scene.gfmodel_debug_animations
    if hasattr(bpy.types.Scene, "gfmodel_debug_bone"):
        del bpy.types.Scene.gfmodel_debug_bone
    if hasattr(bpy.types.Scene, "gfmodel_debug_motion"):
        del bpy.types.Scene.gfmodel_debug_motion
    if hasattr(bpy.types.Object, "gfmodel_runtime_enabled"):
        del bpy.types.Object.gfmodel_runtime_enabled
    if hasattr(bpy.types.Object, "gfmodel_runtime_motion_index"):
        del bpy.types.Object.gfmodel_runtime_motion_index
    if hasattr(bpy.types.Object, "gfmodel_runtime_loop"):
        del bpy.types.Object.gfmodel_runtime_loop
    try:
        bpy.app.handlers.frame_change_post.remove(_gfmodel_frame_change_post)
    except Exception:
        pass
    bpy.utils.unregister_class(VIEW3D_PT_gfmodel)


def _apply_uv_anim_enable(scene: bpy.types.Scene) -> None:
    enabled = bool(getattr(scene, "gfmodel_uv_anims_enabled", False))
    toggled = 0
    for mat in bpy.data.materials:
        if not mat.get("gfmodel_has_uv_anims"):
            continue
        nt = getattr(mat, "node_tree", None)
        if nt is None:
            continue
        nt.animation_data_create()
        action_name = mat.get("gfmodel_uv_action")
        if enabled and isinstance(action_name, str) and action_name:
            nt.animation_data.action = bpy.data.actions.get(action_name)
        else:
            nt.animation_data.action = None
        toggled += 1
                               
    try:
        bpy.context.view_layer.update()
    except Exception:
        pass
    if bool(getattr(scene, "gfmodel_debug_animations", False)):
        print(f"[GFModel] UV toggle={enabled} node_trees={toggled}")


def _uv_toggle_update(scene: bpy.types.Scene, _context: bpy.types.Context) -> None:
    _apply_uv_anim_enable(scene)


def _apply_visibility_anim_enable(scene: bpy.types.Scene) -> None:
    enabled = bool(getattr(scene, "gfmodel_vis_anims_enabled", False))
    toggled = 0
    for obj in bpy.data.objects:
        if not obj.get("gfmodel_has_vis_anims"):
            continue
        obj.animation_data_create()
        action_name = obj.get("gfmodel_vis_action")
        if enabled and isinstance(action_name, str) and action_name:
            obj.animation_data.action = bpy.data.actions.get(action_name)
        else:
            obj.animation_data.action = None
        toggled += 1
    try:
        bpy.context.view_layer.update()
    except Exception:
        pass
    if bool(getattr(scene, "gfmodel_debug_animations", False)):
        print(f"[GFModel] Visibility toggle={enabled} objects={toggled}")


def _vis_toggle_update(scene: bpy.types.Scene, _context: bpy.types.Context) -> None:
    _apply_visibility_anim_enable(scene)


def _runtime_toggle_update(obj: bpy.types.Object, context: bpy.types.Context) -> None:
    if obj.type != "ARMATURE":
        return
    cache = _GF_RUNTIME_CACHE.get(int(obj.as_pointer()))
    if cache is not None:
        cache["last_frame"] = None
        cache["last_motion"] = None
    if not bool(getattr(obj, "gfmodel_runtime_enabled", False)):
                                                        
        for pb in obj.pose.bones:
            pb.location = (0.0, 0.0, 0.0)
            pb.rotation_mode = "QUATERNION"
            pb.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
            pb.scale = (1.0, 1.0, 1.0)
        return
    try:
        mi = int(getattr(obj, "gfmodel_runtime_motion_index", 0))
        _gf_runtime_apply_pose(
            context, obj, int(context.scene.frame_current), motion_index=mi
        )
    except Exception:
        pass


def _runtime_motion_update(obj: bpy.types.Object, context: bpy.types.Context) -> None:
    if not bool(getattr(obj, "gfmodel_runtime_enabled", False)):
        return
    try:
        mi = int(getattr(obj, "gfmodel_runtime_motion_index", 0))
        _gf_runtime_apply_pose(
            context, obj, int(context.scene.frame_current), motion_index=mi
        )
    except Exception:
        pass


def _mat4_to_list(m: Matrix) -> List[List[float]]:
    return [[float(m[r][c]) for c in range(4)] for r in range(4)]


def _compute_rest_world_mats(
    model: _GFModel, conv: Matrix, global_scale: float, *, ssc: bool
) -> Dict[str, Matrix]:
    conv3 = conv.to_3x3()
    rest_world: Dict[str, Matrix] = {}
    by_name = {b.name: b for b in model.skeleton}

    def world_for(name: str) -> Matrix:
        if name in rest_world:
            return rest_world[name]
        b = by_name.get(name)
        if b is None:
            rest_world[name] = Matrix.Identity(4)
            return rest_world[name]
        if b.parent and b.parent in by_name:
            parent = by_name[b.parent]
            ps = parent.scale
            if ssc and _bone_uses_ssc(int(b.flags)):
                inv_ps = Vector(
                    (
                        1.0 / ps.x if ps.x != 0 else 0.0,
                        1.0 / ps.y if ps.y != 0 else 0.0,
                        1.0 / ps.z if ps.z != 0 else 0.0,
                    )
                )
                t_scaled = Vector(
                    (
                        b.translation.x * ps.x,
                        b.translation.y * ps.y,
                        b.translation.z * ps.z,
                    )
                )
                inv_s_parent = Matrix.Diagonal(
                    Vector((inv_ps.x, inv_ps.y, inv_ps.z, 1.0))
                )
            else:
                t_scaled = b.translation
                inv_s_parent = None

            t = Matrix.Translation(conv @ (t_scaled * global_scale))
            q = _transform_quat_basis(_euler_to_quat_xyz(b.rotation), conv3)
            r = q.to_matrix().to_4x4()
            s = Matrix.Diagonal(Vector((b.scale.x, b.scale.y, b.scale.z, 1.0)))
            if inv_s_parent is not None:
                mw = world_for(b.parent) @ inv_s_parent @ (t @ r @ s)
            else:
                mw = world_for(b.parent) @ (t @ r @ s)
        else:
            t = Matrix.Translation(conv @ (b.translation * global_scale))
            q = _transform_quat_basis(_euler_to_quat_xyz(b.rotation), conv3)
            r = q.to_matrix().to_4x4()
            s = Matrix.Diagonal(Vector((b.scale.x, b.scale.y, b.scale.z, 1.0)))
            mw = t @ r @ s
        rest_world[name] = mw
        return mw

    for b in model.skeleton:
        world_for(b.name)
    return rest_world


def _eval_motion_pose_world_mats(
    model: _GFModel,
    mot: _GFMotion,
    conv: Matrix,
    global_scale: float,
    frame: int,
) -> Dict[str, Matrix]:
    conv3 = conv.to_3x3()
    rest_by_name = {b.name: b for b in model.skeleton}
    bt_by_name = {bt.name: bt for bt in mot.bones}
    pose_world: Dict[str, Matrix] = {}

    def pose_for(name: str) -> Matrix:
        if name in pose_world:
            return pose_world[name]
        rb = rest_by_name.get(name)
        if rb is None:
            pose_world[name] = Matrix.Identity(4)
            return pose_world[name]
        bt = bt_by_name.get(name)

        sx = _mot_eval(bt.sx, frame, rb.scale.x) if bt else rb.scale.x
        sy = _mot_eval(bt.sy, frame, rb.scale.y) if bt else rb.scale.y
        sz = _mot_eval(bt.sz, frame, rb.scale.z) if bt else rb.scale.z

        tx = _mot_eval(bt.tx, frame, rb.translation.x) if bt else rb.translation.x
        ty = _mot_eval(bt.ty, frame, rb.translation.y) if bt else rb.translation.y
        tz = _mot_eval(bt.tz, frame, rb.translation.z) if bt else rb.translation.z

        rx = _mot_eval(bt.rx, frame, rb.rotation.x) if bt else rb.rotation.x
        ry = _mot_eval(bt.ry, frame, rb.rotation.y) if bt else rb.rotation.y
        rz = _mot_eval(bt.rz, frame, rb.rotation.z) if bt else rb.rotation.z

        if bt and bt.is_axis_angle:
            axis = Vector((rx, ry, rz))
            angle = axis.length * 2.0
            if angle > 0:
                q_cur = Quaternion(axis.normalized(), angle)
            else:
                q_cur = Quaternion((1.0, 0.0, 0.0, 0.0))
        else:
            q_cur = _euler_to_quat_xyz(Vector((rx, ry, rz)))

        q_cur_t = _transform_quat_basis(q_cur, conv3)
        r_anim = q_cur_t.to_matrix().to_4x4()
        s_anim = Matrix.Diagonal(Vector((sx, sy, sz, 1.0)))

        if rb.parent and rb.parent in rest_by_name:
            prb = rest_by_name[rb.parent]
            pbt = bt_by_name.get(rb.parent)
            psx = _mot_eval(pbt.sx, frame, prb.scale.x) if pbt else prb.scale.x
            psy = _mot_eval(pbt.sy, frame, prb.scale.y) if pbt else prb.scale.y
            psz = _mot_eval(pbt.sz, frame, prb.scale.z) if pbt else prb.scale.z

            if _bone_uses_ssc(int(rb.flags)):
                inv_psx = 1.0 / psx if psx != 0 else 0.0
                inv_psy = 1.0 / psy if psy != 0 else 0.0
                inv_psz = 1.0 / psz if psz != 0 else 0.0
                inv_s_parent = Matrix.Diagonal(Vector((inv_psx, inv_psy, inv_psz, 1.0)))

                t_scaled = Vector((tx * psx, ty * psy, tz * psz))
                t_anim = Matrix.Translation(conv @ (t_scaled * global_scale))
                pose = pose_for(rb.parent) @ inv_s_parent @ (t_anim @ r_anim @ s_anim)
            else:
                t_anim = Matrix.Translation(
                    conv @ (Vector((tx, ty, tz)) * global_scale)
                )
                pose = pose_for(rb.parent) @ (t_anim @ r_anim @ s_anim)
        else:
            t_anim = Matrix.Translation(conv @ (Vector((tx, ty, tz)) * global_scale))
            pose = t_anim @ r_anim @ s_anim

        pose_world[name] = pose
        return pose

    for b in model.skeleton:
        pose_for(b.name)
    return pose_world
