"""Implementation of `EXPORT_SCENE_OT_gfmodel.execute`.

Separated to keep operator class definitions small.
"""

from __future__ import annotations

import json
import os
import struct
from typing import Dict, List, Optional, Tuple

import bpy
from bpy_extras.io_utils import axis_conversion
from mathutils import Matrix

from ...core.export import (
    parse_gf_model_pack_entries,
    write_gf_model_blob,
    write_gf_model_pack,
    write_gf_texture_blob,
)
from ...core.gfpack import parse_gf_model_pack
from ...core.gfpack import write_gf_model_pack as write_gf_model_pack_low
from ...core.io import _load_any, _parse_gf_texture
from .archive_patch_apply import patch_into_source_archive
from .grow_buffers_patch import (
    _apply_uv_strategy_to_mesh,
    _build_temp_mesh_object_for_slot,
    _collect_tris_all,
    _collect_tris_for_material,
    _material_routing_candidates,
    _patch_pack_grow_buffers_tris,
    _route_source_object_to_submesh_slots,
)
from .inplace_patch import (
    _patch_pack_indices_in_place,
    _patch_pack_normals_in_place,
    _patch_pack_pos_nrm_uv0_skin_in_place,
    _patch_pack_positions_in_place,
    _patch_pack_positions_normals_in_place,
    _patch_pack_positions_normals_uv0_in_place,
    _patch_pack_topology_tris_in_place,
    _patch_pack_uv0_in_place,
    _patch_pack_verts_topology_tris_in_place,
)
from .pack_wrappers import rewrap_pack, unwrap_pack
from .textures_patch import (
    _collect_texture_overrides_by_slot,
    _encode_texture_from_image,
    _patch_pack_textures_rgba8,
)


def export_scene_execute(self, context: bpy.types.Context):
    src_path = str(context.scene.get("gfmodel_last_import_path", ""))
    if not src_path:
        self.report({"ERROR"}, "No last import path stored; import a GFModel first")
        return {"CANCELLED"}
    try:
        with open(src_path, "rb") as f:
            src_data = f.read()
    except Exception as e:
        self.report({"ERROR"}, f"Failed to read: {src_path} ({e})")
        return {"CANCELLED"}

    if self.export_meshes and self.mesh_export_mode == "PRESERVE_SOURCE_BYTES":
        out_bytes = bytes(src_data)
        os.makedirs(os.path.dirname(self.filepath) or ".", exist_ok=True)
        with open(self.filepath, "wb") as f:
            f.write(out_bytes)
        try:
            if self.remember_last_export_settings:
                context.scene["gfmodel_last_export_path"] = str(self.filepath)
                context.scene["gfmodel_last_export_export_meshes"] = bool(
                    self.export_meshes
                )
                context.scene["gfmodel_last_export_mesh_export_mode"] = str(
                    self.mesh_export_mode
                )
        except Exception:
            pass

        if self.patch_into_source_archive:
            try:
                out_archive = patch_into_source_archive(context, out_bytes)
                self.report({"INFO"}, f"Patched archive: {out_archive}")
            except Exception as e:
                self.report({"ERROR"}, f"Export succeeded, but patch failed: {e}")

        self.report({"INFO"}, f"Wrote {self.filepath}")
        return {"FINISHED"}

    models, textures, motions, shaders = _load_any(src_data)
    if not models:
        self.report({"ERROR"}, "No models found in source file")
        return {"CANCELLED"}

    model = models[0]

    tagged: Dict[int, bpy.types.Object] = {}
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        if obj.get("gfmodel_model_name") != model.name:
            continue
        smi = obj.get("gfmodel_submesh_index")
        if isinstance(smi, int):
            tagged[int(smi)] = obj

    if tagged:
        try:
            for obj in tagged.values():
                if getattr(obj, "mode", "") == "EDIT":
                    obj.update_from_editmode()
            context.view_layer.update()
        except Exception:
            pass
    if self.export_meshes and self.mesh_export_mode == "REBUILD_FROM_SCENE":
        if not tagged:
            self.report(
                {"WARNING"},
                "No tagged meshes found (gfmodel_submesh_index); REBUILD_FROM_SCENE will preserve original buffers",
            )
        else:
            self.report({"INFO"}, f"Tagged meshes: {len(tagged)}")

    if self.export_meshes and self.mesh_export_mode in (
        "UPDATE_POSITIONS_IN_PLACE",
        "UPDATE_POS_NRM_IN_PLACE",
        "UPDATE_POS_NRM_UV0_IN_PLACE",
        "UPDATE_POS_NRM_UV0_SKIN_IN_PLACE",
        "UPDATE_NORMALS_IN_PLACE",
        "UPDATE_UV0_IN_PLACE",
        "UPDATE_INDICES_IN_PLACE",
        "UPDATE_TOPOLOGY_TRIS_IN_PLACE",
        "UPDATE_VERTS_TOPOLOGY_TRIS_IN_PLACE",
        "GROW_BUFFERS_TRIS",
    ):
        try:
            pack_src, pack_wrap = unwrap_pack(src_data)
        except Exception as e:
            self.report(
                {"ERROR"},
                f"In-place patching requires RAW_PACK/CM/CP->CM source: {e}",
            )
            return {"CANCELLED"}

        axis_forward = str(context.scene.get("gfmodel_last_axis_forward", "-Z"))
        axis_up = str(context.scene.get("gfmodel_last_axis_up", "Y"))
        global_scale = float(context.scene.get("gfmodel_last_global_scale", 1.0))
        conv = axis_conversion(
            from_forward=axis_forward, from_up=axis_up, to_forward="-Y", to_up="Z"
        ).to_4x4()
        gf_from_blender = conv.inverted()

        skeleton_names = [b.name for b in model.skeleton]

        pack = parse_gf_model_pack(pack_src)
        model_slot = None
        for i in range(int(pack.counts[0])):
            e = pack.get(0, i)
            if e is not None:
                model_slot = int(i)
                break
        if model_slot is None:
            self.report({"ERROR"}, "Model section is empty in GFModelPack")
            return {"CANCELLED"}
        try:
            pos_changed = None
            nrm_changed = None
            uv_changed = None
            skin_changed = None
            skin_fallback = None
            if self.mesh_export_mode == "UPDATE_POSITIONS_IN_PLACE":
                out_bytes, changed = _patch_pack_positions_in_place(
                    pack_src,
                    model,
                    tagged=tagged,
                    gf_from_blender=gf_from_blender,
                    global_scale=float(global_scale),
                )
            elif self.mesh_export_mode == "UPDATE_POS_NRM_IN_PLACE":
                out_bytes, pos_changed, nrm_changed = (
                    _patch_pack_positions_normals_in_place(
                        pack_src,
                        model,
                        tagged=tagged,
                        gf_from_blender=gf_from_blender,
                        global_scale=float(global_scale),
                    )
                )
                changed = int(pos_changed)
            elif self.mesh_export_mode == "UPDATE_POS_NRM_UV0_IN_PLACE":
                out_bytes, pos_changed, nrm_changed, uv_changed = (
                    _patch_pack_positions_normals_uv0_in_place(
                        pack_src,
                        model,
                        tagged=tagged,
                        gf_from_blender=gf_from_blender,
                        global_scale=float(global_scale),
                    )
                )
                changed = int(pos_changed)
            elif self.mesh_export_mode == "UPDATE_POS_NRM_UV0_SKIN_IN_PLACE":
                (
                    out_bytes,
                    pos_changed,
                    nrm_changed,
                    uv_changed,
                    skin_changed,
                    skin_fallback,
                ) = _patch_pack_pos_nrm_uv0_skin_in_place(
                    pack_src,
                    model,
                    tagged=tagged,
                    gf_from_blender=gf_from_blender,
                    global_scale=float(global_scale),
                    skeleton_names=skeleton_names,
                )
                changed = int(pos_changed)
            elif self.mesh_export_mode == "UPDATE_NORMALS_IN_PLACE":
                out_bytes, changed = _patch_pack_normals_in_place(
                    pack_src,
                    model,
                    tagged=tagged,
                    gf_from_blender=gf_from_blender,
                )
            elif self.mesh_export_mode == "UPDATE_INDICES_IN_PLACE":
                out_bytes, changed = _patch_pack_indices_in_place(
                    pack_src,
                    model,
                    tagged=tagged,
                )
            elif self.mesh_export_mode == "UPDATE_TOPOLOGY_TRIS_IN_PLACE":
                out_bytes, changed = _patch_pack_topology_tris_in_place(
                    pack_src,
                    model,
                    tagged=tagged,
                )
            elif self.mesh_export_mode == "UPDATE_VERTS_TOPOLOGY_TRIS_IN_PLACE":
                out_bytes, changed = _patch_pack_verts_topology_tris_in_place(
                    pack_src,
                    model,
                    tagged=tagged,
                    gf_from_blender=gf_from_blender,
                    global_scale=float(global_scale),
                    skeleton_names=skeleton_names,
                )
            elif self.mesh_export_mode == "GROW_BUFFERS_TRIS":
                active_obj = context.active_object
                if (
                    active_obj is None
                    or active_obj.type != "MESH"
                    or active_obj.get("gfmodel_model_name") != model.name
                    or active_obj.get("gfmodel_submesh_index") is None
                ):
                    raise ValueError(
                        "Select an imported GFModel mesh object (active) before using Grow Buffers"
                    )

                uv_strategy = str(
                    getattr(self, "grow_buffers_uv_strategy", "DUPLICATE")
                    or "DUPLICATE"
                )
                rebuild_mode = str(
                    getattr(self, "grow_buffers_rebuild_mode", "") or ""
                ).strip()
                if not rebuild_mode:
                    legacy = bool(
                        getattr(self, "grow_buffers_rebuild_submeshes", False)
                    )
                    rebuild_mode = "REBUILD_SPLIT" if legacy else "CLAMP_ROUTE"
                if rebuild_mode not in (
                    "CLAMP_ROUTE",
                    "REBUILD_PALETTE",
                    "REBUILD_SPLIT",
                ):
                    rebuild_mode = "CLAMP_ROUTE"
                allow_palette_split = rebuild_mode == "REBUILD_SPLIT"
                disallow_new_mesh_sections = rebuild_mode != "REBUILD_SPLIT"
                allow_palette_rebuild = rebuild_mode != "CLAMP_ROUTE"
                clamp_conflict_mode = str(
                    getattr(self, "grow_buffers_clamp_conflict_mode", "") or ""
                ).strip()
                if clamp_conflict_mode not in (
                    "CLAMP_BY_WEIGHT",
                    "CLAMP_BY_NEIGHBORS",
                    "DROP_CONFLICTS",
                ):
                    clamp_conflict_mode = "CLAMP_BY_WEIGHT"

                mapping_raw = str(self.grow_buffers_material_sources_json or "").strip()
                if mapping_raw:
                    try:
                        mat_to_obj = json.loads(mapping_raw)
                    except Exception as e:
                        raise ValueError(f"Invalid material source mapping JSON: {e}")
                    if not isinstance(mat_to_obj, dict):
                        raise ValueError(
                            "Invalid material source mapping (expected object)"
                        )

                    tmp_objects: List[bpy.types.Object] = []
                    try:
                        tagged_multi: Dict[int, bpy.types.Object] = {}
                        total_dropped = 0
                        total_changed_slots = 0
                        for mat_name, obj_name in mat_to_obj.items():
                            if not obj_name:
                                continue
                            src_obj = bpy.data.objects.get(str(obj_name))
                            if src_obj is None:
                                raise ValueError(
                                    f"Source object {obj_name!r} not found for material {mat_name!r}"
                                )

                            if rebuild_mode in ("REBUILD_PALETTE", "REBUILD_SPLIT"):
                                tris = _collect_tris_for_material(
                                    src_obj, material_name=str(mat_name)
                                )
                                if not tris:
                                    self.report(
                                        {"WARNING"},
                                        f"{mat_name}: source object has 0 triangles for this material",
                                    )
                                    continue
                                candidates = _material_routing_candidates(
                                    model, material_name=str(mat_name)
                                )
                                if not candidates:
                                    self.report(
                                        {"WARNING"},
                                        f"{mat_name}: no target submesh slots found for this material",
                                    )
                                    continue
                                si0 = int(sorted(candidates)[0])
                                tmp = _build_temp_mesh_object_for_slot(
                                    name=f"__gf_tmp_{model.name}_{mat_name}_{si0}",
                                    src_obj=src_obj,
                                    src_mesh=src_obj.data,                          
                                    tri_verts=tris,
                                )
                                _apply_uv_strategy_to_mesh(
                                    tmp.data,                          
                                    strategy=uv_strategy,
                                )
                                tmp_objects.append(tmp)
                                try:
                                    context.scene.collection.objects.link(tmp)
                                except Exception:
                                    pass
                                tagged_multi[int(si0)] = tmp
                                total_changed_slots += 1
                                continue

                            tris_by_si, weights_override_by_si, stats, extra = (
                                _route_source_object_to_submesh_slots(
                                    model,
                                    material_name=str(mat_name),
                                    src_obj=src_obj,
                                    skeleton_names=skeleton_names,
                                    routing_strategy=str(
                                        self.grow_buffers_routing_strategy
                                    ),
                                    weight_cutoff=float(
                                        getattr(self, "grow_buffers_weight_cutoff", 0.0)
                                    ),
                                    conflict_mode=str(clamp_conflict_mode),
                                )
                            )
                            total_dropped += int(stats.get("dropped", 0))
                            total_routed = sum(len(v) for v in tris_by_si.values())
                            if total_routed == 0:
                                self.report(
                                    {"WARNING"},
                                    f"{mat_name}: routed 0 triangles (check source object materials and weights)",
                                )
                            clamped_count = int(stats.get("clamped", 0) or 0)
                            if clamped_count > 0:
                                self.report(
                                    {"WARNING"},
                                    f"{mat_name}: clamped {clamped_count} triangle(s) to a single palette (weight trimming)",
                                )

                            for si, tris in tris_by_si.items():
                                if not tris:
                                    continue
                                tmp = _build_temp_mesh_object_for_slot(
                                    name=f"__gf_tmp_{model.name}_{mat_name}_{si}",
                                    src_obj=src_obj,
                                    src_mesh=src_obj.data,                          
                                    tri_verts=tris,
                                    weights_override_by_src_vi=weights_override_by_si.get(
                                        int(si)
                                    ),
                                )
                                _apply_uv_strategy_to_mesh(
                                    tmp.data,                          
                                    strategy=uv_strategy,
                                )
                                tmp_objects.append(tmp)
                                try:
                                    context.scene.collection.objects.link(tmp)
                                except Exception:
                                    pass
                                tagged_multi[int(si)] = tmp
                                total_changed_slots += 1

                            if extra.get("unknown_bones"):
                                self.report(
                                    {"WARNING"},
                                    f"{mat_name}: unknown bones (not in skeleton); dropped triangles (sample: {', '.join(extra['unknown_bones'][:8])})",
                                )
                            if extra.get("dropped_unknown_bones"):
                                counts = (
                                    extra.get("dropped_unknown_bones_counts", {}) or {}
                                )
                                top = sorted(
                                    ((str(k), int(v)) for k, v in counts.items()),
                                    key=lambda t: (-int(t[1]), str(t[0])),
                                )[:12]
                                s = ", ".join(f"{k}({v})" for k, v in top)
                                self.report(
                                    {"WARNING"},
                                    f"{mat_name}: dropped triangles due to unknown bones: {s}",
                                )
                            if extra.get("dropped_missing_bones"):
                                counts = (
                                    extra.get("dropped_missing_bones_counts", {}) or {}
                                )
                                top = sorted(
                                    ((str(k), int(v)) for k, v in counts.items()),
                                    key=lambda t: (-int(t[1]), str(t[0])),
                                )[:12]
                                s = ", ".join(f"{k}({v})" for k, v in top)
                                self.report(
                                    {"WARNING"},
                                    f"{mat_name}: dropped triangles (bones not in any {mat_name} palette): {s}",
                                )
                            if extra.get("dropped_conflict_pairs"):
                                counts = (
                                    extra.get("dropped_conflict_pairs_counts", {}) or {}
                                )
                                top = sorted(
                                    ((str(k), int(v)) for k, v in counts.items()),
                                    key=lambda t: (-int(t[1]), str(t[0])),
                                )[:12]
                                s = ", ".join(f"{k}({v})" for k, v in top)
                                verb = (
                                    "dropped"
                                    if str(clamp_conflict_mode) == "DROP_CONFLICTS"
                                    else "clamped"
                                )
                                self.report(
                                    {"WARNING"},
                                    f"{mat_name}: {verb} triangles that spanned palettes (pairs): {s}",
                                )

                        if not tagged_multi:
                            self.report(
                                {"WARNING"},
                                "Material routing enabled, but routed 0 triangles; leaving meshes unchanged",
                            )
                            out_bytes = bytes(pack_src)
                            changed = 0
                        else:
                            out_bytes, changed = _patch_pack_grow_buffers_tris(
                                pack_src,
                                model,
                                tagged=tagged_multi,
                                gf_from_blender=gf_from_blender,
                                global_scale=float(global_scale),
                                skeleton_names=skeleton_names,
                                disallow_new_mesh_sections=bool(
                                    disallow_new_mesh_sections
                                ),
                                allow_palette_rebuild=bool(allow_palette_rebuild),
                                allow_palette_split=bool(allow_palette_split),
                            )
                        if total_dropped > 0:
                            self.report(
                                {"WARNING"},
                                f"Material routing: dropped triangles total={int(total_dropped)}",
                            )
                    finally:
                        for o in tmp_objects:
                            try:
                                if o.name in context.scene.collection.objects:
                                    context.scene.collection.objects.unlink(o)
                            except Exception:
                                pass
                            try:
                                m = o.data
                            except Exception:
                                m = None
                            try:
                                bpy.data.objects.remove(o, do_unlink=True)
                            except Exception:
                                pass
                            if m is not None:
                                try:
                                    bpy.data.meshes.remove(m, do_unlink=True)
                                except Exception:
                                    pass
                else:
                    active_si = int(active_obj.get("gfmodel_submesh_index"))
                    tmp_objects: List[bpy.types.Object] = []
                    try:
                        if uv_strategy != "DUPLICATE":
                            tris = _collect_tris_all(active_obj)
                            tmp = _build_temp_mesh_object_for_slot(
                                name=f"__gf_tmp_{model.name}_uv_{active_si}",
                                src_obj=active_obj,
                                src_mesh=active_obj.data,                          
                                tri_verts=tris,
                            )
                            _apply_uv_strategy_to_mesh(
                                tmp.data,                          
                                strategy=uv_strategy,
                            )
                            tmp_objects.append(tmp)
                            try:
                                context.scene.collection.objects.link(tmp)
                            except Exception:
                                pass
                            tagged_active = {active_si: tmp}
                        else:
                            tagged_active = {active_si: active_obj}
                        out_bytes, changed = _patch_pack_grow_buffers_tris(
                            pack_src,
                            model,
                            tagged=tagged_active,
                            gf_from_blender=gf_from_blender,
                            global_scale=float(global_scale),
                            skeleton_names=skeleton_names,
                            disallow_new_mesh_sections=bool(disallow_new_mesh_sections),
                            allow_palette_rebuild=bool(allow_palette_rebuild),
                            allow_palette_split=bool(allow_palette_split),
                        )
                    finally:
                        for o in tmp_objects:
                            try:
                                if o.name in context.scene.collection.objects:
                                    context.scene.collection.objects.unlink(o)
                            except Exception:
                                pass
                            try:
                                m = o.data
                            except Exception:
                                m = None
                            try:
                                bpy.data.objects.remove(o, do_unlink=True)
                            except Exception:
                                pass
                            if m is not None:
                                try:
                                    bpy.data.meshes.remove(m, do_unlink=True)
                                except Exception:
                                    pass
            else:
                out_bytes, changed = _patch_pack_uv0_in_place(
                    src_data,
                    model,
                    tagged=tagged,
                )
        except Exception as e:
            self.report({"ERROR"}, f"In-place patch failed: {e}")
            return {"CANCELLED"}

        if self.export_textures and str(self.texture_mode) in (
            "RGBA8",
            "RGBA8_SAME_SIZE",
            "RGBA8_ORIGINAL_SIZE",
            "ORIGINAL_FORMAT",
            "OVERRIDE_FORMAT",
        ):
            try:
                overrides = _collect_texture_overrides_by_slot(model)
                out_bytes, tex_changed = _patch_pack_textures_rgba8(
                    out_bytes,
                    overrides=overrides,
                    texture_mode=str(self.texture_mode),
                    texture_override_format=str(
                        getattr(self, "texture_override_format", "RGBA8") or "RGBA8"
                    ),
                    texture_max_size=int(self.texture_max_size),
                )
                if tex_changed > 0:
                    self.report(
                        {"INFO"},
                        f"Patched textures: {int(tex_changed)} (RGBA8 overwrite)",
                    )
            except Exception as e:
                self.report({"ERROR"}, f"Texture export failed: {e}")
                return {"CANCELLED"}

        if changed == 0:
            self.report(
                {"INFO"},
                "In-place patch: no bytes changed (output should be identical)",
            )
        else:
            if (
                self.mesh_export_mode == "UPDATE_POS_NRM_IN_PLACE"
                and nrm_changed is not None
            ):
                self.report(
                    {"INFO"},
                    f"In-place patch: updated vertices pos={int(changed)} nrm={int(nrm_changed)}",
                )
            elif (
                self.mesh_export_mode == "UPDATE_POS_NRM_UV0_IN_PLACE"
                and nrm_changed is not None
                and uv_changed is not None
            ):
                self.report(
                    {"INFO"},
                    f"In-place patch: updated pos={int(changed)} nrm={int(nrm_changed)} uv0={int(uv_changed)}",
                )
            elif (
                self.mesh_export_mode == "UPDATE_POS_NRM_UV0_SKIN_IN_PLACE"
                and nrm_changed is not None
                and uv_changed is not None
                and skin_changed is not None
            ):
                if skin_fallback is not None and int(skin_fallback) > 0:
                    self.report(
                        {"WARNING"},
                        f"In-place patch: updated pos={int(changed)} nrm={int(nrm_changed)} uv0={int(uv_changed)} skin={int(skin_changed)} (fallback_weights={int(skin_fallback)})",
                    )
                else:
                    self.report(
                        {"INFO"},
                        f"In-place patch: updated pos={int(changed)} nrm={int(nrm_changed)} uv0={int(uv_changed)} skin={int(skin_changed)}",
                    )
            elif self.mesh_export_mode == "UPDATE_UV0_IN_PLACE":
                self.report(
                    {"INFO"}, f"In-place patch: updated UV0 for {changed} vertices"
                )
            elif self.mesh_export_mode == "UPDATE_INDICES_IN_PLACE":
                self.report(
                    {"INFO"},
                    f"In-place patch: updated {changed} indices (index-only topology)",
                )
            elif self.mesh_export_mode == "UPDATE_TOPOLOGY_TRIS_IN_PLACE":
                self.report(
                    {"INFO"},
                    f"In-place patch: updated topology for {changed} submesh(es) (triangles)",
                )
            elif self.mesh_export_mode == "UPDATE_VERTS_TOPOLOGY_TRIS_IN_PLACE":
                self.report(
                    {"INFO"},
                    f"In-place patch: updated verts+topology for {changed} submesh(es) (triangles)",
                )
            elif self.mesh_export_mode == "GROW_BUFFERS_TRIS":
                self.report(
                    {"INFO"},
                    f"Grow-buffers patch: rebuilt buffers for {changed} submesh(es) (triangles)",
                )
            else:
                self.report({"INFO"}, f"In-place patch: updated {changed} vertices")

        try:
            out_bytes = rewrap_pack(pack_wrap, out_bytes)
        except Exception as e:
            self.report({"ERROR"}, f"Failed to rewrap patched pack: {e}")
            return {"CANCELLED"}

        os.makedirs(os.path.dirname(self.filepath) or ".", exist_ok=True)
        with open(self.filepath, "wb") as f:
            f.write(out_bytes)

        if self.patch_into_source_archive:
            try:
                out_archive = patch_into_source_archive(context, out_bytes)
                self.report({"INFO"}, f"Patched archive: {out_archive}")
            except Exception as e:
                self.report({"ERROR"}, f"Export succeeded, but patch failed: {e}")

        self.report({"INFO"}, f"Wrote {self.filepath}")
        return {"FINISHED"}

    if self.export_material_textures and self.material_export_mode == "UPDATE_BINDINGS":
        for mat_def in model.materials:
            mat = bpy.data.materials.get(mat_def.name)
            if mat is None or mat.node_tree is None:
                continue
            changed = False
            for tu in mat_def.tex_units:
                if self.texture_mode == "KEEP" and 0 <= int(tu.unit_index) <= 2:
                    img = _find_tex_image_for_unit(mat, int(tu.unit_index))
                    if img is not None and tu.name != img.name:
                        tu.name = img.name
                        changed = True
                mapping = mat.node_tree.nodes.get(f"GF_MAPPING_{tu.unit_index}")
                if mapping is not None:
                    sc = mapping.inputs["Scale"].default_value
                    loc = mapping.inputs["Location"].default_value
                    rot = mapping.inputs["Rotation"].default_value
                    nx, ny = float(sc[0]), float(sc[1])
                    tx, ty = float(loc[0]), float(loc[1])
                    rz = float(rot[2])
                    if (
                        abs(float(tu.scale.x) - nx) > 1e-6
                        or abs(float(tu.scale.y) - ny) > 1e-6
                        or abs(float(tu.translation.x) - tx) > 1e-6
                        or abs(float(tu.translation.y) - ty) > 1e-6
                        or abs(float(tu.rotation) - rz) > 1e-6
                    ):
                        tu.scale.x = nx
                        tu.scale.y = ny
                        tu.translation.x = tx
                        tu.translation.y = ty
                        tu.rotation = rz
                        changed = True
            if changed:
                mat_def.raw_blob = None

        used_tex: List[str] = []
        seen = set()
        for mat_def in model.materials:
            for tu in mat_def.tex_units:
                if not tu.name:
                    continue
                if tu.name not in seen:
                    used_tex.append(tu.name)
                    seen.add(tu.name)
        new_tex_names: List[str] = []
        existing = list(getattr(model, "texture_names", []))
        for n in existing:
            if n in seen:
                new_tex_names.append(n)
        for n in used_tex:
            if n not in new_tex_names:
                new_tex_names.append(n)
        if new_tex_names:
            model.texture_names = new_tex_names

    tex_blobs: Dict[str, bytes] = {}
    if self.export_textures and self.texture_mode in (
        "RGBA8",
        "RGBA8_SAME_SIZE",
        "RGBA8_ORIGINAL_SIZE",
        "ORIGINAL_FORMAT",
        "OVERRIDE_FORMAT",
    ):
        overrides = _collect_texture_overrides_by_slot(model)
        max_size = int(self.texture_max_size)
        if max_size % 8 != 0:
            self.report(
                {"WARNING"}, "Texture Max Size is not multiple of 8; rounding down"
            )
            max_size = max(8, (max_size // 8) * 8)
        for t in textures:
            img = overrides.get(t.name) or bpy.data.images.get(t.name)
            if img is None:
                continue
            override_fmt = str(
                getattr(self, "texture_override_format", "RGBA8") or "RGBA8"
            )
            out_tex = _encode_texture_from_image(
                tex_name=str(t.name),
                img=img,
                original_width=int(t.width),
                original_height=int(t.height),
                original_fmt=int(t.fmt),
                texture_mode=str(self.texture_mode),
                texture_override_format=str(override_fmt),
                texture_max_size=int(max_size),
            )
            t.width = int(out_tex.width)
            t.height = int(out_tex.height)
            t.fmt = int(out_tex.fmt)
            t.raw = bytes(out_tex.raw)
            tex_blobs[t.name] = write_gf_texture_blob(out_tex)

    meshes_by_index: Dict[int, bytes] = {}
    if self.export_meshes and self.mesh_export_mode in (
        "REBUILD_FROM_SCENE",
        "REBUILD_ACTIVE_SUBMESH",
        "UPDATE_POSITIONS_IN_PLACE",
    ):
        mesh_groups: Dict[int, List[Tuple[int, _GFSubMesh]]] = {}
        for submesh_index, sm in enumerate(model.submeshes):
            mesh_groups.setdefault(int(sm.mesh_index), []).append((submesh_index, sm))
        for k in list(mesh_groups.keys()):
            mesh_groups[k] = sorted(mesh_groups[k], key=lambda t: int(t[1].face_index))

        skeleton_names = [b.name for b in model.skeleton]
        axis_forward = str(context.scene.get("gfmodel_last_axis_forward", "-Z"))
        axis_up = str(context.scene.get("gfmodel_last_axis_up", "Y"))
        global_scale = float(context.scene.get("gfmodel_last_global_scale", 1.0))
        conv = axis_conversion(
            from_forward=axis_forward, from_up=axis_up, to_forward="-Y", to_up="Z"
        ).to_4x4()
        gf_from_blender = conv.inverted()

        only_submesh_index: Optional[int] = None
        if self.mesh_export_mode == "REBUILD_ACTIVE_SUBMESH":
            ao = getattr(context, "active_object", None)
            smi = None
            if ao is not None:
                try:
                    smi = ao.get("gfmodel_submesh_index")                              
                except Exception:
                    smi = None
            if (
                ao is None
                or getattr(ao, "type", "") != "MESH"
                or not isinstance(smi, int)
                or ao.get("gfmodel_model_name") != model.name                              
            ):
                raise ValueError(
                    "Rebuild Active Submesh requires selecting an imported GFModel mesh object (tagged gfmodel_submesh_index) as the active object"
                )
            only_submesh_index = int(smi)

        rebuilt_submeshes = 0
        for mesh_index, faces in mesh_groups.items():
            rebuilt: List[_GFSubMesh] = []
            mesh_pos_min = Vector((1e9, 1e9, 1e9))
            mesh_pos_max = Vector((-1e9, -1e9, -1e9))
            mesh_weight_max = 0
            if self.mesh_export_mode == "REBUILD_ACTIVE_SUBMESH":
                for _submesh_index, _sm in faces:
                    try:
                        mesh_pos_min.x = min(mesh_pos_min.x, float(_sm.mesh_bbox_min.x))
                        mesh_pos_min.y = min(mesh_pos_min.y, float(_sm.mesh_bbox_min.y))
                        mesh_pos_min.z = min(mesh_pos_min.z, float(_sm.mesh_bbox_min.z))
                        mesh_pos_max.x = max(mesh_pos_max.x, float(_sm.mesh_bbox_max.x))
                        mesh_pos_max.y = max(mesh_pos_max.y, float(_sm.mesh_bbox_max.y))
                        mesh_pos_max.z = max(mesh_pos_max.z, float(_sm.mesh_bbox_max.z))
                        mesh_weight_max = max(
                            mesh_weight_max,
                            int(getattr(_sm, "mesh_weight_max", 0) or 0),
                        )
                    except Exception:
                        pass
            for submesh_index, sm in faces:
                if only_submesh_index is not None and int(submesh_index) != int(
                    only_submesh_index
                ):
                    obj = None
                else:
                    obj = tagged.get(int(submesh_index))
                if obj is None:
                    rebuilt.append(sm)
                    continue
                rebuilt_submeshes += 1
                if self.mesh_export_mode == "UPDATE_POSITIONS_IN_PLACE":
                    mn, mx = _patch_submesh_positions_in_place(
                        sm,
                        obj=obj,
                        gf_from_blender=gf_from_blender,
                        global_scale=global_scale,
                    )
                    mesh_pos_min.x = min(mesh_pos_min.x, float(mn.x))
                    mesh_pos_min.y = min(mesh_pos_min.y, float(mn.y))
                    mesh_pos_min.z = min(mesh_pos_min.z, float(mn.z))
                    mesh_pos_max.x = max(mesh_pos_max.x, float(mx.x))
                    mesh_pos_max.y = max(mesh_pos_max.y, float(mx.y))
                    mesh_pos_max.z = max(mesh_pos_max.z, float(mx.z))
                else:
                    positions, raw_buffer, indices, max_w = (
                        self._build_export_loop_stream(
                            obj,
                            sm,
                            gf_from_blender=gf_from_blender,
                            global_scale=global_scale,
                            skeleton_names=skeleton_names,
                        )
                    )

                    if self.mesh_export_mode != "REBUILD_ACTIVE_SUBMESH":
                        for p in positions:
                            mesh_pos_min.x = min(mesh_pos_min.x, p.x)
                            mesh_pos_min.y = min(mesh_pos_min.y, p.y)
                            mesh_pos_min.z = min(mesh_pos_min.z, p.z)
                            mesh_pos_max.x = max(mesh_pos_max.x, p.x)
                            mesh_pos_max.y = max(mesh_pos_max.y, p.y)
                            mesh_pos_max.z = max(mesh_pos_max.z, p.z)
                        mesh_weight_max = max(mesh_weight_max, int(max_w))

                    sm.indices = indices
                    sm.primitive_mode = 0
                    sm.vertex_count = len(positions)
                    sm.index_count = len(indices)
                    sm.raw_buffer = raw_buffer
                    if self.mesh_export_mode != "REBUILD_ACTIVE_SUBMESH":
                        try:
                            sm.mesh_weight_max = int(mesh_weight_max)
                            sm.mesh_bbox_min.x = float(mesh_pos_min.x)
                            sm.mesh_bbox_min.y = float(mesh_pos_min.y)
                            sm.mesh_bbox_min.z = float(mesh_pos_min.z)
                            sm.mesh_bbox_max.x = float(mesh_pos_max.x)
                            sm.mesh_bbox_max.y = float(mesh_pos_max.y)
                            sm.mesh_bbox_max.z = float(mesh_pos_max.z)
                        except Exception:
                            pass

                if self.mesh_export_mode != "REBUILD_ACTIVE_SUBMESH":
                    try:
                        sm.mesh_bbox_min.x = float(mesh_pos_min.x)
                        sm.mesh_bbox_min.y = float(mesh_pos_min.y)
                        sm.mesh_bbox_min.z = float(mesh_pos_min.z)
                        sm.mesh_bbox_max.x = float(mesh_pos_max.x)
                        sm.mesh_bbox_max.y = float(mesh_pos_max.y)
                        sm.mesh_bbox_max.z = float(mesh_pos_max.z)
                    except Exception:
                        pass
                rebuilt.append(sm)

            mesh_name = rebuilt[0].mesh_name if rebuilt else f"Mesh{mesh_index}"
            meshes_by_index[int(mesh_index)] = write_gf_mesh_blob(mesh_name, rebuilt)

        if self.mesh_export_mode != "REBUILD_ACTIVE_SUBMESH":
            try:
                mn = Vector((1e9, 1e9, 1e9))
                mx = Vector((-1e9, -1e9, -1e9))
                for sm in model.submeshes:
                    mn.x = min(mn.x, float(sm.mesh_bbox_min.x))
                    mn.y = min(mn.y, float(sm.mesh_bbox_min.y))
                    mn.z = min(mn.z, float(sm.mesh_bbox_min.z))
                    mx.x = max(mx.x, float(sm.mesh_bbox_max.x))
                    mx.y = max(mx.y, float(sm.mesh_bbox_max.y))
                    mx.z = max(mx.z, float(sm.mesh_bbox_max.z))
                model.bbox_min.x = float(mn.x)
                model.bbox_min.y = float(mn.y)
                model.bbox_min.z = float(mn.z)
                model.bbox_max.x = float(mx.x)
                model.bbox_max.y = float(mx.y)
                model.bbox_max.z = float(mx.z)
            except Exception:
                pass
        if rebuilt_submeshes == 0:
            if self.mesh_export_mode == "REBUILD_ACTIVE_SUBMESH":
                self.report(
                    {"ERROR"},
                    "Rebuild Active Submesh ran, but rebuilt 0 submeshes (active object not a tagged imported submesh?)",
                )
                return {"CANCELLED"}
            else:
                self.report(
                    {"WARNING"},
                    "Rebuild mode ran, but rebuilt 0 submeshes (no matching tagged mesh objects); output likely unchanged",
                )
        else:
            self.report({"INFO"}, f"Rebuilt submeshes: {rebuilt_submeshes}")

    model_blob = write_gf_model_blob(model, meshes_by_index=meshes_by_index)

    def parse_container(data: bytes) -> Tuple[str, List[bytes], List[int]]:
        magic = data[:2].decode("ascii", "replace")
        count = struct.unpack_from("<H", data, 2)[0]
        offs = [struct.unpack_from("<I", data, 4 + i * 4)[0] for i in range(count + 1)]
        ents = [data[offs[i] : offs[i + 1]] for i in range(count)]
        return magic, ents, offs

    pack_src: bytes
    cp_present = False
    new_cm_ents: Optional[List[bytes]] = None

    if len(src_data) >= 4 and struct.unpack_from("<I", src_data, 0)[0] == 0x00010000:
        pack_src = src_data
    else:
        top_magic, top_ents, _top_offs = parse_container(src_data)
        cp_present = top_magic == "CP" and len(top_ents) >= 2
        if top_magic == "CP":
            if len(top_ents) < 2:
                self.report({"ERROR"}, "Malformed CP container (missing CM entry)")
                return {"CANCELLED"}
            cm_data = top_ents[1]
        elif top_magic == "CM":
            cm_data = src_data
        else:
            self.report(
                {"ERROR"}, "Exporter v1 supports RAW_PACK, CM, or CP->CM sources"
            )
            return {"CANCELLED"}

        cm_magic, cm_ents, _cm_offs = parse_container(cm_data)
        if cm_magic != "CM":
            self.report({"ERROR"}, "Expected CM container content")
            return {"CANCELLED"}

        pack_i = None
        for i, ent in enumerate(cm_ents):
            if len(ent) >= 4 and struct.unpack_from("<I", ent, 0)[0] == 0x00010000:
                pack_i = i
                break
        if pack_i is None:
            self.report({"ERROR"}, "No GFModelPack found in CM entries")
            return {"CANCELLED"}

        pack_src = cm_ents[pack_i]
        new_cm_ents = list(cm_ents)

    pack = parse_gf_model_pack(pack_src)
    model_slot = None
    model_name = None
    for i in range(int(pack.counts[0])):
        e = pack.get(0, i)
        if e is not None:
            model_slot = int(i)
            model_name = e.name
            break
    if model_slot is None:
        self.report({"ERROR"}, "Model section is empty in GFModelPack")
        return {"CANCELLED"}

    repl: Dict[Tuple[int, int], bytes] = {(0, int(model_slot)): model_blob}
    if tex_blobs:
        for i in range(int(pack.counts[1])):
            e = pack.get(1, i)
            if e is None:
                continue
            blob = tex_blobs.get(e.name)
            if blob is not None:
                repl[(1, int(i))] = blob

    new_pack = write_gf_model_pack_low(pack, replacements=repl)

    if new_cm_ents is not None:
        pack_i = None
        for i, ent in enumerate(new_cm_ents):
            if len(ent) >= 4 and struct.unpack_from("<I", ent, 0)[0] == 0x00010000:
                pack_i = i
                break
        if pack_i is None:
            self.report({"ERROR"}, "Internal error: pack slot not found during rebuild")
            return {"CANCELLED"}
        new_cm_ents[int(pack_i)] = new_pack

    def build_container(magic: str, ents: List[bytes]) -> bytes:
        def align_up(x: int, a: int) -> int:
            return (x + (a - 1)) & ~(a - 1)

        header_base = 4 + (len(ents) + 1) * 4
        header_size = align_up(header_base, 0x80)

        offs: List[int] = []
        cur = header_size
        for e in ents:
            cur = align_up(cur, 0x80)
            offs.append(cur)
            cur = align_up(cur + len(e), 0x80)
        offs.append(cur)

        out = bytearray()
        out += magic.encode("ascii")
        out += struct.pack("<H", len(ents))
        for o in offs:
            out += struct.pack("<I", int(o))
        if len(out) < header_size:
            out += b"\x00" * (header_size - len(out))
        for i, e in enumerate(ents):
            start = offs[i]
            end = offs[i + 1]
            if len(out) < start:
                out += b"\x00" * (start - len(out))
            out += e
            if len(out) < end:
                out += b"\x00" * (end - len(out))
        return bytes(out)

    new_cm = build_container("CM", new_cm_ents) if new_cm_ents is not None else b""
    mode = self.output_container
    if mode == "AUTO":
        mode = "RAW_PACK" if new_cm_ents is None else ("CP" if cp_present else "CM")

    if mode == "RAW_PACK":
        out_bytes = new_pack
    elif mode == "RAW_MODEL":
        out_bytes = model_blob
    elif mode == "CM":
        if new_cm_ents is None:
            self.report(
                {"ERROR"}, "CM output requires a CM/CP source (or choose RAW_PACK)"
            )
            return {"CANCELLED"}
        out_bytes = new_cm
    else:
        if new_cm_ents is None:
            self.report(
                {"ERROR"}, "CP output requires a CP source (or choose RAW_PACK)"
            )
            self.report(
                {"ERROR"}, "CP output requires a CP source (or choose RAW_PACK)"
            )
            return {"CANCELLED"}

        top_magic, top_ents, _top_offs = parse_container(src_data)
        if top_magic == "CP" and len(top_ents) >= 2:
            new_cp_ents = list(top_ents)
            new_cp_ents[1] = new_cm
        else:
            new_cp_ents = [b"\x00" * 0x80, new_cm]
        out_bytes = build_container("CP", new_cp_ents)

    os.makedirs(os.path.dirname(self.filepath) or ".", exist_ok=True)
    with open(self.filepath, "wb") as f:
        f.write(out_bytes)
    try:
        if self.remember_last_export_settings:
            context.scene["gfmodel_last_export_path"] = str(self.filepath)
            context.scene["gfmodel_last_export_export_meshes"] = bool(
                self.export_meshes
            )
            context.scene["gfmodel_last_export_mesh_export_mode"] = str(
                self.mesh_export_mode
            )
            context.scene["gfmodel_last_export_export_material_textures"] = bool(
                self.export_material_textures
            )
            context.scene["gfmodel_last_export_material_export_mode"] = str(
                self.material_export_mode
            )
            context.scene["gfmodel_last_export_export_textures"] = bool(
                self.export_textures
            )
            context.scene["gfmodel_last_export_texture_max_size"] = int(
                self.texture_max_size
            )
            context.scene["gfmodel_last_export_texture_mode"] = str(self.texture_mode)
            context.scene["gfmodel_last_export_texture_override_format"] = str(
                getattr(self, "texture_override_format", "RGBA8") or "RGBA8"
            )
            context.scene["gfmodel_last_export_output_container"] = str(
                self.output_container
            )
    except Exception:
        pass

    if self.patch_into_source_archive:
        try:
            out_archive = patch_into_source_archive(context, out_bytes)
            self.report({"INFO"}, f"Patched archive: {out_archive}")
        except Exception as e:
            self.report({"ERROR"}, f"Export succeeded, but patch failed: {e}")

    self.report({"INFO"}, f"Wrote {self.filepath}")
    return {"FINISHED"}
