

from __future__ import annotations

import json

import os
import struct
from typing import Dict, Optional

import bpy
from bpy_extras.io_utils import axis_conversion
from mathutils import Matrix

from ....core.gfpack import parse_gf_model_pack
from ..archive_patch_apply import patch_into_source_archive
from ..grow_buffers_patch import (
    _apply_uv_strategy_to_mesh,
    _build_temp_mesh_object_for_slot,
    _build_temp_mesh_object_for_sources,
    _collect_tris_all,
    _collect_tris_for_material,
    _material_routing_candidates,
    _patch_pack_grow_buffers_tris,
    _route_source_object_to_submesh_slots,
)
from ..grow_buffers_rewrite import _rewrite_model_blob_grow_buffers_tris
from ..inplace_patch import (
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
from ..pack_wrappers import rewrap_pack, unwrap_pack
from ..textures_patch import _collect_texture_overrides_by_slot, _patch_pack_textures_rgba8


def maybe_export_inplace_or_grow(
    self,
    context: bpy.types.Context,
    *,
    src_data: bytes,
    model: "_GFModel",
    tagged: Dict[int, bpy.types.Object],
    coll: Optional[bpy.types.Collection],
):
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
        raw_model_source = (
            len(src_data) >= 4
            and struct.unpack_from("<I", src_data, 0)[0] == 0x15122117
        )
        if raw_model_source:
            if self.mesh_export_mode != "GROW_BUFFERS_TRIS":
                self.report(
                    {"ERROR"},
                    "In-place patching requires RAW_PACK/CM/CP->CM source (raw GFModel only supports Grow Buffers)",
                )
                return {"CANCELLED"}
            pack_src, pack_wrap = bytes(src_data), None
        else:
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

        if not raw_model_source:
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
                active_mesh_obj = None
                if active_obj is not None and getattr(active_obj, "type", "") == "MESH":
                    active_mesh_obj = active_obj
                elif active_obj is not None and getattr(active_obj, "type", "") == "ARMATURE":
                    try:
                        ucs = getattr(active_obj, "users_collection", None)
                        objs2 = list(ucs[0].all_objects) if ucs else []
                        for o2 in objs2:
                            if o2 is None or getattr(o2, "type", "") != "MESH":
                                continue
                            if o2.get("gfmodel_model_name") != model.name:
                                continue
                            if o2.get("gfmodel_submesh_index") is None:
                                continue
                            active_mesh_obj = o2
                            break
                    except Exception:
                        active_mesh_obj = None
                if (
                    active_mesh_obj is None
                    or active_mesh_obj.get("gfmodel_model_name") != model.name
                    or active_mesh_obj.get("gfmodel_submesh_index") is None
                ):
                    raise ValueError(
                        "Select an imported GFModel mesh (or its armature) before using Grow Buffers"
                    )
                active_obj = active_mesh_obj

                uv_strategy = str(
                    getattr(self, "grow_buffers_uv_strategy", "DUPLICATE")
                    or "DUPLICATE"
                )
                patch_all = bool(
                    getattr(self, "grow_buffers_patch_all_tagged_submeshes", False)
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

                expand_palettes = bool(
                    getattr(self, "grow_buffers_expand_bone_palettes", False)
                )
                split_across_existing = bool(
                    getattr(self, "grow_buffers_split_across_existing_slots", False)
                )

                palette_append_only = bool(
                    getattr(self, "grow_buffers_palette_append_only", False)
                )

                palette_prune_unused = bool(
                    getattr(self, "grow_buffers_palette_prune_unused", False)
                )

                if patch_all:
                    if str(uv_strategy) != "DUPLICATE":
                        raise ValueError(
                            "Robust patch mode currently requires UV Strategy=DUPLICATE"
                        )


                    objs = None
                    try:
                        ucs = getattr(active_obj, "users_collection", None)
                        if ucs:

                            objs = list(ucs[0].all_objects)
                    except Exception:
                        objs = None
                    if objs is None:
                        objs = list(bpy.data.objects)

                    tagged_all: Dict[int, bpy.types.Object] = {}
                    duplicates: List[int] = []
                    for o in objs:
                        if o is None or getattr(o, "type", "") != "MESH":
                            continue
                        if o.get("gfmodel_model_name") != model.name:
                            continue
                        si = o.get("gfmodel_submesh_index")
                        if si is None:
                            continue
                        try:
                            si_i = int(si)
                        except Exception:
                            continue
                        if si_i in tagged_all:
                            duplicates.append(int(si_i))
                            continue
                        tagged_all[int(si_i)] = o

                    if duplicates:
                        self.report(
                            {"WARNING"},
                            f"Robust patch: duplicate gfmodel_submesh_index values found (keeping first): {sorted(set(duplicates))[:12]}",
                        )
                    if not tagged_all:
                        raise ValueError(
                            "No tagged meshes found (gfmodel_submesh_index); import via GFModel Archive and select a mesh object"
                        )

                    auto_route_new = bool(
                        getattr(self, "grow_buffers_auto_route_new_meshes", False)
                    )
                    tmp_objects: List[bpy.types.Object] = []
                    try:
                        if auto_route_new:
                            if str(rebuild_mode) != "CLAMP_ROUTE" and not palette_append_only:
                                raise ValueError(
                                    "Auto-route new meshes currently requires Rebuild Mode=Clamp/Route (No Rebuild), unless Palette Append Only is enabled"
                                )

                            skel_set = set(str(n) for n in (skeleton_names or []))

                            def _base_mat_name(n: str) -> str:
                                s = str(n or "")
                                if len(s) > 4 and s[-4] == "." and s[-3:].isdigit():
                                    return s[:-4]
                                return s

                            model_mat_names: List[str] = []
                            try:
                                seen = set()
                                for sm in getattr(model, "submeshes", []) or []:
                                    mn = str(getattr(sm, "name", "") or "").strip()
                                    if not mn or mn in seen:
                                        continue
                                    seen.add(mn)
                                    model_mat_names.append(mn)
                            except Exception:
                                model_mat_names = []

                            new_meshes: List[bpy.types.Object] = []
                            for o in objs:
                                if o is None or getattr(o, "type", "") != "MESH":
                                    continue
                                if o.get("gfmodel_submesh_index") is not None:
                                    continue
                                if o.get("gfmodel_model_name") == model.name:
                                    continue

                                ok = False
                                try:
                                    for vg in getattr(o, "vertex_groups", []) or []:
                                        if str(getattr(vg, "name", "") or "") in skel_set:
                                            ok = True
                                            break
                                except Exception:
                                    ok = False
                                if not ok:
                                    try:
                                        for m in getattr(o, "modifiers", []) or []:
                                            if str(getattr(m, "type", "") or "") == "ARMATURE":
                                                ok = True
                                                break
                                    except Exception:
                                        ok = False
                                if not ok:
                                    continue
                                new_meshes.append(o)

                            additions_by_si: Dict[
                                int,
                                List[
                                    Tuple[
                                        bpy.types.Object,
                                        List[Tuple[int, int, int, int, int, int]],
                                        Optional[Dict[int, List[Tuple[str, float]]]],
                                    ]
                                ],
                            ] = {}
                            total_dropped = 0
                            total_clamped = 0
                            total_routed = 0

                            for src_obj in new_meshes:
                                obj_mats = set()
                                try:
                                    for ms in getattr(src_obj, "material_slots", []) or []:
                                        bm = getattr(ms, "material", None)
                                        nm = str(getattr(bm, "name", "") or "")
                                        if nm:
                                            obj_mats.add(_base_mat_name(nm))
                                except Exception:
                                    obj_mats = set()

                                for mat_name in model_mat_names:
                                    if obj_mats and _base_mat_name(mat_name) not in obj_mats:
                                        continue
                                    (
                                        tris_by_si,
                                        weights_override_by_si,
                                        stats,
                                        extra,
                                    ) = _route_source_object_to_submesh_slots(
                                        model,
                                        material_name=str(mat_name),
                                        src_obj=src_obj,
                                        skeleton_names=skeleton_names,
                                        routing_strategy=str(self.grow_buffers_routing_strategy),
                                        weight_cutoff=float(getattr(self, "grow_buffers_weight_cutoff", 0.0)),
                                        conflict_mode=str(clamp_conflict_mode),
                                        allow_palette_expand=bool(expand_palettes),
                                        split_across_slots=bool(split_across_existing),
                                    )
                                    total_dropped += int(stats.get("dropped", 0) or 0)
                                    total_clamped += int(stats.get("clamped", 0) or 0)
                                    for si, tris in (tris_by_si or {}).items():
                                        if not tris:
                                            continue
                                        total_routed += int(len(tris))
                                        additions_by_si.setdefault(int(si), []).append(
                                            (
                                                src_obj,
                                                list(tris),
                                                (weights_override_by_si or {}).get(int(si)),
                                            )
                                        )
                                    if extra.get("unknown_bones"):
                                        self.report(
                                            {"WARNING"},
                                            "Auto-route: unknown bones; dropped triangles (see console for details)",
                                        )

                            if total_clamped > 0:
                                self.report(
                                    {"WARNING"},
                                    f"Auto-route: clamped {int(total_clamped)} triangle(s) (weight trimming)",
                                )
                            if total_dropped > 0:
                                self.report(
                                    {"WARNING"},
                                    f"Auto-route: dropped {int(total_dropped)} triangle(s) (no slot / unknown bones)",
                                )

                            for si, adds in additions_by_si.items():
                                base_obj = tagged_all.get(int(si))
                                if base_obj is None or getattr(base_obj, "type", "") != "MESH":
                                    continue
                                base_tris = _collect_tris_all(base_obj)
                                sources = [(base_obj, base_obj.data, base_tris, None)]
                                for src_obj, tris, ow in adds:
                                    sources.append((src_obj, src_obj.data, tris, ow))
                                tmp = _build_temp_mesh_object_for_sources(
                                    name=f"__gf_tmp_autoroute_{model.name}_{si}",
                                    sources=sources,
                                )
                                _apply_uv_strategy_to_mesh(tmp.data, strategy=uv_strategy)
                                tmp_objects.append(tmp)
                                tagged_all[int(si)] = tmp

                            if total_routed > 0:
                                self.report(
                                    {"INFO"},
                                    f"Auto-route: routed {int(total_routed)} triangle(s) from {len(new_meshes)} mesh(es)",
                                )

                        if raw_model_source:
                            out_bytes = _rewrite_model_blob_grow_buffers_tris(
                                pack_src=pack_src,
                                model_blob=pack_src,
                                model_blob_start_in_pack=0,
                                model=model,
                                tagged=tagged_all,
                                gf_from_blender=gf_from_blender,
                                global_scale=float(global_scale),
                                skeleton_names=skeleton_names,
                                disallow_new_mesh_sections=bool(disallow_new_mesh_sections),
                                allow_palette_rebuild=bool(allow_palette_rebuild),
                                allow_palette_split=bool(allow_palette_split),
                                palette_append_only=bool(palette_append_only),
                                palette_prune_unused=bool(palette_prune_unused),
                            )
                            changed = int(len(tagged_all))
                        else:
                            out_bytes, changed = _patch_pack_grow_buffers_tris(
                                pack_src,
                                model,
                                tagged=tagged_all,
                                gf_from_blender=gf_from_blender,
                                global_scale=float(global_scale),
                                skeleton_names=skeleton_names,
                                disallow_new_mesh_sections=bool(disallow_new_mesh_sections),
                                allow_palette_rebuild=bool(allow_palette_rebuild),
                                allow_palette_split=bool(allow_palette_split),
                                palette_append_only=bool(palette_append_only),
                                palette_prune_unused=bool(palette_prune_unused),
                            )
                    finally:
                        for o in tmp_objects:
                            try:
                                m = getattr(o, "data", None)
                                bpy.data.objects.remove(o, do_unlink=True)
                                if m is not None:
                                    bpy.data.meshes.remove(m, do_unlink=True)
                            except Exception:
                                pass
                mapping_raw = str(self.grow_buffers_material_sources_json or "").strip()
                if (not patch_all) and mapping_raw:
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
                            if raw_model_source:
                                out_bytes = _rewrite_model_blob_grow_buffers_tris(
                                    pack_src=pack_src,
                                    model_blob=pack_src,
                                    model_blob_start_in_pack=0,
                                    model=model,
                                    tagged=tagged_multi,
                                    gf_from_blender=gf_from_blender,
                                    global_scale=float(global_scale),
                                    skeleton_names=skeleton_names,
                                    disallow_new_mesh_sections=bool(
                                        disallow_new_mesh_sections
                                    ),
                                    allow_palette_rebuild=bool(allow_palette_rebuild),
                                    allow_palette_split=bool(allow_palette_split),
                                palette_append_only=bool(palette_append_only),
                                palette_prune_unused=bool(palette_prune_unused),
                                )
                                changed = int(len(tagged_multi))
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
                                palette_append_only=bool(palette_append_only),
                                palette_prune_unused=bool(palette_prune_unused),
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
                elif not patch_all:
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
                        if raw_model_source:
                            out_bytes = _rewrite_model_blob_grow_buffers_tris(
                                pack_src=pack_src,
                                model_blob=pack_src,
                                model_blob_start_in_pack=0,
                                model=model,
                                tagged=tagged_active,
                                gf_from_blender=gf_from_blender,
                                global_scale=float(global_scale),
                                skeleton_names=skeleton_names,
                                disallow_new_mesh_sections=bool(disallow_new_mesh_sections),
                                allow_palette_rebuild=bool(allow_palette_rebuild),
                                allow_palette_split=bool(allow_palette_split),
                                palette_append_only=bool(palette_append_only),
                                palette_prune_unused=bool(palette_prune_unused),
                            )
                            changed = int(len(tagged_active))
                        else:
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
                                palette_append_only=bool(palette_append_only),
                                palette_prune_unused=bool(palette_prune_unused),
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

        if (not raw_model_source) and self.export_textures and str(self.texture_mode) in (
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
            if pack_wrap is not None:
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


    return None
