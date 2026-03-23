
from __future__ import annotations

import copy
import struct
from typing import Dict, List, Optional, Tuple
import bmesh
import bpy
from mathutils import Matrix, Vector
from ....core.gfpack import parse_gf_model_pack
from ....core.gfpack import write_gf_model_pack as write_gf_model_pack_low
from ..grow_buffers_rewrite import _rewrite_model_blob_grow_buffers_tris
from .tris import _tri_bones_used


def _material_routing_candidates(
    model: "_GFModel",
    *,
    material_name: str,
) -> List[int]:
    out: List[int] = []
    for si, sm in enumerate(model.submeshes):
        if str(getattr(sm, "name", "") or "") == str(material_name):
            out.append(int(si))
    return out


def _route_source_object_to_submesh_slots(
    model: "_GFModel",
    *,
    material_name: str,
    src_obj: bpy.types.Object,
    skeleton_names: List[str],
    routing_strategy: str,
    weight_cutoff: float = 0.0,
    conflict_mode: str = "CLAMP_BY_WEIGHT",
) -> Tuple[
    Dict[int, List[Tuple[int, int, int, int, int, int]]],
    Dict[int, Dict[int, List[Tuple[str, float]]]],
    Dict[str, int],
    Dict[str, List[str]],
]:
    if src_obj.type != "MESH":
        raise ValueError(f"Source object for {material_name!r} is not a mesh")
    try:
        if getattr(src_obj, "mode", "") == "EDIT":
            src_obj.update_from_editmode()
    except Exception:
        pass
    src_mesh: bpy.types.Mesh = src_obj.data                            
    try:
        src_mesh.calc_loop_triangles()
    except Exception:
        pass

                                        
    sk_index_by_name: Dict[str, int] = {
        str(n): int(i) for i, n in enumerate(skeleton_names)
    }

                                                                                 
    bones_by_v: List[List[int]] = [[] for _ in range(len(src_mesh.vertices))]
    unknown_bones: Dict[str, None] = {}
    unknown_by_v: List[List[str]] = [[] for _ in range(len(src_mesh.vertices))]
    weights_by_v: List[List[Tuple[int, float]]] = [
        [] for _ in range(len(src_mesh.vertices))
    ]
    vg_by_index: Dict[int, str] = {}
    cutoff = float(weight_cutoff)
    try:
        for i, vg in enumerate(src_obj.vertex_groups):
            vg_by_index[int(i)] = str(vg.name)
    except Exception:
        vg_by_index = {}

    for vi, v in enumerate(src_mesh.vertices):
        used: Dict[int, None] = {}
        unknown_here: Dict[str, None] = {}
        wl: List[Tuple[int, float]] = []
        for g in getattr(v, "groups", []) or []:
            try:
                w = float(g.weight)
                if w <= 0.0 or w < cutoff:
                    continue
                gname = vg_by_index.get(int(g.group), "")
                if not gname:
                    continue
                si = sk_index_by_name.get(gname)
                if si is None:
                    unknown_bones[gname] = None
                    unknown_here[gname] = None
                    continue
                used[int(si)] = None
                wl.append((int(si), float(w)))
            except Exception:
                continue
        bones_by_v[int(vi)] = list(used.keys())
        unknown_by_v[int(vi)] = list(unknown_here.keys())
        weights_by_v[int(vi)] = wl

    candidates = _material_routing_candidates(model, material_name=material_name)
    if not candidates:
        raise ValueError(
            f"No target submesh slots found for material {material_name!r}"
        )
    mode = str(conflict_mode or "CLAMP_BY_WEIGHT").strip()
    if mode not in ("CLAMP_BY_WEIGHT", "CLAMP_BY_NEIGHBORS", "DROP_CONFLICTS"):
        mode = "CLAMP_BY_WEIGHT"

    palette_by_si: Dict[int, Dict[int, None]] = {}
    palette_size_by_si: Dict[int, int] = {}
    palette_union: Dict[int, None] = {}
    palette_pair_ok: Dict[Tuple[int, int], None] = {}
    palette_list_by_si: Dict[int, List[int]] = {}
    for si in candidates:
        sm = model.submeshes[int(si)]
        pal_list = [int(bi) for bi in (getattr(sm, "bone_indices", []) or [])]
        pal = {int(bi): None for bi in pal_list}
        palette_by_si[int(si)] = pal
        palette_size_by_si[int(si)] = int(len(pal))
        palette_list_by_si[int(si)] = pal_list
        for bi in pal_list:
            palette_union[int(bi)] = None
                                                             
        for i in range(len(pal_list)):
            a = int(pal_list[i])
            for j in range(i, len(pal_list)):
                b = int(pal_list[j])
                if a <= b:
                    palette_pair_ok[(a, b)] = None
                else:
                    palette_pair_ok[(b, a)] = None

                       
    tris_by_si: Dict[int, List[Tuple[int, int, int, int, int, int]]] = {
        int(si): [] for si in candidates
    }
    assigned_tri_count: Dict[int, int] = {int(si): 0 for si in candidates}
    weights_override_by_si: Dict[int, Dict[int, List[Tuple[str, float]]]] = {
        int(si): {} for si in candidates
    }
    dropped = 0
    dropped_unknown = 0
    dropped_no_slot = 0
    clamped = 0
    dropped_unknown_bones: Dict[str, int] = {}
    dropped_missing_bones: Dict[str, int] = {}
    dropped_conflict_pairs: Dict[str, int] = {}

    def choose_slot(used_bones: List[int]) -> Optional[int]:
        used_set = {int(b): None for b in used_bones}
        valid: List[int] = []
        for si in candidates:
            pal = palette_by_si[int(si)]
            ok = True
            for b in used_set.keys():
                if int(b) not in pal:
                    ok = False
                    break
            if ok:
                valid.append(int(si))
        if not valid:
            return None
        if routing_strategy == "BALANCE":
            valid.sort(key=lambda s: (assigned_tri_count.get(int(s), 0), int(s)))
            return int(valid[0])
        if routing_strategy == "ORIGINAL_ORDER":
            return int(sorted(valid)[0])
                               
        valid.sort(key=lambda s: (palette_size_by_si.get(int(s), 1 << 30), int(s)))
        return int(valid[0])

    def _base_mat_name(n: str) -> str:
        s = str(n or "")
        if len(s) > 4 and s[-4] == "." and s[-3:].isdigit():
            return s[:-4]
        return s

    def choose_slot_by_weight(tri_verts: Tuple[int, int, int]) -> int:
        best_si: Optional[int] = None
        best_score = -1.0
        for si in candidates:
            pal = palette_by_si[int(si)]
            score = 0.0
            for vi in tri_verts:
                if 0 <= int(vi) < len(weights_by_v):
                    for bi, w in weights_by_v[int(vi)]:
                        if int(bi) in pal:
                            score += float(w)
            if score > best_score:
                best_score = float(score)
                best_si = int(si)
        return int(best_si) if best_si is not None else int(candidates[0])

    def clamped_weights_for_vertex(si: int, vi: int) -> List[Tuple[str, float]]:
        pal = palette_by_si[int(si)]
        wl = weights_by_v[int(vi)] if 0 <= int(vi) < len(weights_by_v) else []
        kept: List[Tuple[int, float]] = [
            (int(bi), float(w)) for bi, w in wl if int(bi) in pal and float(w) > 0.0
        ]
        if not kept:
            pal_list = palette_list_by_si.get(int(si), [])
            bi = int(pal_list[0]) if pal_list else 0
            name = str(skeleton_names[bi]) if 0 <= bi < len(skeleton_names) else str(bi)
            return [(name, 1.0)]
        kept.sort(key=lambda t: -float(t[1]))
        kept = kept[:4]
        s = float(sum(float(w) for _bi, w in kept))
        if s <= 0.0:
            s = 1.0
        out: List[Tuple[str, float]] = []
        for bi, w in kept:
            name = (
                str(skeleton_names[int(bi)])
                if 0 <= int(bi) < len(skeleton_names)
                else str(int(bi))
            )
            out.append((name, float(w) / s))
        return out

                                                                                     
                                                                 
    tri_assigned_si: Dict[int, int] = {}
    tri_verts_by_idx: Dict[int, Tuple[int, int, int]] = {}
    tri_set_by_idx: Dict[int, Tuple[int, int, int]] = {}
    tri_indices_by_v: Dict[int, List[int]] = {}

    for tri in getattr(src_mesh, "loop_triangles", []) or []:
                                                                                            
                                                                                
        try:
            pi = int(getattr(tri, "polygon_index", -1))
            if 0 <= pi < len(src_mesh.polygons):
                poly = src_mesh.polygons[pi]
                mi = int(getattr(poly, "material_index", -1))
                if 0 <= mi < len(getattr(src_obj, "material_slots", []) or []):
                    ms = src_obj.material_slots[mi]
                    bm = getattr(ms, "material", None)
                    bm_name = str(getattr(bm, "name", "") or "")
                    if bm_name and _base_mat_name(bm_name) != _base_mat_name(
                        str(material_name)
                    ):
                        continue
        except Exception:
            pass
        a, b, c = (int(tri.vertices[0]), int(tri.vertices[1]), int(tri.vertices[2]))
        la, lb, lc = (int(tri.loops[0]), int(tri.loops[1]), int(tri.loops[2]))
        tri_ref = (int(a), int(b), int(c), int(la), int(lb), int(lc))
        tri_idx = int(getattr(tri, "index", -1))
        if tri_idx < 0:
                                                                                      
            tri_idx = int(len(tri_verts_by_idx))
        tri_verts_by_idx[int(tri_idx)] = (int(a), int(b), int(c))
        tri_set_by_idx[int(tri_idx)] = tuple(sorted((int(a), int(b), int(c))))
        for vi in (a, b, c):
            tri_indices_by_v.setdefault(int(vi), []).append(int(tri_idx))
        used = _tri_bones_used(bones_by_v, (a, b, c))
                                                                                      
        unk = []
        for vi in (a, b, c):
            if 0 <= int(vi) < len(unknown_by_v):
                unk.extend(unknown_by_v[int(vi)])
        if unk:
            dropped += 1
            dropped_unknown += 1
            for bn in unk[:64]:
                bns = str(bn)
                dropped_unknown_bones[bns] = int(dropped_unknown_bones.get(bns, 0) + 1)
            continue
        si = choose_slot(used)
        if si is None:
                                                                                                   
            missing_any: List[int] = [
                int(bi) for bi in used if int(bi) not in palette_union
            ]
            if missing_any:
                dropped += 1
                dropped_no_slot += 1
                for bi in missing_any[:64]:
                    try:
                        bn = str(skeleton_names[int(bi)])
                    except Exception:
                        bn = str(int(bi))
                    dropped_missing_bones[bn] = int(
                        dropped_missing_bones.get(bn, 0) + 1
                    )
                continue
            else:
                if mode == "DROP_CONFLICTS":
                    dropped += 1
                                                                            
                    ub = sorted(set(int(bi) for bi in used))
                    pair = None
                    for i in range(len(ub)):
                        for j in range(i + 1, len(ub)):
                            aa, bb = int(ub[i]), int(ub[j])
                            key = (aa, bb) if aa <= bb else (bb, aa)
                            if key not in palette_pair_ok:
                                pair = key
                                break
                        if pair is not None:
                            break
                    if pair is not None:
                        try:
                            an = str(skeleton_names[int(pair[0])])
                        except Exception:
                            an = str(int(pair[0]))
                        try:
                            bn = str(skeleton_names[int(pair[1])])
                        except Exception:
                            bn = str(int(pair[1]))
                        k = f"{an}+{bn}"
                        dropped_conflict_pairs[k] = int(
                            dropped_conflict_pairs.get(k, 0) + 1
                        )
                    continue

                chosen = None
                if mode == "CLAMP_BY_NEIGHBORS":
                    scores: Dict[int, int] = {}
                    tri_vset = {int(a): None, int(b): None, int(c): None}
                    for vi in tri_vset.keys():
                        for nidx in tri_indices_by_v.get(int(vi), []):
                            if int(nidx) == int(tri_idx):
                                continue
                            nsi = tri_assigned_si.get(int(nidx))
                            if nsi is None:
                                continue
                            na, nb, nc = tri_verts_by_idx.get(int(nidx), (0, 0, 0))
                            common = 0
                            for vv in (int(na), int(nb), int(nc)):
                                if vv in tri_vset:
                                    common += 1
                            if common <= 0:
                                continue
                            scores[int(nsi)] = int(scores.get(int(nsi), 0) + common)
                    if scores:
                        best = max(scores.values())
                        tied = [int(si0) for si0, sc in scores.items() if sc == best]
                        if len(tied) == 1:
                            chosen = int(tied[0])
                        else:
                                                           
                            best_si = None
                            best_score = -1.0
                            for si0 in tied:
                                pal = palette_by_si[int(si0)]
                                score = 0.0
                                for vi in (a, b, c):
                                    if 0 <= int(vi) < len(weights_by_v):
                                        for bi, w in weights_by_v[int(vi)]:
                                            if int(bi) in pal:
                                                score += float(w)
                                if score > best_score:
                                    best_score = float(score)
                                    best_si = int(si0)
                            if best_si is not None:
                                chosen = int(best_si)

                if chosen is None:
                    chosen = choose_slot_by_weight((a, b, c))
                for vi in (a, b, c):
                    weights_override_by_si[int(chosen)][int(vi)] = (
                        clamped_weights_for_vertex(int(chosen), int(vi))
                    )
                tris_by_si[int(chosen)].append(tri_ref)
                assigned_tri_count[int(chosen)] = int(
                    assigned_tri_count[int(chosen)] + 1
                )
                clamped += 1
                                                                        
                ub = sorted(set(int(bi) for bi in used))
                pair = None
                for i in range(len(ub)):
                    for j in range(i + 1, len(ub)):
                        aa, bb = int(ub[i]), int(ub[j])
                        key = (aa, bb) if aa <= bb else (bb, aa)
                        if key not in palette_pair_ok:
                            pair = key
                            break
                    if pair is not None:
                        break
                if pair is not None:
                    try:
                        an = str(skeleton_names[int(pair[0])])
                    except Exception:
                        an = str(int(pair[0]))
                    try:
                        bn = str(skeleton_names[int(pair[1])])
                    except Exception:
                        bn = str(int(pair[1]))
                    k = f"{an}+{bn}"
                    dropped_conflict_pairs[k] = int(
                        dropped_conflict_pairs.get(k, 0) + 1
                    )
                tri_assigned_si[int(tri_idx)] = int(chosen)
                continue
        tris_by_si[int(si)].append(tri_ref)
        assigned_tri_count[int(si)] = int(assigned_tri_count[int(si)] + 1)
        tri_assigned_si[int(tri_idx)] = int(si)

    stats = {
        "dropped": int(dropped),
        "dropped_unknown_bones": int(dropped_unknown),
        "dropped_no_slot": int(dropped_no_slot),
        "clamped": int(clamped),
    }
    extra = {
        "conflict_mode": str(mode),
        "unknown_bones": sorted(list(unknown_bones.keys())),
        "dropped_unknown_bones": sorted(list(dropped_unknown_bones.keys())),
        "dropped_missing_bones": sorted(list(dropped_missing_bones.keys())),
        "dropped_conflict_pairs": sorted(list(dropped_conflict_pairs.keys())),
        "dropped_unknown_bones_counts": dropped_unknown_bones,
        "dropped_missing_bones_counts": dropped_missing_bones,
        "dropped_conflict_pairs_counts": dropped_conflict_pairs,
    }
    return tris_by_si, weights_override_by_si, stats, extra
