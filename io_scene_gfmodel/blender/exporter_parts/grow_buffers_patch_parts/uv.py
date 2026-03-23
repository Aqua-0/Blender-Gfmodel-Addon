
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

def _apply_uv_strategy_to_mesh(
    mesh: bpy.types.Mesh,
    *,
    strategy: str,
    tol: float = 1e-6,
) -> None:
    strat = str(strategy or "DUPLICATE").upper()
    if strat == "DUPLICATE":
        return
    if not getattr(mesh, "uv_layers", None):
        return

    bm = bmesh.new()
    try:
        bm.from_mesh(mesh)
        bm.verts.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        uv_layer = bm.loops.layers.uv.active or bm.loops.layers.uv[0]
        if uv_layer is None:
            return

        if strat == "SMEAR":
            uv_by_v: Dict[int, Tuple[float, float]] = {}
            for f in bm.faces:
                for l in f.loops:
                    vi = int(l.vert.index)
                    if vi not in uv_by_v:
                        uv = l[uv_layer].uv
                        uv_by_v[vi] = (float(uv.x), float(uv.y))
            for f in bm.faces:
                for l in f.loops:
                    u, v = uv_by_v.get(int(l.vert.index), (0.0, 0.0))
                    l[uv_layer].uv.x = float(u)
                    l[uv_layer].uv.y = float(v)
        elif strat == "STITCH_TRANSLATE":
            tol_q = max(1.0, float(tol) * 1e6)

            def uv_q(uv: Tuple[float, float]) -> Tuple[int, int]:
                return (int(round(float(uv[0]) * 1e6)), int(round(float(uv[1]) * 1e6)))

            def close(a: Tuple[float, float], b: Tuple[float, float]) -> bool:
                ax, ay = uv_q(a)
                bx, by = uv_q(b)
                return abs(ax - bx) <= tol_q and abs(ay - by) <= tol_q

            def face_edge_uv_connected(
                fa: bmesh.types.BMFace, fb: bmesh.types.BMFace, e: bmesh.types.BMEdge
            ) -> bool:
                for v in e.verts:
                    la = next((l for l in fa.loops if l.vert == v), None)
                    lb = next((l for l in fb.loops if l.vert == v), None)
                    if la is None or lb is None:
                        return False
                    uva = la[uv_layer].uv
                    uvb = lb[uv_layer].uv
                    if not close(
                        (float(uva.x), float(uva.y)), (float(uvb.x), float(uvb.y))
                    ):
                        return False
                return True

                                                                                           
            face_seen: Dict[int, None] = {}
            islands: List[List[bmesh.types.BMFace]] = []
            for f in bm.faces:
                if int(f.index) in face_seen:
                    continue
                stack = [f]
                face_seen[int(f.index)] = None
                group: List[bmesh.types.BMFace] = []
                while stack:
                    cur = stack.pop()
                    group.append(cur)
                    for e in cur.edges:
                        if not e.is_manifold or len(e.link_faces) != 2:
                            continue
                        other = (
                            e.link_faces[0]
                            if e.link_faces[1] == cur
                            else e.link_faces[1]
                        )
                        if int(other.index) in face_seen:
                            continue
                        if face_edge_uv_connected(cur, other, e):
                            face_seen[int(other.index)] = None
                            stack.append(other)
                islands.append(group)

            if len(islands) > 1:
                face_to_island: Dict[int, int] = {}
                for ii, isl in enumerate(islands):
                    for f in isl:
                        face_to_island[int(f.index)] = int(ii)

                                                  
                seam_edges: List[bmesh.types.BMEdge] = []
                for e in bm.edges:
                    if not e.is_manifold or len(e.link_faces) != 2:
                        continue
                    f0, f1 = e.link_faces[0], e.link_faces[1]
                    a = face_to_island.get(int(f0.index))
                    b = face_to_island.get(int(f1.index))
                    if a is None or b is None or a == b:
                        continue
                    for v in e.verts:
                        l0 = next((l for l in f0.loops if l.vert == v), None)
                        l1 = next((l for l in f1.loops if l.vert == v), None)
                        if l0 is None or l1 is None:
                            continue
                        u0 = l0[uv_layer].uv
                        u1 = l1[uv_layer].uv
                        if not close(
                            (float(u0.x), float(u0.y)), (float(u1.x), float(u1.y))
                        ):
                            seam_edges.append(e)
                            break

                                                               
                adj: Dict[int, List[Tuple[int, Tuple[float, float]]]] = {
                    int(i): [] for i in range(len(islands))
                }

                def edge_translation(
                    fa: bmesh.types.BMFace,
                    fb: bmesh.types.BMFace,
                    e: bmesh.types.BMEdge,
                ) -> Optional[Tuple[float, float]]:
                    deltas: List[Tuple[float, float]] = []
                    for v in e.verts:
                        la = next((l for l in fa.loops if l.vert == v), None)
                        lb = next((l for l in fb.loops if l.vert == v), None)
                        if la is None or lb is None:
                            return None
                        ua = la[uv_layer].uv
                        ub = lb[uv_layer].uv
                        deltas.append((float(ua.x - ub.x), float(ua.y - ub.y)))
                    if len(deltas) != 2:
                        return None
                    if (
                        abs(deltas[0][0] - deltas[1][0]) <= float(tol) * 4.0
                        and abs(deltas[0][1] - deltas[1][1]) <= float(tol) * 4.0
                    ):
                        dx = 0.5 * (deltas[0][0] + deltas[1][0])
                        dy = 0.5 * (deltas[0][1] + deltas[1][1])
                        return (dx, dy)
                    return None

                for e in seam_edges:
                    if len(e.link_faces) != 2:
                        continue
                    f0, f1 = e.link_faces[0], e.link_faces[1]
                    a = face_to_island.get(int(f0.index))
                    b = face_to_island.get(int(f1.index))
                    if a is None or b is None or a == b:
                        continue
                    t_ab = edge_translation(f0, f1, e)
                    t_ba = edge_translation(f1, f0, e)
                    if t_ab is not None:
                        adj[int(a)].append((int(b), t_ab))
                    if t_ba is not None:
                        adj[int(b)].append((int(a), t_ba))

                assigned: Dict[int, Tuple[float, float]] = {0: (0.0, 0.0)}
                q: List[int] = [0]
                while q:
                    cur = q.pop(0)
                    base = assigned[cur]
                    for nxt, d in adj.get(cur, []):
                        if nxt in assigned:
                            continue
                        assigned[nxt] = (base[0] + d[0], base[1] + d[1])
                        q.append(nxt)

                for ii, (dx, dy) in assigned.items():
                    if ii == 0:
                        continue
                    if abs(dx) <= 0.0 and abs(dy) <= 0.0:
                        continue
                    for f in islands[ii]:
                        for l in f.loops:
                            l[uv_layer].uv.x = float(l[uv_layer].uv.x + dx)
                            l[uv_layer].uv.y = float(l[uv_layer].uv.y + dy)
        else:
            return

        bm.to_mesh(mesh)
        mesh.update()
    finally:
        bm.free()
