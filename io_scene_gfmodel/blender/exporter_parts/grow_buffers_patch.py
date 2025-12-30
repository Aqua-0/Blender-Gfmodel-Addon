"""Grow Buffers patching helpers (Tier-2), including material routing and UV strategy."""

from __future__ import annotations

import copy
import struct
from typing import Dict, List, Optional, Tuple

import bmesh
import bpy
from mathutils import Matrix, Vector

from ...core.gfpack import parse_gf_model_pack
from ...core.gfpack import write_gf_model_pack as write_gf_model_pack_low
from .grow_buffers_rewrite import _rewrite_model_blob_grow_buffers_tris


def _patch_pack_grow_buffers_tris(
    pack_src: bytes,
    model: "_GFModel",
    *,
    tagged: Dict[int, bpy.types.Object],
    gf_from_blender: Matrix,
    global_scale: float,
    skeleton_names: List[str],
    disallow_new_mesh_sections: bool = False,
    allow_palette_rebuild: bool = True,
    allow_palette_split: bool = True,
) -> Tuple[bytes, int]:
    """Tier-2: grow allocated vtx/idx buffers by rewriting the GFModel blob and pack."""
    if not tagged:
        raise ValueError(
            "No tagged meshes found (gfmodel_submesh_index); import via GFModel Archive and select a mesh object"
        )
    pack = parse_gf_model_pack(pack_src)
    model_slot = None
    for i in range(int(pack.counts[0])):
        e = pack.get(0, i)
        if e is not None:
            model_slot = int(i)
            break
    if model_slot is None:
        raise ValueError("Model section is empty in GFModelPack")
    entry = pack.get(0, int(model_slot))
    if entry is None:
        raise ValueError("Missing model entry in GFModelPack")

    new_model_blob = _rewrite_model_blob_grow_buffers_tris(
        pack_src=pack_src,
        model_blob=entry.blob,
        model_blob_start_in_pack=int(entry.addr),
        model=model,
        tagged=tagged,
        gf_from_blender=gf_from_blender,
        global_scale=float(global_scale),
        skeleton_names=skeleton_names,
        disallow_new_mesh_sections=bool(disallow_new_mesh_sections),
        allow_palette_rebuild=bool(allow_palette_rebuild),
        allow_palette_split=bool(allow_palette_split),
    )
    out = write_gf_model_pack_low(
        pack,
        replacements={(0, int(model_slot)): bytes(new_model_blob)},
        align_blobs=0x80,
    )
    return bytes(out), int(len(tagged))


def _rgba8_bytes_from_image(
    img: bpy.types.Image,
    *,
    width: int,
    height: int,
    allow_scale: bool,
) -> bytes:
    """Return linear RGBA8 bytes from a Blender image."""
    img.pixels[0]                 

    w0, h0 = int(img.size[0]), int(img.size[1])
    if w0 <= 0 or h0 <= 0:
        raise ValueError(f"Invalid image size for {img.name!r}: {w0}x{h0}")

    w = int(width)
    h = int(height)
    if not allow_scale:
        if w0 != w or h0 != h:
            raise ValueError(
                f"Image size mismatch for {img.name!r}: image={w0}x{h0} expected={w}x{h}"
            )
        px_f = img.pixels[: w * h * 4]
    else:
        img2 = img.copy()
        try:
            img2.scale(w, h)
            px_f = img2.pixels[: w * h * 4]
        finally:
            bpy.data.images.remove(img2)

    return bytes(int(max(0, min(255, round(float(c) * 255.0)))) for c in px_f)


def _collect_texture_overrides_by_slot(model: "_GFModel") -> Dict[str, bpy.types.Image]:
    """Map existing GF texture names -> Blender images used by materials."""
    overrides: Dict[str, bpy.types.Image] = {}
    for mat_def in getattr(model, "materials", []) or []:
        mat = bpy.data.materials.get(str(getattr(mat_def, "name", "") or ""))
        if mat is None or getattr(mat, "node_tree", None) is None:
            continue
        for tu in getattr(mat_def, "tex_units", []) or []:
            try:
                unit_index = int(getattr(tu, "unit_index", -1))
            except Exception:
                unit_index = -1
            if unit_index < 0:
                continue
            img = _find_tex_image_for_unit(mat, unit_index)
            if img is None:
                continue
            slot_name = str(getattr(tu, "name", "") or "").strip()
            if not slot_name:
                continue
            overrides.setdefault(slot_name, img)
    return overrides


def _patch_pack_textures_rgba8(
    pack_src: bytes,
    *,
    overrides: Dict[str, bpy.types.Image],
    texture_mode: str,
    texture_max_size: int,
) -> Tuple[bytes, int]:
    """Overwrite existing texture slots with RGBA8 bytes, without adding new slots."""
    if texture_mode not in ("RGBA8", "RGBA8_SAME_SIZE", "RGBA8_ORIGINAL_SIZE"):
        return pack_src, 0

    pack = parse_gf_model_pack(pack_src)
    if int(pack.counts[1]) <= 0:
        return pack_src, 0

    max_size = int(texture_max_size)
    if max_size % 8 != 0:
        max_size = max(8, (max_size // 8) * 8)

    replacements: Dict[Tuple[int, int], bytes] = {}
    changed = 0
    for i in range(int(pack.counts[1])):
        e = pack.get(1, int(i))
        if e is None:
            continue
        tex = _parse_gf_texture(e.blob)
        img = overrides.get(tex.name) or bpy.data.images.get(tex.name)
        if img is None:
            continue

        if texture_mode in ("RGBA8_SAME_SIZE", "RGBA8_ORIGINAL_SIZE"):
            w = int(tex.width)
            h = int(tex.height)
            allow_scale = texture_mode == "RGBA8_ORIGINAL_SIZE"
        else:
            w0, h0 = int(img.size[0]), int(img.size[1])
            w = int(min(w0, max_size))
            h = int(min(h0, max_size))
            w = max(8, (w // 8) * 8)
            h = max(8, (h // 8) * 8)
            allow_scale = True

        if w % 8 != 0 or h % 8 != 0:
            raise ValueError(
                f"Texture size must be multiple of 8 for tiled formats: {tex.name!r} ({w}x{h})"
            )

        px = _rgba8_bytes_from_image(img, width=w, height=h, allow_scale=allow_scale)
        raw = encode_pica_rgba8_swizzled_abgr(px, w, h)
        out_tex = _GFTexture(name=tex.name, width=w, height=h, fmt=0x4, raw=raw)
        replacements[(1, int(i))] = write_gf_texture_blob(out_tex)
        changed += 1

    if changed == 0:
        return pack_src, 0
    out = write_gf_model_pack_low(pack, replacements=replacements, align_blobs=0x80)
    return bytes(out), int(changed)


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


def _tri_bones_used(
    bones_by_v: List[List[int]],
    tri_verts: Tuple[int, int, int],
) -> List[int]:
    used: Dict[int, None] = {}
    for vi in tri_verts:
        if 0 <= int(vi) < len(bones_by_v):
            for bi in bones_by_v[int(vi)]:
                used[int(bi)] = None
    return list(used.keys())


def _collect_tris_for_material(
    obj: bpy.types.Object,
    *,
    material_name: str,
) -> List[Tuple[int, int, int, int, int, int]]:
    if obj.type != "MESH":
        return []
    try:
        if getattr(obj, "mode", "") == "EDIT":
            obj.update_from_editmode()
    except Exception:
        pass
    mesh: bpy.types.Mesh = obj.data                            
    try:
        mesh.calc_loop_triangles()
    except Exception:
        pass

    def base(n: str) -> str:
        s = str(n or "")
        if len(s) > 4 and s[-4] == "." and s[-3:].isdigit():
            return s[:-4]
        return s

    target = base(str(material_name))
                                                                                     
    out: List[Tuple[int, int, int, int, int, int]] = []
    for tri in getattr(mesh, "loop_triangles", []) or []:
        try:
            pi = int(getattr(tri, "polygon_index", -1))
            if 0 <= pi < len(mesh.polygons):
                poly = mesh.polygons[pi]
                mi = int(getattr(poly, "material_index", -1))
                if 0 <= mi < len(getattr(obj, "material_slots", []) or []):
                    ms = obj.material_slots[mi]
                    bm = getattr(ms, "material", None)
                    bm_name = base(str(getattr(bm, "name", "") or ""))
                    if bm_name and bm_name != target:
                        continue
        except Exception:
            pass
        a, b, c = tri.vertices
        la, lb, lc = tri.loops
        out.append((int(a), int(b), int(c), int(la), int(lb), int(lc)))
    return out


def _collect_tris_all(
    obj: bpy.types.Object,
) -> List[Tuple[int, int, int, int, int, int]]:
    if obj.type != "MESH":
        return []
    try:
        if getattr(obj, "mode", "") == "EDIT":
            obj.update_from_editmode()
    except Exception:
        pass
    mesh: bpy.types.Mesh = obj.data                            
    try:
        mesh.calc_loop_triangles()
    except Exception:
        pass
    out: List[Tuple[int, int, int, int, int, int]] = []
    for tri in getattr(mesh, "loop_triangles", []) or []:
        a, b, c = tri.vertices
        la, lb, lc = tri.loops
        out.append((int(a), int(b), int(c), int(la), int(lb), int(lc)))
    return out


def _apply_uv_strategy_to_mesh(
    mesh: bpy.types.Mesh,
    *,
    strategy: str,
    tol: float = 1e-6,
) -> None:
    """Mutate UVs on `mesh` according to `strategy`.

    - `DUPLICATE`: keep per-loop UVs as-is (seams preserved; exporter may duplicate verts)
    - `SMEAR`: force one UV per 3D vertex (collapses seams; may smear)
    - `STITCH_TRANSLATE`: try to remove seams by translating UV islands to align (translation-only heuristic)
    """
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


def _build_temp_mesh_object_for_slot(
    *,
    name: str,
    src_obj: bpy.types.Object,
    src_mesh: bpy.types.Mesh,
    tri_verts: List[Tuple[int, int, int, int, int, int]],
    weights_override_by_src_vi: Optional[Dict[int, List[Tuple[str, float]]]] = None,
) -> bpy.types.Object:
    """Create a temporary object whose mesh contains only the specified triangles.

    Vertex positions are baked in world space (obj.matrix_world applied). Vertex groups
    are copied for the used vertices only.
    """
    tmp_mesh = bpy.data.meshes.new(name)

    if not tri_verts:
        tmp_mesh.from_pydata([], [], [])
        tmp_obj = bpy.data.objects.new(name + "_obj", tmp_mesh)
        return tmp_obj

    mw = Matrix(src_obj.matrix_world)
    verts_world: List[Tuple[float, float, float]] = []
    faces: List[Tuple[int, int, int]] = []
    face_uvs: List[
        Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]
    ] = []
    remap: Dict[int, int] = {}
    src_vs = src_mesh.vertices
    uv_src = None
    try:
        if getattr(src_mesh, "uv_layers", None):
            uv_src = src_mesh.uv_layers.active or src_mesh.uv_layers[0]
    except Exception:
        uv_src = None

                                             
    for a, b, c, la, lb, lc in tri_verts:
        for vi in (int(a), int(b), int(c)):
            if vi not in remap:
                remap[vi] = len(verts_world)
                co = mw @ Vector(src_vs[vi].co)
                verts_world.append((float(co.x), float(co.y), float(co.z)))
        faces.append((remap[int(a)], remap[int(b)], remap[int(c)]))
        if uv_src is not None:
            try:
                uva = uv_src.data[int(la)].uv
                uvb = uv_src.data[int(lb)].uv
                uvc = uv_src.data[int(lc)].uv
                face_uvs.append(
                    (
                        (float(uva.x), float(uva.y)),
                        (float(uvb.x), float(uvb.y)),
                        (float(uvc.x), float(uvc.y)),
                    )
                )
            except Exception:
                face_uvs.append(((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)))
        else:
            face_uvs.append(((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)))

    tmp_mesh.from_pydata(verts_world, [], faces)
                                                                                            
    if uv_src is not None and face_uvs:
        try:
            uv_dst = tmp_mesh.uv_layers.new(name=str(getattr(uv_src, "name", "UVMap")))
            for fi, poly in enumerate(tmp_mesh.polygons):
                if fi >= len(face_uvs):
                    break
                luvs = face_uvs[fi]
                for ci, li in enumerate(poly.loop_indices):
                    if ci >= 3:
                        break
                    u, v = luvs[ci]
                    uv_dst.data[int(li)].uv = (float(u), float(v))
        except Exception:
            pass
    try:
        tmp_mesh.calc_normals()
    except Exception:
        pass
    try:
        tmp_mesh.calc_loop_triangles()
    except Exception:
        pass

    tmp_obj = bpy.data.objects.new(name + "_obj", tmp_mesh)

                                 
                                         
    vg_names: Dict[int, str] = {}
    try:
        for i, vg in enumerate(src_obj.vertex_groups):
            vg_names[int(i)] = str(vg.name)
    except Exception:
        vg_names = {}

    dst_groups: Dict[str, bpy.types.VertexGroup] = {}
    for src_vi, dst_vi in remap.items():
        if weights_override_by_src_vi is not None:
            ow = weights_override_by_src_vi.get(int(src_vi))
        else:
            ow = None
        if ow:
            for gname, w in ow:
                try:
                    if float(w) <= 0.0:
                        continue
                    gname_s = str(gname)
                    vg_dst = dst_groups.get(gname_s)
                    if vg_dst is None:
                        vg_dst = tmp_obj.vertex_groups.new(name=gname_s)
                        dst_groups[gname_s] = vg_dst
                    vg_dst.add([int(dst_vi)], float(w), "REPLACE")
                except Exception:
                    continue
            continue
        try:
            v = src_vs[int(src_vi)]
        except Exception:
            continue
        for g in getattr(v, "groups", []) or []:
            try:
                w = float(g.weight)
                if w <= 0.0:
                    continue
                gname = vg_names.get(int(g.group))
                if not gname:
                    continue
                vg_dst = dst_groups.get(gname)
                if vg_dst is None:
                    vg_dst = tmp_obj.vertex_groups.new(name=gname)
                    dst_groups[gname] = vg_dst
                vg_dst.add([int(dst_vi)], float(w), "REPLACE")
            except Exception:
                continue

    return tmp_obj


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
    """Return mapping: submesh_index -> list of loop triangle tuples (v0,v1,v2,l0,l1,l2).

    Drops triangles that can't fit any slot for this material; reports counts.
    """
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
