

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

def _build_temp_mesh_object_for_slot(
    *,
    name: str,
    src_obj: bpy.types.Object,
    src_mesh: bpy.types.Mesh,
    tri_verts: List[Tuple[int, int, int, int, int, int]],
    weights_override_by_src_vi: Optional[Dict[int, List[Tuple[str, float]]]] = None,
) -> bpy.types.Object:

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


def _build_temp_mesh_object_for_sources(
    *,
    name: str,
    sources: List[
        Tuple[
            bpy.types.Object,
            bpy.types.Mesh,
            List[Tuple[int, int, int, int, int, int]],
            Optional[Dict[int, List[Tuple[str, float]]]],
        ]
    ],
) -> bpy.types.Object:


    tmp_mesh = bpy.data.meshes.new(name)

    verts_world: List[Tuple[float, float, float]] = []
    faces: List[Tuple[int, int, int]] = []
    face_uvs: List[
        Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]
    ] = []

    per_source: List[
        Tuple[
            bpy.types.Object,
            bpy.types.Mesh,
            Dict[int, int],
            Optional[Dict[int, List[Tuple[str, float]]]],
        ]
    ] = []

    for src_obj, src_mesh, tri_verts, weights_override_by_src_vi in sources:
        if src_obj is None or getattr(src_obj, "type", "") != "MESH":
            continue
        if not tri_verts:
            continue
        try:
            if getattr(src_obj, "mode", "") == "EDIT":
                src_obj.update_from_editmode()
        except Exception:
            pass

        mw = Matrix(src_obj.matrix_world)
        uv_src = None
        try:
            if getattr(src_mesh, "uv_layers", None):
                uv_src = src_mesh.uv_layers.active or src_mesh.uv_layers[0]
        except Exception:
            uv_src = None

        remap: Dict[int, int] = {}
        src_vs = src_mesh.vertices
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

        per_source.append((src_obj, src_mesh, remap, weights_override_by_src_vi))

    if not faces:
        tmp_mesh.from_pydata([], [], [])
        tmp_obj = bpy.data.objects.new(name + "_obj", tmp_mesh)
        return tmp_obj

    tmp_mesh.from_pydata(verts_world, [], faces)
    if face_uvs:
        try:
            uv_dst = tmp_mesh.uv_layers.new(name="UVMap")
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

    dst_groups: Dict[str, bpy.types.VertexGroup] = {}
    for src_obj, src_mesh, remap, weights_override_by_src_vi in per_source:
        vg_names: Dict[int, str] = {}
        try:
            for i, vg in enumerate(src_obj.vertex_groups):
                vg_names[int(i)] = str(vg.name)
        except Exception:
            vg_names = {}

        src_vs = src_mesh.vertices
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
