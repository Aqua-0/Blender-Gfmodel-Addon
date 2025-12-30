"""Vertex attribute packing and weight extraction helpers."""

from __future__ import annotations

import struct
from typing import Dict, List, Optional, Tuple

import bpy
from mathutils import Vector

from ...core.types import _GFSubMesh


def _pack_attr_value(fmt: int, scale: float, v: float) -> bytes:
    inv = 1.0 / float(scale) if float(scale) != 0.0 else 1.0
    x = float(v) * inv
    if fmt == 0:      
        n = float(round(x))
        if abs(x - n) < 1e-6:
            x = n
        xi = int(round(x))
        xi = max(-128, min(127, xi))
        return struct.pack("<b", int(xi))
    if fmt == 1:      
        n = float(round(x))
        if abs(x - n) < 1e-6:
            x = n
        xi = int(round(x))
        xi = max(0, min(255, xi))
        return int(xi).to_bytes(1, "little", signed=False)
    if fmt == 2:       
        n = float(round(x))
        if abs(x - n) < 1e-6:
            x = n
        xi = int(round(x))
        xi = max(-32768, min(32767, xi))
        return struct.pack("<h", int(xi))
    return struct.pack("<f", float(x))


def _pack_vertex_bytes(
    sm: _GFSubMesh,
    *,
    position: Vector,
    normal: Vector,
    uv0: Tuple[float, float],
    weights: List[Tuple[int, float]],
) -> bytes:
    stride = int(sm.vertex_stride)
    if stride <= 0:
        raise ValueError("vertex_stride<=0")
    buf = bytearray(stride)
    local = 0

    def align2(fmt: int) -> None:
        nonlocal local
        if fmt not in (0, 1):
            local += local & 1

    for attr in sm.attributes:
        align2(int(attr.fmt))
        elems = [0.0, 0.0, 0.0, 1.0]
        if attr.name == 0:            
            elems = [float(position.x), float(position.y), float(position.z), 1.0]
        elif attr.name == 1:          
            elems = [float(normal.x), float(normal.y), float(normal.z), 0.0]
        elif attr.name == 4:       
            u, v = uv0
            elems = [float(u), float(v), 0.0, 0.0]
        elif attr.name == 3:         
            elems = [1.0, 1.0, 1.0, 1.0]
        elif attr.name == 7:                               
            bi = [0, 0, 0, 0]
            for i, (pi, _w) in enumerate(weights[:4]):
                bi[i] = int(pi)
            elems = [float(bi[0]), float(bi[1]), float(bi[2]), float(bi[3])]
        elif attr.name == 8:              
            bw = [0.0, 0.0, 0.0, 0.0]
            for i, (_pi, wgt) in enumerate(weights[:4]):
                bw[i] = float(wgt)
            elems = bw

        for ei in range(int(attr.elements)):
            b = _pack_attr_value(int(attr.fmt), float(attr.scale), float(elems[ei]))
            buf[local : local + len(b)] = b
            local += len(b)

    return bytes(buf)


def _vertex_attr_offsets(sm: _GFSubMesh) -> Dict[int, int]:
    stride = int(sm.vertex_stride)
    if stride <= 0:
        raise ValueError("vertex_stride<=0")
    local = 0
    out: Dict[int, int] = {}

    def align2(fmt: int) -> None:
        nonlocal local
        if fmt not in (0, 1):
            local += local & 1

    for attr in sm.attributes:
        align2(int(attr.fmt))
        out[int(attr.name)] = int(local)
        for _ei in range(int(attr.elements)):
            local += len(_pack_attr_value(int(attr.fmt), float(attr.scale), 0.0))
    if local > stride:
        raise ValueError(f"vertex attribute layout exceeds stride: {local} > {stride}")
    return out


def _pack_submesh_vertex_buffer(
    sm: _GFSubMesh,
    *,
    positions: List[Vector],
    normals: List[Vector],
    uvs: List[Tuple[float, float]],
    weights: List[List[Tuple[int, float]]],
) -> bytes:
    stride = int(sm.vertex_stride)
    if stride <= 0:
        raise ValueError("vertex_stride<=0")
    count = len(positions)
    buf = bytearray(count * stride)

    for vi in range(count):
        local = vi * stride

        def align2(fmt: int) -> None:
            nonlocal local
            if fmt not in (0, 1):
                local += local & 1

        for attr in sm.attributes:
            align2(int(attr.fmt))
            elems = [0.0, 0.0, 0.0, 1.0]
            if attr.name == 0:            
                p = positions[vi]
                elems = [float(p.x), float(p.y), float(p.z), 1.0]
            elif attr.name == 1:          
                n = normals[vi]
                elems = [float(n.x), float(n.y), float(n.z), 0.0]
            elif attr.name == 4:       
                u, v = uvs[vi]
                elems = [float(u), float(v), 0.0, 0.0]
            elif attr.name == 3:         
                elems = [1.0, 1.0, 1.0, 1.0]
            elif attr.name == 7:                               
                bi = [0, 0, 0, 0]
                wl = weights[vi]
                for i, (pi, _w) in enumerate(wl[:4]):
                    bi[i] = int(pi)
                elems = [float(bi[0]), float(bi[1]), float(bi[2]), float(bi[3])]
            elif attr.name == 8:              
                bw = [0.0, 0.0, 0.0, 0.0]
                wl = weights[vi]
                for i, (_pi, wgt) in enumerate(wl[:4]):
                    bw[i] = float(wgt)
                elems = bw

            for ei in range(int(attr.elements)):
                b = _pack_attr_value(int(attr.fmt), float(attr.scale), float(elems[ei]))
                buf[local : local + len(b)] = b
                local += len(b)

    return bytes(buf)


def _gather_weights_palette_indices(
    obj: bpy.types.Object, sm: _GFSubMesh, skeleton_names: List[str]
) -> List[List[Tuple[int, float]]]:
    palette = list(sm.bone_indices or [])[: int(sm.bone_indices_count)]
    skel_name_to_index = {n: i for i, n in enumerate(skeleton_names)}
    skel_to_palette: Dict[int, int] = {int(s): int(i) for i, s in enumerate(palette)}

    has_dynamic = any(a.name in (7, 8) for a in sm.attributes)
    if not has_dynamic:
        return [[] for _ in range(len(obj.data.vertices))]                            

    vg_by_index = {vg.index: vg.name for vg in obj.vertex_groups}
    weights: List[List[Tuple[int, float]]] = []
    mesh: bpy.types.Mesh = obj.data                            
    for v in mesh.vertices:
        wl: List[Tuple[int, float]] = []
        for g in v.groups:
            bone_name = vg_by_index.get(int(g.group))
            if bone_name is None:
                continue
            sk = skel_name_to_index.get(bone_name)
            if sk is None:
                continue
            pi = skel_to_palette.get(int(sk))
            if pi is None:
                continue
            wl.append((int(pi), float(g.weight)))
        wl.sort(key=lambda t: t[1], reverse=True)
        wl = wl[:4]
        s = sum(w for _, w in wl)
        if s > 0:
            wl = [(pi, w / s) for pi, w in wl]
        weights.append(wl)
    return weights


def _gather_weights_palette_indices_checked(
    obj: bpy.types.Object, sm: _GFSubMesh, skeleton_names: List[str]
) -> Tuple[List[List[Tuple[int, float]]], List[str], List[str]]:
    palette = list(sm.bone_indices or [])[: int(sm.bone_indices_count)]
    skel_name_to_index = {n: i for i, n in enumerate(skeleton_names)}
    skel_to_palette: Dict[int, int] = {int(s): int(i) for i, s in enumerate(palette)}

    has_dynamic = any(a.name in (7, 8) for a in sm.attributes)
    if not has_dynamic:
        return (
            [[] for _ in range(len(obj.data.vertices))],                            
            [],
            [],
        )

    vg_by_index = {vg.index: vg.name for vg in obj.vertex_groups}
    unknown = set()
    not_in_palette = set()

    weights: List[List[Tuple[int, float]]] = []
    mesh: bpy.types.Mesh = obj.data                            
    for v in mesh.vertices:
        wl: List[Tuple[int, float]] = []
        for g in v.groups:
            if float(g.weight) <= 0.0:
                continue
            bone_name = vg_by_index.get(int(g.group))
            if bone_name is None:
                continue
            sk = skel_name_to_index.get(str(bone_name))
            if sk is None:
                unknown.add(str(bone_name))
                continue
            pi = skel_to_palette.get(int(sk))
            if pi is None:
                not_in_palette.add(str(bone_name))
                continue
            wl.append((int(pi), float(g.weight)))
        wl.sort(key=lambda t: t[1], reverse=True)
        wl = wl[:4]
        s = sum(w for _, w in wl)
        if s > 0:
            wl = [(pi, w / s) for pi, w in wl]
        weights.append(wl)

    return weights, sorted(unknown), sorted(not_in_palette)


def _gather_weights_skeleton_indices_checked(
    obj: bpy.types.Object, skeleton_names: List[str]
) -> Tuple[List[List[Tuple[int, float]]], List[str]]:
    vg_by_index = {vg.index: vg.name for vg in obj.vertex_groups}
    skel_name_to_index = {n: i for i, n in enumerate(skeleton_names)}

    unknown = set()
    weights: List[List[Tuple[int, float]]] = []
    mesh: bpy.types.Mesh = obj.data                            
    for v in mesh.vertices:
        wl: List[Tuple[int, float]] = []
        for g in v.groups:
            w = float(g.weight)
            if w <= 0.0:
                continue
            bone_name = vg_by_index.get(int(g.group))
            if bone_name is None:
                continue
            sk = skel_name_to_index.get(str(bone_name))
            if sk is None:
                unknown.add(str(bone_name))
                continue
            wl.append((int(sk), w))
        wl.sort(key=lambda t: t[1], reverse=True)
        wl = wl[:4]
        s = float(sum(w for _sk, w in wl))
        if s > 0.0:
            wl = [(sk, w / s) for sk, w in wl]
        weights.append(wl)
    return weights, sorted(unknown)
