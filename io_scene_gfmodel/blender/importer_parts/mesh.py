
from __future__ import annotations

import json
import os
import struct
from typing import Dict, List, Optional, Sequence, Tuple

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, StringProperty
from bpy_extras.io_utils import ImportHelper, axis_conversion
from mathutils import Matrix, Quaternion, Vector

from ...core.io import _load_any
from ...core.patch_plan import PatchPlan, steps_to_breadcrumb
from ...core.pica import (
    _bgra_to_rgba_floats,
    _decode_rgba_u32,
    _decode_texenv_update_buffer,
    _flip_bgra_y,
    _pica_decode_bitmap_to_bgra,
)
from ...core.types import (
    _GFMaterial,
    _GFModel,
    _GFMotion,
    _GFShader,
    _GFSubMesh,
    _GFTexture,
)
from ..anim import (
    _apply_uv_anim_enable,
    _apply_visibility_anim_enable,
    _compute_rest_world_mats,
    _euler_to_quat_xyz,
    _gf_runtime_cache_armature,
    _mot_eval,
    _transform_quat_basis,
)


def _bone_uses_ssc(flags: int) -> bool:
    return (int(flags) & 0x02) != 0


def _triangulate(indices: Sequence[int], prim: int) -> List[Tuple[int, int, int]]:
    if prim == 0:
        tris: List[Tuple[int, int, int]] = []
        for i in range(0, len(indices) - 2, 3):
            a, b, c = indices[i], indices[i + 1], indices[i + 2]
            if a == b or b == c or a == c:
                continue
            tris.append((a, b, c))
        return tris
    if prim == 1:
        tris = []
        flip = False
        for i in range(len(indices) - 2):
            a, b, c = indices[i], indices[i + 1], indices[i + 2]
            if a == b or b == c or a == c:
                flip = not flip
                continue
            tris.append((a, c, b) if flip else (a, b, c))
            flip = not flip
        return tris
    if prim == 2:
        if len(indices) < 3:
            return []
        center = indices[0]
        tris = []
        for i in range(1, len(indices) - 1):
            a, b = indices[i], indices[i + 1]
            if center == a or a == b or center == b:
                continue
            tris.append((center, a, b))
        return tris
    return []


def _read_vertices(
    submesh: _GFSubMesh,
) -> Tuple[
    List[Vector],
    List[Vector],
    List[Vector],
    List[Vector],
    List[List[Tuple[int, float]]],
]:
    buf = memoryview(submesh.raw_buffer)
    stride = submesh.vertex_stride
    if stride <= 0:
        return [], [], [], []
    count = len(buf) // stride

    pos: List[Vector] = [Vector((0.0, 0.0, 0.0))] * count
    nrm: List[Vector] = [Vector((0.0, 0.0, 1.0))] * count
    uv0: List[Vector] = [Vector((0.0, 0.0))] * count
    col: List[Vector] = [Vector((1.0, 1.0, 1.0, 1.0))] * count
    weights: List[List[Tuple[int, float]]] = [[] for _ in range(count)]

    fixed_indices = next((fa for fa in submesh.fixed_attributes if fa.name == 7), None)
    fixed_weights = next((fa for fa in submesh.fixed_attributes if fa.name == 8), None)
    fixed_color = next((fa for fa in submesh.fixed_attributes if fa.name == 3), None)

    for vi in range(count):
        o = vi * stride

        local = o
        bone_indices: List[int] = []
        bone_weights: List[float] = []

        def align2(fmt: int) -> None:
            nonlocal local
            if fmt not in (0, 1):
                local += local & 1

        def read_elem(fmt: int) -> float:
            nonlocal local
            if fmt == 0:
                v = struct.unpack_from("<b", buf, local)[0]
                local += 1
                return float(v)
            if fmt == 1:
                v = buf[local]
                local += 1
                return float(v)
            if fmt == 2:
                v = struct.unpack_from("<h", buf, local)[0]
                local += 2
                return float(v)
            v = struct.unpack_from("<f", buf, local)[0]
            local += 4
            return float(v)

        for attr in submesh.attributes:
            align2(attr.fmt)
            elems = [0.0, 0.0, 0.0, 0.0]
            for ei in range(attr.elements):
                elems[ei] = read_elem(attr.fmt)
            v4 = Vector((elems[0], elems[1], elems[2], elems[3])) * attr.scale
            if attr.name == 0:
                pos[vi] = Vector((v4.x, v4.y, v4.z))
            elif attr.name == 1:
                nrm[vi] = Vector((v4.x, v4.y, v4.z))
            elif attr.name == 3:
                col[vi] = Vector((v4.x, v4.y, v4.z, v4.w))
            elif attr.name == 4:
                uv0[vi] = Vector((v4.x, v4.y))
            elif attr.name == 7:
                bone_indices.extend(
                    [int(v4.x), int(v4.y), int(v4.z), int(v4.w)][: attr.elements]
                )
            elif attr.name == 8:
                bone_weights.extend(
                    [float(v4.x), float(v4.y), float(v4.z), float(v4.w)][
                        : attr.elements
                    ]
                )

        if fixed_color is not None and (col[vi].x, col[vi].y, col[vi].z, col[vi].w) == (
            1.0,
            1.0,
            1.0,
            1.0,
        ):
            col[vi] = Vector(
                (
                    float(fixed_color.x),
                    float(fixed_color.y),
                    float(fixed_color.z),
                    float(fixed_color.w),
                )
            )

        if (not bone_indices) and fixed_indices is not None:
            bone_indices = [
                int(fixed_indices.x),
                int(fixed_indices.y),
                int(fixed_indices.z),
                int(fixed_indices.w),
            ]
        if (not bone_weights) and fixed_weights is not None:
            bone_weights = [
                float(fixed_weights.x),
                float(fixed_weights.y),
                float(fixed_weights.z),
                float(fixed_weights.w),
            ]

        wlist: List[Tuple[int, float]] = []
        for bi, bw in zip(bone_indices, bone_weights):
            if bw <= 0:
                continue
            wlist.append((bi, bw))
        weights[vi] = wlist

    return pos, nrm, uv0, col, weights


