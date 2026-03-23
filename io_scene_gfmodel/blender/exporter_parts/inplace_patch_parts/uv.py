
from __future__ import annotations

import struct
from typing import Dict, List, Optional, Tuple

import bpy
from mathutils import Matrix, Vector

from ....core.types import _GFSubMesh
from ..common import mesh_tris_indices as _mesh_tris_indices
from ..common import (
    pica_iter_cmds_with_param_indices as _pica_iter_cmds_with_param_indices,
)
from ..common import (
    pica_patch_reg_all_in_cmd_bytes as _pica_patch_reg_all_in_cmd_bytes,
)
from ..vertex_pack import (
    _gather_weights_palette_indices_checked,
    _gather_weights_skeleton_indices_checked,
    _pack_attr_value,
    _pack_submesh_vertex_buffer,
    _vertex_attr_offsets,
)


def _patch_pack_uv0_in_place(
    pack_src: bytes,
    model: "_GFModel",
    *,
    tagged: Dict[int, bpy.types.Object],
) -> Tuple[bytes, int]:
    out = bytearray(pack_src)
    changed = 0

    for submesh_index, sm in enumerate(model.submeshes):
        obj = tagged.get(int(submesh_index))
        if obj is None:
            continue
        mesh: bpy.types.Mesh = obj.data                            
        if int(len(mesh.vertices)) != int(sm.vertex_count):
            raise ValueError(
                f"Vertex count mismatch for submesh {sm.name!r}: scene={len(mesh.vertices)} file={sm.vertex_count}"
            )

        uv_layer = None
        if getattr(mesh, "uv_layers", None):
            uv_layer = mesh.uv_layers.active or mesh.uv_layers[0]
        if uv_layer is None:
            continue

                                                                                             
                                                      
        uv_by_v: List[Optional[Tuple[float, float]]] = [None] * int(len(mesh.vertices))
        try:
            for poly in mesh.polygons:
                for li in poly.loop_indices:
                    vi = int(mesh.loops[li].vertex_index)
                    if 0 <= vi < len(uv_by_v) and uv_by_v[vi] is None:
                        uv = uv_layer.data[li].uv
                        uv_by_v[vi] = (float(uv.x), float(uv.y))
        except Exception:
            pass
        uv_by_v2: List[Tuple[float, float]] = [
            (0.0, 0.0) if v is None else v for v in uv_by_v
        ]

        offs = _vertex_attr_offsets(sm)
        uv_off = offs.get(4)
        if uv_off is None:
            continue
        uv_attr = next((a for a in sm.attributes if int(a.name) == 4), None)
        if uv_attr is None or int(uv_attr.elements) < 2:
            continue

        comp_size = len(_pack_attr_value(int(uv_attr.fmt), float(uv_attr.scale), 0.0))
        stride = int(sm.vertex_stride)
        base = int(getattr(sm, "raw_buffer_off", 0))
        if base <= 0:
            raise ValueError("Missing/invalid raw_buffer_off for submesh")

        for i, (u, v) in enumerate(uv_by_v2):
            vb = int(base) + int(i) * stride + int(uv_off)
            if vb < 0 or vb + comp_size * 2 > len(out):
                raise ValueError("UV write out of range (bad offsets/stride)")
            bu = _pack_attr_value(int(uv_attr.fmt), float(uv_attr.scale), float(u))
            bv = _pack_attr_value(int(uv_attr.fmt), float(uv_attr.scale), float(v))
            old = bytes(out[vb : vb + comp_size * 2])
            new = bu + bv
            if old != new:
                out[vb : vb + comp_size * 2] = new
                changed += 1

    return bytes(out), int(changed)


