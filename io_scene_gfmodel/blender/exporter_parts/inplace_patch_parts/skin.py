
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


def _patch_pack_skin_in_place(
    pack_src: bytes,
    model: "_GFModel",
    *,
    tagged: Dict[int, bpy.types.Object],
    skeleton_names: List[str],
) -> Tuple[bytes, int, int]:
    out = bytearray(pack_src)
    changed = 0
    fallback = 0

    for submesh_index, sm in enumerate(model.submeshes):
        obj = tagged.get(int(submesh_index))
        if obj is None:
            continue
        mesh: bpy.types.Mesh = obj.data
        if int(len(mesh.vertices)) != int(sm.vertex_count):
            raise ValueError(
                f"Vertex count mismatch for submesh {sm.name!r}: scene={len(mesh.vertices)} file={sm.vertex_count}"
            )

        attr_names = set(int(a.name) for a in (sm.attributes or []))
        if 7 not in attr_names and 8 not in attr_names:

            continue
        if not (7 in attr_names and 8 in attr_names):
            raise ValueError(
                f"Submesh {sm.name!r} has partial skinning attributes (need both 7 and 8)"
            )

        offs = _vertex_attr_offsets(sm)
        bi_off = offs.get(7)
        bw_off = offs.get(8)
        if bi_off is None or bw_off is None:
            raise ValueError("Missing BoneIndex/BoneWeight offsets")

        bi_attr = next((a for a in sm.attributes if int(a.name) == 7), None)
        bw_attr = next((a for a in sm.attributes if int(a.name) == 8), None)
        if bi_attr is None or bw_attr is None:
            raise ValueError("Missing BoneIndex/BoneWeight attributes")

        bi_elems = int(bi_attr.elements)
        bw_elems = int(bw_attr.elements)
        if bi_elems <= 0 or bw_elems <= 0:
            raise ValueError("Invalid BoneIndex/BoneWeight element counts")
        elems = min(4, bi_elems, bw_elems)

        bi_comp = len(_pack_attr_value(int(bi_attr.fmt), float(bi_attr.scale), 0.0))
        bw_comp = len(_pack_attr_value(int(bw_attr.fmt), float(bw_attr.scale), 0.0))

        stride = int(sm.vertex_stride)
        base = int(getattr(sm, "raw_buffer_off", 0))
        if base <= 0:
            raise ValueError("Missing/invalid raw_buffer_off for submesh")

        weights_by_v, unknown_bones, not_in_palette = (
            _gather_weights_palette_indices_checked(obj, sm, skeleton_names)
        )
        if unknown_bones:
            sample = ", ".join(unknown_bones[:10])
            raise ValueError(
                f"Vertex groups reference bones not in skeleton (first {min(10, len(unknown_bones))}): {sample}"
            )
        if not_in_palette:
            sample = ", ".join(not_in_palette[:10])
            raise ValueError(
                f"Vertex groups reference bones not in this submesh palette (first {min(10, len(not_in_palette))}): {sample}"
            )
        pal_count = int(getattr(sm, "bone_indices_count", 0) or 0)
        if pal_count <= 0:
            raise ValueError("Submesh has no bone palette (bone_indices_count<=0)")

        for vi in range(int(sm.vertex_count)):
            wl = weights_by_v[vi] if vi < len(weights_by_v) else []
            indices = [0] * elems
            weights = [0.0] * elems
            for i, (pi, w) in enumerate(wl[:elems]):
                if not (0 <= int(pi) < pal_count):
                    raise ValueError(
                        f"Palette index out of range at v={vi}: {pi} (pal_count={pal_count})"
                    )
                indices[i] = int(pi)
                weights[i] = float(w)
            s = float(sum(max(0.0, w) for w in weights))
            if s > 0:
                weights = [max(0.0, w) / s for w in weights]
            else:


                indices[0] = 0
                weights[0] = 1.0
                fallback += 1

            vbi = int(base) + vi * stride + int(bi_off)
            vbw = int(base) + vi * stride + int(bw_off)
            if vbi < 0 or vbi + bi_comp * elems > len(out):
                raise ValueError("BoneIndex write out of range (bad offsets/stride)")
            if vbw < 0 or vbw + bw_comp * elems > len(out):
                raise ValueError("BoneWeight write out of range (bad offsets/stride)")

            bi_new = b"".join(
                _pack_attr_value(int(bi_attr.fmt), float(bi_attr.scale), float(i))
                for i in indices
            )
            bw_new = b"".join(
                _pack_attr_value(int(bw_attr.fmt), float(bw_attr.scale), float(w))
                for w in weights
            )

            bi_old = bytes(out[vbi : vbi + len(bi_new)])
            bw_old = bytes(out[vbw : vbw + len(bw_new)])
            if bi_old != bi_new or bw_old != bw_new:
                out[vbi : vbi + len(bi_new)] = bi_new
                out[vbw : vbw + len(bw_new)] = bw_new
                changed += 1

    return bytes(out), int(changed), int(fallback)
