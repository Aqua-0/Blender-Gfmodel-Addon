

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


def _patch_pack_indices_in_place(
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

        if int(sm.primitive_mode) != 0:
            raise ValueError(
                f"Index in-place patch currently supports primitive_mode=0 (Triangles) only; submesh {sm.name!r} has {int(sm.primitive_mode)}"
            )


        try:
            mesh.calc_loop_triangles()
        except Exception:
            pass
        tris = getattr(mesh, "loop_triangles", None)
        if not tris:
            new_indices: List[int] = []
        else:
            new_indices = []
            for tri in tris:
                a, b, c = tri.vertices
                new_indices.extend([int(a), int(b), int(c)])

        old_count = int(len(sm.indices))
        if int(len(new_indices)) != old_count:
            raise ValueError(
                f"Index count mismatch for submesh {sm.name!r}: scene={len(new_indices)} file={old_count} (index-only edits must keep triangle/index count identical)"
            )

        elem_size = int(getattr(sm, "index_elem_size", 0) or 0)
        if elem_size not in (1, 2):
            raise ValueError(
                f"Missing/invalid index_elem_size for submesh {sm.name!r}: {elem_size}"
            )
        base = int(getattr(sm, "index_buffer_off", 0) or 0)
        if base <= 0:
            raise ValueError(
                f"Missing/invalid index_buffer_off for submesh {sm.name!r}: {base}"
            )

        max_index = int(sm.vertex_count) - 1
        for i in new_indices:
            if i < 0 or i > max_index:
                raise ValueError(
                    f"Index out of range for submesh {sm.name!r}: {i} (vcount={int(sm.vertex_count)})"
                )
            if elem_size == 1 and i > 0xFF:
                raise ValueError(
                    f"Index too large for u8 index buffer for submesh {sm.name!r}: {i}"
                )


        raw_len = int(old_count) * int(elem_size)
        if base < 0 or base + raw_len > len(out):
            raise ValueError("Index write out of range (bad offsets/length)")

        old_bytes = bytes(out[base : base + raw_len])
        if elem_size == 2:
            new_bytes = b"".join(struct.pack("<H", int(i)) for i in new_indices)
        else:
            new_bytes = bytes(int(i) & 0xFF for i in new_indices)

        if len(new_bytes) != raw_len:
            raise ValueError("Internal error: encoded index byte length mismatch")
        if old_bytes != new_bytes:
            out[base : base + raw_len] = new_bytes

            if elem_size == 2:
                for j in range(old_count):
                    if old_bytes[j * 2 : j * 2 + 2] != new_bytes[j * 2 : j * 2 + 2]:
                        changed += 1
            else:
                for j in range(old_count):
                    if old_bytes[j] != new_bytes[j]:
                        changed += 1

    return bytes(out), int(changed)


