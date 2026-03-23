
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


def _pica_iter_cmds_with_param_indices(
    cmds: Sequence[int],
) -> Iterable[Tuple[int, int, List[int]]]:
    i = 0
    n = int(len(cmds))
    while i + 1 < n:
        param0 = int(cmds[i])
        cmd = int(cmds[i + 1])
        start_param_index = int(i)
        i += 2

        reg = int(cmd & 0xFFFF)
        extra = int((cmd >> 20) & 0x7FF)
        consecutive = (cmd >> 31) != 0
        if consecutive:
                                                           
            for j in range(extra + 1):
                yield (int(reg + j), int(start_param_index + j), [int(param0)])
                if j < extra:
                    if i >= n:
                        break
                    param0 = int(cmds[i])
                    i += 1
        else:
            params = [int(param0)]
            for _ in range(extra):
                if i >= n:
                    break
                params.append(int(cmds[i]))
                i += 1
            yield (int(reg), int(start_param_index), params)

        if (i & 1) != 0:
            i += 1


def _patch_pack_topology_tris_in_place(
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
                f"Topology in-place patch currently supports primitive_mode=0 (Triangles) only; submesh {sm.name!r} has {int(sm.primitive_mode)}"
            )

                                           
        try:
            mesh.calc_loop_triangles()                              
        except Exception:
            pass
        tris = getattr(mesh, "loop_triangles", None)
        new_indices: List[int] = []
        if tris:
            for tri in tris:
                a, b, c = tri.vertices
                new_indices.extend([int(a), int(b), int(c)])

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

        idx_len = int(getattr(sm, "index_data_len", 0) or 0)
        if idx_len <= 0:
            raise ValueError(
                f"Missing/invalid index_data_len for submesh {sm.name!r}: {idx_len}"
            )
        max_indices = int(idx_len) // int(elem_size)
        if int(len(new_indices)) > max_indices:
            raise ValueError(
                f"New index stream too large for submesh {sm.name!r}: new={len(new_indices)} max={max_indices} (idx_len={idx_len}, elem_size={elem_size})"
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

                        
        if elem_size == 2:
            new_bytes = b"".join(struct.pack("<H", int(i)) for i in new_indices)
            zero = b"\x00\x00"
        else:
            new_bytes = bytes(int(i) & 0xFF for i in new_indices)
            zero = b"\x00"

        new_raw_len = int(len(new_bytes))
        if base < 0 or base + idx_len > len(out):
            raise ValueError("Index write out of range (bad offsets/length)")

        old_bytes = bytes(out[base : base + idx_len])
                                                                   
        out[base : base + idx_len] = new_bytes + (
            zero * (max_indices - len(new_indices))
        )
        if bytes(out[base : base + idx_len]) != old_bytes:
            changed += 1

                                                          
        idx_count_off = int(getattr(sm, "index_count_off", 0) or 0)
        if idx_count_off <= 0 or idx_count_off + 4 > len(out):
            raise ValueError(
                f"Missing/invalid index_count_off for submesh {sm.name!r}: {idx_count_off}"
            )
        old_decl = struct.unpack_from("<i", out, idx_count_off)[0]
        if int(old_decl) != int(len(sm.indices)):
                                                                                          
            pass
        struct.pack_into("<i", out, idx_count_off, int(len(new_indices)))

                                                                                                   
        index_cmds_off = int(getattr(sm, "index_cmds_off", 0) or 0)
        index_cmds_len_u32 = int(getattr(sm, "index_cmds_len_u32", 0) or 0)
        if index_cmds_off <= 0 or index_cmds_len_u32 <= 0:
            raise ValueError(
                f"Missing/invalid index_cmds_off/len for submesh {sm.name!r}: off={index_cmds_off} len_u32={index_cmds_len_u32}"
            )
        end = index_cmds_off + index_cmds_len_u32 * 4
        if index_cmds_off < 0 or end > len(out):
            raise ValueError("index_cmds range out of file bounds")
        patched = _pica_patch_reg_all_in_cmd_bytes(
            out,
            cmd_u32_off=index_cmds_off,
            cmd_u32_len=index_cmds_len_u32,
            reg=0x0228,
            new_param=int(len(new_indices)),
        )
        if patched == 0:
            raise ValueError(
                f"Could not find PICA reg 0x0228 in index_cmds for {sm.name!r}"
            )

    return bytes(out), int(changed)


