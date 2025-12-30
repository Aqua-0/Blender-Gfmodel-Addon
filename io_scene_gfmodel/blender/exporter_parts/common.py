"""Shared helpers used by exporter/patch operators.

Important: PICA command parsing must match the file format used by GFModel mesh command
buffers. The command stream stores pairs of (param0, cmd) u32 words, plus optional extra
param words depending on the command header. Do not “simplify” this without verifying
against real files, because tools patch registers (notably 0x0228 index count).
"""

from __future__ import annotations

import struct
from typing import Iterable, List, Sequence, Tuple

import bpy


def align_up(n: int, a: int) -> int:
    a = int(a)
    if a <= 1:
        return int(n)
    n = int(n)
    return (n + (a - 1)) & ~(a - 1)


def mesh_get_uv0_per_vertex(mesh: bpy.types.Mesh) -> List[Tuple[float, float]]:
    if not mesh.uv_layers:
        return [(0.0, 0.0)] * len(mesh.vertices)
    uv_layer = mesh.uv_layers.active or mesh.uv_layers[0]
    uv_by_v = [None] * len(mesh.vertices)
    for poly in mesh.polygons:
        for li in poly.loop_indices:
            vi = mesh.loops[li].vertex_index
            if uv_by_v[vi] is None:
                uv = uv_layer.data[li].uv
                uv_by_v[vi] = (float(uv.x), float(uv.y))
    return [(0.0, 0.0) if v is None else v for v in uv_by_v]


def mesh_get_normals(mesh: bpy.types.Mesh) -> List[Tuple[float, float, float]]:
    mesh.calc_normals_split()
    return [
        (float(v.normal.x), float(v.normal.y), float(v.normal.z)) for v in mesh.vertices
    ]


def mesh_tris_indices(mesh: bpy.types.Mesh) -> List[int]:
    mesh.calc_loop_triangles()
    out: List[int] = []
    for tri in mesh.loop_triangles:
        out.extend([int(tri.vertices[0]), int(tri.vertices[1]), int(tri.vertices[2])])
    return out


def mesh_tris(mesh: bpy.types.Mesh) -> List[Tuple[int, int, int]]:
    mesh.calc_loop_triangles()
    return [
        (int(t.vertices[0]), int(t.vertices[1]), int(t.vertices[2]))
        for t in mesh.loop_triangles
    ]


def pica_iter_cmds_with_param_indices(
    cmds: Sequence[int],
) -> Iterable[Tuple[int, int, List[int]]]:
    """Yield (reg, first_param_index, params_u32) for a PICA command stream."""
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


def pica_patch_reg_all_in_cmd_bytes(
    cmd_bytes: bytearray,
    *,
    cmd_u32_off: int,
    cmd_u32_len: int,
    reg: int,
    new_param: int,
) -> int:
    if cmd_u32_len <= 0:
        return 0
    cmds = list(
        struct.unpack_from(f"<{int(cmd_u32_len)}I", cmd_bytes, int(cmd_u32_off))
    )
    patched = 0
    for r, param_index, _params in pica_iter_cmds_with_param_indices(cmds):
        if int(r) == int(reg):
            struct.pack_into(
                "<I",
                cmd_bytes,
                int(cmd_u32_off) + int(param_index) * 4,
                int(new_param) & 0xFFFFFFFF,
            )
            patched += 1
    return int(patched)
