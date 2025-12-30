"""In-place patching helpers (edit existing buffers without reallocation)."""

from __future__ import annotations

import struct
from typing import Dict, List, Optional, Tuple

import bpy
from mathutils import Matrix, Vector

from ...core.types import _GFSubMesh
from .common import mesh_tris_indices as _mesh_tris_indices
from .common import (
    pica_iter_cmds_with_param_indices as _pica_iter_cmds_with_param_indices,
)
from .common import (
    pica_patch_reg_all_in_cmd_bytes as _pica_patch_reg_all_in_cmd_bytes,
)
from .vertex_pack import (
    _gather_weights_palette_indices_checked,
    _gather_weights_skeleton_indices_checked,
    _pack_attr_value,
    _pack_submesh_vertex_buffer,
    _vertex_attr_offsets,
)


def _patch_submesh_positions_in_place(
    sm: _GFSubMesh,
    *,
    obj: bpy.types.Object,
    gf_from_blender: Matrix,
    global_scale: float,
) -> Tuple[Vector, Vector]:
    mesh: bpy.types.Mesh = obj.data                            
    if int(len(mesh.vertices)) != int(sm.vertex_count):
        raise ValueError(
            f"Vertex count mismatch for submesh {sm.name!r}: scene={len(mesh.vertices)} file={sm.vertex_count}"
        )
    stride = int(sm.vertex_stride)
    raw = bytearray(sm.raw_buffer)
    if len(raw) != int(sm.vertex_count) * stride:
        raise ValueError(
            f"Raw buffer size mismatch for submesh {sm.name!r}: len={len(raw)} expected={int(sm.vertex_count) * stride}"
        )

    offs = _vertex_attr_offsets(sm)
    pos_off = offs.get(0)
    if pos_off is None:
        raise ValueError("Submesh has no Position attribute (attr 0)")
    pos_attr = next((a for a in sm.attributes if int(a.name) == 0), None)
    if pos_attr is None:
        raise ValueError("Submesh has no Position attribute (attr 0)")
    if int(pos_attr.elements) < 3:
        raise ValueError("Position attribute has <3 elements")
    comp_size = len(_pack_attr_value(int(pos_attr.fmt), float(pos_attr.scale), 0.0))

    mn = Vector((1e9, 1e9, 1e9))
    mx = Vector((-1e9, -1e9, -1e9))

    for i, v in enumerate(mesh.vertices):
        co = gf_from_blender @ (Vector(v.co) / float(global_scale))
        mn.x = min(mn.x, float(co.x))
        mn.y = min(mn.y, float(co.y))
        mn.z = min(mn.z, float(co.z))
        mx.x = max(mx.x, float(co.x))
        mx.y = max(mx.y, float(co.y))
        mx.z = max(mx.z, float(co.z))

        base = int(i) * stride + int(pos_off)
        raw[base : base + comp_size] = _pack_attr_value(
            int(pos_attr.fmt), float(pos_attr.scale), float(co.x)
        )
        base += comp_size
        raw[base : base + comp_size] = _pack_attr_value(
            int(pos_attr.fmt), float(pos_attr.scale), float(co.y)
        )
        base += comp_size
        raw[base : base + comp_size] = _pack_attr_value(
            int(pos_attr.fmt), float(pos_attr.scale), float(co.z)
        )

    sm.raw_buffer = bytes(raw)
    return mn, mx


def _patch_pack_positions_in_place(
    pack_src: bytes,
    model: "_GFModel",
    *,
    tagged: Dict[int, bpy.types.Object],
    gf_from_blender: Matrix,
    global_scale: float,
) -> Tuple[bytes, int]:
    """Return patched pack bytes where only position bytes inside vertex buffers change."""
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

        offs = _vertex_attr_offsets(sm)
        pos_off = offs.get(0)
        if pos_off is None:
            raise ValueError("Submesh has no Position attribute (attr 0)")
        pos_attr = next((a for a in sm.attributes if int(a.name) == 0), None)
        if pos_attr is None or int(pos_attr.elements) < 3:
            raise ValueError("Invalid Position attribute layout")

        comp_size = len(_pack_attr_value(int(pos_attr.fmt), float(pos_attr.scale), 0.0))
        stride = int(sm.vertex_stride)
                                                                                             
                                                                          
        base = int(getattr(sm, "raw_buffer_off", 0))
        if base <= 0:
            raise ValueError("Missing/invalid raw_buffer_off for submesh")

        for i, v in enumerate(mesh.vertices):
            co = gf_from_blender @ (Vector(v.co) / float(global_scale))
            vb = int(base) + int(i) * stride + int(pos_off)
            if vb < 0 or vb + comp_size * 3 > len(out):
                raise ValueError("Position write out of range (bad offsets/stride)")
            bx = _pack_attr_value(int(pos_attr.fmt), float(pos_attr.scale), float(co.x))
            by = _pack_attr_value(int(pos_attr.fmt), float(pos_attr.scale), float(co.y))
            bz = _pack_attr_value(int(pos_attr.fmt), float(pos_attr.scale), float(co.z))
            old = bytes(out[vb : vb + comp_size * 3])
            new = bx + by + bz
            if old != new:
                out[vb : vb + comp_size * 3] = new
                changed += 1

    return bytes(out), int(changed)


def _patch_pack_normals_in_place(
    pack_src: bytes,
    model: "_GFModel",
    *,
    tagged: Dict[int, bpy.types.Object],
    gf_from_blender: Matrix,
) -> Tuple[bytes, int]:
    """Return patched pack bytes where only normal bytes inside vertex buffers change."""
    out = bytearray(pack_src)
    changed = 0

    gf_nrm = gf_from_blender.to_3x3()

    for submesh_index, sm in enumerate(model.submeshes):
        obj = tagged.get(int(submesh_index))
        if obj is None:
            continue
        mesh: bpy.types.Mesh = obj.data                            

        try:
            if hasattr(mesh, "calc_normals_split"):
                mesh.calc_normals_split()                              
            elif hasattr(mesh, "calc_normals"):
                mesh.calc_normals()                              
        except Exception:
            pass

        if int(len(mesh.vertices)) != int(sm.vertex_count):
            raise ValueError(
                f"Vertex count mismatch for submesh {sm.name!r}: scene={len(mesh.vertices)} file={sm.vertex_count}"
            )

        offs = _vertex_attr_offsets(sm)
        nrm_off = offs.get(1)
        if nrm_off is None:
                                                                
            continue
        nrm_attr = next((a for a in sm.attributes if int(a.name) == 1), None)
        if nrm_attr is None or int(nrm_attr.elements) < 3:
            continue

        comp_size = len(_pack_attr_value(int(nrm_attr.fmt), float(nrm_attr.scale), 0.0))
        stride = int(sm.vertex_stride)
        base = int(getattr(sm, "raw_buffer_off", 0))
        if base <= 0:
            raise ValueError("Missing/invalid raw_buffer_off for submesh")

        for i, v in enumerate(mesh.vertices):
            no = Vector(v.normal)
            no = (gf_nrm @ no).normalized()
            vb = int(base) + int(i) * stride + int(nrm_off)
            if vb < 0 or vb + comp_size * 3 > len(out):
                raise ValueError("Normal write out of range (bad offsets/stride)")
            bx = _pack_attr_value(int(nrm_attr.fmt), float(nrm_attr.scale), float(no.x))
            by = _pack_attr_value(int(nrm_attr.fmt), float(nrm_attr.scale), float(no.y))
            bz = _pack_attr_value(int(nrm_attr.fmt), float(nrm_attr.scale), float(no.z))
            old = bytes(out[vb : vb + comp_size * 3])
            new = bx + by + bz
            if old != new:
                out[vb : vb + comp_size * 3] = new
                changed += 1

    return bytes(out), int(changed)


def _patch_pack_positions_normals_in_place(
    pack_src: bytes,
    model: "_GFModel",
    *,
    tagged: Dict[int, bpy.types.Object],
    gf_from_blender: Matrix,
    global_scale: float,
) -> Tuple[bytes, int, int]:
    out1, pos_changed = _patch_pack_positions_in_place(
        pack_src,
        model,
        tagged=tagged,
        gf_from_blender=gf_from_blender,
        global_scale=float(global_scale),
    )
    out2, nrm_changed = _patch_pack_normals_in_place(
        out1,
        model,
        tagged=tagged,
        gf_from_blender=gf_from_blender,
    )
    return bytes(out2), int(pos_changed), int(nrm_changed)


def _patch_pack_positions_normals_uv0_in_place(
    pack_src: bytes,
    model: "_GFModel",
    *,
    tagged: Dict[int, bpy.types.Object],
    gf_from_blender: Matrix,
    global_scale: float,
) -> Tuple[bytes, int, int, int]:
    out1, pos_changed = _patch_pack_positions_in_place(
        pack_src,
        model,
        tagged=tagged,
        gf_from_blender=gf_from_blender,
        global_scale=float(global_scale),
    )
    out2, nrm_changed = _patch_pack_normals_in_place(
        out1,
        model,
        tagged=tagged,
        gf_from_blender=gf_from_blender,
    )
    out3, uv_changed = _patch_pack_uv0_in_place(out2, model, tagged=tagged)
    return bytes(out3), int(pos_changed), int(nrm_changed), int(uv_changed)


def _patch_pack_pos_nrm_uv0_skin_in_place(
    pack_src: bytes,
    model: "_GFModel",
    *,
    tagged: Dict[int, bpy.types.Object],
    gf_from_blender: Matrix,
    global_scale: float,
    skeleton_names: List[str],
) -> Tuple[bytes, int, int, int, int, int]:
    out1, pos_changed, nrm_changed, uv_changed = (
        _patch_pack_positions_normals_uv0_in_place(
            pack_src,
            model,
            tagged=tagged,
            gf_from_blender=gf_from_blender,
            global_scale=float(global_scale),
        )
    )
    out2, skin_changed, skin_fallback = _patch_pack_skin_in_place(
        out1, model, tagged=tagged, skeleton_names=skeleton_names
    )
    return (
        bytes(out2),
        int(pos_changed),
        int(nrm_changed),
        int(uv_changed),
        int(skin_changed),
        int(skin_fallback),
    )


def _patch_pack_uv0_in_place(
    pack_src: bytes,
    model: "_GFModel",
    *,
    tagged: Dict[int, bpy.types.Object],
) -> Tuple[bytes, int]:
    """Return patched pack bytes where only UV0 bytes inside vertex buffers change."""
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


def _patch_pack_indices_in_place(
    pack_src: bytes,
    model: "_GFModel",
    *,
    tagged: Dict[int, bpy.types.Object],
) -> Tuple[bytes, int]:
    """Patch only index buffer bytes in-place (requires no index count change)."""
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


def _pica_iter_cmds_with_param_indices(
    cmds: Sequence[int],
) -> Iterable[Tuple[int, int, List[int]]]:
    """Yield (reg, first_param_index, params_u32)."""
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
    """Patch index buffer bytes + index counts in-place (triangles only, vertex count fixed).

    This supports deleting faces (reducing triangle/index count) and adding faces only as long
    as the new index stream fits within the existing allocated `index_data_len`.
    """
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


def _patch_pack_verts_topology_tris_in_place(
    pack_src: bytes,
    model: "_GFModel",
    *,
    tagged: Dict[int, bpy.types.Object],
    gf_from_blender: Matrix,
    global_scale: float,
    skeleton_names: List[str],
) -> Tuple[bytes, int]:
    """Patch vertex+index buffers in-place (triangles only), allowing vertex/index count changes.

    Constraints:
    - No stride/layout changes (uses source `sm.attributes` + `sm.vertex_stride`).
    - New vertex count must fit within allocated vertex buffer length (len(sm.raw_buffer)).
    - New index count must fit within allocated index buffer length (`sm.index_data_len`).
    """
    out = bytearray(pack_src)
    changed = 0

    gf_nrm = gf_from_blender.to_3x3()

    for submesh_index, sm in enumerate(model.submeshes):
        obj = tagged.get(int(submesh_index))
        if obj is None:
            continue
        mesh: bpy.types.Mesh = obj.data                            

        if int(sm.primitive_mode) != 0:
            raise ValueError(
                f"Verts+Topo in-place patch supports primitive_mode=0 (Triangles) only; submesh {sm.name!r} has {int(sm.primitive_mode)}"
            )

        stride = int(sm.vertex_stride)
        if stride <= 0:
            raise ValueError(f"vertex_stride<=0 for submesh {sm.name!r}")
        base_vtx = int(getattr(sm, "raw_buffer_off", 0) or 0)
        if base_vtx <= 0:
            raise ValueError(f"Missing/invalid raw_buffer_off for submesh {sm.name!r}")

        vtx_len = int(len(getattr(sm, "raw_buffer", b"") or b""))
        if vtx_len <= 0:
            raise ValueError(
                f"Missing/invalid vertex buffer length for submesh {sm.name!r}"
            )
        cap_verts = int(vtx_len) // int(stride)
        new_vcount = int(len(mesh.vertices))
        if new_vcount > cap_verts:
            raise ValueError(
                f"New vertex count exceeds allocated capacity for submesh {sm.name!r}: new={new_vcount} cap={cap_verts} (vtx_len={vtx_len}, stride={stride})"
            )

                                                                   
        uv_layer = None
        if getattr(mesh, "uv_layers", None):
            uv_layer = mesh.uv_layers.active or mesh.uv_layers[0]
        uv_by_v: List[Tuple[float, float]] = [(0.0, 0.0)] * new_vcount
        if uv_layer is not None:
            seen = [False] * new_vcount
            try:
                for poly in mesh.polygons:
                    for li in poly.loop_indices:
                        vi = int(mesh.loops[li].vertex_index)
                        if 0 <= vi < new_vcount and not seen[vi]:
                            uv = uv_layer.data[li].uv
                            uv_by_v[vi] = (float(uv.x), float(uv.y))
                            seen[vi] = True
            except Exception:
                pass

                                                             
        col_by_v: List[Tuple[float, float, float, float]] = [
            (1.0, 1.0, 1.0, 1.0)
        ] * new_vcount
        try:
            ca = getattr(mesh, "color_attributes", None)
            col_attr = ca.get("Col") if ca is not None and hasattr(ca, "get") else None
            if col_attr is not None and getattr(col_attr, "data", None) is not None:
                seen = [False] * new_vcount
                for poly in mesh.polygons:
                    for li in poly.loop_indices:
                        vi = int(mesh.loops[li].vertex_index)
                        if 0 <= vi < new_vcount and not seen[vi]:
                            c = col_attr.data[li].color
                            col_by_v[vi] = (
                                float(c[0]),
                                float(c[1]),
                                float(c[2]),
                                float(c[3]),
                            )
                            seen[vi] = True
        except Exception:
            pass

        offs = _vertex_attr_offsets(sm)
        pos_off = offs.get(0)
        nrm_off = offs.get(1)
        uv_off = offs.get(4)
        col_off = offs.get(3)
        bi_off = offs.get(7)
        bw_off = offs.get(8)

        pos_attr = next((a for a in sm.attributes if int(a.name) == 0), None)
        nrm_attr = next((a for a in sm.attributes if int(a.name) == 1), None)
        uv_attr = next((a for a in sm.attributes if int(a.name) == 4), None)
        col_attr_layout = next((a for a in sm.attributes if int(a.name) == 3), None)
        bi_attr = next((a for a in sm.attributes if int(a.name) == 7), None)
        bw_attr = next((a for a in sm.attributes if int(a.name) == 8), None)

                                                                                      
        weights_by_v: List[List[Tuple[int, float]]] = [[] for _ in range(new_vcount)]
        pal_count = int(getattr(sm, "bone_indices_count", 0) or 0)
        if bi_attr is not None or bw_attr is not None:
            if not (
                bi_attr is not None
                and bw_attr is not None
                and bi_off is not None
                and bw_off is not None
            ):
                raise ValueError(
                    f"Submesh {sm.name!r} has partial skinning attributes (need both 7 and 8)"
                )
            wtmp, unknown_bones, not_in_palette = (
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
            if pal_count <= 0:
                raise ValueError("Submesh has no bone palette (bone_indices_count<=0)")
            weights_by_v = wtmp[:new_vcount]

        old_vcount = int(sm.vertex_count)
        if old_vcount <= 0:
            old_vcount = 1
        if base_vtx + old_vcount * stride > len(out):
            raise ValueError("Vertex buffer range out of file bounds")
        templ_bytes = bytes(out[base_vtx : base_vtx + stride])

                                                                                              
        for i, v in enumerate(mesh.vertices):
            src = (
                bytes(out[base_vtx + i * stride : base_vtx + (i + 1) * stride])
                if i < old_vcount
                else templ_bytes
            )
            dst = bytearray(src)

            if (
                pos_off is not None
                and pos_attr is not None
                and int(pos_attr.elements) >= 3
            ):
                comp = len(
                    _pack_attr_value(int(pos_attr.fmt), float(pos_attr.scale), 0.0)
                )
                co = gf_from_blender @ (Vector(v.co) / float(global_scale))
                o = int(pos_off)
                dst[o : o + comp] = _pack_attr_value(
                    int(pos_attr.fmt), float(pos_attr.scale), float(co.x)
                )
                o += comp
                dst[o : o + comp] = _pack_attr_value(
                    int(pos_attr.fmt), float(pos_attr.scale), float(co.y)
                )
                o += comp
                dst[o : o + comp] = _pack_attr_value(
                    int(pos_attr.fmt), float(pos_attr.scale), float(co.z)
                )

            if (
                nrm_off is not None
                and nrm_attr is not None
                and int(nrm_attr.elements) >= 3
            ):
                comp = len(
                    _pack_attr_value(int(nrm_attr.fmt), float(nrm_attr.scale), 0.0)
                )
                no = (gf_nrm @ Vector(v.normal)).normalized()
                o = int(nrm_off)
                dst[o : o + comp] = _pack_attr_value(
                    int(nrm_attr.fmt), float(nrm_attr.scale), float(no.x)
                )
                o += comp
                dst[o : o + comp] = _pack_attr_value(
                    int(nrm_attr.fmt), float(nrm_attr.scale), float(no.y)
                )
                o += comp
                dst[o : o + comp] = _pack_attr_value(
                    int(nrm_attr.fmt), float(nrm_attr.scale), float(no.z)
                )

            if (
                uv_off is not None
                and uv_attr is not None
                and int(uv_attr.elements) >= 2
                and i < len(uv_by_v)
            ):
                comp = len(
                    _pack_attr_value(int(uv_attr.fmt), float(uv_attr.scale), 0.0)
                )
                u, vv = uv_by_v[i]
                o = int(uv_off)
                dst[o : o + comp] = _pack_attr_value(
                    int(uv_attr.fmt), float(uv_attr.scale), float(u)
                )
                o += comp
                dst[o : o + comp] = _pack_attr_value(
                    int(uv_attr.fmt), float(uv_attr.scale), float(vv)
                )

            if (
                col_off is not None
                and col_attr_layout is not None
                and int(col_attr_layout.elements) >= 3
                and i < len(col_by_v)
            ):
                comp = len(
                    _pack_attr_value(
                        int(col_attr_layout.fmt), float(col_attr_layout.scale), 0.0
                    )
                )
                cr, cg, cb, ca = col_by_v[i]
                o = int(col_off)
                dst[o : o + comp] = _pack_attr_value(
                    int(col_attr_layout.fmt),
                    float(col_attr_layout.scale),
                    float(cr),
                )
                o += comp
                dst[o : o + comp] = _pack_attr_value(
                    int(col_attr_layout.fmt),
                    float(col_attr_layout.scale),
                    float(cg),
                )
                o += comp
                dst[o : o + comp] = _pack_attr_value(
                    int(col_attr_layout.fmt),
                    float(col_attr_layout.scale),
                    float(cb),
                )
                if int(col_attr_layout.elements) >= 4:
                    o += comp
                    dst[o : o + comp] = _pack_attr_value(
                        int(col_attr_layout.fmt),
                        float(col_attr_layout.scale),
                        float(ca),
                    )

            if (
                bi_off is not None
                and bw_off is not None
                and bi_attr is not None
                and bw_attr is not None
                and i < len(weights_by_v)
            ):
                bi_elems = int(bi_attr.elements)
                bw_elems = int(bw_attr.elements)
                elems = min(4, bi_elems, bw_elems)
                bi_comp = len(
                    _pack_attr_value(int(bi_attr.fmt), float(bi_attr.scale), 0.0)
                )
                bw_comp = len(
                    _pack_attr_value(int(bw_attr.fmt), float(bw_attr.scale), 0.0)
                )
                wl = weights_by_v[i]
                indices = [0] * elems
                weights = [0.0] * elems
                for wi, (pi, w) in enumerate(wl[:elems]):
                    if not (0 <= int(pi) < pal_count):
                        raise ValueError(
                            f"Palette index out of range at v={i}: {pi} (pal_count={pal_count})"
                        )
                    indices[wi] = int(pi)
                    weights[wi] = float(w)
                s = float(sum(max(0.0, w) for w in weights))
                if s > 0:
                    weights = [max(0.0, w) / s for w in weights]
                else:
                    indices[0] = 0
                    weights[0] = 1.0

                o = int(bi_off)
                for j in range(elems):
                    dst[o : o + bi_comp] = _pack_attr_value(
                        int(bi_attr.fmt), float(bi_attr.scale), float(indices[j])
                    )
                    o += bi_comp
                o = int(bw_off)
                for j in range(elems):
                    dst[o : o + bw_comp] = _pack_attr_value(
                        int(bw_attr.fmt), float(bw_attr.scale), float(weights[j])
                    )
                    o += bw_comp

            out[base_vtx + i * stride : base_vtx + (i + 1) * stride] = dst

        vtx_count_off = int(getattr(sm, "vertex_count_off", 0) or 0)
        if vtx_count_off <= 0 or vtx_count_off + 4 > len(out):
            raise ValueError(
                f"Missing/invalid vertex_count_off for submesh {sm.name!r}: {vtx_count_off}"
            )
        struct.pack_into("<i", out, vtx_count_off, int(new_vcount))

                                      
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

        idx_len = int(getattr(sm, "index_data_len", 0) or 0)
        elem_size = int(getattr(sm, "index_elem_size", 0) or 0)
        base_idx = int(getattr(sm, "index_buffer_off", 0) or 0)
        if elem_size not in (1, 2) or idx_len <= 0 or base_idx <= 0:
            raise ValueError(f"Missing/invalid index buffer metadata for {sm.name!r}")
        max_indices = int(idx_len) // int(elem_size)
        if int(len(new_indices)) > max_indices:
            raise ValueError(
                f"New index stream too large for submesh {sm.name!r}: new={len(new_indices)} max={max_indices} (idx_len={idx_len}, elem_size={elem_size})"
            )
        for ii in new_indices:
            if ii < 0 or ii >= new_vcount:
                raise ValueError(
                    f"Index out of range for submesh {sm.name!r}: {ii} (new_vcount={new_vcount})"
                )
            if elem_size == 1 and ii > 0xFF:
                raise ValueError(
                    f"Index too large for u8 index buffer for submesh {sm.name!r}: {ii}"
                )

        if base_idx < 0 or base_idx + idx_len > len(out):
            raise ValueError("Index write out of range (bad offsets/length)")

        if elem_size == 2:
            new_bytes = b"".join(struct.pack("<H", int(i)) for i in new_indices)
            zero = b"\x00\x00"
        else:
            new_bytes = bytes(int(i) & 0xFF for i in new_indices)
            zero = b"\x00"
        out[base_idx : base_idx + idx_len] = new_bytes + (
            zero * (max_indices - len(new_indices))
        )

        idx_count_off = int(getattr(sm, "index_count_off", 0) or 0)
        if idx_count_off <= 0 or idx_count_off + 4 > len(out):
            raise ValueError(
                f"Missing/invalid index_count_off for submesh {sm.name!r}: {idx_count_off}"
            )
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

        changed += 1


    return bytes(out), int(changed)

def _patch_pack_skin_in_place(
    pack_src: bytes,
    model: "_GFModel",
    *,
    tagged: Dict[int, bpy.types.Object],
    skeleton_names: List[str],
) -> Tuple[bytes, int, int]:
    """Patch BoneIndex(7) + BoneWeight(8) attributes in-place (requires both as dynamic attributes)."""
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
