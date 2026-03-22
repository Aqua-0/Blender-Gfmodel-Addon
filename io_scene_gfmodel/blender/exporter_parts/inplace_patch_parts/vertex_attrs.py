
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

