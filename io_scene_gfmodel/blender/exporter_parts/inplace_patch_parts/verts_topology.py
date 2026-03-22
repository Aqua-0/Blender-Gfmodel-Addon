
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


def _patch_pack_verts_topology_tris_in_place(
    pack_src: bytes,
    model: "_GFModel",
    *,
    tagged: Dict[int, bpy.types.Object],
    gf_from_blender: Matrix,
    global_scale: float,
    skeleton_names: List[str],
) -> Tuple[bytes, int]:
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
