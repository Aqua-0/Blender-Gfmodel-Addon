"""Grow Buffers mesh-section rewrite logic (triangles).

Kept separate from operator/UI code to keep file sizes manageable.
"""

from __future__ import annotations

import copy
import struct
from typing import Dict, List, Optional, Tuple

import bpy
from mathutils import Matrix, Vector

from ...core.export import write_gf_mesh_blob, write_gf_model_blob
from ...core.io import _load_any
from ...core.types import _GFSubMesh
from .common import align_up as _align_up
from .common import mesh_tris_indices as _mesh_tris_indices
from .common import pica_patch_reg_all_in_cmd_bytes as _pica_patch_reg_all_in_cmd_bytes
from .vertex_pack import (
    _gather_weights_skeleton_indices_checked,
    _pack_attr_value,
    _vertex_attr_offsets,
)


def _rewrite_model_blob_grow_buffers_tris(
    *,
    pack_src: bytes,
    model_blob: bytes,
    model_blob_start_in_pack: int,
    model: "_GFModel",
    tagged: Dict[int, bpy.types.Object],
    gf_from_blender: Matrix,
    global_scale: float,
    skeleton_names: List[str],
    disallow_new_mesh_sections: bool = False,
    allow_palette_rebuild: bool = True,
    allow_palette_split: bool = True,
) -> bytes:
    """Rewrite a GFModel blob by rebuilding mesh sections with grown vtx/idx buffers (triangles only)."""
    out_model = bytearray(model_blob)
    gf_nrm = gf_from_blender.to_3x3()
    overflow_mesh_parts: Dict[str, List[_GFSubMesh]] = {}

    def _build_uv_seam_split_tris(
        mesh: bpy.types.Mesh,
    ) -> Tuple[int, List[int], List[Tuple[int, Tuple[float, float]]]]:
        """Build triangle indices while optionally splitting vertices by UV seams.

        Returns:
          - new_vcount: number of unique (vertex, seam-uv) keys
          - new_indices: triangle indices into the compacted/split vertex list
          - used_keys: list mapping new vertex index -> (source vertex index, uv0)
        """
        try:
            mesh.calc_loop_triangles()
        except Exception:
            pass
        loop_tris = list(getattr(mesh, "loop_triangles", []) or [])
        if not loop_tris:
            raise ValueError("Mesh has no triangles")

        uv_layer = None
        if getattr(mesh, "uv_layers", None):
            uv_layer = mesh.uv_layers.active or mesh.uv_layers[0]

        def uv_q(uv: Tuple[float, float]) -> Tuple[int, int]:
            return (
                int(round(float(uv[0]) * 1e6)),
                int(round(float(uv[1]) * 1e6)),
            )

        uv_first: List[Optional[Tuple[float, float]]] = [None] * int(len(mesh.vertices))
        uv_split: List[bool] = [False] * int(len(mesh.vertices))
        if uv_layer is not None:
            for tri in loop_tris:
                for li in tri.loops:
                    vi0 = int(mesh.loops[int(li)].vertex_index)
                    if vi0 < 0 or vi0 >= len(uv_first):
                        continue
                    uv = uv_layer.data[int(li)].uv
                    uv_t = (float(uv.x), float(uv.y))
                    prev = uv_first[vi0]
                    if prev is None:
                        uv_first[vi0] = uv_t
                    else:
                        if uv_q(prev) != uv_q(uv_t):
                            uv_split[vi0] = True

        def get_corner_uv(vi0: int, li: int) -> Tuple[float, float]:
            if uv_layer is None:
                return (0.0, 0.0)
            uv = uv_layer.data[int(li)].uv
            loop_uv = (float(uv.x), float(uv.y))
            if 0 <= int(vi0) < len(uv_split) and uv_split[int(vi0)]:
                return loop_uv
            base_uv = uv_first[int(vi0)] if 0 <= int(vi0) < len(uv_first) else None
            return base_uv or loop_uv

        remap: Dict[Tuple[int, Tuple[int, int]], int] = {}
        used_keys: List[Tuple[int, Tuple[float, float]]] = []
        new_indices: List[int] = []
        for tri in loop_tris:
            for vi0, li in zip((int(x) for x in tri.vertices), tri.loops):
                use_uv = get_corner_uv(int(vi0), int(li))
                uv_key = (
                    uv_q(use_uv)
                    if 0 <= int(vi0) < len(uv_split) and uv_split[int(vi0)]
                    else (0, 0)
                )
                key = (int(vi0), uv_key)
                idx = remap.get(key)
                if idx is None:
                    idx = len(used_keys)
                    remap[key] = idx
                    used_keys.append((int(vi0), use_uv))
                new_indices.append(int(idx))

        new_vcount = int(len(used_keys))
        for ii in new_indices:
            if ii < 0 or ii >= new_vcount:
                raise ValueError(
                    f"Index out of range after UV seam split: {ii} (new_vcount={new_vcount})"
                )

        return new_vcount, new_indices, used_keys

    def _submesh_layout_equal(a: _GFSubMesh, b: _GFSubMesh) -> bool:
        if int(a.vertex_stride) != int(b.vertex_stride):
            return False
        if int(a.primitive_mode) != int(b.primitive_mode):
            return False
        if int(getattr(a, "index_elem_size", 0) or 0) != int(
            getattr(b, "index_elem_size", 0) or 0
        ):
            return False
        if [int(x) for x in (a.enable_cmds or [])] != [
            int(x) for x in (b.enable_cmds or [])
        ]:
            return False
        if [int(x) for x in (a.disable_cmds or [])] != [
            int(x) for x in (b.disable_cmds or [])
        ]:
            return False
        if [int(x) for x in (a.index_cmds or [])] != [
            int(x) for x in (b.index_cmds or [])
        ]:
            return False
        a_attrs = [
            (int(x.name), int(x.fmt), int(x.elements), float(x.scale))
            for x in (a.attributes or [])
        ]
        b_attrs = [
            (int(x.name), int(x.fmt), int(x.elements), float(x.scale))
            for x in (b.attributes or [])
        ]
        if a_attrs != b_attrs:
            return False
        a_fixed = [
            (int(x.name), float(x.x), float(x.y), float(x.z), float(x.w))
            for x in (a.fixed_attributes or [])
        ]
        b_fixed = [
            (int(x.name), float(x.x), float(x.y), float(x.z), float(x.w))
            for x in (b.fixed_attributes or [])
        ]
        return a_fixed == b_fixed

    def build_split_faces_for_submesh(
        *,
        sm: _GFSubMesh,
        obj: bpy.types.Object,
        skeleton_names: List[str],
    ) -> List[_GFSubMesh]:
        """Return one or more new binary submeshes derived from a single Blender object.

        Splits triangles into multiple submeshes when the local bone palette would overflow 0x1F.
        """
        mesh: bpy.types.Mesh = obj.data                            
        if int(sm.primitive_mode) != 0:
            raise ValueError(
                f"Grow-buffers patch supports primitive_mode=0 (Triangles) only; submesh {sm.name!r} has {int(sm.primitive_mode)}"
            )
        stride = int(sm.vertex_stride)
        if stride <= 0:
            raise ValueError(f"vertex_stride<=0 for submesh {sm.name!r}")

        try:
            mesh.calc_loop_triangles()
        except Exception:
            pass
        try:
            if hasattr(mesh, "calc_normals_split"):
                mesh.calc_normals_split()                              
            elif hasattr(mesh, "calc_normals"):
                mesh.calc_normals()                              
        except Exception:
            pass

        loop_tris = list(getattr(mesh, "loop_triangles", []) or [])
        if not loop_tris:
            raise ValueError(f"Mesh has no triangles for submesh {sm.name!r}")

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

        has_skin = bi_attr is not None or bw_attr is not None
        if has_skin and not (
            bi_attr is not None
            and bw_attr is not None
            and bi_off is not None
            and bw_off is not None
        ):
            raise ValueError(
                f"Submesh {sm.name!r} has partial skinning attributes (need both 7 and 8)"
            )

        sk_w_by_v: List[List[Tuple[int, float]]] = [
            [] for _ in range(len(mesh.vertices))
        ]
        if has_skin:
            sk_w_by_v, unknown_bones = _gather_weights_skeleton_indices_checked(
                obj, skeleton_names
            )
            if unknown_bones:
                sample = ", ".join(unknown_bones[:10])
                raise ValueError(
                    f"Vertex groups reference bones not in skeleton (first {min(10, len(unknown_bones))}): {sample}"
                )

        bones_by_v: List[List[int]] = []
        for wl in sk_w_by_v:
            bones = [int(sk) for sk, w in wl if float(w) > 0.0]
            if not bones:
                bones = [0]
            bones_by_v.append(bones)

        max_palette = 0x1F

        def build_palette(required: List[int]) -> List[int]:
            palette_raw = list(sm.bone_indices or [])[
                : int(getattr(sm, "bone_indices_count", 0) or 0)
            ]
            req_set = set(int(x) for x in required)
            if not allow_palette_rebuild:
                palette: List[int] = []
                seen = set()
                for b in palette_raw:
                    bi = int(b)
                    if bi in seen:
                        continue
                    palette.append(int(bi))
                    seen.add(int(bi))
                missing = sorted(int(x) for x in (req_set - seen))
                if missing:
                    sample = ", ".join(str(x) for x in missing[:12])
                    raise ValueError(
                        f"Triangles require bones not present in the existing palette for {sm.name!r}: {sample} "
                        "(use Rebuild Palette Only / Rebuild + Split, or adjust routing/weights)"
                    )
                return palette

            palette: List[int] = []
            palette_set = set()
            for b in palette_raw:
                bi = int(b)
                if bi in palette_set:
                    continue
                palette.append(int(bi))
                palette_set.add(int(bi))

            if req_set:
                palette = [sk for sk in palette if int(sk) in req_set]
                palette_set = set(int(sk) for sk in palette)
                for sk in sorted(req_set):
                    if int(sk) in palette_set:
                        continue
                    palette.append(int(sk))
                    palette_set.add(int(sk))

            return palette

        groups: List[Tuple[set, List[object]]] = []
        if not allow_palette_split:
            used: set = set()
            for tri in loop_tris:
                a, b, c = (int(x) for x in tri.vertices)
                tb = set()
                for vi in (a, b, c):
                    if 0 <= int(vi) < len(bones_by_v):
                        tb.update(int(x) for x in bones_by_v[int(vi)])
                if len(tb) > max_palette:
                    raise ValueError(
                        f"Single triangle exceeds palette limit for {sm.name!r}: needs {len(tb)} bones (max={max_palette})"
                    )
                used |= tb
                if len(used) > max_palette:
                    raise ValueError(
                        f"Submesh {sm.name!r} requires {len(used)} bones across triangles (max={max_palette}); "
                        "use Rebuild + Split or adjust routing/weights"
                    )
            groups = [(set(used), list(loop_tris))]
        else:
            for tri in loop_tris:
                a, b, c = (int(x) for x in tri.vertices)
                tb = set()
                for vi in (a, b, c):
                    if 0 <= int(vi) < len(bones_by_v):
                        tb.update(int(x) for x in bones_by_v[int(vi)])
                if len(tb) > max_palette:
                    raise ValueError(
                        f"Single triangle exceeds palette limit for {sm.name!r}: needs {len(tb)} bones (max={max_palette})"
                    )
                placed = False
                for gi, (gb, gtris) in enumerate(groups):
                    if len(gb | tb) <= max_palette:
                        gb |= tb
                        gtris.append(tri)
                        placed = True
                        break
                if not placed:
                    groups.append((set(tb), [tri]))

        uv_layer = None
        if getattr(mesh, "uv_layers", None):
            uv_layer = mesh.uv_layers.active or mesh.uv_layers[0]

        def uv_q(uv: Tuple[float, float]) -> Tuple[int, int]:
            return (int(round(float(uv[0]) * 1e6)), int(round(float(uv[1]) * 1e6)))

        uv_first: List[Optional[Tuple[float, float]]] = [None] * len(mesh.vertices)
        uv_split: List[bool] = [False] * len(mesh.vertices)
        if uv_layer is not None:
            for tri in loop_tris:
                for li in tri.loops:
                    vi = int(mesh.loops[int(li)].vertex_index)
                    if vi < 0 or vi >= len(mesh.vertices):
                        continue
                    uv = uv_layer.data[int(li)].uv
                    uv_t = (float(uv.x), float(uv.y))
                    prev = uv_first[vi]
                    if prev is None:
                        uv_first[vi] = uv_t
                    else:
                        if uv_q(prev) != uv_q(uv_t):
                            uv_split[vi] = True

        col_by_v = [(1.0, 1.0, 1.0, 1.0)] * len(mesh.vertices)
        try:
            ca = getattr(mesh, "color_attributes", None)
            col_attr = ca.get("Col") if ca is not None and hasattr(ca, "get") else None
            if col_attr is not None and getattr(col_attr, "data", None) is not None:
                seen = [False] * len(mesh.vertices)
                for poly in mesh.polygons:
                    for li in poly.loop_indices:
                        vi = int(mesh.loops[li].vertex_index)
                        if 0 <= vi < len(mesh.vertices) and not seen[vi]:
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

        out_faces: List[_GFSubMesh] = []
        old_raw = bytes(getattr(sm, "raw_buffer", b"") or b"")
        old_vcount = int(getattr(sm, "vertex_count", 0) or 0)
        templ = old_raw[:stride] if len(old_raw) >= stride else (b"\x00" * stride)

        for gi, (gb, gtris) in enumerate(groups):
            required = sorted(int(x) for x in gb)
            palette = build_palette(required)
            if len(palette) > max_palette:
                raise ValueError(
                    f"Submesh palette overflow for {sm.name!r} (group {gi}): needs {len(palette)} bones, max={max_palette}"
                )
            if any(int(sk) < 0 or int(sk) > 0xFF for sk in palette):
                raise ValueError(
                    f"Submesh palette contains invalid skeleton indices for {sm.name!r}"
                )
            sk_to_pi = {int(sk): int(i) for i, sk in enumerate(palette)}

            remap: Dict[Tuple[int, Tuple[int, int]], int] = {}
            new_indices: List[int] = []
            used_keys: List[Tuple[int, Tuple[int, int], Tuple[float, float]]] = []

            def get_corner_uv(vi: int, li: int) -> Tuple[float, float]:
                if uv_layer is None:
                    return (0.0, 0.0)
                uv = uv_layer.data[int(li)].uv
                loop_uv = (float(uv.x), float(uv.y))
                if 0 <= int(vi) < len(uv_split) and uv_split[int(vi)]:
                    return loop_uv
                base_uv = uv_first[int(vi)] if 0 <= int(vi) < len(uv_first) else None
                return base_uv or loop_uv

            for tri in gtris:
                for vi, li in zip((int(x) for x in tri.vertices), tri.loops):
                    use_uv = get_corner_uv(int(vi), int(li))
                    uv_key = (
                        uv_q(use_uv)
                        if 0 <= int(vi) < len(uv_split) and uv_split[int(vi)]
                        else (0, 0)
                    )
                    key = (int(vi), uv_key)
                    idx = remap.get(key)
                    if idx is None:
                        idx = len(used_keys)
                        remap[key] = idx
                        used_keys.append((int(vi), uv_key, use_uv))
                    new_indices.append(int(idx))

            new_vcount = len(used_keys)

            idx16 = int(getattr(sm, "index_elem_size", 0) or 0) == 2 or any(
                int(ii) > 0xFF for ii in new_indices
            )
            elem_size = 2 if idx16 else 1
            if not idx16 and new_vcount > 0xFF:
                raise ValueError(
                    f"u8 index buffer cannot address {new_vcount} vertices for {sm.name!r} (group {gi})"
                )

            vbytes = bytearray()
            for vi_old, _uv_key, use_uv in used_keys:
                src = (
                    old_raw[int(vi_old) * stride : (int(vi_old) + 1) * stride]
                    if 0 <= int(vi_old) < old_vcount
                    else templ
                )
                if len(src) != stride:
                    src = templ
                dst = bytearray(src)

                v = mesh.vertices[int(vi_old)]
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
                ):
                    comp = len(
                        _pack_attr_value(int(uv_attr.fmt), float(uv_attr.scale), 0.0)
                    )
                    u, vv = use_uv
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
                    and 0 <= int(vi_old) < len(col_by_v)
                ):
                    comp = len(
                        _pack_attr_value(
                            int(col_attr_layout.fmt), float(col_attr_layout.scale), 0.0
                        )
                    )
                    cr, cg, cb, ca = col_by_v[int(vi_old)]
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
                    has_skin
                    and bi_off is not None
                    and bw_off is not None
                    and bi_attr is not None
                    and bw_attr is not None
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
                    wl = sk_w_by_v[int(vi_old)] if int(vi_old) < len(sk_w_by_v) else []
                    idxs = [0] * elems
                    wts = [0.0] * elems
                    out_i = 0
                    for sk, w in wl[:elems]:
                        pi = sk_to_pi.get(int(sk))
                        if pi is None:
                            continue
                        idxs[out_i] = int(pi)
                        wts[out_i] = float(w)
                        out_i += 1
                        if out_i >= elems:
                            break
                    s = float(sum(max(0.0, w) for w in wts))
                    if s > 0:
                        wts = [max(0.0, w) / s for w in wts]
                    else:
                        idxs[0] = 0
                        wts[0] = 1.0
                    o = int(bi_off)
                    for j in range(elems):
                        dst[o : o + bi_comp] = _pack_attr_value(
                            int(bi_attr.fmt), float(bi_attr.scale), float(idxs[j])
                        )
                        o += bi_comp
                    o = int(bw_off)
                    for j in range(elems):
                        dst[o : o + bw_comp] = _pack_attr_value(
                            int(bw_attr.fmt), float(bw_attr.scale), float(wts[j])
                        )
                        o += bw_comp

                vbytes += dst

            vtx_payload = int(new_vcount) * int(stride)
            vtx_len = _align_up(int(vtx_payload), 0x10)
            if len(vbytes) > vtx_len:
                raise ValueError("vertex bytes exceed allocated length")
            if len(vbytes) < vtx_len:
                vbytes += b"\x00" * (int(vtx_len) - int(len(vbytes)))

            idx_payload = int(len(new_indices)) * int(elem_size)
            idx_len = _align_up(int(idx_payload), 4)
            idx_pad = b"\x00" * max(0, int(idx_len) - int(idx_payload))

            part = copy.deepcopy(sm)
            part.name = str(sm.name)
            part.bone_indices_count = int(len(palette))
            part.bone_indices = list(int(x) for x in palette)
            part.vertex_count = int(new_vcount)
            part.index_count = int(len(new_indices))
            part.indices = list(int(x) for x in new_indices)
            part.raw_buffer = bytes(vbytes)
            part.index_data_len = int(idx_len)
            part.index_pad_bytes = bytes(idx_pad)
            part.index_elem_size = int(elem_size)
            out_faces.append(part)

        return out_faces

    def _remap_bone_indices_in_raw_buffer(
        sm: _GFSubMesh, *, raw: bytes, old_palette: List[int], new_palette: List[int]
    ) -> bytes:
        attr_names = set(int(a.name) for a in (sm.attributes or []))
        if not (7 in attr_names and 8 in attr_names):
            return bytes(raw)
        offs = _vertex_attr_offsets(sm)
        bi_off = offs.get(7)
        if bi_off is None:
            return bytes(raw)
        bi_attr = next((a for a in sm.attributes if int(a.name) == 7), None)
        if bi_attr is None:
            return bytes(raw)

        sk_to_pi = {int(sk): int(i) for i, sk in enumerate(new_palette)}
        elem = len(_pack_attr_value(int(bi_attr.fmt), float(bi_attr.scale), 0.0))
        elems = min(4, int(bi_attr.elements))
        stride = int(sm.vertex_stride)
        if stride <= 0 or elem <= 0 or elems <= 0:
            return bytes(raw)

        out = bytearray(raw)
        vcount = int(sm.vertex_count)
        for vi in range(vcount):
            base = int(vi) * int(stride) + int(bi_off)
            for j in range(elems):
                o = base + j * elem
                if o < 0 or o + elem > len(out):
                    break
                if int(bi_attr.fmt) == 0:      
                    pi = struct.unpack_from("<b", out, o)[0]
                elif int(bi_attr.fmt) == 1:      
                    pi = out[o]
                elif int(bi_attr.fmt) == 2:       
                    pi = struct.unpack_from("<h", out, o)[0]
                else:       
                    pi = int(round(struct.unpack_from("<f", out, o)[0]))
                if int(pi) < 0:
                    pi = 0
                if int(pi) >= len(old_palette):
                    sk = int(old_palette[0]) if old_palette else 0
                else:
                    sk = int(old_palette[int(pi)])
                new_pi = int(sk_to_pi.get(int(sk), 0))
                out[o : o + elem] = _pack_attr_value(
                    int(bi_attr.fmt), float(bi_attr.scale), float(new_pi)
                )
        return bytes(out)

    def _merge_faces_preserving_palette_limit(
        a: _GFSubMesh, b: _GFSubMesh
    ) -> Optional[_GFSubMesh]:
        if str(a.name) != str(b.name):
            return None
        if int(a.vertex_stride) != int(b.vertex_stride):
            return None
        if [
            tuple((int(x.name), int(x.fmt), int(x.elements), float(x.scale)))
            for x in (a.attributes or [])
        ] != [
            tuple((int(x.name), int(x.fmt), int(x.elements), float(x.scale)))
            for x in (b.attributes or [])
        ]:
            return None
        if [
            tuple((int(x.name), float(x.x), float(x.y), float(x.z), float(x.w)))
            for x in (a.fixed_attributes or [])
        ] != [
            tuple((int(x.name), float(x.x), float(x.y), float(x.z), float(x.w)))
            for x in (b.fixed_attributes or [])
        ]:
            return None
        if [int(x) for x in (a.enable_cmds or [])] != [
            int(x) for x in (b.enable_cmds or [])
        ]:
            return None
        if [int(x) for x in (a.disable_cmds or [])] != [
            int(x) for x in (b.disable_cmds or [])
        ]:
            return None
        if [int(x) for x in (a.index_cmds or [])] != [
            int(x) for x in (b.index_cmds or [])
        ]:
            return None

        max_palette = 0x1F
        pal_a = list(a.bone_indices or [])[: int(a.bone_indices_count)]
        pal_b = list(b.bone_indices or [])[: int(b.bone_indices_count)]
        new_palette: List[int] = []
        seen = set()
        for sk in pal_a + pal_b:
            sk = int(sk)
            if sk in seen:
                continue
            new_palette.append(int(sk))
            seen.add(int(sk))
        if len(new_palette) > max_palette:
            return None

        merged = copy.deepcopy(a)
        merged.bone_indices_count = int(len(new_palette))
        merged.bone_indices = list(int(x) for x in new_palette)

        raw_a = bytes(getattr(a, "raw_buffer", b"") or b"")
        raw_b = bytes(getattr(b, "raw_buffer", b"") or b"")
        raw_a = _remap_bone_indices_in_raw_buffer(
            a, raw=raw_a, old_palette=pal_a, new_palette=new_palette
        )
        raw_b = _remap_bone_indices_in_raw_buffer(
            b, raw=raw_b, old_palette=pal_b, new_palette=new_palette
        )

        stride = int(merged.vertex_stride)
        merged.vertex_count = int(a.vertex_count) + int(b.vertex_count)
        merged.raw_buffer = raw_a + raw_b
        pad = (-len(merged.raw_buffer)) & 0xF
        if pad:
            merged.raw_buffer += b"\x00" * int(pad)

        base = int(a.vertex_count)
        merged.indices = [int(ii) for ii in (a.indices or [])] + [
            int(ii) + base for ii in (b.indices or [])
        ]
        merged.index_count = int(len(merged.indices))
        merged.index_elem_size = (
            2
            if int(getattr(a, "index_elem_size", 0) or 0) == 2
            or int(getattr(b, "index_elem_size", 0) or 0) == 2
            else 1
        )
        elem = (
            int(merged.index_elem_size) if int(merged.index_elem_size) in (1, 2) else 2
        )
        idx_payload = int(len(merged.indices)) * int(elem)
        merged.index_data_len = _align_up(int(idx_payload), 4)
        merged.index_pad_bytes = b"\x00" * max(
            0, int(merged.index_data_len) - int(idx_payload)
        )
        return merged

    for obj in (tagged or {}).values():
        try:
            if getattr(obj, "mode", "") == "EDIT":
                obj.update_from_editmode()
        except Exception:
            pass

    by_section: Dict[int, List[int]] = {}
    for si, sm in enumerate(model.submeshes):
        sec = int(getattr(sm, "mesh_section_off", 0) or 0)
        if sec <= 0:
            continue
        by_section.setdefault(sec, []).append(int(si))

    for sec_off_pack in sorted(by_section.keys(), reverse=True):
        sub_idxs = by_section[sec_off_pack]
        if not sub_idxs:
            continue

        planned_main: Dict[int, _GFSubMesh] = {}

        sec_off_rel = int(sec_off_pack) - int(model_blob_start_in_pack)
        if sec_off_rel < 0 or sec_off_rel + 0x10 > len(out_model):
            raise ValueError("mesh section offset out of model blob bounds")

        old_payload_len = struct.unpack_from("<I", out_model, sec_off_rel + 8)[0]
        old_sec_end = int(sec_off_rel) + 0x10 + int(old_payload_len)
        if old_sec_end > len(out_model):
            raise ValueError("mesh section length out of model blob bounds")

        if bool(allow_palette_split):
            for si in sub_idxs:
                obj = tagged.get(int(si))
                if obj is None:
                    continue
                sm = model.submeshes[int(si)]
                attr_names = set(int(a.name) for a in (sm.attributes or []))
                if not (7 in attr_names and 8 in attr_names):
                    continue
                mesh: bpy.types.Mesh = obj.data                            
                sk_w_by_v, unknown_bones = _gather_weights_skeleton_indices_checked(
                    obj, skeleton_names
                )
                if unknown_bones:
                    sample = ", ".join(unknown_bones[:10])
                    raise ValueError(
                        f"Vertex groups reference bones not in skeleton (first {min(10, len(unknown_bones))}): {sample}"
                    )
                required = set()
                for wl in sk_w_by_v[: int(len(mesh.vertices))]:
                    for sk, w in wl:
                        if float(w) > 0.0:
                            required.add(int(sk))
                if any(not wl for wl in sk_w_by_v[: int(len(mesh.vertices))]):
                    required.add(0)
                if len(required) <= 0x1F:
                    continue

                if not allow_palette_rebuild:
                    raise ValueError(
                        f"Submesh {sm.name!r} needs {len(required)} bones (max=31); "
                        "palette splitting requires palette rebuild (use Rebuild + Split)"
                    )

                parts = build_split_faces_for_submesh(
                    sm=sm, obj=obj, skeleton_names=skeleton_names
                )
                if not parts:
                    raise ValueError("internal error: split returned no parts")
                planned_main[int(si)] = parts[0]
                if len(parts) > 1:
                    candidates = []
                    for sj in sub_idxs:
                        if int(sj) == int(si):
                            continue
                        if int(sj) in planned_main:
                            continue
                        other = model.submeshes[int(sj)]
                        if str(other.name) != str(sm.name):
                            continue
                        if not _submesh_layout_equal(other, sm):
                            continue
                        candidates.append(int(sj))
                    candidates.sort(key=lambda x: 1 if int(x) in tagged else 0)
                    need = int(len(parts) - 1)
                    for part, sj in zip(parts[1:], candidates[:need]):
                        planned_main[int(sj)] = part
                    for part in parts[1 + len(candidates) :]:
                        overflow_mesh_parts.setdefault(str(sm.mesh_name), []).append(
                            part
                        )

        buffers_start_pack = min(
            int(model.submeshes[i].raw_buffer_off) for i in sub_idxs
        )
        buffers_start_rel = int(buffers_start_pack) - int(model_blob_start_in_pack)
        if buffers_start_rel < sec_off_rel or buffers_start_rel > old_sec_end:
            raise ValueError("mesh buffers start out of expected bounds")

        prefix = bytes(out_model[sec_off_rel:buffers_start_rel])
        rebuilt = bytearray(prefix)

        ordered = sorted(sub_idxs, key=lambda i: int(model.submeshes[i].face_index))
        sec_end_abs = int(model_blob_start_in_pack) + int(old_sec_end)
        bounds: Dict[int, Tuple[int, int, int]] = {}
        for oi, si in enumerate(ordered):
            sm = model.submeshes[si]
            vtx_off = int(getattr(sm, "raw_buffer_off", 0) or 0)
            idx_off = int(getattr(sm, "index_buffer_off", 0) or 0)
            next_vtx_off = int(sec_end_abs)
            if oi + 1 < len(ordered):
                next_vtx_off = int(
                    getattr(model.submeshes[ordered[oi + 1]], "raw_buffer_off", 0) or 0
                )
            if vtx_off <= 0 or idx_off <= 0 or idx_off < vtx_off:
                raise ValueError("Invalid face offsets (vtx/idx)")
            if next_vtx_off < idx_off:
                raise ValueError("Invalid face offsets (next_vtx)")
            if next_vtx_off > len(pack_src):
                raise ValueError("Face bounds exceed pack size")
            bounds[int(si)] = (int(vtx_off), int(idx_off), int(next_vtx_off))

        unplanned_geom: Dict[
            int, Tuple[int, List[int], List[Tuple[int, Tuple[float, float]]]]
        ] = {}
        for si in sub_idxs:
            sm = model.submeshes[si]
            planned = planned_main.get(int(si))
            obj = tagged.get(int(si))
            if obj is None and planned is None:
                continue
            mesh: Optional[bpy.types.Mesh] = (
                obj.data if obj is not None else None                            
            )
            if int(sm.primitive_mode) != 0:
                raise ValueError(
                    f"Grow-buffers patch supports primitive_mode=0 (Triangles) only; submesh {sm.name!r} has {int(sm.primitive_mode)}"
                )
            if int(sm.vertex_stride) <= 0:
                raise ValueError(f"vertex_stride<=0 for submesh {sm.name!r}")

            if planned is not None:
                new_vcount = int(getattr(planned, "vertex_count", 0) or 0)
                new_indices = list(getattr(planned, "indices", []) or [])
                elem = int(getattr(planned, "index_elem_size", 0) or 0) or int(
                    getattr(sm, "index_elem_size", 0) or 0
                )
            else:
                if mesh is None:
                    raise ValueError("internal error: missing mesh for unplanned face")
                new_vcount, new_indices, used_keys = _build_uv_seam_split_tris(mesh)
                unplanned_geom[int(si)] = (
                    int(new_vcount),
                    list(int(x) for x in new_indices),
                    list(used_keys),
                )
                elem = int(getattr(sm, "index_elem_size", 0) or 0)
            if elem not in (1, 2):
                raise ValueError(
                    f"Missing/invalid index_elem_size for submesh {sm.name!r}"
                )
            if elem == 1 and int(new_vcount) > 0xFF:
                raise ValueError(
                    f"u8 index buffer cannot address {new_vcount} vertices for {sm.name!r}; "
                    "use Rebuild + Split or reduce vertex count"
                )

            b = bounds.get(int(si))
            if b is None:
                raise ValueError("Missing face bounds")
            vtx_off, idx_off, next_vtx_off = b
            orig_vtx_len = int(idx_off) - int(vtx_off)
            orig_idx_len = int(next_vtx_off) - int(idx_off)
            if orig_vtx_len < 0 or orig_idx_len < 0:
                raise ValueError("Invalid inferred vtx/idx lengths")

            new_vtx_payload = int(new_vcount) * int(sm.vertex_stride)
            new_vtx_len = _align_up(int(new_vtx_payload), 0x10)
            new_idx_payload = int(len(new_indices)) * int(elem)
            new_idx_len = _align_up(int(new_idx_payload), 4)
            if planned is not None:
                vb = bytes(getattr(planned, "raw_buffer", b"") or b"")
                if len(vb) > int(new_vtx_len):
                    new_vtx_len = int(len(vb))
                il = int(getattr(planned, "index_data_len", 0) or 0)
                if il > int(new_idx_len):
                    new_idx_len = int(il)
            if int(orig_vtx_len) > int(new_vtx_len):
                new_vtx_len = int(orig_vtx_len)
            if int(orig_idx_len) > int(new_idx_len):
                new_idx_len = int(orig_idx_len)

            vc_off = int(getattr(sm, "vertex_count_off", 0) or 0) - int(
                model_blob_start_in_pack
            )
            ic_off = int(getattr(sm, "index_count_off", 0) or 0) - int(
                model_blob_start_in_pack
            )
            vl_off = int(getattr(sm, "vertex_data_len_off", 0) or 0) - int(
                model_blob_start_in_pack
            )
            il_off = int(getattr(sm, "index_data_len_off", 0) or 0) - int(
                model_blob_start_in_pack
            )
            for o in (vc_off, ic_off, vl_off, il_off):
                if o < sec_off_rel or o + 4 > buffers_start_rel:
                    raise ValueError("submesh info offsets out of prefix bounds")
            struct.pack_into(
                "<i", rebuilt, int(vc_off) - int(sec_off_rel), int(new_vcount)
            )
            struct.pack_into(
                "<i", rebuilt, int(ic_off) - int(sec_off_rel), int(len(new_indices))
            )
            struct.pack_into(
                "<i", rebuilt, int(vl_off) - int(sec_off_rel), int(new_vtx_len)
            )
            struct.pack_into(
                "<i", rebuilt, int(il_off) - int(sec_off_rel), int(new_idx_len)
            )

            if planned is not None:
                pal_count = int(getattr(planned, "bone_indices_count", 0) or 0)
                palette = list(getattr(planned, "bone_indices", []) or [])[:pal_count]
                if pal_count <= 0 or pal_count > 0x1F:
                    raise ValueError("invalid planned palette size")
                bone_count_off = int(getattr(sm, "vertex_count_off", 0) or 0) - 0x20
                bone_count_rel = int(bone_count_off) - int(model_blob_start_in_pack)
                if (
                    bone_count_rel < sec_off_rel
                    or bone_count_rel + 0x20 > buffers_start_rel
                ):
                    raise ValueError("bone palette offsets out of prefix bounds")
                local_bc = int(bone_count_rel) - int(sec_off_rel)
                rebuilt[local_bc] = int(pal_count) & 0xFF
                for j in range(0x1F):
                    rebuilt[local_bc + 1 + j] = (
                        int(palette[j]) & 0xFF if j < pal_count else 0
                    )

            cmd_off = int(getattr(sm, "index_cmds_off", 0) or 0) - int(
                model_blob_start_in_pack
            )
            cmd_len_u32 = int(getattr(sm, "index_cmds_len_u32", 0) or 0)
            if cmd_off < sec_off_rel or cmd_off + cmd_len_u32 * 4 > buffers_start_rel:
                raise ValueError("index_cmds range out of prefix bounds")
            local_cmd_off = int(cmd_off) - int(sec_off_rel)
            _pica_patch_reg_all_in_cmd_bytes(
                rebuilt,
                cmd_u32_off=local_cmd_off,
                cmd_u32_len=cmd_len_u32,
                reg=0x0228,
                new_param=int(len(new_indices)),
            )

        for oi, si in enumerate(ordered):
            sm = model.submeshes[si]
            stride = int(sm.vertex_stride)
            if stride <= 0:
                raise ValueError(f"vertex_stride<=0 for submesh {sm.name!r}")
            planned = planned_main.get(int(si))
            elem = (
                int(getattr(planned, "index_elem_size", 0) or 0)
                if planned is not None
                else int(getattr(sm, "index_elem_size", 0) or 0)
            )
            if elem not in (1, 2):
                raise ValueError(
                    f"Missing/invalid index_elem_size for submesh {sm.name!r}"
                )

            b = bounds.get(int(si))
            if b is None:
                raise ValueError("Missing face bounds")
            vtx_off, idx_off, next_vtx_off = b
            orig_vtx_len = int(idx_off) - int(vtx_off)
            orig_idx_len = int(next_vtx_off) - int(idx_off)
            if orig_vtx_len < 0 or orig_idx_len < 0:
                raise ValueError("Invalid inferred vtx/idx lengths")

            obj = tagged.get(int(si))
            if obj is None and planned is None:
                rebuilt += bytes(pack_src[int(vtx_off) : int(next_vtx_off)])
                continue

            mesh: Optional[bpy.types.Mesh] = (
                obj.data if obj is not None else None                            
            )
            used_keys: Optional[List[Tuple[int, Tuple[float, float]]]] = None
            if planned is not None:
                new_vcount = int(getattr(planned, "vertex_count", 0) or 0)
                new_indices = list(getattr(planned, "indices", []) or [])
            else:
                if mesh is None:
                    raise ValueError("internal error: missing mesh for unplanned face")
                cached = unplanned_geom.get(int(si))
                if cached is None:
                    new_vcount, new_indices, used_keys = _build_uv_seam_split_tris(mesh)
                else:
                    new_vcount, new_indices, used_keys = cached
            for ii in new_indices:
                if ii < 0 or ii >= new_vcount:
                    raise ValueError(
                        f"Index out of range for submesh {sm.name!r}: {ii} (new_vcount={new_vcount})"
                    )

            new_vtx_payload = int(new_vcount) * int(stride)
            new_vtx_len = _align_up(int(new_vtx_payload), 0x10)
            new_idx_payload = int(len(new_indices)) * int(elem)
            new_idx_len = _align_up(int(new_idx_payload), 4)
            if int(orig_vtx_len) > int(new_vtx_len):
                new_vtx_len = int(orig_vtx_len)
            if int(orig_idx_len) > int(new_idx_len):
                new_idx_len = int(orig_idx_len)

            if planned is not None:
                vbytes = bytearray(bytes(getattr(planned, "raw_buffer", b"") or b""))
                if len(vbytes) > new_vtx_len:
                    raise ValueError("vertex bytes exceed allocated length")
                if len(vbytes) < new_vtx_len:
                    tail = bytes(
                        pack_src[
                            int(vtx_off) + int(len(vbytes)) : int(vtx_off)
                            + int(min(orig_vtx_len, new_vtx_len))
                        ]
                    )
                    pad_len = int(new_vtx_len) - int(len(vbytes)) - int(len(tail))
                    if pad_len < 0:
                        pad_len = 0
                    vbytes += tail + (b"\x00" * pad_len)
                rebuilt += vbytes

                if elem == 2:
                    ib = b"".join(struct.pack("<H", int(i)) for i in new_indices)
                else:
                    ib = bytes(int(i) & 0xFF for i in new_indices)
                if len(ib) > new_idx_len:
                    raise ValueError("index bytes exceed allocated length")
                if len(ib) < new_idx_len:
                    tail = bytes(
                        pack_src[
                            int(idx_off) + int(len(ib)) : int(idx_off)
                            + int(min(orig_idx_len, new_idx_len))
                        ]
                    )
                    pad_len = int(new_idx_len) - int(len(ib)) - int(len(tail))
                    if pad_len < 0:
                        pad_len = 0
                    ib += tail + (b"\x00" * pad_len)
                rebuilt += ib
                continue

            src_base_pack = int(vtx_off)
            old_vcount = int(sm.vertex_count)
            if old_vcount <= 0:
                old_vcount = 1
            templ = pack_src[src_base_pack : src_base_pack + stride]
            if len(templ) != stride:
                templ = bytes([0]) * stride

            if used_keys is None:
                used_keys = [(int(i), (0.0, 0.0)) for i in range(int(new_vcount))]
            src_vcount = int(len(mesh.vertices))
            uv_by_v = [uv for _src_vi, uv in used_keys]
            col_by_src = [(1.0, 1.0, 1.0, 1.0)] * src_vcount
            try:
                ca = getattr(mesh, "color_attributes", None)
                col_attr = (
                    ca.get("Col") if ca is not None and hasattr(ca, "get") else None
                )
                if col_attr is not None and getattr(col_attr, "data", None) is not None:
                    seen = [False] * src_vcount
                    for poly in mesh.polygons:
                        for li in poly.loop_indices:
                            vi = int(mesh.loops[li].vertex_index)
                            if 0 <= vi < src_vcount and not seen[vi]:
                                c = col_attr.data[li].color
                                col_by_src[vi] = (
                                    float(c[0]),
                                    float(c[1]),
                                    float(c[2]),
                                    float(c[3]),
                                )
                                seen[vi] = True
            except Exception:
                pass
            col_by_v = [col_by_src[int(src_vi)] for src_vi, _uv in used_keys]

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

            weights_by_src: List[List[Tuple[int, float]]] = [
                [] for _ in range(src_vcount)
            ]
            weights_by_v: List[List[Tuple[int, float]]] = [
                [] for _ in range(new_vcount)
            ]
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
                sk_w_by_v, unknown_bones = _gather_weights_skeleton_indices_checked(
                    obj, skeleton_names
                )
                if unknown_bones:
                    sample = ", ".join(unknown_bones[:10])
                    raise ValueError(
                        f"Vertex groups reference bones not in skeleton (first {min(10, len(unknown_bones))}): {sample}"
                    )

                max_palette = 0x1F
                palette_raw = list(sm.bone_indices or [])[
                    : int(getattr(sm, "bone_indices_count", 0) or 0)
                ]
                palette: List[int] = []
                palette_set = set()
                for b in palette_raw:
                    bi = int(b)
                    if bi in palette_set:
                        continue
                    palette.append(int(bi))
                    palette_set.add(int(bi))

                required = set()
                for wl in sk_w_by_v[:src_vcount]:
                    for sk, w in wl:
                        if float(w) > 0.0:
                            required.add(int(sk))

                if allow_palette_rebuild and required:
                    palette = [sk for sk in palette if int(sk) in required]
                    palette_set = set(int(sk) for sk in palette)
                    for sk in sorted(required):
                        if int(sk) in palette_set:
                            continue
                        palette.append(int(sk))
                        palette_set.add(int(sk))

                if len(palette) > max_palette:
                    if allow_palette_split:
                        raise ValueError(
                            f"internal error: palette overflow reached unplanned path for {sm.name!r}"
                        )
                    required_names: List[str] = []
                    for sk in sorted(required):
                        if 0 <= int(sk) < len(skeleton_names):
                            required_names.append(str(skeleton_names[int(sk)]))
                        else:
                            required_names.append(f"<skel:{int(sk)}>")
                    sample = ", ".join(required_names[:16])
                    raise ValueError(
                        f"Submesh palette overflow for {sm.name!r}: needs {len(palette)} bones, max={max_palette}. "
                        "Use Rebuild + Split, or adjust routing/weights."
                        + (
                            f" Required bones (first {min(16, len(required_names))}): {sample}"
                            if required_names
                            else ""
                        )
                    )

                pal_count = len(palette)
                if pal_count <= 0:
                    raise ValueError("Submesh has empty bone palette")

                if allow_palette_rebuild:
                    bone_count_off = int(getattr(sm, "vertex_count_off", 0) or 0) - 0x20
                    bone_count_rel = int(bone_count_off) - int(model_blob_start_in_pack)
                    if (
                        bone_count_rel < sec_off_rel
                        or bone_count_rel + 0x20 > buffers_start_rel
                    ):
                        raise ValueError("bone palette offsets out of prefix bounds")
                    local_bc = int(bone_count_rel) - int(sec_off_rel)
                    rebuilt[local_bc] = int(pal_count) & 0xFF
                    for j in range(max_palette):
                        rebuilt[local_bc + 1 + j] = (
                            int(palette[j]) & 0xFF if j < pal_count else 0
                        )

                sk_to_pi = {int(sk): int(i) for i, sk in enumerate(palette)}
                fallback_pi = 0
                for wi, wl in enumerate(sk_w_by_v[:src_vcount]):
                    out_wl: List[Tuple[int, float]] = []
                    for sk, w in wl[:4]:
                        pi = sk_to_pi.get(int(sk))
                        if pi is None:
                            continue
                        out_wl.append((int(pi), float(w)))
                    if not out_wl:
                        out_wl = [(int(fallback_pi), 1.0)]
                    if 0 <= int(wi) < len(weights_by_src):
                        weights_by_src[int(wi)] = out_wl
                weights_by_v = [
                    weights_by_src[int(src_vi)]
                    if 0 <= int(src_vi) < len(weights_by_src)
                    else []
                    for src_vi, _uv in used_keys
                ]

            vbytes = bytearray()
            for vi, (src_vi, _uv) in enumerate(used_keys):
                src = (
                    pack_src[
                        src_base_pack + int(src_vi) * stride : src_base_pack
                        + (int(src_vi) + 1) * stride
                    ]
                    if int(src_vi) < old_vcount
                    else templ
                )
                if len(src) != stride:
                    src = templ
                dst = bytearray(src)
                v = mesh.vertices[int(src_vi)]
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
                    and vi < len(uv_by_v)
                ):
                    comp = len(
                        _pack_attr_value(int(uv_attr.fmt), float(uv_attr.scale), 0.0)
                    )
                    u, vv = uv_by_v[vi]
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
                    and vi < len(col_by_v)
                ):
                    comp = len(
                        _pack_attr_value(
                            int(col_attr_layout.fmt), float(col_attr_layout.scale), 0.0
                        )
                    )
                    cr, cg, cb, ca = col_by_v[vi]
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
                    and vi < len(weights_by_v)
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
                    wl = weights_by_v[vi]
                    idxs = [0] * elems
                    wts = [0.0] * elems
                    for wi, (pi, w) in enumerate(wl[:elems]):
                        idxs[wi] = int(pi)
                        wts[wi] = float(w)
                    s = float(sum(max(0.0, w) for w in wts))
                    if s > 0:
                        wts = [max(0.0, w) / s for w in wts]
                    else:
                        idxs[0] = 0
                        wts[0] = 1.0
                    o = int(bi_off)
                    for j in range(elems):
                        dst[o : o + bi_comp] = _pack_attr_value(
                            int(bi_attr.fmt), float(bi_attr.scale), float(idxs[j])
                        )
                        o += bi_comp
                    o = int(bw_off)
                    for j in range(elems):
                        dst[o : o + bw_comp] = _pack_attr_value(
                            int(bw_attr.fmt), float(bw_attr.scale), float(wts[j])
                        )
                        o += bw_comp

                vbytes += dst
            if len(vbytes) > new_vtx_len:
                raise ValueError("vertex bytes exceed allocated length")
            if len(vbytes) < new_vtx_len:
                tail = bytes(
                    pack_src[
                        int(vtx_off) + int(len(vbytes)) : int(vtx_off)
                        + int(min(orig_vtx_len, new_vtx_len))
                    ]
                )
                pad_len = int(new_vtx_len) - int(len(vbytes)) - int(len(tail))
                if pad_len < 0:
                    pad_len = 0
                vbytes += tail + (b"\x00" * pad_len)
            rebuilt += vbytes

            if elem == 2:
                ib = b"".join(struct.pack("<H", int(i)) for i in new_indices)
            else:
                ib = bytes(int(i) & 0xFF for i in new_indices)
            if len(ib) > new_idx_len:
                raise ValueError("index bytes exceed allocated length")
            if len(ib) < new_idx_len:
                tail = bytes(
                    pack_src[
                        int(idx_off) + int(len(ib)) : int(idx_off)
                        + int(min(orig_idx_len, new_idx_len))
                    ]
                )
                pad_len = int(new_idx_len) - int(len(ib)) - int(len(tail))
                if pad_len < 0:
                    pad_len = 0
                ib += tail + (b"\x00" * pad_len)
            rebuilt += ib

        cur_payload = len(rebuilt) - 0x10
        pad_len = (0x10 - (cur_payload & 0xF)) if (cur_payload & 0xF) else 0x10
        rebuilt += b"\x00" * int(pad_len)
        new_payload_len = len(rebuilt) - 0x10
        struct.pack_into("<I", rebuilt, 8, int(new_payload_len))
        out_model[sec_off_rel:old_sec_end] = rebuilt

    if overflow_mesh_parts:
        if disallow_new_mesh_sections:
            summary = ", ".join(
                f"{k}:+{len(v)}" for k, v in sorted(overflow_mesh_parts.items())
            )
            raise ValueError(
                "Palette splitting required more face slots than available in the existing mesh section(s); "
                "new mesh sections are disallowed in this mode. "
                f"Overflow parts by mesh name: {summary}"
            )
        parsed_models, _, _, _ = _load_any(bytes(out_model))
        if not parsed_models:
            raise ValueError("internal error: failed to re-parse patched model blob")
        model2 = parsed_models[0]

        by_mi: Dict[int, List[_GFSubMesh]] = {}
        for sm in model2.submeshes:
            by_mi.setdefault(int(sm.mesh_index), []).append(sm)

        meshes_by_index: Dict[int, bytes] = {}
        for mi in range(len(model2.mesh_names)):
            faces = sorted(by_mi.get(int(mi), []), key=lambda s: int(s.face_index))
            if not faces:
                continue
            fc = int(len(faces))
            for fi, sm in enumerate(faces):
                sm.mesh_index = int(mi)
                sm.face_index = int(fi)
                sm.mesh_face_count = int(fc)
                sm.mesh_name = str(model2.mesh_names[int(mi)])
            meshes_by_index[int(mi)] = write_gf_mesh_blob(
                str(model2.mesh_names[int(mi)]), faces
            )

        for base_name, parts in overflow_mesh_parts.items():
            new_mesh_name = f"{base_name}_Overflow"
            if new_mesh_name in model2.mesh_names:
                k = 2
                while f"{new_mesh_name}{k}" in model2.mesh_names:
                    k += 1
                new_mesh_name = f"{new_mesh_name}{k}"

            new_index = int(len(model2.mesh_names))
            model2.mesh_names.append(str(new_mesh_name))

            faces: List[_GFSubMesh] = []
            for fi, sm in enumerate(parts):
                x = copy.deepcopy(sm)
                x.mesh_name = str(new_mesh_name)
                x.mesh_index = int(new_index)
                x.face_index = int(fi)
                faces.append(x)
            fc = int(len(faces))
            for sm in faces:
                sm.mesh_face_count = int(fc)
            meshes_by_index[int(new_index)] = write_gf_mesh_blob(
                str(new_mesh_name), faces
            )

        return write_gf_model_blob(model2, meshes_by_index=meshes_by_index)

    return bytes(out_model)
