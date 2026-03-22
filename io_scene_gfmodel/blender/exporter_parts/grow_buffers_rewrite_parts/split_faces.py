
from __future__ import annotations

import copy
import struct
from typing import Dict, List, Optional, Tuple

import bpy
from mathutils import Matrix, Vector

from ....core.types import _GFSubMesh
from ..common import align_up as _align_up
from ..vertex_pack import (
    _gather_weights_skeleton_indices_checked,
    _pack_attr_value,
    _vertex_attr_offsets,
)

def _build_uv_seam_split_tris(
    mesh: bpy.types.Mesh,
) -> Tuple[int, List[int], List[Tuple[int, Tuple[float, float]]]]:
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








    has_full_skin = (
        bi_attr is not None
        and bw_attr is not None
        and bi_off is not None
        and bw_off is not None
    )
    has_indices_only_skin = (
        bi_attr is not None
        and bi_off is not None
        and (bw_attr is None or bw_off is None)
    )
    has_skin = has_full_skin or has_indices_only_skin
    if (bw_attr is not None or bw_off is not None) and not has_full_skin:
        raise ValueError(
            f"Submesh {sm.name!r} has partial skinning attributes (need both 7 and 8, or indices-only 7)"
        )

    sk_w_by_v: List[List[Tuple[int, float]]] = [
        [] for _ in range(len(mesh.vertices))
    ]
    if has_full_skin:
        sk_w_by_v, unknown_bones = _gather_weights_skeleton_indices_checked(
            obj, skeleton_names
        )
        if unknown_bones:
            sample = ", ".join(unknown_bones[:10])
            raise ValueError(
                f"Vertex groups reference bones not in skeleton (first {min(10, len(unknown_bones))}): {sample}"
            )

    palette_raw = list(sm.bone_indices or [])[
        : int(getattr(sm, "bone_indices_count", 0) or 0)
    ]

    def _read_pi(b: bytes, o: int, fmt: int) -> int:
        if o < 0 or o >= len(b):
            return 0
        if int(fmt) == 0:
            return int(struct.unpack_from("<b", b, o)[0])
        if int(fmt) == 1:
            return int(b[o])
        if int(fmt) == 2:
            return int(struct.unpack_from("<h", b, o)[0])
        return int(round(float(struct.unpack_from("<f", b, o)[0])))

    bones_by_v: List[List[int]] = []
    if has_indices_only_skin:
        raw = bytes(getattr(sm, "raw_buffer", b"") or b"")
        vcount = int(getattr(sm, "vertex_count", 0) or 0)
        stride_i = int(stride)
        for vi in range(max(0, vcount)):
            base = int(vi) * stride_i + int(bi_off)
            pi = _read_pi(raw, base, int(bi_attr.fmt))
            if pi < 0:
                pi = 0
            if palette_raw and pi >= len(palette_raw):
                sk = int(palette_raw[0])
            else:
                sk = int(palette_raw[int(pi)]) if palette_raw else 0
            bones_by_v.append([int(sk)])
    else:
        for wl in sk_w_by_v:
            bones = [int(sk) for sk, w in wl if float(w) > 0.0]
            if not bones:
                bones = [0]
            bones_by_v.append(bones)

    max_palette = 0x1F

    def build_palette(required: List[int]) -> List[int]:
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
                has_full_skin
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
            elif has_indices_only_skin and bi_off is not None and bi_attr is not None:
                bi_comp = len(
                    _pack_attr_value(int(bi_attr.fmt), float(bi_attr.scale), 0.0)
                )
                elems = max(1, min(4, int(bi_attr.elements)))
                o0 = int(bi_off)
                for j in range(elems):
                    pi0 = _read_pi(bytes(dst), o0 + j * bi_comp, int(bi_attr.fmt))
                    if pi0 < 0:
                        pi0 = 0
                    if palette_raw and pi0 >= len(palette_raw):
                        sk = int(palette_raw[0])
                    else:
                        sk = int(palette_raw[int(pi0)]) if palette_raw else 0
                    new_pi = int(sk_to_pi.get(int(sk), 0))
                    dst[o0 + j * bi_comp : o0 + (j + 1) * bi_comp] = _pack_attr_value(
                        int(bi_attr.fmt), float(bi_attr.scale), float(new_pi)
                    )

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
    if 7 not in attr_names:
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
