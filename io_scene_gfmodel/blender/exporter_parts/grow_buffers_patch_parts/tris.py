

from __future__ import annotations

import copy
import struct
from typing import Dict, List, Optional, Tuple
import bmesh
import bpy
from mathutils import Matrix, Vector
from ....core.gfpack import parse_gf_model_pack
from ....core.gfpack import write_gf_model_pack as write_gf_model_pack_low
from ..grow_buffers_rewrite import _rewrite_model_blob_grow_buffers_tris

def _patch_pack_grow_buffers_tris(
    pack_src: bytes,
    model: "_GFModel",
    *,
    tagged: Dict[int, bpy.types.Object],
    gf_from_blender: Matrix,
    global_scale: float,
    skeleton_names: List[str],
    disallow_new_mesh_sections: bool = False,
    allow_palette_rebuild: bool = True,
    allow_palette_split: bool = True,
    palette_append_only: bool = False,
    palette_prune_unused: bool = False,
) -> Tuple[bytes, int]:

    if not tagged:
        raise ValueError(
            "No tagged meshes found (gfmodel_submesh_index); import via GFModel Archive and select a mesh object"
        )
    pack = parse_gf_model_pack(pack_src)
    model_slot = None
    for i in range(int(pack.counts[0])):
        e = pack.get(0, i)
        if e is not None:
            model_slot = int(i)
            break
    if model_slot is None:
        raise ValueError("Model section is empty in GFModelPack")
    entry = pack.get(0, int(model_slot))
    if entry is None:
        raise ValueError("Missing model entry in GFModelPack")

    new_model_blob = _rewrite_model_blob_grow_buffers_tris(
        pack_src=pack_src,
        model_blob=entry.blob,
        model_blob_start_in_pack=int(entry.addr),
        model=model,
        tagged=tagged,
        gf_from_blender=gf_from_blender,
        global_scale=float(global_scale),
        skeleton_names=skeleton_names,
        disallow_new_mesh_sections=bool(disallow_new_mesh_sections),
        allow_palette_rebuild=bool(allow_palette_rebuild),
        allow_palette_split=bool(allow_palette_split),
        palette_append_only=bool(palette_append_only),
        palette_prune_unused=bool(palette_prune_unused),
    )
    out = write_gf_model_pack_low(
        pack,
        replacements={(0, int(model_slot)): bytes(new_model_blob)},
        align_blobs=0x80,
    )
    return bytes(out), int(len(tagged))


def _tri_bones_used(
    bones_by_v: List[List[int]],
    tri_verts: Tuple[int, int, int],
) -> List[int]:
    used: Dict[int, None] = {}
    for vi in tri_verts:
        if 0 <= int(vi) < len(bones_by_v):
            for bi in bones_by_v[int(vi)]:
                used[int(bi)] = None
    return list(used.keys())


def _collect_tris_for_material(
    obj: bpy.types.Object,
    *,
    material_name: str,
) -> List[Tuple[int, int, int, int, int, int]]:
    if obj.type != "MESH":
        return []
    try:
        if getattr(obj, "mode", "") == "EDIT":
            obj.update_from_editmode()
    except Exception:
        pass
    mesh: bpy.types.Mesh = obj.data
    try:
        mesh.calc_loop_triangles()
    except Exception:
        pass

    def base(n: str) -> str:
        s = str(n or "")
        if len(s) > 4 and s[-4] == "." and s[-3:].isdigit():
            return s[:-4]
        return s

    target = base(str(material_name))

    out: List[Tuple[int, int, int, int, int, int]] = []
    for tri in getattr(mesh, "loop_triangles", []) or []:
        try:
            pi = int(getattr(tri, "polygon_index", -1))
            if 0 <= pi < len(mesh.polygons):
                poly = mesh.polygons[pi]
                mi = int(getattr(poly, "material_index", -1))
                if 0 <= mi < len(getattr(obj, "material_slots", []) or []):
                    ms = obj.material_slots[mi]
                    bm = getattr(ms, "material", None)
                    bm_name = base(str(getattr(bm, "name", "") or ""))
                    if bm_name and bm_name != target:
                        continue
        except Exception:
            pass
        a, b, c = tri.vertices
        la, lb, lc = tri.loops
        out.append((int(a), int(b), int(c), int(la), int(lb), int(lc)))
    return out


def _collect_tris_all(
    obj: bpy.types.Object,
) -> List[Tuple[int, int, int, int, int, int]]:
    if obj.type != "MESH":
        return []
    try:
        if getattr(obj, "mode", "") == "EDIT":
            obj.update_from_editmode()
    except Exception:
        pass
    mesh: bpy.types.Mesh = obj.data
    try:
        mesh.calc_loop_triangles()
    except Exception:
        pass
    out: List[Tuple[int, int, int, int, int, int]] = []
    for tri in getattr(mesh, "loop_triangles", []) or []:
        a, b, c = tri.vertices
        la, lb, lc = tri.loops
        out.append((int(a), int(b), int(c), int(la), int(lb), int(lc)))
    return out
