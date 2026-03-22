
from __future__ import annotations

from .inplace_patch_parts.vertex_attrs import (
    _patch_pack_normals_in_place,
    _patch_pack_pos_nrm_uv0_skin_in_place,
    _patch_pack_positions_in_place,
    _patch_pack_positions_normals_in_place,
    _patch_pack_positions_normals_uv0_in_place,
    _patch_submesh_positions_in_place,
)
from .inplace_patch_parts.uv import _patch_pack_uv0_in_place
from .inplace_patch_parts.indices import _patch_pack_indices_in_place
from .inplace_patch_parts.topology import _patch_pack_topology_tris_in_place
from .inplace_patch_parts.verts_topology import _patch_pack_verts_topology_tris_in_place
from .inplace_patch_parts.skin import _patch_pack_skin_in_place

__all__ = [
    '_patch_pack_indices_in_place',
    '_patch_pack_normals_in_place',
    '_patch_pack_pos_nrm_uv0_skin_in_place',
    '_patch_pack_positions_in_place',
    '_patch_pack_positions_normals_in_place',
    '_patch_pack_positions_normals_uv0_in_place',
    '_patch_pack_skin_in_place',
    '_patch_pack_topology_tris_in_place',
    '_patch_pack_uv0_in_place',
    '_patch_pack_verts_topology_tris_in_place',
    '_patch_submesh_positions_in_place',
]
