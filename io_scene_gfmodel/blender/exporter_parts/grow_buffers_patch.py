
from __future__ import annotations

from .grow_buffers_patch_parts.routing import _material_routing_candidates, _route_source_object_to_submesh_slots
from .grow_buffers_patch_parts.temp_mesh import _build_temp_mesh_object_for_slot, _build_temp_mesh_object_for_sources
from .grow_buffers_patch_parts.textures import _collect_texture_overrides_by_slot, _patch_pack_textures_rgba8, _rgba8_bytes_from_image
from .grow_buffers_patch_parts.tris import _collect_tris_all, _collect_tris_for_material, _patch_pack_grow_buffers_tris, _tri_bones_used
from .grow_buffers_patch_parts.uv import _apply_uv_strategy_to_mesh

__all__ = [
    '_apply_uv_strategy_to_mesh',
    '_build_temp_mesh_object_for_slot',
    '_build_temp_mesh_object_for_sources',
    '_collect_texture_overrides_by_slot',
    '_collect_tris_all',
    '_collect_tris_for_material',
    '_material_routing_candidates',
    '_patch_pack_grow_buffers_tris',
    '_patch_pack_textures_rgba8',
    '_rgba8_bytes_from_image',
    '_route_source_object_to_submesh_slots',
    '_tri_bones_used',
]
