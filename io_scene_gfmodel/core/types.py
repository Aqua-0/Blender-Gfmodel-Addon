"""Shared data structures for the GFModel Blender add-on.

This module intentionally contains no Blender registration logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .math_compat import Vector


@dataclass
class _GFLUT:
    texture_hash: int
    command_bytes: bytes
    command_words: List[int]


@dataclass
class _GFTexture:
    name: str
    width: int
    height: int
    fmt: int
    raw: bytes


@dataclass
class _GFBone:
    name: str
    parent: str
    flags: int
    scale: Vector
    rotation: Vector
    translation: Vector


@dataclass
class _GFTextureCoord:
    name: Optional[str]


@dataclass
class _GFTextureUnit:
    name: str
    unit_index: int
    mapping_type: int
    scale: Vector
    rotation: float
    translation: Vector
    sampler_words: List[int]


@dataclass
class _GFMaterial:
    name: str
    shader_name: str
    vtx_shader: str
    frag_shader: str
    lut_hashes: Tuple[int, int, int]
    tex0: Optional[str]
    tex1: Optional[str]
    tex2: Optional[str]
    tex_units: List[_GFTextureUnit]
    bump_texture: int
    edge_type: int
    id_edge_enable: int
    edge_id: int
    projection_type: int
    rim_pow: float
    rim_scale: float
    phong_pow: float
    phong_scale: float
    id_edge_offset_enable: int
    edge_map_alpha_mask: int
    bake_ops: List[int]
    vertex_shader_type: int
    shader_params: Tuple[float, float, float, float]
    const_assignments: List[int]
    colors_rgba: List[Tuple[int, int, int, int]]
    alpha_test_enabled: bool
    alpha_test_ref: float
    alpha_test_func: int
    blend_func: Optional[dict]
    blend_color_rgba: Optional[Tuple[int, int, int, int]]
    stencil_test: Optional[dict]
    stencil_op: Optional[dict]
    depth_test_enabled: Optional[bool]
    depth_test_func: Optional[int]
    depth_write: Optional[bool]
    color_write_mask: Optional[Tuple[bool, bool, bool, bool]]
    face_culling: Optional[int]
    render_priority: int
    render_layer: int
    header_hashes: Tuple[int, int, int, int]
    unk_render: int
    pica_commands: List[int]
    pica_regs: Dict[int, int]
                                                                                              
                                                                  
    raw_blob: Optional[bytes] = None


@dataclass
class _GFShaderTexEnvStage:
    stage: int
    source: Optional[int]
    operand: Optional[int]
    combiner: Optional[int]
    color: Optional[int]
    scale: Optional[int]


@dataclass
class _GFShader:
    name: str
    texenv_stages: List[_GFShaderTexEnvStage]
    texenv_buffer_color: Optional[int]
    texenv_update_buffer: Optional[int]
    pica_commands: List[int]
    pica_regs: Dict[int, int]


def _jsonify_pica_regs(regs: Dict[int, int]) -> Dict[str, int]:
                                                                   
    return {f"0x{k:04X}": int(v) for k, v in sorted(regs.items())}


@dataclass
class _PICAAttribute:
    name: int
    fmt: int
    elements: int
    scale: float


@dataclass
class _PICAFixedAttribute:
    name: int
    x: float
    y: float
    z: float
    w: float


@dataclass
class _GFSubMesh:
    name: str
    mesh_index: int
    face_index: int
    mesh_name: str
    mesh_bbox_min: Vector
    mesh_bbox_max: Vector
    mesh_is_blend_shape: bool
    mesh_face_count: int
    mesh_weight_max: int
    bone_indices_count: int
    bone_indices: List[int]
    vertex_count: int
    index_count: int
    vertex_stride: int
    primitive_mode: int
    indices: List[int]
    raw_buffer: bytes
    attributes: List[_PICAAttribute]
    fixed_attributes: List[_PICAFixedAttribute]
    enable_cmds: List[int]
    disable_cmds: List[int]
    index_cmds: List[int]
                                                                                          
                                                                                        
                             
    index_data_len: int = 0
    index_pad_bytes: bytes = b""
                                                                                             
                                                                   
    index_buffer_off: int = 0
                                           
    index_elem_size: int = 0
                                                                                                     
                                                                                         
                                                                                         
    index_count_off: int = 0
    index_cmds_off: int = 0
    index_cmds_len_u32: int = 0
                                                                              
    vertex_count_off: int = 0
                                                                                                   
    vertex_data_len_off: int = 0
    index_data_len_off: int = 0
                                                                                   
    mesh_section_off: int = 0
    mesh_section_len_off: int = 0
                                                                                             
                                                                                              
                                                                                         
                                           
    raw_buffer_off: int = 0


@dataclass
class _GFModel:
    name: str
    shader_names: List[str]
    texture_names: List[str]
    material_names: List[str]
    mesh_names: List[str]
    bbox_min: Vector
    bbox_max: Vector
    transform_rows: Tuple[Tuple[float, float, float, float], ...]
    unknown_blob: bytes
    luts: List[_GFLUT]
    skeleton: List[_GFBone]
    materials: List[_GFMaterial]
    submeshes: List[_GFSubMesh]
    unknown_off: int = 0


@dataclass
class _GFMotKeyFrame:
    frame: int
    value: float
    slope: float


@dataclass
class _GFMotBoneTransform:
    name: str
    is_axis_angle: bool
    sx: List[_GFMotKeyFrame]
    sy: List[_GFMotKeyFrame]
    sz: List[_GFMotKeyFrame]
    rx: List[_GFMotKeyFrame]
    ry: List[_GFMotKeyFrame]
    rz: List[_GFMotKeyFrame]
    tx: List[_GFMotKeyFrame]
    ty: List[_GFMotKeyFrame]
    tz: List[_GFMotKeyFrame]


@dataclass
class _GFMotion:
    index: int
    frames_count: int
    is_looping: bool
    is_blended: bool
    anim_region_min: Vector
    anim_region_max: Vector
    anim_hash: int
    bones: List[_GFMotBoneTransform]
    uv_transforms: List["__GFMotUVTransform"]
    visibility_tracks: List["__GFMotVisibilityTrack"]
    unknown_sections: List[Tuple[int, int, int, bytes]]


@dataclass
class __GFMotUVTransform:
    name: str
    unit_index: int
    sx: List[_GFMotKeyFrame]
    sy: List[_GFMotKeyFrame]
    rot: List[_GFMotKeyFrame]
    tx: List[_GFMotKeyFrame]
    ty: List[_GFMotKeyFrame]


@dataclass
class __GFMotVisibilityTrack:
    name: str
    values: List[bool]
