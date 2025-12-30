"""Export operator definitions."""

from __future__ import annotations

import bpy
from bpy.props import (
    BoolProperty,
    CollectionProperty,
    EnumProperty,
    FloatProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)
from bpy_extras.io_utils import ExportHelper

from .export_scene_execute import export_scene_execute
from .props import GFModel_GrowBuffersMaterialSource


class EXPORT_SCENE_OT_gfmodel(bpy.types.Operator, ExportHelper):
    bl_idname = "export_scene.gfmodel"
    bl_label = "Export GFModel (Scaffold v1)"
    bl_options = {"UNDO"}

    filename_ext = ""
    filter_glob: StringProperty(default="*.*", options={"HIDDEN"})

    export_meshes: BoolProperty(name="Export Meshes/Weights", default=True)
    mesh_export_mode: EnumProperty(
        name="Mesh Export Mode",
        items=[
            (
                "REBUILD_FROM_SCENE",
                "Rebuild From Scene",
                "Repack vertex/index buffers from the current Blender meshes (enables geometry edits, but may diverge from retail layout)",
            ),
            (
                "REBUILD_ACTIVE_SUBMESH",
                "Rebuild Active Submesh (Topology, Experimental)",
                "Rebuild vertex/index buffers only for the active tagged submesh (enables add/remove verts on that submesh); other submeshes keep original bytes",
            ),
            (
                "PRESERVE_SOURCE_BYTES",
                "Preserve Source Bytes (No-Op)",
                "Write the exact imported bytes back out (useful to validate patching / game invariants); ignores any scene mesh edits",
            ),
            (
                "UPDATE_POSITIONS_IN_PLACE",
                "Update Positions In-Place",
                "Update only vertex positions inside the existing vertex buffer layout (no topology/index changes); safest for in-game patching",
            ),
            (
                "UPDATE_POS_NRM_IN_PLACE",
                "Update Positions+Normals In-Place",
                "Update vertex positions and normals inside the existing vertex buffer layout (no topology/index changes)",
            ),
            (
                "UPDATE_POS_NRM_UV0_IN_PLACE",
                "Update Pos+Nrm+UV0 In-Place",
                "Update vertex positions, normals, and UV0 inside the existing vertex buffer layout (no topology/index changes)",
            ),
            (
                "UPDATE_POS_NRM_UV0_SKIN_IN_PLACE",
                "Update Pos+Nrm+UV0+Skin In-Place",
                "Update vertex positions, normals, UV0, and skinning (BoneIndex+BoneWeight) inside the existing vertex buffer layout (no topology/index changes)",
            ),
            (
                "UPDATE_NORMALS_IN_PLACE",
                "Update Normals In-Place",
                "Update only vertex normals inside the existing vertex buffer layout (no topology/index changes)",
            ),
            (
                "UPDATE_UV0_IN_PLACE",
                "Update UV0 In-Place",
                "Update only UV0 inside the existing vertex buffer layout (no topology/index changes)",
            ),
            (
                "UPDATE_INDICES_IN_PLACE",
                "Update Indices In-Place (Index-Only Topology)",
                "Update only the index buffer inside the existing layout (requires identical vertex/index counts); enables index-only topology edits",
            ),
            (
                "UPDATE_TOPOLOGY_TRIS_IN_PLACE",
                "Update Topology In-Place (Tris, Resize Index Count)",
                "Update the index buffer and index counts inside the existing allocated index buffer; supports deleting/adding faces without changing vertex count (triangles only)",
            ),
            (
                "UPDATE_VERTS_TOPOLOGY_TRIS_IN_PLACE",
                "Update Verts+Topo In-Place (Tris, Resize Counts)",
                "Update vertex+index buffers and counts inside the existing allocated buffers; supports add/remove vertices and faces as long as they fit (triangles only)",
            ),
            (
                "GROW_BUFFERS_TRIS",
                "Grow Buffers (Tris, Rewrite Pack)",
                "Rewrite the GFModel blob to grow vertex/index buffers for tagged submeshes (triangles only); enables topology edits beyond original capacity",
            ),
        ],
        default="REBUILD_FROM_SCENE",
    )
    export_material_textures: BoolProperty(
        name="Export Material Texture Bindings", default=True
    )
    material_export_mode: EnumProperty(
        name="Material Export Mode",
        items=[
            (
                "PRESERVE_RAW",
                "Preserve Original (Ohana-safe)",
                "Keep original material section bytes; ignores Blender-side texture/mapping edits",
            ),
            (
                "UPDATE_BINDINGS",
                "Update From Blender",
                "Rewrite materials whose texture/mapping values changed in Blender",
            ),
        ],
        default="PRESERVE_RAW",
    )
    export_textures: BoolProperty(
        name="Export Texture Data (Overwrite Slots)",
        default=False,
        description="If enabled, overwrites existing GFModelPack texture slots (does not add new slots)",
    )
    texture_max_size: IntProperty(
        name="Texture Max Size",
        default=256,
        min=8,
        description="Downscale textures to fit within this size (keeps aspect); must be multiple of 8 for tiled formats",
    )
    texture_mode: EnumProperty(
        name="Texture Mode",
        items=[
            (
                "KEEP",
                "Keep Original",
                "Keep original texture blobs (only repoint names)",
            ),
            (
                "ORIGINAL_FORMAT",
                "Original Format",
                "Encode to the original texture format+size from the file (errors if encoder not implemented for that fmt)",
            ),
            ("RGBA8", "RGBA8", "Write RGBA8 swizzled (no ETC encoding)"),
            (
                "RGBA8_SAME_SIZE",
                "RGBA8 (Same Size)",
                "Write RGBA8 swizzled (no ETC encoding); requires image width/height match the original texture",
            ),
            (
                "RGBA8_ORIGINAL_SIZE",
                "RGBA8 (Auto-Resize to Original)",
                "Write RGBA8 swizzled (no ETC encoding); auto-resizes images to the original texture size",
            ),
            (
                "OVERRIDE_FORMAT",
                "Override Format (Original Size)",
                "Encode to a selected format using the original file's texture size",
            ),
        ],
        default="KEEP",
    )
    texture_override_format: EnumProperty(
        name="Texture Override Format",
        items=[
            ("RGBA8", "RGBA8", "PICA RGBA8 (GF fmt=0x4)"),
            ("RGB8", "RGB8", "PICA RGB8 (GF fmt=0x3)"),
            ("RGB565", "RGB565", "PICA RGB565 (GF fmt=0x2)"),
            ("RGBA4", "RGBA4", "PICA RGBA4 (GF fmt=0x16)"),
            ("RGBA5551", "RGBA5551", "PICA RGBA5551 (GF fmt=0x17)"),
            ("LA8", "LA8", "PICA LA8 (GF fmt=0x23)"),
            ("L8", "L8", "PICA L8 (GF fmt=0x25)"),
            ("A8", "A8", "PICA A8 (GF fmt=0x26)"),
            ("LA4", "LA4", "PICA LA4 (GF fmt=0x27)"),
            ("L4", "L4", "PICA L4 (GF fmt=0x28)"),
            ("A4", "A4", "PICA A4 (GF fmt=0x29)"),
            ("ETC1", "ETC1", "PICA ETC1 (GF fmt=0x2A)"),
            ("ETC1A4", "ETC1A4", "PICA ETC1A4 (GF fmt=0x2B)"),
        ],
        default="RGBA8",
    )
    output_container: EnumProperty(
        name="Output Container",
        items=[
            ("AUTO", "Auto (Same as Source)", "Match the source container type"),
            (
                "RAW_PACK",
                "Raw GFModelPack",
                "Write only the GFModelPack blob (0x00010000)",
            ),
            ("RAW_MODEL", "Raw GFModel", "Write only the GFModel blob (0x15122117)"),
            ("CM", "CM Only (No CP)", "Write only the CM container (SPICA-friendly)"),
            ("CP", "CP + CM", "Wrap CM inside a CP container (game-like)"),
        ],
        default="AUTO",
    )
    patch_into_source_archive: BoolProperty(
        name="Patch Back Into Source Archive",
        default=False,
        description="If the last import came from GFModel Archive (any depth), export and patch the embedded blob back into the archive in one step",
    )
    remember_last_export_settings: BoolProperty(
        name="Remember As Last Export",
        default=True,
        options={"HIDDEN"},
        description="Internal: whether to update gfmodel_last_export_* scene settings",
    )
    grow_buffers_routing_strategy: EnumProperty(
        name="Grow Buffers Routing Strategy",
        items=[
            (
                "MOST_SPECIFIC",
                "Most Specific Palette",
                "Prefer the smallest matching bone palette",
            ),
            (
                "ORIGINAL_ORDER",
                "Original Slot Order",
                "Use the first matching submesh slot in file order",
            ),
            ("BALANCE", "Balance", "Spread triangles across matching slots"),
        ],
        default="MOST_SPECIFIC",
        options={"HIDDEN"},
    )
    grow_buffers_material_sources_json: StringProperty(
        name="Grow Buffers Material Sources",
        default="",
        options={"HIDDEN"},
    )
    grow_buffers_weight_cutoff: FloatProperty(
        name="Grow Buffers Weight Cutoff",
        default=0.01,
        min=0.0,
        max=1.0,
        options={"HIDDEN"},
    )
    grow_buffers_uv_strategy: EnumProperty(
        name="Grow Buffers UV Strategy",
        items=[
            (
                "DUPLICATE",
                "Duplicate Verts (Accurate)",
                "Preserve UV seams by duplicating vertices where needed (GPU-style; most accurate)",
            ),
            (
                "SMEAR",
                "Smear (Low Verts)",
                "Force one UV per 3D vertex (collapses seams; can smear textures; reduces vertex duplication)",
            ),
            (
                "STITCH_TRANSLATE",
                "Stitch Translate (Try)",
                "Try to remove seams by translating UV islands to align across seams (translation-only heuristic)",
            ),
        ],
        default="DUPLICATE",
        options={"HIDDEN"},
    )
    grow_buffers_rebuild_mode: EnumProperty(
        name="Grow Buffers Rebuild Mode",
        items=[
            (
                "CLAMP_ROUTE",
                "Clamp/Route (No Rebuild)",
                "Route triangles to existing submesh slots/palettes; clamp or drop when they don't fit",
            ),
            (
                "REBUILD_PALETTE",
                "Rebuild Palette Only",
                "Rebuild each affected submesh's bone palette (up to 31) and remap weights, but do not split/add submeshes (errors if >31 needed)",
            ),
            (
                "REBUILD_SPLIT",
                "Rebuild + Split",
                "Rebuild palettes and split triangles into new parts/submeshes when needed to stay within 31 bones",
            ),
        ],
        default="CLAMP_ROUTE",
        options={"HIDDEN"},
    )
    grow_buffers_clamp_conflict_mode: EnumProperty(
        name="Grow Buffers Clamp/Route Conflicts",
        items=[
            (
                "CLAMP_BY_WEIGHT",
                "Clamp (By Weight)",
                "When a triangle spans palettes, pick the best slot by weight and trim weights to that palette",
            ),
            (
                "CLAMP_BY_NEIGHBORS",
                "Clamp (Prefer Neighbors)",
                "When a triangle spans palettes, prefer the same slot as adjacent triangles, then trim weights",
            ),
            (
                "DROP_CONFLICTS",
                "Drop Conflicts",
                "Drop triangles that don't fit any single existing palette",
            ),
        ],
        default="CLAMP_BY_WEIGHT",
        options={"HIDDEN"},
    )

    @staticmethod
    def _build_export_loop_stream(
        obj: bpy.types.Object,
        sm: _GFSubMesh,
        *,
        gf_from_blender: Matrix,
        global_scale: float,
        skeleton_names: List[str],
    ) -> Tuple[
        List[Vector],
        bytes,
        List[int],
        int,
    ]:
        mesh: bpy.types.Mesh = obj.data                            
        mesh.calc_loop_triangles()
                                                                                       
                                                                                       
                                                         
        try:
            if hasattr(mesh, "calc_normals_split"):
                mesh.calc_normals_split()                              
            elif hasattr(mesh, "calc_normals"):
                mesh.calc_normals()                              
            else:
                                                                                  
                try:
                    mesh.update()                              
                except Exception:
                    pass
        except Exception:
            pass

        uv_layer = None
        if mesh.uv_layers:
            uv_layer = mesh.uv_layers.active or mesh.uv_layers[0]

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
        has_dynamic_skin = any(int(a.name) in (7, 8) for a in (sm.attributes or []))

        positions: List[Vector] = []
        packed_vertices: List[bytes] = []
        indices: List[int] = []
        remap: Dict[Tuple[int, Tuple[int, int]], int] = {}
        max_weights = 0

                                                                
        uv_first: List[Optional[Tuple[float, float]]] = [None] * len(mesh.vertices)
        uv_split: List[bool] = [False] * len(mesh.vertices)

        def uv_q(uv: Tuple[float, float]) -> Tuple[int, int]:
                                                
            return (int(round(float(uv[0]) * 1e6)), int(round(float(uv[1]) * 1e6)))

        for tri in mesh.loop_triangles:
            for li in tri.loops:
                vi = mesh.loops[li].vertex_index
                if uv_layer is not None:
                    uv = uv_layer.data[li].uv
                    uv_t = (float(uv.x), float(uv.y))
                    prev = uv_first[vi]
                    if prev is None:
                        uv_first[vi] = uv_t
                    else:
                        if uv_q(prev) != uv_q(uv_t):
                            uv_split[vi] = True

        for tri in mesh.loop_triangles:
            for li in tri.loops:
                vi = int(mesh.loops[li].vertex_index)
                co = gf_from_blender @ (Vector(mesh.vertices[vi].co) / global_scale)
                no = Vector(mesh.vertices[vi].normal)
                no = (gf_from_blender.to_3x3() @ no).normalized()

                if uv_layer is not None:
                    uv = uv_layer.data[li].uv
                    loop_uv = (float(uv.x), float(uv.y))
                else:
                    loop_uv = (0.0, 0.0)
                use_uv = loop_uv if uv_split[vi] else (uv_first[vi] or loop_uv)
                uv_key = uv_q(use_uv) if uv_split[vi] else (0, 0)

                wl = weights_by_v[vi] if vi < len(weights_by_v) else []
                if has_dynamic_skin:
                    s = float(sum(max(0.0, float(w)) for _pi, w in wl))
                    if s <= 0.0:
                        wl = [(0, 1.0)]
                max_weights = max(max_weights, len(wl))
                key = (vi, uv_key)
                idx = remap.get(key)
                if idx is None:
                    idx = len(packed_vertices)
                    remap[key] = idx
                    pv = _pack_vertex_bytes(
                        sm,
                        position=co,
                        normal=no,
                        uv0=use_uv,
                        weights=wl,
                    )
                    positions.append(co)
                    packed_vertices.append(pv)
                indices.append(int(idx))

                                                                                               
                                                                                  
        if max_weights == 0:
            try:
                if any(int(fa.name) in (7, 8) for fa in (sm.fixed_attributes or [])):
                    max_weights = 1
            except Exception:
                pass

        return positions, b"".join(packed_vertices), indices, int(max_weights)

    def execute(self, context: bpy.types.Context):
        return export_scene_execute(self, context)
