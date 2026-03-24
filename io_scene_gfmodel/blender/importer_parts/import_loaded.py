

from __future__ import annotations

import os
from typing import List

import bpy

from ...core.types import _GFModel, _GFTexture, _GFMotion, _GFShader
from .import_core import _import_model_to_blender
from ..anim import _apply_uv_anim_enable

def _import_gfmodel_loaded(
    context: bpy.types.Context,
    *,
    models: List[_GFModel],
    textures: List[_GFTexture],
    motions: List[_GFMotion],
    shaders: List[_GFShader],
    source_path: str,
    import_textures: bool,
    import_animations: bool,
    import_material_animations: bool = True,
    import_visibility_animations: bool = True,
    global_scale: float = 1.0,
    axis_forward: str = "-Z",
    axis_up: str = "Y",
) -> bool:

    try:
        sp = str(source_path)
        if os.path.isfile(sp):
            context.scene["gfmodel_last_import_path"] = sp

            bc = str(context.scene.get("gfmodel_last_import_breadcrumb", "")).strip()
            if not bc:
                bc = sp
            context.scene["gfmodel_last_import_source"] = bc
            context.scene["gfmodel_last_import_breadcrumb"] = bc
        else:
            context.scene["gfmodel_last_import_source"] = sp
            context.scene["gfmodel_last_import_breadcrumb"] = sp
        context.scene["gfmodel_last_axis_forward"] = str(axis_forward)
        context.scene["gfmodel_last_axis_up"] = str(axis_up)
        context.scene["gfmodel_last_global_scale"] = float(global_scale)
    except Exception:
        pass

    print(
        f"[GFModel] Loaded: models={len(models)} textures={len(textures)} motions={len(motions)} shaders={len(shaders)}"
    )
    if motions:
        for mot in motions[:10]:
            print(
                f"[GFModel] Motion {mot.index}: frames={mot.frames_count} bones={len(mot.bones)} uv={len(mot.uv_transforms)} vis={len(mot.visibility_tracks)}"
            )

    if not models:
        return False

    for i, model in enumerate(models):
        _import_model_to_blender(
            context,
            model,
            textures,
            motions if i == 0 else [],
            shaders,
            import_textures=bool(import_textures),
            import_animations=bool(import_animations) and i == 0,
            import_material_animations=bool(import_material_animations) and i == 0,
            import_visibility_animations=bool(import_visibility_animations) and i == 0,
            global_scale=float(global_scale),
            axis_forward=str(axis_forward),
            axis_up=str(axis_up),
        )

    _apply_uv_anim_enable(context.scene)
    return True



