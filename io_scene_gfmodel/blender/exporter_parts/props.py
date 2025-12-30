"""Blender PropertyGroup definitions used by exporter UI."""

from __future__ import annotations

import bpy
from bpy.props import PointerProperty, StringProperty


class GFModel_GrowBuffersMaterialSource(bpy.types.PropertyGroup):
    material_name: StringProperty(name="Material", default="")
    source_object: PointerProperty(name="Source", type=bpy.types.Object)


