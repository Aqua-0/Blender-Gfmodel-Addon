"""Blender UI: browse pk3DS-style 'Mini' containers (legacy panel)."""

from __future__ import annotations

import os

import bpy
from bpy.props import CollectionProperty, IntProperty, StringProperty

from ..core.mini import parse_mini
from .importer import _import_gfmodel_bytes


class GFModelMiniFileEntry(bpy.types.PropertyGroup):
    index: bpy.props.IntProperty()                            
    start_hex: bpy.props.StringProperty()                            
    end_hex: bpy.props.StringProperty()                            
    length_hex: bpy.props.StringProperty()                            
    magic4: bpy.props.StringProperty()                            
    mini_info: bpy.props.StringProperty()                            


class GFModel_UL_mini_file_entries(bpy.types.UIList):
    def draw_item(
        self,
        context: bpy.types.Context,
        layout: bpy.types.UILayout,
        data: object,
        item: object,
        icon: int,
        active_data: object,
        active_propname: str,
        index: int,
    ) -> None:
        e: GFModelMiniFileEntry = item                            
        row = layout.row(align=True)
        row.label(text=f"{int(e.index):04d}")
        row.label(text=str(getattr(e, "magic4", "")))
        row.label(text=str(getattr(e, "length_hex", "")))
        mi = str(getattr(e, "mini_info", ""))
        if mi:
            row.label(text=mi)


def _guess_magic4(b: bytes) -> str:
    if len(b) < 4:
        return ""
    s = b[:4]
    try:
        return s.decode("ascii")
    except Exception:
        return "".join(chr(x) if 32 <= x < 127 else "." for x in s)


class GFModel_OT_mini_scan(bpy.types.Operator):
    bl_idname = "gfmodel.mini_scan"
    bl_label = "GFModel: Scan Mini"
    bl_options = {"UNDO"}

    def execute(self, context: bpy.types.Context):
        path = str(getattr(context.scene, "gfmodel_mini_path", "")).strip()
        if not path:
            self.report({"ERROR"}, "Set a mini path first")
            return {"CANCELLED"}
        if not os.path.exists(path) or not os.path.isfile(path):
            self.report({"ERROR"}, f"File not found: {path}")
            return {"CANCELLED"}
        try:
            blob = open(path, "rb").read()
        except Exception as e:
            self.report({"ERROR"}, f"Failed to read file: {e}")
            return {"CANCELLED"}

        try:
            mini = parse_mini(blob)
        except Exception as e:
            self.report({"ERROR"}, f"Not a Mini container: {e}")
            return {"CANCELLED"}

        context.scene["gfmodel_mini_ident"] = str(mini.ident)
        context.scene["gfmodel_mini_count"] = int(mini.count)

        entries = context.scene.gfmodel_mini_entries
        entries.clear()
        for i in range(mini.count):
            sub = mini.extract(blob, i)
            it = entries.add()
            it.index = int(i)
            it.start_hex = f"0x{int(mini.offsets[i]):X}"
            it.end_hex = f"0x{int(mini.offsets[i + 1]):X}"
            it.length_hex = f"0x{len(sub):X}"
            it.magic4 = _guess_magic4(sub[:4])
            try:
                m2 = parse_mini(sub)
                it.mini_info = f"mini={m2.ident}({m2.count})"
            except Exception:
                it.mini_info = ""

        context.scene.gfmodel_mini_selected = min(
            int(context.scene.gfmodel_mini_selected), max(0, len(entries) - 1)
        )
        context.scene.gfmodel_mini2_entries.clear()
        return {"FINISHED"}


class GFModel_OT_mini_import_selected(bpy.types.Operator):
    bl_idname = "gfmodel.mini_import_selected"
    bl_label = "GFModel: Import Selected Mini File"
    bl_options = {"UNDO"}

    def execute(self, context: bpy.types.Context):
        path = str(getattr(context.scene, "gfmodel_mini_path", "")).strip()
        if not path:
            self.report({"ERROR"}, "Set a mini path first")
            return {"CANCELLED"}
        if not os.path.exists(path) or not os.path.isfile(path):
            self.report({"ERROR"}, f"File not found: {path}")
            return {"CANCELLED"}

        try:
            blob = open(path, "rb").read()
        except Exception as e:
            self.report({"ERROR"}, f"Failed to read file: {e}")
            return {"CANCELLED"}

        try:
            mini = parse_mini(blob)
        except Exception as e:
            self.report({"ERROR"}, f"Not a Mini container: {e}")
            return {"CANCELLED"}

        sel = int(getattr(context.scene, "gfmodel_mini_selected", 0))
        if sel < 0 or sel >= mini.count:
            self.report({"ERROR"}, f"Selection out of range: {sel}")
            return {"CANCELLED"}

        sub = mini.extract(blob, sel)
        ok = _import_gfmodel_bytes(
            context,
            sub,
            source_path=f"{path}#mini{mini.ident}[{sel}]",
            import_textures=True,
            import_animations=True,
        )
        if not ok:
            self.report({"ERROR"}, "No GFModel content found in selected mini file")
            return {"CANCELLED"}
        return {"FINISHED"}


class GFModel_OT_mini_scan_nested(bpy.types.Operator):
    bl_idname = "gfmodel.mini_scan_nested"
    bl_label = "GFModel: Scan Nested Mini"
    bl_options = {"UNDO"}

    def execute(self, context: bpy.types.Context):
        path = str(getattr(context.scene, "gfmodel_mini_path", "")).strip()
        if not path:
            self.report({"ERROR"}, "Set a mini path first")
            return {"CANCELLED"}
        if not os.path.exists(path) or not os.path.isfile(path):
            self.report({"ERROR"}, f"File not found: {path}")
            return {"CANCELLED"}

        try:
            blob = open(path, "rb").read()
        except Exception as e:
            self.report({"ERROR"}, f"Failed to read file: {e}")
            return {"CANCELLED"}

        try:
            mini = parse_mini(blob)
        except Exception as e:
            self.report({"ERROR"}, f"Not a Mini container: {e}")
            return {"CANCELLED"}

        sel = int(getattr(context.scene, "gfmodel_mini_selected", 0))
        if sel < 0 or sel >= mini.count:
            self.report({"ERROR"}, f"Selection out of range: {sel}")
            return {"CANCELLED"}

        sub = mini.extract(blob, sel)
        try:
            mini2 = parse_mini(sub)
        except Exception as e:
            self.report({"ERROR"}, f"Selected subfile is not a Mini: {e}")
            return {"CANCELLED"}

        context.scene["gfmodel_mini2_ident"] = str(mini2.ident)
        context.scene["gfmodel_mini2_count"] = int(mini2.count)

        entries = context.scene.gfmodel_mini2_entries
        entries.clear()
        for i in range(mini2.count):
            sub2 = mini2.extract(sub, i)
            it = entries.add()
            it.index = int(i)
            it.start_hex = f"0x{int(mini2.offsets[i]):X}"
            it.end_hex = f"0x{int(mini2.offsets[i + 1]):X}"
            it.length_hex = f"0x{len(sub2):X}"
            it.magic4 = _guess_magic4(sub2[:4])
            try:
                m3 = parse_mini(sub2)
                it.mini_info = f"mini={m3.ident}({m3.count})"
            except Exception:
                it.mini_info = ""

        context.scene.gfmodel_mini2_selected = min(
            int(context.scene.gfmodel_mini2_selected), max(0, len(entries) - 1)
        )
        return {"FINISHED"}


class GFModel_OT_mini_import_nested(bpy.types.Operator):
    bl_idname = "gfmodel.mini_import_nested"
    bl_label = "GFModel: Import Nested Mini Selected"
    bl_options = {"UNDO"}

    def execute(self, context: bpy.types.Context):
        path = str(getattr(context.scene, "gfmodel_mini_path", "")).strip()
        if not path:
            self.report({"ERROR"}, "Set a mini path first")
            return {"CANCELLED"}
        if not os.path.exists(path) or not os.path.isfile(path):
            self.report({"ERROR"}, f"File not found: {path}")
            return {"CANCELLED"}

        try:
            blob = open(path, "rb").read()
        except Exception as e:
            self.report({"ERROR"}, f"Failed to read file: {e}")
            return {"CANCELLED"}

        try:
            mini = parse_mini(blob)
        except Exception as e:
            self.report({"ERROR"}, f"Not a Mini container: {e}")
            return {"CANCELLED"}

        sel = int(getattr(context.scene, "gfmodel_mini_selected", 0))
        if sel < 0 or sel >= mini.count:
            self.report({"ERROR"}, f"Selection out of range: {sel}")
            return {"CANCELLED"}
        sub = mini.extract(blob, sel)

        try:
            mini2 = parse_mini(sub)
        except Exception as e:
            self.report({"ERROR"}, f"Selected subfile is not a Mini: {e}")
            return {"CANCELLED"}

        sel2 = int(getattr(context.scene, "gfmodel_mini2_selected", 0))
        if sel2 < 0 or sel2 >= mini2.count:
            self.report({"ERROR"}, f"Nested selection out of range: {sel2}")
            return {"CANCELLED"}
        sub2 = mini2.extract(sub, sel2)

        ok = _import_gfmodel_bytes(
            context,
            sub2,
            source_path=f"{path}#mini{mini.ident}[{sel}]/mini{mini2.ident}[{sel2}]",
            import_textures=True,
            import_animations=True,
        )
        if not ok:
            self.report(
                {"ERROR"}, "No GFModel content found in selected nested mini file"
            )
            return {"CANCELLED"}
        return {"FINISHED"}


class VIEW3D_PT_gfmodel_mini(bpy.types.Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "GFModel"
    bl_label = "GFModel Mini"

    def draw(self, context: bpy.types.Context):
        layout = self.layout
        layout.prop(context.scene, "gfmodel_mini_path", text="Mini")
        row = layout.row(align=True)
        row.operator("gfmodel.mini_scan", text="Scan")
        row.operator("gfmodel.mini_import_selected", text="Import Selected")

        ident = str(context.scene.get("gfmodel_mini_ident", ""))
        count = int(context.scene.get("gfmodel_mini_count", 0))
        if ident:
            layout.label(text=f"Ident: {ident}  Count: {count}")

        entries = getattr(context.scene, "gfmodel_mini_entries", None)
        if entries is None or len(entries) == 0:
            layout.label(text="(No mini entries scanned)")
            return
        layout.template_list(
            "GFModel_UL_mini_file_entries",
            "",
            context.scene,
            "gfmodel_mini_entries",
            context.scene,
            "gfmodel_mini_selected",
            rows=8,
        )

        layout.separator()
        layout.label(text="Nested mini (selected subfile)")
        row = layout.row(align=True)
        row.operator("gfmodel.mini_scan_nested", text="Scan Nested")
        row.operator("gfmodel.mini_import_nested", text="Import Nested Selected")

        ident2 = str(context.scene.get("gfmodel_mini2_ident", ""))
        count2 = int(context.scene.get("gfmodel_mini2_count", 0))
        if ident2:
            layout.label(text=f"Nested Ident: {ident2}  Count: {count2}")

        entries2 = getattr(context.scene, "gfmodel_mini2_entries", None)
        if entries2 is None or len(entries2) == 0:
            layout.label(text="(No nested mini scanned)")
            return
        layout.template_list(
            "GFModel_UL_mini_file_entries",
            "NESTED",
            context.scene,
            "gfmodel_mini2_entries",
            context.scene,
            "gfmodel_mini2_selected",
            rows=6,
        )


classes = (
    GFModelMiniFileEntry,
    GFModel_UL_mini_file_entries,
    GFModel_OT_mini_scan,
    GFModel_OT_mini_import_selected,
    GFModel_OT_mini_scan_nested,
    GFModel_OT_mini_import_nested,
    VIEW3D_PT_gfmodel_mini,
)


def register() -> None:
    for c in classes:
        bpy.utils.register_class(c)

    bpy.types.Scene.gfmodel_mini_path = StringProperty(                              
        name="Mini Path",
        default="",
        subtype="FILE_PATH",
    )
    bpy.types.Scene.gfmodel_mini_entries = CollectionProperty(                              
        type=GFModelMiniFileEntry
    )
    bpy.types.Scene.gfmodel_mini_selected = IntProperty(                              
        name="Selected Mini File",
        default=0,
        min=0,
    )

    bpy.types.Scene.gfmodel_mini2_entries = CollectionProperty(                              
        type=GFModelMiniFileEntry
    )
    bpy.types.Scene.gfmodel_mini2_selected = IntProperty(                              
        name="Selected Nested Mini File",
        default=0,
        min=0,
    )


def unregister() -> None:
    if hasattr(bpy.types.Scene, "gfmodel_mini2_selected"):
        del bpy.types.Scene.gfmodel_mini2_selected
    if hasattr(bpy.types.Scene, "gfmodel_mini2_entries"):
        del bpy.types.Scene.gfmodel_mini2_entries
    if hasattr(bpy.types.Scene, "gfmodel_mini_selected"):
        del bpy.types.Scene.gfmodel_mini_selected
    if hasattr(bpy.types.Scene, "gfmodel_mini_entries"):
        del bpy.types.Scene.gfmodel_mini_entries
    if hasattr(bpy.types.Scene, "gfmodel_mini_path"):
        del bpy.types.Scene.gfmodel_mini_path

    for c in reversed(classes):
        bpy.utils.unregister_class(c)
