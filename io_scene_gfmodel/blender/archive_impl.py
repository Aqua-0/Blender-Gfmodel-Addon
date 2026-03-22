
from __future__ import annotations

import bpy
from bpy.props import (
    BoolProperty,
    CollectionProperty,
    EnumProperty,
    IntProperty,
    StringProperty,
)

from .archive_parts.types_ui import (
    GFModelArchiveEntry,
    GFModelMiniEntry,
    GFModelContainerEntry,
    GFModel_UL_archive_entries,
    GFModel_UL_mini_entries,
    GFModel_UL_container_entries,
)
from .archive_parts.ops_archive import (
    GFModel_OT_archive_scan,
    GFModel_OT_archive_import_entry,
    GFModel_OT_archive_patch_entry_from_file,
    GFModel_OT_archive_verify_entry_payload,
)
from .archive_parts.ops_nested import (
    GFModel_OT_archive_patch_mini_from_file,
    GFModel_OT_archive_verify_mini_payload,
    GFModel_OT_archive_patch_container_from_file,
    GFModel_OT_archive_verify_container_payload,
    GFModel_OT_archive_patch_container2_from_file,
    GFModel_OT_archive_verify_container2_payload,
)
from .archive_parts.ops_mini import (
    GFModel_OT_archive_scan_mini,
    GFModel_OT_archive_import_mini,
    GFModel_OT_archive_scan_container,
    GFModel_OT_archive_import_container_entry,
    GFModel_OT_archive_scan_container2,
    GFModel_OT_archive_import_container2_entry,
    GFModel_OT_archive_import_container2_pair01,
)
from .archive_parts.panel import VIEW3D_PT_gfmodel_archive


classes = (
    GFModelArchiveEntry,
    GFModelMiniEntry,
    GFModelContainerEntry,
    GFModel_UL_archive_entries,
    GFModel_UL_mini_entries,
    GFModel_UL_container_entries,
    GFModel_OT_archive_scan,
    GFModel_OT_archive_import_entry,
    GFModel_OT_archive_patch_entry_from_file,
    GFModel_OT_archive_verify_entry_payload,
    GFModel_OT_archive_patch_mini_from_file,
    GFModel_OT_archive_verify_mini_payload,
    GFModel_OT_archive_patch_container_from_file,
    GFModel_OT_archive_verify_container_payload,
    GFModel_OT_archive_patch_container2_from_file,
    GFModel_OT_archive_verify_container2_payload,
    GFModel_OT_archive_scan_mini,
    GFModel_OT_archive_import_mini,
    GFModel_OT_archive_scan_container,
    GFModel_OT_archive_import_container_entry,
    GFModel_OT_archive_scan_container2,
    GFModel_OT_archive_import_container2_entry,
    GFModel_OT_archive_import_container2_pair01,
    VIEW3D_PT_gfmodel_archive,
)

def register() -> None:
    for c in classes:
        bpy.utils.register_class(c)

    bpy.types.Scene.gfmodel_archive_path = StringProperty(
        name="GARC/CRAG Archive Path",
        default="",
        subtype="FILE_PATH",
    )
    bpy.types.Scene.gfmodel_archive_auto_resolve_a094_group = BoolProperty(
        name="Auto a094 textures/anims",
        default=True,
        description="When importing from romfs/a/0/9/4 (a094), scan the 9-member group to auto-attach textures and motions",
    )
    bpy.types.Scene.gfmodel_a094_motion_pack = EnumProperty(
        name="a094 Motion Pack",
        items=(
            ("BATTLE", "Battle", "Use battle motion ordering (BT)"),
            ("KAWAIGARI", "Kawaigari", "Use kawai (refresh) motion ordering (KW)"),
            ("FIELD", "Field", "Use field/walk motion ordering (FI)"),
            ("POKE_FINDER", "Poke Finder", "Use Poke Finder combined ordering (PF)"),
            ("ALL", "All (BT+KW+FI)", "Import battle, kawai, and field packs"),
        ),
        default="ALL",
    )
    bpy.types.Scene.gfmodel_a094_name_motions = BoolProperty(
        name="Name a094 motions",
        default=True,
        description="When auto-resolving a094 motion packs, name actions using the game's standard slot ordering",
    )

    bpy.types.Scene.gfmodel_archive_patch_payload_path = StringProperty(
        name="Payload Path",
        default="",
        subtype="FILE_PATH",
        description="Raw bytes to inject into the selected archive entry (bit 0 by default)",
    )
    bpy.types.Scene.gfmodel_mini_patch_payload_path = StringProperty(
        name="Mini Payload Path",
        default="",
        subtype="FILE_PATH",
        description="Raw bytes to inject into the selected Mini subfile",
    )
    bpy.types.Scene.gfmodel_container_patch_payload_path = StringProperty(
        name="Container Payload Path",
        default="",
        subtype="FILE_PATH",
        description="Raw bytes to inject into the selected CP/CM container entry",
    )
    bpy.types.Scene.gfmodel_container2_patch_payload_path = StringProperty(
        name="Nested Payload Path",
        default="",
        subtype="FILE_PATH",
        description="Raw bytes to inject into the selected nested CP/CM container entry",
    )
    bpy.types.Scene.gfmodel_archive_patch_output_path = StringProperty(
        name="Output Archive Path",
        default="",
        subtype="FILE_PATH",
        description="Where to write the patched archive (empty => '<archive>.patched')",
    )
    bpy.types.Scene.gfmodel_archive_patch_bit = IntProperty(
        name="Bit",
        default=0,
        min=0,
        max=31,
        description="Subentry bit to patch (0 is the primary payload in most GARCs)",
    )
    bpy.types.Scene.gfmodel_archive_patch_inplace = BoolProperty(
        name="In-Place",
        default=False,
        description="Replace the archive file directly (requires Backup)",
    )
    bpy.types.Scene.gfmodel_archive_patch_backup = BoolProperty(
        name="Backup",
        default=True,
        description="When patching in-place, rename the original to '<archive>.bak' first",
    )
    bpy.types.Scene.gfmodel_archive_entries = CollectionProperty(
        type=GFModelArchiveEntry
    )
    bpy.types.Scene.gfmodel_archive_selected = IntProperty(
        name="Selected Entry",
        default=0,
        min=0,
    )
    bpy.types.Scene.gfmodel_archive_search = StringProperty(
        name="Find Entry",
        default="",
        description="Filter the archive list by entry index (decimal or 0x... hex)",
    )
    bpy.types.Scene.gfmodel_archive_show_species_names = BoolProperty(
        name="Names",
        default=False,
        description="Show species labels in the archive entry list (a094 only)",
    )
    bpy.types.Scene.gfmodel_mini_entries = CollectionProperty(
        type=GFModelMiniEntry
    )
    bpy.types.Scene.gfmodel_mini_selected = IntProperty(
        name="Selected Mini File",
        default=0,
        min=0,
    )
    bpy.types.Scene.gfmodel_mini_search = StringProperty(
        name="Find Mini",
        default="",
        description="Filter the mini list by subfile index (decimal or 0x... hex)",
    )
    bpy.types.Scene.gfmodel_mini_filter = EnumProperty(
        name="Mini Folder",
        items=[
            ("ALL", "All", ""),
            ("MODEL", "Model", ""),
            ("MOTION", "Motion", ""),
            ("TEXTURE", "Texture", ""),
            ("CONTAINER", "Container", ""),
            ("MINI", "Mini", ""),
            ("LZ11", "LZ11", ""),
            ("OTHER", "Other", ""),
        ],
        default="ALL",
    )
    bpy.types.Scene.gfmodel_container_entries = CollectionProperty(
        type=GFModelContainerEntry
    )
    bpy.types.Scene.gfmodel_container_selected = IntProperty(
        name="Selected Container Entry",
        default=0,
        min=0,
    )
    bpy.types.Scene.gfmodel_container_search = StringProperty(
        name="Find Container",
        default="",
        description="Filter the container list by entry index (decimal or 0x... hex)",
    )
    bpy.types.Scene.gfmodel_container2_entries = CollectionProperty(
        type=GFModelContainerEntry
    )
    bpy.types.Scene.gfmodel_container2_selected = IntProperty(
        name="Selected Nested Container Entry",
        default=0,
        min=0,
    )
    bpy.types.Scene.gfmodel_container2_search = StringProperty(
        name="Find Nested",
        default="",
        description="Filter the nested container list by entry index (decimal or 0x... hex)",
    )


def unregister() -> None:
    if hasattr(bpy.types.Scene, "gfmodel_mini_filter"):
        del bpy.types.Scene.gfmodel_mini_filter
    if hasattr(bpy.types.Scene, "gfmodel_container2_search"):
        del bpy.types.Scene.gfmodel_container2_search
    if hasattr(bpy.types.Scene, "gfmodel_container2_selected"):
        del bpy.types.Scene.gfmodel_container2_selected
    if hasattr(bpy.types.Scene, "gfmodel_container2_entries"):
        del bpy.types.Scene.gfmodel_container2_entries
    if hasattr(bpy.types.Scene, "gfmodel_container_search"):
        del bpy.types.Scene.gfmodel_container_search
    if hasattr(bpy.types.Scene, "gfmodel_container_selected"):
        del bpy.types.Scene.gfmodel_container_selected
    if hasattr(bpy.types.Scene, "gfmodel_container_entries"):
        del bpy.types.Scene.gfmodel_container_entries
    if hasattr(bpy.types.Scene, "gfmodel_mini_search"):
        del bpy.types.Scene.gfmodel_mini_search
    if hasattr(bpy.types.Scene, "gfmodel_mini_selected"):
        del bpy.types.Scene.gfmodel_mini_selected
    if hasattr(bpy.types.Scene, "gfmodel_mini_entries"):
        del bpy.types.Scene.gfmodel_mini_entries
    if hasattr(bpy.types.Scene, "gfmodel_archive_search"):
        del bpy.types.Scene.gfmodel_archive_search
    if hasattr(bpy.types.Scene, "gfmodel_archive_selected"):
        del bpy.types.Scene.gfmodel_archive_selected
    if hasattr(bpy.types.Scene, "gfmodel_archive_entries"):
        del bpy.types.Scene.gfmodel_archive_entries
    if hasattr(bpy.types.Scene, "gfmodel_archive_auto_resolve_a094_group"):
        del bpy.types.Scene.gfmodel_archive_auto_resolve_a094_group
    if hasattr(bpy.types.Scene, "gfmodel_archive_show_species_names"):
        del bpy.types.Scene.gfmodel_archive_show_species_names
    if hasattr(bpy.types.Scene, "gfmodel_a094_motion_pack"):
        del bpy.types.Scene.gfmodel_a094_motion_pack
    if hasattr(bpy.types.Scene, "gfmodel_a094_name_motions"):
        del bpy.types.Scene.gfmodel_a094_name_motions
    if hasattr(bpy.types.Scene, "gfmodel_archive_path"):
        del bpy.types.Scene.gfmodel_archive_path
    if hasattr(bpy.types.Scene, "gfmodel_archive_patch_backup"):
        del bpy.types.Scene.gfmodel_archive_patch_backup
    if hasattr(bpy.types.Scene, "gfmodel_archive_patch_inplace"):
        del bpy.types.Scene.gfmodel_archive_patch_inplace
    if hasattr(bpy.types.Scene, "gfmodel_archive_patch_bit"):
        del bpy.types.Scene.gfmodel_archive_patch_bit
    if hasattr(bpy.types.Scene, "gfmodel_archive_patch_output_path"):
        del bpy.types.Scene.gfmodel_archive_patch_output_path
    if hasattr(bpy.types.Scene, "gfmodel_archive_patch_payload_path"):
        del bpy.types.Scene.gfmodel_archive_patch_payload_path
    if hasattr(bpy.types.Scene, "gfmodel_mini_patch_payload_path"):
        del bpy.types.Scene.gfmodel_mini_patch_payload_path
    if hasattr(bpy.types.Scene, "gfmodel_container_patch_payload_path"):
        del bpy.types.Scene.gfmodel_container_patch_payload_path
    if hasattr(bpy.types.Scene, "gfmodel_container2_patch_payload_path"):
        del bpy.types.Scene.gfmodel_container2_patch_payload_path

    for c in reversed(classes):
        bpy.utils.unregister_class(c)
