

from __future__ import annotations

from .blender import anim, archive, dump, exporter, importer, motion_patch, patch_ui


def register() -> None:
    importer.register()
    anim.register()
    archive.register()
    exporter.register()
    patch_ui.register()
    motion_patch.register()
    dump.register()


def unregister() -> None:
    dump.unregister()
    motion_patch.unregister()
    patch_ui.unregister()
    exporter.unregister()
    archive.unregister()
    anim.unregister()
    importer.unregister()
