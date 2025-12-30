"""Addon coordinator.

`__init__.py` exposes `register`/`unregister` from this module.
This file stays intentionally small; implementation lives in submodules.
"""

from __future__ import annotations

from .blender import anim, archive, dump, exporter, importer, patch_ui


def register() -> None:
    importer.register()
    anim.register()
    archive.register()
    exporter.register()
    patch_ui.register()
    dump.register()


def unregister() -> None:
    dump.unregister()
    patch_ui.unregister()
    exporter.unregister()
    archive.unregister()
    anim.unregister()
    importer.unregister()
