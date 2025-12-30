"""Patch helpers for nested archive formats (GARC -> LZ11 -> Mini -> CP/CM -> nested CP/CM).

This module is Blender-independent. The Blender UI/operator layer should supply:
- archive path + entry index
- mini index / container index / nested index
- replacement leaf bytes
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .garc import parse_garc_file, rewrite_garc_file
from .lz11 import compress, decompress, looks_like_lz11
from .mini import parse_mini, patch_mini
from .pkmn_container import parse_container, patch_container


@dataclass(frozen=True)
class LeafContext:
    archive_path: str
    entry_index: int
    mini_index: int
    container_index: int
    nested_index: int


def patch_entry_leaf_bytes(
    entry_bytes: bytes,
    *,
    mini_index: int,
    container_index: int,
    nested_index: int,
    replacement_leaf_bytes: bytes,
) -> bytes:
    """Return new entry bytes after replacing a nested leaf (auto (de)compresses)."""
    entry_was_lz11 = looks_like_lz11(entry_bytes)
    entry_dec = decompress(entry_bytes) if entry_was_lz11 else entry_bytes

    mini = parse_mini(entry_dec)
    if mini_index < 0 or mini_index >= mini.count:
        raise IndexError("mini index out of range")
    mini_seg_raw = mini.extract(entry_dec, int(mini_index))

    mini_seg_was_lz11 = looks_like_lz11(mini_seg_raw)
    mini_seg_dec = decompress(mini_seg_raw) if mini_seg_was_lz11 else mini_seg_raw

    cont = parse_container(mini_seg_dec)
    if container_index < 0 or container_index >= cont.count:
        raise IndexError("container index out of range")
    outer_raw = cont.extract(mini_seg_dec, int(container_index))

    outer_was_lz11 = looks_like_lz11(outer_raw)
    outer_dec = decompress(outer_raw) if outer_was_lz11 else outer_raw

    cont2 = parse_container(outer_dec)
    if nested_index < 0 or nested_index >= cont2.count:
        raise IndexError("nested index out of range")
    leaf_raw = cont2.extract(outer_dec, int(nested_index))

    leaf_was_lz11 = looks_like_lz11(leaf_raw)
                                                                                                
                                                                                          
                                                                             
    if leaf_was_lz11:
        if looks_like_lz11(replacement_leaf_bytes):
            leaf_new_raw = bytes(replacement_leaf_bytes)
        else:
                                                                                                 
                                                                                                
            try:
                if bytes(decompress(leaf_raw)) == bytes(replacement_leaf_bytes):
                    leaf_new_raw = bytes(leaf_raw)
                else:
                    leaf_new_raw = compress(replacement_leaf_bytes)
            except Exception:
                leaf_new_raw = compress(replacement_leaf_bytes)
    else:
        if looks_like_lz11(replacement_leaf_bytes):
            raise ValueError(
                "Replacement leaf looks like LZ11 but target leaf is not LZ11-compressed"
            )
        leaf_new_raw = bytes(replacement_leaf_bytes)

    outer_dec_new = patch_container(
        outer_dec, index=int(nested_index), replacement=leaf_new_raw
    )
    if outer_dec_new == outer_dec:
        outer_new_raw = outer_raw
    else:
        outer_new_raw = compress(outer_dec_new) if outer_was_lz11 else outer_dec_new

    mini_seg_dec_new = patch_container(
        mini_seg_dec, index=int(container_index), replacement=outer_new_raw
    )
    if mini_seg_dec_new == mini_seg_dec:
        mini_seg_new_raw = mini_seg_raw
    else:
        mini_seg_new_raw = (
            compress(mini_seg_dec_new) if mini_seg_was_lz11 else mini_seg_dec_new
        )

    entry_dec_new = patch_mini(
        entry_dec, index=int(mini_index), replacement=mini_seg_new_raw
    )
    if entry_dec_new == entry_dec:
        return entry_bytes
    return compress(entry_dec_new) if entry_was_lz11 else entry_dec_new


def patch_archive_leaf_file(
    ctx: LeafContext,
    *,
    replacement_leaf_bytes: bytes,
    dst_path: str,
    bit: int = 0,
) -> None:
    """Patch one leaf into a GARC file and write `dst_path`."""
    garc = parse_garc_file(ctx.archive_path)
    entry = garc.read_primary_bytes(int(ctx.entry_index))
    new_entry = patch_entry_leaf_bytes(
        entry,
        mini_index=int(ctx.mini_index),
        container_index=int(ctx.container_index),
        nested_index=int(ctx.nested_index),
        replacement_leaf_bytes=replacement_leaf_bytes,
    )
                                                                                         
                                                                                            
    if bytes(new_entry) == bytes(entry):
        import shutil

        if str(ctx.archive_path) == str(dst_path):
            return
        shutil.copyfile(str(ctx.archive_path), str(dst_path))
        return
    rewrite_garc_file(
        ctx.archive_path,
        str(dst_path),
        replacements={(int(ctx.entry_index), int(bit)): new_entry},
    )
