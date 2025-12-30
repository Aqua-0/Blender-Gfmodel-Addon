"""Helpers to unwrap/rewrap GFModelPack inside simple CP/CM containers.

Used by in-place patch modes (including Grow Buffers) so they work when the imported
source blob is `CM` or `CP->CM`, not only a raw GFModelPack.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Optional

from ...core.pkmn_container import parse_container, patch_container


def _is_pack(data: bytes) -> bool:
    return len(data) >= 4 and struct.unpack_from("<I", data, 0)[0] == 0x00010000


def _find_pack_index_in_cm(cm_bytes: bytes) -> int:
    cont = parse_container(cm_bytes)
    if cont.magic2 != "CM":
        raise ValueError("expected CM container")
    for i in range(cont.count):
        ent = cont.extract(cm_bytes, i)
        if _is_pack(ent):
            return int(i)
    raise ValueError("no GFModelPack found in CM container")


@dataclass(frozen=True)
class PackWrapper:
    kind: str               
    cm_pack_index: int = -1

                                                  
    cm_bytes: bytes = b""

                                                                                                
    cp_bytes: bytes = b""


def unwrap_pack(src_data: bytes) -> tuple[bytes, PackWrapper]:
    """Return (pack_bytes, wrapper) for RAW_PACK / CM / CP->CM sources."""
    if _is_pack(src_data):
        return bytes(src_data), PackWrapper(kind="RAW")

    top = parse_container(src_data)
    if top.magic2 == "CM":
        pack_i = _find_pack_index_in_cm(src_data)
        pack_bytes = parse_container(src_data).extract(src_data, int(pack_i))
        return bytes(pack_bytes), PackWrapper(
            kind="CM", cm_pack_index=int(pack_i), cm_bytes=bytes(src_data)
        )

    if top.magic2 == "CP":
        if top.count < 2:
            raise ValueError("malformed CP container (missing CM entry)")
        cm_bytes = top.extract(src_data, 1)
        cm = parse_container(cm_bytes)
        if cm.magic2 != "CM":
            raise ValueError("CP slot 1 is not a CM container")
        pack_i = _find_pack_index_in_cm(cm_bytes)
        pack_bytes = cm.extract(cm_bytes, int(pack_i))
        return bytes(pack_bytes), PackWrapper(
            kind="CP",
            cm_pack_index=int(pack_i),
            cm_bytes=bytes(cm_bytes),
            cp_bytes=bytes(src_data),
        )

    raise ValueError("expected RAW_PACK, CM, or CP->CM source")


def rewrap_pack(wrapper: PackWrapper, new_pack: bytes) -> bytes:
    """Return a new blob with `new_pack` inserted back into its original wrapper."""
    if wrapper.kind == "RAW":
        return bytes(new_pack)
    if wrapper.kind == "CM":
        if wrapper.cm_pack_index < 0:
            raise ValueError("invalid CM wrapper")
        return patch_container(
            wrapper.cm_bytes,
            index=int(wrapper.cm_pack_index),
            replacement=bytes(new_pack),
        )
    if wrapper.kind == "CP":
        if wrapper.cm_pack_index < 0:
            raise ValueError("invalid CP wrapper")
        cm_new = patch_container(
            wrapper.cm_bytes,
            index=int(wrapper.cm_pack_index),
            replacement=bytes(new_pack),
        )
        return patch_container(wrapper.cp_bytes, index=1, replacement=bytes(cm_new))
    raise ValueError(f"unknown wrapper kind: {wrapper.kind!r}")
