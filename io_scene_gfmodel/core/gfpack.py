"""GFModelPack (0x00010000) reader/writer helpers.

This is the lowest-level container used by many extracted `*.bin` files (ex: `0.bin`).
It is independent of Blender and does not require `mathutils`.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class GFPackEntry:
    section: int
    index: int
    name: str
    addr: int
    end: int
    blob: bytes


@dataclass(frozen=True)
class GFModelPack:
    counts: Tuple[int, int, int, int, int]
    slots: Tuple[Tuple[Optional[GFPackEntry], ...], ...]              

    def get(self, section: int, index: int) -> Optional[GFPackEntry]:
        return self.slots[int(section)][int(index)]


def _u32(b: bytes, o: int) -> int:
    return int(struct.unpack_from("<I", b, o)[0])


def parse_gf_model_pack(data: bytes) -> GFModelPack:
    if len(data) < 4 + 5 * 4:
        raise ValueError("data too small for GFModelPack header")
    magic = _u32(data, 0)
    if magic != 0x00010000:
        raise ValueError("not a GFModelPack (magic!=0x00010000)")

    counts = tuple(_u32(data, 4 + i * 4) for i in range(5))
    ptr_base = 4 + 5 * 4

                                                 
    ptr_tables: List[List[int]] = []
    off = ptr_base
    for sect in range(5):
        c = int(counts[sect])
        ptrs = [
            (_u32(data, off + i * 4) if off + i * 4 + 4 <= len(data) else 0)
            for i in range(c)
        ]
        ptr_tables.append(ptrs)
        off += c * 4

                                                                    
    headers: List[Tuple[int, int, str, int]] = []
    for sect in range(5):
        for idx, ptr in enumerate(ptr_tables[sect]):
            if ptr == 0 or ptr >= len(data):
                continue
            name_len = data[ptr]
            name_start = ptr + 1
            name_end = name_start + int(name_len)
            if name_end + 4 > len(data):
                continue
            name = data[name_start:name_end].decode("ascii", "replace")
            addr = _u32(data, name_end)
            if addr == 0 or addr >= len(data):
                continue
            headers.append((int(sect), int(idx), name, int(addr)))

                                                         
    by_addr = sorted(headers, key=lambda t: t[3])
    addr_to_end: Dict[int, int] = {}
    for i, (_s, _idx, _n, addr) in enumerate(by_addr):
        end = len(data)
        if i + 1 < len(by_addr):
            end = int(by_addr[i + 1][3])
        addr_to_end[int(addr)] = int(end)

                          
    slots: List[List[Optional[GFPackEntry]]] = [
        [None for _ in range(int(counts[s]))] for s in range(5)
    ]
    for sect, idx, name, addr in headers:
        end = addr_to_end.get(int(addr), len(data))
        blob = data[int(addr) : int(end)]
        entry = GFPackEntry(
            section=int(sect),
            index=int(idx),
            name=name,
            addr=int(addr),
            end=int(end),
            blob=blob,
        )
        if 0 <= sect < 5 and 0 <= idx < len(slots[sect]):
            slots[sect][idx] = entry

    return GFModelPack(counts=counts, slots=tuple(tuple(s) for s in slots))


def write_gf_model_pack(
    pack: GFModelPack,
    *,
    replacements: Optional[Dict[Tuple[int, int], bytes]] = None,
    align_blobs: int = 0x80,
) -> bytes:
    """Rewrite a GFModelPack while preserving the original section counts and slot indices.

    This is intended for fast patching workflows (swap one entry blob) and for a
    functional round-trip baseline. It does not attempt to preserve byte-identical
    layout; only the logical structure and payloads.
    """
    replacements = replacements or {}
    counts = [int(c) for c in pack.counts]

    out = bytearray()
    out += struct.pack("<I", 0x00010000)
    out += struct.pack("<5I", *counts)

                                                             
    ptr_tables_off = len(out)
    total_ptrs = sum(counts)
    out += b"\x00" * (total_ptrs * 4)

                                                                                      
    addr_patch_by_slot: Dict[Tuple[int, int], int] = {}
    ptr_cursor = 0
    for sect in range(5):
        for idx in range(counts[sect]):
            entry = pack.get(sect, idx)
            if entry is None:
                ptr_cursor += 1
                continue
            ptr_off = ptr_tables_off + ptr_cursor * 4
            struct.pack_into("<I", out, ptr_off, len(out))
            name_b = (entry.name or "").encode("ascii", "replace")
            name_b = name_b[:0xFF]
            out.append(len(name_b) & 0xFF)
            out += name_b
            addr_patch_by_slot[(sect, idx)] = len(out)
            out += b"\x00\x00\x00\x00"
            ptr_cursor += 1

                                                                        
     
                                                                                            
                                                                                   
    def align(n: int) -> None:
        if n <= 1:
            return
        pad = (-len(out)) % int(n)
        if pad:
            out.extend(b"\x00" * pad)

    for sect in range(5):
        for idx in range(counts[sect]):
            entry = pack.get(sect, idx)
            if entry is None:
                continue
            align(int(align_blobs))
            addr = len(out)
            patch_off = addr_patch_by_slot.get((sect, idx))
            if patch_off is None:
                raise ValueError("internal error: missing addr patch offset")
            struct.pack_into("<I", out, patch_off, int(addr))
            blob = replacements.get((sect, idx), entry.blob)
            out += blob

    return bytes(out)
