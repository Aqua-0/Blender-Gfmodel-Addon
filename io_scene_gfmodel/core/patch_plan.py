"""Generalized "unwrap/rewrap" patch plans for replacing an embedded GFModel blob.

Motivation:
- The Blender exporter naturally writes a GFModelPack/CM/CP blob.
- Archives often wrap those blobs in additional layers (LZ11, Mini, CP/CM nesting, etc).
- When the importer loads a model from somewhere inside a GARC entry, we record a list of
  unwrap steps that lead to an exportable blob; later we can patch a rebuilt blob back
  into the original entry while preserving outer wrappers.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .lz11 import compress, decompress, looks_like_lz11
from .mini import parse_mini, patch_mini
from .pkmn_container import parse_container, patch_container

Step = Dict[str, Any]


def _is_pack(data: bytes) -> bool:
    return len(data) >= 4 and struct.unpack_from("<I", data, 0)[0] == 0x00010000


def _container_has_pack(data: bytes) -> bool:
    try:
        cont = parse_container(data)
    except Exception:
        return False
    for i in range(cont.count):
        try:
            ent = cont.extract(data, i)
        except Exception:
            continue
        if _is_pack(ent):
            return True
    return False


def _is_exportable_blob(data: bytes) -> bool:
    """True if Blender exporter v1 can treat this as a source blob."""
    if _is_pack(data):
        return True
    try:
        cont = parse_container(data)
    except Exception:
        return False
    if cont.magic2 == "CM":
        return _container_has_pack(data)
    if cont.magic2 == "CP":
                                                                                       
        try:
            if cont.count < 2:
                return False
            cm = cont.extract(data, 1)
            cmc = parse_container(cm)
            if cmc.magic2 != "CM":
                return False
            return _container_has_pack(cm)
        except Exception:
            return False
    return False


def steps_to_breadcrumb(steps: Sequence[Step]) -> str:
    parts: List[str] = []
    for st in steps:
        op = str(st.get("op", "")).strip()
        if op == "lz11":
            parts.append("lz11")
        elif op == "mini":
            parts.append(f"mini[{int(st.get('index', -1))}]")
        elif op == "container":
            magic = str(st.get("magic", "")).strip() or "??"
            parts.append(f"{magic}[{int(st.get('index', -1))}]")
        else:
            parts.append(op or "??")
    return "/".join(parts)


def extract_via_steps(data: bytes, steps: Sequence[Step]) -> bytes:
    cur = bytes(data)
    for st in steps:
        op = str(st.get("op", "")).strip()
        if op == "lz11":
            if not looks_like_lz11(cur):
                raise ValueError("patch plan expects LZ11, but data is not LZ11")
            cur = decompress(cur)
            continue
        if op == "mini":
            mini = parse_mini(cur)
            idx = int(st.get("index", -1))
            cur = mini.extract(cur, idx)
            continue
        if op == "container":
            cont = parse_container(cur)
            want_magic = str(st.get("magic", "")).strip()
            if want_magic and want_magic != cont.magic2:
                raise ValueError(
                    f"patch plan container magic mismatch: expected={want_magic} got={cont.magic2}"
                )
            idx = int(st.get("index", -1))
            cur = cont.extract(cur, idx)
            continue
        raise ValueError(f"unknown patch plan step: {op!r}")
    return bytes(cur)


def patch_via_steps(data: bytes, steps: Sequence[Step], replacement: bytes) -> bytes:
    """Return new bytes after replacing the sub-blob addressed by `steps`."""

    def _patch(cur: bytes, i: int) -> bytes:
        if i >= len(steps):
            return bytes(replacement)
        st = steps[i]
        op = str(st.get("op", "")).strip()
        if op == "lz11":
            if not looks_like_lz11(cur):
                raise ValueError("patch plan expects LZ11, but data is not LZ11")
            dec = decompress(cur)
            dec_new = _patch(dec, i + 1)
            if bytes(dec_new) == bytes(dec):
                return bytes(cur)
            return compress(dec_new)
        if op == "mini":
            mini = parse_mini(cur)
            idx = int(st.get("index", -1))
            child = mini.extract(cur, idx)
            child_new = _patch(child, i + 1)
            if bytes(child_new) == bytes(child):
                return bytes(cur)
            return patch_mini(cur, index=idx, replacement=child_new)
        if op == "container":
            cont = parse_container(cur)
            want_magic = str(st.get("magic", "")).strip()
            if want_magic and want_magic != cont.magic2:
                raise ValueError(
                    f"patch plan container magic mismatch: expected={want_magic} got={cont.magic2}"
                )
            idx = int(st.get("index", -1))
            child = cont.extract(cur, idx)
            child_new = _patch(child, i + 1)
            if bytes(child_new) == bytes(child):
                return bytes(cur)
            return patch_container(cur, index=idx, replacement=child_new)
        raise ValueError(f"unknown patch plan step: {op!r}")

    return _patch(bytes(data), 0)


def find_exportable_blob(
    data: bytes,
    *,
    max_depth: int = 32,
) -> Tuple[bytes, List[Step]]:
    """Return (exportable_blob, steps_to_reach_it).

    The returned blob is one of:
    - GFModelPack (raw)
    - CM container containing a GFModelPack
    - CP container whose slot 1 is a CM containing a GFModelPack
    """

    def _walk(cur: bytes, depth: int) -> Optional[Tuple[bytes, List[Step]]]:
        if depth > int(max_depth):
            return None
        if _is_exportable_blob(cur):
            return bytes(cur), []

                       
        if looks_like_lz11(cur):
            try:
                dec = decompress(cur)
            except Exception:
                dec = b""
            if dec:
                res = _walk(dec, depth + 1)
                if res is not None:
                    blob, st = res
                    return blob, [{"op": "lz11"}] + st

                       
        try:
            mini = parse_mini(cur)
        except Exception:
            mini = None
        if mini is not None:
            for i in range(mini.count):
                try:
                    child = mini.extract(cur, i)
                except Exception:
                    continue
                res = _walk(child, depth + 1)
                if res is not None:
                    blob, st = res
                    return blob, [
                        {"op": "mini", "index": int(i), "ident": mini.ident}
                    ] + st

                            
        try:
            cont = parse_container(cur)
        except Exception:
            cont = None
        if cont is not None:
            for i in range(cont.count):
                try:
                    child = cont.extract(cur, i)
                except Exception:
                    continue
                res = _walk(child, depth + 1)
                if res is not None:
                    blob, st = res
                    return (
                        blob,
                        [{"op": "container", "index": int(i), "magic": cont.magic2}]
                        + st,
                    )

        return None

    out = _walk(bytes(data), 0)
    if out is None:
        raise ValueError("no exportable GFModel blob found")
    return out


@dataclass(frozen=True)
class PatchPlan:
    version: int
    archive_path: str
    entry_index: int
    bit: int
    steps: List[Step]
    breadcrumb: str

    def to_json(self) -> str:
        return json.dumps(
            {
                "v": int(self.version),
                "archive_path": str(self.archive_path),
                "entry_index": int(self.entry_index),
                "bit": int(self.bit),
                "steps": list(self.steps),
                "breadcrumb": str(self.breadcrumb),
            },
            separators=(",", ":"),
            sort_keys=True,
        )


def patch_plan_from_json(s: str) -> PatchPlan:
    obj = json.loads(s)
    if not isinstance(obj, dict):
        raise ValueError("patch plan json must be an object")
    v = int(obj.get("v", 0))
    if v != 1:
        raise ValueError(f"unsupported patch plan version: {v}")
    archive_path = str(obj.get("archive_path", "")).strip()
    entry_index = int(obj.get("entry_index", -1))
    bit = int(obj.get("bit", 0))
    steps = obj.get("steps", [])
    if not isinstance(steps, list):
        raise ValueError("patch plan steps must be a list")
    breadcrumb = str(obj.get("breadcrumb", "")).strip()
    if not archive_path or entry_index < 0 or bit < 0 or bit > 31:
        raise ValueError("patch plan missing required fields")
    return PatchPlan(
        version=1,
        archive_path=archive_path,
        entry_index=entry_index,
        bit=bit,
        steps=[dict(x) for x in steps],
        breadcrumb=breadcrumb,
    )
