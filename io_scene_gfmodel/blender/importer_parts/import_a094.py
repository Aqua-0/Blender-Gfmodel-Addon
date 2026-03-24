

from __future__ import annotations

from .a094_slot_names import a094_slot_name as _a094_slot_name
from .a094_slot_names import motion_short_tag as _motion_short_tag
import os
import struct
from typing import Dict, Sequence

import bpy

from ...core.io import _load_any
from ...core.patch_plan import PatchPlan, steps_to_breadcrumb
from ...core.types import _GFTexture, _GFShader, _GFMotion
from .import_loaded import _import_gfmodel_loaded








def _a094_pack_from_pc_count(count: int) -> str | None:
    if int(count) == 32:
        return "BT"
    if int(count) == 40:
        return "KW"
    if int(count) == 27:
        return "FI"
    if int(count) == 71:
        return "PF"
    return None


def _parse_pc_container_safe(data: bytes) -> tuple[str | None, list[bytes]]:
    if not data or len(data) < 4:
        return None, []
    try:
        magic = data[0:2].decode("ascii", "replace")
    except Exception:
        return None, []
    if magic.strip() == "":
        return None, []
    count = int.from_bytes(data[2:4], "little", signed=False)
    table_len = 4 + 4 * (count + 1)
    if count < 0 or count > 0x4000 or len(data) < table_len:
        return magic, []

    try:
        offsets = [struct.unpack_from("<I", data, 4 + i * 4)[0] for i in range(count + 1)]
    except Exception:
        return magic, []

    entries: list[bytes] = []
    for i in range(count):
        start, end = int(offsets[i]), int(offsets[i + 1])
        if start < 0 or end < start or end > len(data):
            entries.append(b"")
            continue
        entries.append(data[start:end])
    return magic, entries


def _import_gfmodel_bytes_with_a094_group(
    context: bpy.types.Context,
    data: bytes,
    *,
    a094_group_members: Sequence[object],
    a094_motion_pack: str = "ALL",
    a094_name_motions: bool = True,
    source_path: str,
    import_textures: bool,
    import_animations: bool,
    import_material_animations: bool = True,
    import_visibility_animations: bool = True,
    global_scale: float = 1.0,
    axis_forward: str = "-Z",
    axis_up: str = "Y",
) -> bool:


    models, textures, motions, shaders = _load_any(data)

    tex_by_name: Dict[str, _GFTexture] = {t.name: t for t in textures if getattr(t, "name", None)}
    sh_accum: list[_GFShader] = list(shaders)
    mot_accum: list[_GFMotion] = list(motions)

    want = str(a094_motion_pack or "BATTLE").upper().strip()
    allow_bt = want in ("BATTLE", "BT", "ALL")
    allow_kw = want in ("KAWAIGARI", "KAWAII", "KW", "ALL")
    allow_fi = want in ("FIELD", "FI", "ALL")
    allow_pf = want in ("POKEFINDER", "POKE_FINDER", "PF")

    for member in list(a094_group_members or []):
        entry_idx = -1
        bit = 0
        pre_steps: list[dict] = []
        archive_root = ''
        blob = b''
        try:
            if isinstance(member, dict):
                entry_idx = int(member.get('entry_index', -1))
                bit = int(member.get('bit', 0) or 0)
                pre_steps = list(member.get('pre_steps', []) or [])
                archive_root = str(member.get('archive_path', '') or '').strip()
                blob = bytes(member.get('payload', b''))
            elif isinstance(member, (tuple, list)) and len(member) >= 2:
                entry_idx = int(member[0])
                blob = bytes(member[1])
        except Exception:
            continue

        b = bytes(blob)
        if not archive_root:

            sp = str(source_path or '')
            if '#' in sp:
                archive_root = sp.split('#', 1)[0]

        magic, entries = _parse_pc_container_safe(b)
        pack = _a094_pack_from_pc_count(len(entries)) if (magic == "PC" and entries) else None

        if pack in ("BT", "KW", "FI", "PF"):
            if (pack == "BT" and not allow_bt) or (pack == "KW" and not allow_kw) or (pack == "FI" and not allow_fi) or (pack == "PF" and not allow_pf):
                continue
            if not import_animations:
                continue

            for slot_i, ent in enumerate(entries):
                if not ent:
                    continue
                try:
                    _m, _t, a2, s2 = _load_any(ent)
                except Exception:
                    continue
                if s2:
                    sh_accum.extend(s2)
                for mot in a2:
                    mot.index = int(slot_i)
                    setattr(mot, "gfmodel_pack", pack)
                    if a094_name_motions:
                        nm = _a094_slot_name(pack, int(slot_i))
                        if nm:
                            setattr(mot, "gfmodel_slot_name", nm)
                    try:
                        steps = list(pre_steps) + [{"op": "container", "magic": "PC", "index": int(slot_i)}]
                        bc = f"{archive_root}#{int(entry_idx)}"
                        st = steps_to_breadcrumb(steps)
                        if st:
                            bc = f"{bc}/{st}"
                        plan = PatchPlan(
                            version=1,
                            archive_path=str(archive_root),
                            entry_index=int(entry_idx),
                            bit=int(bit),
                            steps=[dict(x) for x in steps],
                            breadcrumb=str(bc),
                        )
                        setattr(mot, "gfmodel_patch_plan_json", plan.to_json())
                    except Exception:
                        pass
                    mot_accum.append(mot)
            continue


        try:
            _m2, t2, _a2, s2 = _load_any(b)
        except Exception:
            continue
        for tex in t2:
            if getattr(tex, "name", None) and tex.name not in tex_by_name:
                tex_by_name[tex.name] = tex
        if s2:
            sh_accum.extend(s2)

    textures_out = list(tex_by_name.values())

    pack_order = {"BT": 0, "KW": 1, "FI": 2, "PF": 3, "": 9}

    def mot_key(m: _GFMotion):
        p = str(getattr(m, "gfmodel_pack", "") or "")
        return (pack_order.get(p, 8), int(getattr(m, "index", 0)))

    mot_accum.sort(key=mot_key)
    source_path_real = str(source_path)
    try:
        if os.path.isfile(str(source_path_real)):
            pass
        else:
            import hashlib
            import tempfile

            h = hashlib.md5(bytes(data)).hexdigest()[:12]
            tmp_root = ''
            try:
                tmp_root = str(getattr(bpy.app, 'tempdir', '') or '').strip()
            except Exception:
                tmp_root = ''
            if not tmp_root:
                tmp_root = tempfile.gettempdir()
            base = os.path.join(tmp_root, 'gfmodel_imports')
            os.makedirs(base, exist_ok=True)
            source_path_real = os.path.join(base, f'import_{h}.bin')
            if (not os.path.exists(source_path_real)) or (os.path.getsize(source_path_real) != len(data)):
                with open(source_path_real, 'wb') as f:
                    f.write(bytes(data))
    except Exception:
        source_path_real = str(source_path)




    try:
        context.scene['gfmodel_last_import_source'] = str(source_path)
        context.scene['gfmodel_last_import_breadcrumb'] = str(source_path)
    except Exception:
        pass

    return _import_gfmodel_loaded(
        context,
        models=models,
        textures=textures_out,
        motions=mot_accum,
        shaders=sh_accum,
        source_path=str(source_path_real),
        import_textures=import_textures,
        import_animations=import_animations,
        import_material_animations=import_material_animations,
        import_visibility_animations=import_visibility_animations,
        global_scale=global_scale,
        axis_forward=axis_forward,
        axis_up=axis_up,
    )

