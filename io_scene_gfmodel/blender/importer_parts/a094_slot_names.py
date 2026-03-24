

from __future__ import annotations

from typing import Sequence


_A094_BT_NAMES: list[str] = [
    "waitA01",
    "waitB01",
    "waitC01",
    "waitD01",
    "walkA01",
    "walkB01",
    "runA01",
    "runB01",
    "turnA01",
    "turnB01",
    "jumpA01",
    "jumpB01",
    "fallA01",
    "fallB01",
    "landingA01",
    "landingB01",
    "attackA01",
    "attackA02",
    "attackA03",
    "attackB01",
    "attackB02",
    "attackB03",
    "damageA01",
    "damageA02",
    "damageB01",
    "damageB02",
    "faintA01",
    "faintB01",
    "faintC01",
    "faintD01",
    "startA01",
    "endA01",
]

_A094_KW_NAMES: list[str] = [
    "kw_waitA01",
    "kw_waitB01",
    "kw_waitC01",
    "kw_waitD01",
    "kw_walkA01",
    "kw_walkB01",
    "kw_runA01",
    "kw_runB01",
    "kw_turnA01",
    "kw_turnB01",
    "kw_jumpA01",
    "kw_jumpB01",
    "kw_fallA01",
    "kw_fallB01",
    "kw_landingA01",
    "kw_landingB01",
    "kw_attackA01",
    "kw_attackA02",
    "kw_attackA03",
    "kw_attackB01",
    "kw_attackB02",
    "kw_attackB03",
    "kw_damageA01",
    "kw_damageA02",
    "kw_damageB01",
    "kw_damageB02",
    "kw_faintA01",
    "kw_faintB01",
    "kw_faintC01",
    "kw_faintD01",
    "kw_startA01",
    "kw_endA01",
    "kw_guruguruA01",
    "kw_guruguruB01",
    "kw_guruguruC01",
    "kw_guruguruD01",
    "kw_guruguruE01",
    "kw_guruguruF01",
    "kw_guruguruG01",
    "kw_guruguruH01",
]

_A094_FI_NAMES: list[str] = [
    "fi_waitA01",
    "fi_walkA01",
    "fi_runA01",
    "fi_turnA01",
    "fi_jumpA01",
    "fi_fallA01",
    "fi_landingA01",
    "fi_attackA01",
    "fi_attackA02",
    "fi_attackA03",
    "fi_damageA01",
    "fi_damageA02",
    "fi_faintA01",
    "fi_startA01",
    "fi_endA01",
    "fi_flyA01",
    "fi_flyB01",
    "fi_flyC01",
    "fi_flyD01",
    "fi_flyE01",
    "fi_flyF01",
    "fi_flyG01",
    "fi_flyH01",
    "fi_flyI01",
    "fi_flyJ01",
    "fi_flyK01",
    "fi_flyL01",
]

_A094_PF_NAMES: list[str] = _A094_BT_NAMES + _A094_KW_NAMES + _A094_FI_NAMES

_A094_EXTRA_NAMES: list[str] = [
    "extra00",
    "extra01",
    "extra02",
    "extra03",
    "extra04",
    "extra05",
    "extra06",
    "extra07",
    "extra08",
    "extra09",
]


def a094_slot_name(pack: str, slot: int) -> str | None:
    pack = str(pack or "")
    slot_i = int(slot)
    if pack == "BT":
        base = _A094_BT_NAMES
    elif pack == "KW":
        base = _A094_KW_NAMES
    elif pack == "FI":
        base = _A094_FI_NAMES
    elif pack == "PF":
        base = _A094_PF_NAMES
    else:
        return None

    if 0 <= slot_i < len(base):
        return base[slot_i]

    extra_i = slot_i - len(base)
    if 0 <= extra_i < len(_A094_EXTRA_NAMES):
        return _A094_EXTRA_NAMES[extra_i]

    return None


def motion_short_tag(mot: object) -> str:
    pack = str(getattr(mot, "gfmodel_pack", "") or "").strip()
    idx = int(getattr(mot, "index", 0))
    slot_name = str(getattr(mot, "gfmodel_slot_name", "") or "").strip()
    if pack and slot_name:
        return f"{pack}_M{idx:02d}_{slot_name}"
    if pack:
        return f"{pack}_M{idx:02d}"
    if slot_name:
        return f"M{idx:02d}_{slot_name}"
    return f"Motion_{idx}"


__all__ = [
    'a094_slot_name',
    'motion_short_tag',
]
