
from __future__ import annotations

import struct
from typing import List, Tuple

from ..lz11 import decompress as _lz11_decompress
from ..binlinker import looks_like_binlinker, parse_binlinker
from ..motion import _parse_gf_motion
from ..pkmn_container import parse_container as _parse_container
from ..types import _GFModel, _GFMotion, _GFShader, _GFTexture
from .parse_model import _parse_gf_model, _parse_gf_model_pack
from .parse_shader import _parse_gf_shader
from .parse_texture import _parse_gf_texture
from .reader import _BinReader, _looks_like_lz11, _lzss_ninty_decompress

def _parse_pkmn_container(data: bytes) -> Tuple[str, List[bytes]]:
    cont = _parse_container(bytes(data))
    files: List[bytes] = [cont.extract(bytes(data), i) for i in range(int(cont.count))]
    return str(cont.magic2), files


def _load_any(
    data: bytes,
) -> Tuple[List[_GFModel], List[_GFTexture], List[_GFMotion], List[_GFShader]]:
    models: List[_GFModel] = []
    textures: List[_GFTexture] = []
    motions: List[_GFMotion] = []
    shaders: List[_GFShader] = []

                                                                                          
    if looks_like_binlinker(data):
        try:
            bl = parse_binlinker(data)
            for i in range(int(bl.file_count)):
                try:
                    ent = bl.extract(data, i)
                except Exception:
                    continue
                m, t, a, s = _load_any(ent)
                models.extend(m)
                textures.extend(t)
                motions.extend(a)
                shaders.extend(s)
                                                                                         
                                                                                     
                                                                                   
            if models or textures or motions or shaders:
                return models, textures, motions, shaders
        except Exception:
                                               
            pass


    if len(data) >= 4 and 65 <= data[0] <= 90 and 65 <= data[1] <= 90:
        try:
            magic, entries = _parse_pkmn_container(data)
        except Exception:
            magic, entries = "", []
        if magic == "CP" and len(entries) >= 2:
            ent = entries[1]
            if _looks_like_lz11(ent):
                ent = _lz11_decompress(ent)
            return _load_any(ent)
        for ent in entries:
            if _looks_like_lz11(ent):
                ent = _lz11_decompress(ent)
            m, t, a, s = _load_any(ent)
            models.extend(m)
            textures.extend(t)
            motions.extend(a)
            shaders.extend(s)
        if entries:
            return models, textures, motions, shaders

    if len(data) < 4:
        return models, textures, motions, shaders

    u0 = struct.unpack_from("<I", data, 0)[0]






    if u0 == 0x15052616 and len(data) >= 16:
        try:
            section_count = struct.unpack_from("<I", data, 4)[0]
            fake = struct.pack("<II", 0x15122117, int(section_count)) + data[8:]
            model, _ = _parse_gf_model(fake, 0, "Model")
            models.append(model)
            return models, textures, motions, shaders
        except Exception:
            pass
    if u0 == 0x14110400 and len(data) >= 24:
        try:
            section_count = struct.unpack_from("<I", data, 4)[0]
            fake = struct.pack("<II", 0x15041213, int(section_count)) + data[8:]
            textures.append(_parse_gf_texture(fake))
            return models, textures, motions, shaders
        except Exception:
            pass
    if u0 == 0x00050000 and len(data) >= 32:
        try:
            fake = struct.pack("<I", 0x00060000) + data[4:]
            motions.append(_parse_gf_motion(fake, index=len(motions)))
            return models, textures, motions, shaders
        except Exception:
            pass

    if u0 == 0x00010000:
        m, t, s = _parse_gf_model_pack(data)
        models.extend(m)
        textures.extend(t)
        shaders.extend(s)
        return models, textures, motions, shaders
    if u0 == 0x15041213:



        try:
            sect8 = ""
            if len(data) >= 0x10:
                sect8 = data[0x08:0x10].split(b"\0", 1)[0].decode("ascii", "replace")
            sect10 = ""
            if len(data) >= 0x18:
                sect10 = data[0x10:0x18].split(b"\0", 1)[0].decode("ascii", "replace")

            if sect8 == "texture":
                textures.append(_parse_gf_texture(data))
            elif sect10 == "shader":
                sh, _ = _parse_gf_shader(data, 0)
                shaders.append(sh)
            else:

                try:
                    textures.append(_parse_gf_texture(data))
                except Exception:
                    sh, _ = _parse_gf_shader(data, 0)
                    shaders.append(sh)
        except Exception:
            pass
        return models, textures, motions, shaders
    if u0 == 0x15122117:
        model, _ = _parse_gf_model(data, 0, "Model")
        models.append(model)
        return models, textures, motions, shaders
    if u0 == 0x00060000:

        motions.append(_parse_gf_motion(data, index=len(motions)))
        return models, textures, motions, shaders


    anims_count = u0
    if 0 < anims_count < 4096 and len(data) >= 4 + anims_count * 4:
        r = _BinReader(data)
        _ = r.u32()
        offsets = [r.u32() for _ in range(anims_count)]
        for i, off in enumerate(offsets):
            if off == 0:
                continue
            if off >= len(data):
                continue
            sub = data[4 + off :]
            if len(sub) >= 4 and struct.unpack_from("<I", sub, 0)[0] == 0x00060000:
                motions.append(_parse_gf_motion(sub, index=i))
        if motions:
            return models, textures, motions, shaders

    return models, textures, motions, shaders
