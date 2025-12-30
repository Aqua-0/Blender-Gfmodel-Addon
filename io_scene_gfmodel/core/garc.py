"""GARC/CRAG container parsing (minimal).

This is intended for selectively reading files inside archives without extracting
everything to disk. For now, it focuses on the container layout (CRAG + FATO/FATB/FIMB)
and reading raw entry bytes (no decompression or nested-format tracing here yet).
"""

from __future__ import annotations

import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


def _u16le(buf: memoryview, o: int) -> int:
    return int(struct.unpack_from("<H", buf, o)[0])


def _u32le(buf: memoryview, o: int) -> int:
    return int(struct.unpack_from("<I", buf, o)[0])


def _ascii4(buf: memoryview, o: int) -> str:
    return buf[o : o + 4].tobytes().decode("ascii", "replace")


@dataclass(frozen=True)
class _GarcSubEntry:
    bit: int
    start: int
    end: int
    length: int


@dataclass(frozen=True)
class _GarcEntry:
    index: int
    flags: int
    subentries: List[_GarcSubEntry]

    def primary(self) -> Optional[_GarcSubEntry]:
        for se in self.subentries:
            if se.bit == 0:
                return se
        return self.subentries[0] if self.subentries else None


@dataclass(frozen=True)
class _Garc:
    version: int
    data_offset: int
    entries: List[_GarcEntry]

    def extract(self, blob: bytes, index: int, *, bit: int = 0) -> bytes:
        if index < 0 or index >= len(self.entries):
            raise IndexError("entry index out of range")
        ent = self.entries[index]
        se = next((x for x in ent.subentries if x.bit == bit), None)
        if se is None:
            raise ValueError(
                f"entry has no subentry for bit {bit} (flags=0x{ent.flags:08X})"
            )
        start = int(self.data_offset) + int(se.start)
        end = int(self.data_offset) + int(se.end)
        if start < 0 or end < start or end > len(blob):
            raise ValueError("entry slice out of range")
        return blob[start:end]


@dataclass(frozen=True)
class _GarcPrimaryEntry:
    index: int
    flags: int
    bit: int
    start: int
    end: int
    length: int


@dataclass(frozen=True)
class _GarcFileIndex:
    path: str
    version: int
    data_offset: int
    entries: List[_GarcPrimaryEntry]

    def read_primary_bytes(self, index: int) -> bytes:
        if index < 0 or index >= len(self.entries):
            raise IndexError("entry index out of range")
        e = self.entries[index]
        abs_off = int(self.data_offset) + int(e.start)
        length = int(e.end) - int(e.start)
        if length < 0:
            raise ValueError("entry length invalid")
        with open(self.path, "rb") as f:
            f.seek(abs_off)
            return f.read(length)

    def read_primary_magic4(self, index: int) -> bytes:
        if index < 0 or index >= len(self.entries):
            raise IndexError("entry index out of range")
        e = self.entries[index]
        abs_off = int(self.data_offset) + int(e.start)
        with open(self.path, "rb") as f:
            f.seek(abs_off)
            return f.read(4)


def _ctz32(x: int) -> int:
    x = int(x) & 0xFFFFFFFF
    if x == 0:
        return 32
                          
    return (x & -x).bit_length() - 1


def parse_garc_file(path: str) -> _GarcFileIndex:
                                                                                    
    with open(path, "rb") as f:
        head = f.read(0x40)
        if len(head) < 0x1C:
            raise ValueError("file too small")
        h = memoryview(head)
        magic = _ascii4(h, 0x00)
        if magic not in ("CRAG", "GARC"):
            raise ValueError(f"not a GARC (magic={magic!r})")
        fato_off = _u32le(h, 0x04)
                                                                                                   
                                                                                           
        version = _u16le(h, 0x0A)
        data_off = _u32le(h, 0x10)

        if fato_off <= 0:
            raise ValueError("invalid FATO offset")
        if data_off <= 0:
            raise ValueError("invalid data offset")

              
        f.seek(int(fato_off))
        fato_hdr = f.read(0x0C)
        if len(fato_hdr) < 0x0C:
            raise ValueError("FATO header out of range")
        fh = memoryview(fato_hdr)
        fato_magic = _ascii4(fh, 0x00)
        fato_size = _u32le(fh, 0x04)
        entry_count = _u16le(fh, 0x08)
        if fato_magic != "OTAF":
            raise ValueError(f"missing FATO (magic={fato_magic!r})")
        if fato_size < 0x0C:
            raise ValueError("FATO size too small")
                       
        fato_offsets = f.read(int(entry_count) * 4)
        if len(fato_offsets) < int(entry_count) * 4:
            raise ValueError("FATO offsets out of range")
        fo = memoryview(fato_offsets)
        offsets = [_u32le(fo, i * 4) for i in range(int(entry_count))]

              
        fatb_off = int(fato_off) + int(fato_size)
        f.seek(fatb_off)
        fatb_hdr = f.read(0x0C)
        if len(fatb_hdr) < 0x0C:
            raise ValueError("FATB header out of range")
        bh = memoryview(fatb_hdr)
        fatb_magic = _ascii4(bh, 0x00)
        fatb_size = _u32le(bh, 0x04)
        fatb_count = _u32le(bh, 0x08)
        if fatb_magic != "BTAF":
            raise ValueError(f"missing FATB (magic={fatb_magic!r})")
        if int(fatb_count) != int(entry_count):
            raise ValueError("FATO/FATB count mismatch")
        if fatb_size < 0x0C:
            raise ValueError("FATB size too small")
        fatb_end = fatb_off + int(fatb_size)

                                                                                                     
                                                                                                                
        entries: List[_GarcPrimaryEntry] = []
        fatb_table_base = fatb_off + 0x0C
        for i in range(int(entry_count)):
            ent_off = fatb_table_base + int(offsets[i])
            if ent_off + 4 > fatb_end:
                raise ValueError("FATB entry out of range")
            f.seek(ent_off)
            flags_bytes = f.read(4)
            if len(flags_bytes) < 4:
                raise ValueError("FATB entry flags out of range")
            flags = _u32le(memoryview(flags_bytes), 0)
            bit = _ctz32(flags)
            start = end = length = 0
            if flags != 0 and bit < 32:
                sub = f.read(12)
                if len(sub) < 12:
                    raise ValueError("FATB subentry out of range")
                sm = memoryview(sub)
                start = _u32le(sm, 0)
                end = _u32le(sm, 4)
                length = _u32le(sm, 8)
            entries.append(
                _GarcPrimaryEntry(
                    index=int(i),
                    flags=int(flags),
                    bit=int(bit if bit < 32 else 0),
                    start=int(start),
                    end=int(end),
                    length=int(length),
                )
            )

    return _GarcFileIndex(
        path=str(path), version=int(version), data_offset=int(data_off), entries=entries
    )


def parse_garc(blob: bytes) -> _Garc:
    b = memoryview(blob)
    if len(b) < 0x1C:
        raise ValueError("file too small")

    magic = _ascii4(b, 0x00)
                                                                                          
    if magic not in ("CRAG", "GARC"):
        raise ValueError(f"not a GARC (magic={magic!r})")

    fato_off = _u32le(b, 0x04)
    _bom = _u16le(b, 0x08)
    version = _u16le(b, 0x0A)
    _section_count = _u32le(b, 0x0C)
    data_off = _u32le(b, 0x10)

    if fato_off <= 0 or fato_off >= len(b):
        raise ValueError("invalid FATO offset")
    if data_off <= 0 or data_off > len(b):
        raise ValueError("invalid data offset")

          
    o = int(fato_off)
    fato_magic = _ascii4(b, o + 0x00)
    fato_size = _u32le(b, o + 0x04)
    if fato_magic != "OTAF":
        raise ValueError(f"missing FATO (magic={fato_magic!r})")
    if fato_size < 0x0C or o + fato_size > len(b):
        raise ValueError("FATO out of range")
    entry_count = _u16le(b, o + 0x08)
    fato_offsets = [_u32le(b, o + 0x0C + i * 4) for i in range(int(entry_count))]

          
    fatb_off = o + int(fato_size)
    fatb_magic = _ascii4(b, fatb_off + 0x00)
    fatb_size = _u32le(b, fatb_off + 0x04)
    fatb_count = _u32le(b, fatb_off + 0x08)
    if fatb_magic != "BTAF":
        raise ValueError(f"missing FATB (magic={fatb_magic!r})")
    if fatb_size < 0x0C or fatb_off + fatb_size > len(b):
        raise ValueError("FATB out of range")
    if int(fatb_count) != int(entry_count):
        raise ValueError("FATO/FATB count mismatch")

    fatb_table_base = fatb_off + 0x0C
    fatb_end = fatb_off + int(fatb_size)
    entries: List[_GarcEntry] = []
    for i in range(int(entry_count)):
        ent_off = fatb_table_base + int(fato_offsets[i])
        if ent_off + 4 > fatb_end:
            raise ValueError("FATB entry out of range")
        flags = _u32le(b, ent_off)
        cur = ent_off + 4
        sub: List[_GarcSubEntry] = []
        for bit in range(32):
            if (flags & (1 << bit)) == 0:
                continue
            if cur + 12 > fatb_end:
                raise ValueError("FATB subentry out of range")
            start = _u32le(b, cur + 0)
            end = _u32le(b, cur + 4)
            length = _u32le(b, cur + 8)
            sub.append(
                _GarcSubEntry(
                    bit=int(bit), start=int(start), end=int(end), length=int(length)
                )
            )
            cur += 12
        entries.append(_GarcEntry(index=int(i), flags=int(flags), subentries=sub))

    return _Garc(version=int(version), data_offset=int(data_off), entries=entries)


def _align_up(x: int, a: int) -> int:
    a = int(a)
    if a <= 1:
        return int(x)
    return (int(x) + (a - 1)) & ~(a - 1)


@dataclass(frozen=True)
class _GarcPatchSub:
    bit: int
    start: int
    end: int
    length: int
    start_off: int                                   
    end_off: int                                 
    length_off: int                                    


@dataclass(frozen=True)
class _GarcPatchEntry:
    index: int
    flags: int
    subs: List[_GarcPatchSub]


def rewrite_garc_file(
    src_path: str,
    dst_path: str,
    *,
    replacements: dict[tuple[int, int], bytes],
    pad_size_override: Optional[int] = None,
) -> None:
    """Rewrite a GARC file while replacing selected (entry_index, bit) payloads.

    - Reads the header + FATO/FATB tables.
    - Rebuilds the data section sequentially using the archive's `pad_size` alignment.
    - Patches FATB subentry start/end/length fields to match the new layout.
    - Updates header arc size + largest entry sizes.

    This is intended as the first safe step toward in-Blender patch workflows.
    """
    src_path = str(src_path)
    dst_path = str(dst_path)
    src = Path(src_path)
    if not src.exists() or not src.is_file():
        raise FileNotFoundError(src_path)

    with open(src_path, "rb") as f:
        head = f.read(0x40)
        if len(head) < 0x24:
            raise ValueError("file too small for GARC header")
        h = memoryview(head)
        magic = _ascii4(h, 0x00)
        if magic not in ("CRAG", "GARC"):
            raise ValueError(f"not a GARC (magic={magic!r})")

        header_size = _u32le(h, 0x04)
        version = _u16le(h, 0x0A)
        data_off = _u32le(h, 0x10)
                                                                                 
                                                                                    
        pad_size = 0
        if int(header_size) >= 0x24 and len(h) >= 0x24:
            pad_size = _u32le(h, 0x20)
        else:
            pad_size = 4
        if pad_size_override is not None:
            pad_size = int(pad_size_override)
        if pad_size <= 0:
            pad_size = 4

        fato_off = int(header_size)
        if fato_off <= 0:
            raise ValueError("invalid header size / FATO offset")
        if data_off <= 0:
            raise ValueError("invalid data offset")

              
        f.seek(fato_off)
        fato_hdr = f.read(0x0C)
        if len(fato_hdr) < 0x0C:
            raise ValueError("FATO header out of range")
        fh = memoryview(fato_hdr)
        fato_magic = _ascii4(fh, 0x00)
        fato_size = _u32le(fh, 0x04)
        entry_count = _u16le(fh, 0x08)
        if fato_magic != "OTAF":
            raise ValueError(f"missing FATO (magic={fato_magic!r})")
        if fato_size < 0x0C:
            raise ValueError("FATO size too small")
        fato_offsets = f.read(int(entry_count) * 4)
        if len(fato_offsets) < int(entry_count) * 4:
            raise ValueError("FATO offsets out of range")
        fo = memoryview(fato_offsets)
        offsets = [_u32le(fo, i * 4) for i in range(int(entry_count))]

              
        fatb_off = int(fato_off) + int(fato_size)
        f.seek(fatb_off)
        fatb_hdr = f.read(0x0C)
        if len(fatb_hdr) < 0x0C:
            raise ValueError("FATB header out of range")
        bh = memoryview(fatb_hdr)
        fatb_magic = _ascii4(bh, 0x00)
        fatb_size = _u32le(bh, 0x04)
        fatb_count = _u32le(bh, 0x08)
        if fatb_magic != "BTAF":
            raise ValueError(f"missing FATB (magic={fatb_magic!r})")
        if int(fatb_count) != int(entry_count):
            raise ValueError("FATO/FATB count mismatch")
        if fatb_size < 0x0C:
            raise ValueError("FATB size too small")
        fatb_table_base = fatb_off + 0x0C
        fatb_end = fatb_off + int(fatb_size)

                                                   
        entries: List[_GarcPatchEntry] = []
        for i in range(int(entry_count)):
            ent_off = fatb_table_base + int(offsets[i])
            if ent_off + 4 > fatb_end:
                raise ValueError("FATB entry out of range")
            f.seek(ent_off)
            flags_bytes = f.read(4)
            if len(flags_bytes) < 4:
                raise ValueError("FATB entry flags out of range")
            flags = _u32le(memoryview(flags_bytes), 0)
            cur = int(ent_off) + 4
            subs: List[_GarcPatchSub] = []
            for bit in range(32):
                if (int(flags) & (1 << bit)) == 0:
                    continue
                if cur + 12 > fatb_end:
                    raise ValueError("FATB subentry out of range")
                f.seek(cur)
                sub = f.read(12)
                if len(sub) < 12:
                    raise ValueError("FATB subentry out of range")
                sm = memoryview(sub)
                start = _u32le(sm, 0)
                end = _u32le(sm, 4)
                length = _u32le(sm, 8)
                subs.append(
                    _GarcPatchSub(
                        bit=int(bit),
                        start=int(start),
                        end=int(end),
                        length=int(length),
                        start_off=int(cur + 0),
                        end_off=int(cur + 4),
                        length_off=int(cur + 8),
                    )
                )
                cur += 12
            entries.append(_GarcPatchEntry(index=int(i), flags=int(flags), subs=subs))

                                                                             
                                                                                              
        pad_size = _infer_garc_pad_size(entries, default=int(pad_size))

                                                                                           
                                                                                       
        if replacements:
            import shutil

            sub_by_key: dict[tuple[int, int], _GarcPatchSub] = {}
            for ent in entries:
                for sub in ent.subs:
                    sub_by_key[(int(ent.index), int(sub.bit))] = sub

            all_noop = True
            for key, payload_raw in replacements.items():
                sub = sub_by_key.get((int(key[0]), int(key[1])))
                if sub is None:
                    raise ValueError(
                        f"replacement refers to missing subentry: entry={key[0]} bit={key[1]}"
                    )
                orig_stored_len = int(sub.end) - int(sub.start)
                if orig_stored_len < 0:
                    raise ValueError("stored entry length invalid")
                f.seek(int(data_off) + int(sub.start))
                orig_stored = f.read(int(orig_stored_len))
                if len(orig_stored) != int(orig_stored_len):
                    raise ValueError("failed to read original stored bytes")

                rep = bytes(payload_raw)
                if len(rep) > int(orig_stored_len):
                    all_noop = False
                    break
                if rep + orig_stored[len(rep) :] != orig_stored:
                    all_noop = False
                    break

            if all_noop:
                if str(src_path) == str(dst_path):
                    return
                shutil.copyfile(str(src_path), str(dst_path))
                return

                                                                                                    
        f.seek(0)
        prefix = bytearray(f.read(int(data_off)))
        if len(prefix) != int(data_off):
            raise ValueError("failed to read archive prefix")

                                                                                
        data_out = bytearray()
        largest_padded = 0
        largest = 0

        def patch_u32(off: int, v: int) -> None:
            if off < 0 or off + 4 > len(prefix):
                raise ValueError("patch offset out of prefix range")
            struct.pack_into("<I", prefix, int(off), int(v) & 0xFFFFFFFF)

        for ent in entries:
            for sub in ent.subs:
                                                  
                aligned = _align_up(len(data_out), int(pad_size))
                if aligned > len(data_out):
                    data_out.extend(b"\xff" * (aligned - len(data_out)))
                start = len(data_out)

                key = (int(ent.index), int(sub.bit))
                if key in replacements:
                    payload_raw = bytes(replacements[key])

                                                                                                        
                    orig_stored_len = int(sub.end) - int(sub.start)
                    if orig_stored_len < 0:
                        raise ValueError("stored entry length invalid")

                                                                                              
                    f.seek(int(data_off) + int(sub.start))
                    orig_stored = f.read(int(orig_stored_len))
                    if len(orig_stored) != int(orig_stored_len):
                        raise ValueError("failed to read original stored bytes")

                                                                                                 
                                                                                           
                     
                                                                                          
                                                                                              
                                                                                                    
                    if int(len(payload_raw)) == int(orig_stored_len):
                        length_field = int(sub.length)
                        stored_len = int(orig_stored_len)
                        payload = payload_raw
                    else:
                        length_field = int(len(payload_raw))
                        if length_field <= int(orig_stored_len):
                            stored_len = int(orig_stored_len)
                            payload = payload_raw + orig_stored[length_field:stored_len]
                        else:
                            stored_len = int(_align_up(length_field, int(pad_size)))
                            payload = payload_raw + (
                                b"\xff" * (stored_len - length_field)
                            )
                else:
                                                                                       
                    stored_len = int(sub.end) - int(sub.start)
                    if stored_len < 0:
                        raise ValueError("stored entry length invalid")
                    f.seek(int(data_off) + int(sub.start))
                    payload = f.read(int(stored_len))
                    if len(payload) != int(stored_len):
                        raise ValueError("failed to read original stored bytes")
                    length_field = int(sub.length)

                data_out.extend(payload)
                end = len(data_out)

                largest = max(int(largest), int(length_field))
                largest_padded = max(int(largest_padded), int(end - start))

                                                                     
                patch_u32(sub.start_off, int(start))
                patch_u32(sub.end_off, int(end))
                patch_u32(sub.length_off, int(length_field))

                                                
        arc_size = int(data_off) + int(len(data_out))
        patch_u32(0x14, int(arc_size))
        patch_u32(0x18, int(largest_padded))
        patch_u32(0x1C, int(largest))

                                          
    out_path = Path(dst_path)
                                                                                      
    tmp_path = out_path.with_name(out_path.name + f".tmp.{int(time.time() * 1000)}")
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp_path, "wb") as w:
        w.write(prefix)
        w.write(data_out)

                                                                                     
                                                                                 
    last_err: Exception | None = None
    for attempt in range(15):
        try:
            tmp_path.replace(out_path)
            last_err = None
            break
        except PermissionError as e:
            last_err = e
            time.sleep(0.05 * (attempt + 1))
        except OSError as e:
                                                                           
            last_err = e
            time.sleep(0.05 * (attempt + 1))

    if last_err is not None:
        try:
                                  
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        raise PermissionError(
            f"Patch failed: {last_err}\n"
            f"Could not replace:\n  {tmp_path}\nwith:\n  {out_path}\n"
            "If you are patching directly into Citra's mods folder, close Citra (or any tool "
            "watching the file) and try again."
        )


def _infer_garc_pad_size(entries: List[_GarcPatchEntry], *, default: int) -> int:
    starts: List[int] = []
    for ent in entries:
        for sub in ent.subs:
            starts.append(int(sub.start))

    if not starts:
        return max(1, int(default))

    default = int(default)
    if default > 1 and all((s % default) == 0 for s in starts):
        return default

    candidates = [4, 0x10, 0x20, 0x40, 0x80, 0x100, 0x200, 0x400, 0x800, 0x1000]
    for a in reversed(candidates):
        if all((s % a) == 0 for s in starts):
            return a

    return 4
