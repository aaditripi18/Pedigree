"""Read GGUF files (llama.cpp's format) from the Hub with range requests.

A GGUF file starts with metadata and a table of tensors (name, shape, quantisation type,
offset), so like safetensors it can be sampled without downloading it. Quantised tensors
are stored in blocks (32 or 256 values); sampling reads whole leading blocks and
dequantises them with llama.cpp's own reference code (the `gguf` package, MIT).

GGUF renames tensors (model.layers.3.self_attn.q_proj.weight -> blk.3.attn_q.weight);
`hf_names` maps them back to a Hugging Face model's names with llama.cpp's own table.
llama.cpp also reorders the rows of some attention projections for its rotary layout,
a row permutation the invariant check is built to ignore.
"""
from __future__ import annotations

import os
import re
import struct

import numpy as np

from .torchzip import RangeFile

SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
# Correction to readers v6 (2026-11-23, PLAN.md): GGUF's specification allows big-endian files
# (llama.cpp's `gguf_convert_endian.py` writes them; s390x builds load them). The version field tells
# the byte order. Read as little-endian, a big-endian file passed the version check (3 reads as
# 50331648), and every count and string length after it was garbage. A string "length" of ~1.4e18
# made the reader fetch block after block until Kaggle stopped H28's session at 12 hours.
# PEDIGREE_READERS_V6=0 keeps the earlier reading, so H27 and E2y reproduce as recorded.
READERS_V6 = os.environ.get("PEDIGREE_READERS_V6", "1") != "0"
# per llama.cpp's converter: the fp16 fields of each block it byte-swaps (offset within the block)
BE_BLOCK_FIELDS = {"Q4_0": (0,), "Q8_0": (0,), "Q4_K": (0, 2), "Q6_K": (208,), "MXFP4": (), "NVFP4": ()}
BE_ELEMENT = {"F32": 4, "F16": 2, "BF16": 2, "F64": 8, "I8": 1, "I16": 2, "I32": 4, "I64": 8}


def byte_order(version_bytes: bytes) -> str:
    """'<' or '>' for a GGUF file, from its 4 version bytes."""
    le, be = struct.unpack("<I", version_bytes)[0], struct.unpack(">I", version_bytes)[0]
    if not READERS_V6 or 2 <= le <= 0xFFFF or le < 2:
        return "<"
    if 2 <= be <= 0xFFFF:
        return ">"
    raise ValueError(f"GGUF version {le} is not plausible in either byte order")


def to_little(raw: bytes, qtype: str, type_size: int) -> bytes:
    """Bytes of a big-endian GGUF tensor as the little-endian file would hold them, by llama.cpp's
    own rule for each type; a type its converter does not define has no published byte order."""
    a = np.frombuffer(raw, dtype=np.uint8)
    if qtype in BE_ELEMENT:
        k = BE_ELEMENT[qtype]
        return a.reshape(-1, k)[:, ::-1].tobytes()
    if qtype not in BE_BLOCK_FIELDS:
        raise ValueError(f"big-endian GGUF tensor type {qtype}: llama.cpp defines no byte order for it")
    b = a.reshape(-1, type_size).copy()
    for o in BE_BLOCK_FIELDS[qtype]:
        b[:, [o, o + 1]] = b[:, [o + 1, o]]
    return b.tobytes()


PREFERENCE = ["Q8_0", "BF16", "F16", "F32", "Q6_K", "Q5_K_M", "Q5_K_S", "Q5_0", "Q4_K_M", "Q4_K_S", "Q4_0"]


def choose_variant(names: list[str]) -> list[str]:
    """One quantisation variant of a repository's .gguf files (all shards of it), the most
    precise available; multimodal projector files are left out."""
    ggufs = [n for n in names if n.lower().endswith(".gguf") and "mmproj" not in n.lower()]
    if not ggufs:
        return []
    for tag in PREFERENCE:
        pick = [n for n in ggufs if re.search(rf"(?<![A-Za-z0-9]){tag}(?![A-Za-z0-9])", n, re.I)]
        if pick:
            return sorted(pick)
    first = sorted(ggufs)[0]
    stem = re.sub(r"-\d{5}-of-\d{5}\.gguf$", "", first)
    return sorted(n for n in ggufs if n.startswith(stem))


class _Reader:
    def __init__(self, fh: RangeFile):
        self.fh, self.order = fh, "<"

    def version(self) -> int:
        b = self.fh.read(4)
        self.order = byte_order(b)
        (v,) = struct.unpack(self.order + "I", b)
        if v < 2:
            raise ValueError(f"GGUF version {v} is not supported")
        return v

    def take(self, fmt: str):
        if self.order != "<" and fmt[0] == "<":
            fmt = self.order + fmt[1:]
        size = struct.calcsize(fmt)
        return struct.unpack(fmt, self.fh.read(size))

    def string(self) -> str:
        (n,) = self.take("<Q")
        return self.fh.read(n).decode("utf-8", errors="replace")

    def value(self, vtype: int):
        if vtype in SCALARS:
            return self.take(SCALARS[vtype])[0]
        if vtype == 8:
            return self.string()
        if vtype == 9:
            (itype,), (n,) = self.take("<I"), self.take("<Q")
            if itype in SCALARS and n > 64:     # long numeric arrays (token scores): skip
                self.fh.seek(n * struct.calcsize(SCALARS[itype]), 1)
                return None
            out = [self.value(itype) for _ in range(n)]
            return out if n <= 64 else None     # long string arrays (vocabularies): not kept
        raise ValueError(f"unknown GGUF value type {vtype}")


def n_tensors(url: str) -> int:
    """How many tensors this file holds: the count in its first 24 bytes (magic, version, tensor
    count). A split set's total is the sum over its shards, which is what its `split.tensors.count`
    states; reading only these bytes spares walking the metadata (vocabularies of several MB)."""
    from .remote import fetch_range
    head = fetch_range(url, 0, 23)
    if head[:4] != b"GGUF":
        raise ValueError("not a GGUF file")
    order = byte_order(head[4:8])
    version, n = struct.unpack(order + "IQ", head[4:16])
    if version < 2:
        raise ValueError(f"GGUF version {version} is not supported")
    return int(n)


def file_type(url: str, size: int) -> str | None:
    """`general.type` of a GGUF file ("model", "adapter", "imatrix", ...), or None if its leading
    `general.*` keys do not state one (writers put those keys first; reading stops at the first other
    key, before any vocabulary)."""
    fh = RangeFile(url, size)
    r = _Reader(fh)
    if fh.read(4) != b"GGUF":
        raise ValueError("not a GGUF file")
    r.version()
    _, n_kv = r.take("<Q")[0], r.take("<Q")[0]
    for _ in range(n_kv):
        key = r.string()
        (vtype,) = r.take("<I")
        v = r.value(vtype)
        if key == "general.type":
            return str(v)
        if not key.startswith("general."):
            return None
    return None


def tensor_count(url: str, size: int) -> tuple[int, int | None]:
    """(tensors in this file, tensors in the whole split set or None), from the header and metadata
    alone (tensor descriptions are not read)."""
    fh = RangeFile(url, size)
    r = _Reader(fh)
    if fh.read(4) != b"GGUF":
        raise ValueError("not a GGUF file")
    r.version()
    n_tensors, n_kv = r.take("<Q")[0], r.take("<Q")[0]
    split = None
    for _ in range(n_kv):
        key = r.string()
        (vtype,) = r.take("<I")
        v = r.value(vtype)
        if key == "split.tensors.count":
            split = int(v)
    return n_tensors, split


def read_header(url: str, size: int) -> tuple[dict, dict]:
    """(metadata of interest, {tensor name: spec}) with absolute data offsets."""
    fh = RangeFile(url, size)
    r = _Reader(fh)
    if fh.read(4) != b"GGUF":
        raise ValueError("not a GGUF file")
    r.version()
    n_tensors, n_kv = r.take("<Q")[0], r.take("<Q")[0]
    meta = {}
    for _ in range(n_kv):
        key = r.string()
        (vtype,) = r.take("<I")
        v = r.value(vtype)
        if v is not None and not isinstance(v, list):
            meta[key] = v
    infos = []
    for _ in range(n_tensors):
        name = r.string()
        (nd,) = r.take("<I")
        dims = [r.take("<Q")[0] for _ in range(nd)]
        (qtype,) = r.take("<I")
        (offset,) = r.take("<Q")
        infos.append((name, dims, qtype, offset))
    align = int(meta.get("general.alignment", 32))
    start = (fh.tell() + align - 1) // align * align
    tensors = {name: {"dtype": "GGUF", "qtype": qtype, "shape": list(reversed(dims)), "start": start + offset,
                      **({"order": ">"} if r.order == ">" else {})}
               for name, dims, qtype, offset in infos}
    return meta, tensors


def sample(url: str, spec: dict, elements: int) -> np.ndarray:
    """The first `elements` values (float64) of a GGUF tensor, dequantised."""
    import gguf
    from . import ggufextra
    from .remote import fetch_range
    try:
        qtype = gguf.GGMLQuantizationType(spec["qtype"])
    except ValueError:
        qtype = None
    if qtype is None or qtype not in gguf.GGML_QUANT_SIZES or spec["qtype"] in ggufextra.TYPES:
        # newer than the installed gguf package, or listed there with no decoder (Q1_0): decoded
        # from the public source (pedigree/ggufextra.py)
        if spec["qtype"] not in ggufextra.TYPES:
            raise ValueError(f"GGUF tensor type {spec['qtype']} is not in llama.cpp's published list "
                             "(a fork's own format)")
        if spec.get("order") == ">":
            raise ValueError(f"big-endian GGUF tensor type {spec['qtype']}: llama.cpp defines no byte order for it")
        _, block, type_size = ggufextra.TYPES[spec["qtype"]]
        total = int(np.prod(spec["shape"]))
        n = min(elements, total)
        group = ggufextra.ROW_GROUP.get(spec["qtype"])
        if group:       # rows stored interleaved in groups: read whole groups of rows
            cols = int(spec["shape"][-1])
            per_group = group * cols // block * type_size
            nbytes = -(-n // (group * cols)) * per_group
            raw = fetch_range(url, spec["start"], spec["start"] + nbytes - 1)
            return np.asarray(ggufextra.dequantize(raw, spec["qtype"], cols), dtype=np.float64)[:n]
        blocks = -(-n // block)
        raw = fetch_range(url, spec["start"], spec["start"] + blocks * type_size - 1)
        return np.asarray(ggufextra.dequantize(raw, spec["qtype"]), dtype=np.float64)[:n]
    block, type_size = gguf.GGML_QUANT_SIZES[qtype]
    total = int(np.prod(spec["shape"]))
    n = min(elements, total)
    plain = {"I8": "<i1", "I16": "<i2", "I32": "<i4", "I64": "<i8", "F64": "<f8"}.get(qtype.name)
    big = spec.get("order") == ">"
    if plain:                       # integer and float64 tensors are stored as they are, not quantised
        raw = fetch_range(url, spec["start"], spec["start"] + n * type_size - 1)
        if big:
            raw = to_little(raw, qtype.name, type_size)
        return np.frombuffer(raw, dtype=plain).astype(np.float64)[:n]
    blocks = -(-n // block)
    raw = fetch_range(url, spec["start"], spec["start"] + blocks * type_size - 1)
    if big:
        raw = to_little(raw, qtype.name, type_size)
    vals = gguf.quants.dequantize(np.frombuffer(raw, dtype=np.uint8), qtype)
    return np.asarray(vals, dtype=np.float64).ravel()[:n]


def hf_names(meta: dict, gguf_names, hf_candidates) -> dict[str, str]:
    """{gguf tensor name: Hugging Face name} for the HF names a GGUF file holds.

    llama.cpp's table knows a language model's own names (model.layers.3...). A multimodal
    checkpoint nests them (model.language_model.layers.3...), so a name the table does not know
    is tried again with its leading parts dropped (and "model." put back). Such a match is kept
    only when the HF tensor's shape equals the GGUF tensor's and no other candidate claims the same
    GGUF tensor that way, so a vision tower's layers.3 cannot stand in for the language model's.
    Shapes come with dicts {name: (kind, shape)}; with plain lists only exact names are mapped."""
    import gguf
    arch_name = meta.get("general.architecture")
    arch = next((a for a, n in gguf.MODEL_ARCH_NAMES.items() if n == arch_name), None)
    if arch is None:
        return {}
    blocks = int(meta.get(f"{arch_name}.block_count", 0) or 0)
    tmap = gguf.get_tensor_name_map(arch, blocks)
    gshape = gguf_names if isinstance(gguf_names, dict) else {}
    hshape = hf_candidates if isinstance(hf_candidates, dict) else {}
    have = set(gguf_names)

    def lookup(name):
        return tmap.get_name(name, try_suffixes=(".weight", ".bias"))

    out, later = {}, []
    for hf in hf_candidates:
        g = lookup(hf)
        if g and g in have:
            if g not in out:
                out[g] = hf
        elif gshape and hshape:
            later.append(hf)
    stripped: dict[str, list[str]] = {}
    for hf in later:
        parts = hf.split(".")
        for k in range(1, len(parts) - 1):
            rest = ".".join(parts[k:])
            g = next((x for x in (lookup(rest), lookup("model." + rest)) if x and x in have), None)
            if g:
                if tuple(gshape[g][1]) == tuple(hshape[hf][1]):
                    stripped.setdefault(g, []).append(hf)
                break
    for g, cands in stripped.items():
        if g not in out and len(cands) == 1:
            out[g] = cands[0]
    return out
