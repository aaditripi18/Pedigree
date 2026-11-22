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

import re
import struct

import numpy as np

from .torchzip import RangeFile

SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
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
        self.fh = fh

    def take(self, fmt: str):
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
    version, n = struct.unpack("<IQ", head[4:16])
    if version < 2:
        raise ValueError(f"GGUF version {version} is not supported")
    return int(n)


def tensor_count(url: str, size: int) -> tuple[int, int | None]:
    """(tensors in this file, tensors in the whole split set or None), from the header and metadata
    alone (tensor descriptions are not read)."""
    fh = RangeFile(url, size)
    r = _Reader(fh)
    if fh.read(4) != b"GGUF":
        raise ValueError("not a GGUF file")
    (version,) = r.take("<I")
    if version < 2:
        raise ValueError(f"GGUF version {version} is not supported")
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
    (version,) = r.take("<I")
    if version < 2:
        raise ValueError(f"GGUF version {version} is not supported")
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
    tensors = {name: {"dtype": "GGUF", "qtype": qtype, "shape": list(reversed(dims)), "start": start + offset}
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
    if plain:                       # integer and float64 tensors are stored as they are, not quantised
        raw = fetch_range(url, spec["start"], spec["start"] + n * type_size - 1)
        return np.frombuffer(raw, dtype=plain).astype(np.float64)[:n]
    blocks = -(-n // block)
    raw = fetch_range(url, spec["start"], spec["start"] + blocks * type_size - 1)
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
