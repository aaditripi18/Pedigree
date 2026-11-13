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
    from .remote import fetch_range
    try:
        qtype = gguf.GGMLQuantizationType(spec["qtype"])
    except ValueError:
        raise ValueError(f"GGUF tensor type {spec['qtype']} is not in llama.cpp's published list "
                         "(a fork's own format)") from None
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


def hf_names(meta: dict, gguf_names: list[str], hf_candidates: list[str]) -> dict[str, str]:
    """{gguf tensor name: Hugging Face name} for the HF names a GGUF file holds."""
    import gguf
    arch_name = meta.get("general.architecture")
    arch = next((a for a, n in gguf.MODEL_ARCH_NAMES.items() if n == arch_name), None)
    if arch is None:
        return {}
    blocks = int(meta.get(f"{arch_name}.block_count", 0) or 0)
    tmap = gguf.get_tensor_name_map(arch, blocks)
    have = set(gguf_names)
    out = {}
    for hf in hf_candidates:
        g = tmap.get_name(hf, try_suffixes=(".weight", ".bias"))
        if g and g in have and g not in out:
            out[g] = hf
    return out
