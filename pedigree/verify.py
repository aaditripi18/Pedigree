"""Compare two models' weights without downloading them.

A Remote is a model on the Hub pinned to a commit: its safetensors files (one, or
the shards an index lists), their headers, and range reads of any tensor. compare()
aligns the two models' float tensors by name and shape, samples the same leading
bytes of up to N of the paired tensors (spread through the network), decodes them
to float64 whatever the stored float type, and reports the cosine similarity per
tensor and its median. The decision rule (the threshold on the median) is chosen
on ground truth in E1, not here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import hub
from .align import align
from .remote import ITEMSIZE, read_header, resolve_url, tensor_bytes

FLOATS = ("F64", "F32", "F16", "BF16")
SAMPLE_BYTES = 32 * 1024
N_TENSORS = 24
MIN_ELEMENTS = 4096


def decode(dtype: str, raw: bytes) -> np.ndarray:
    """Float values (float64) of a little-endian tensor slice."""
    if dtype == "BF16":
        u = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
        return u.view(np.float32).astype(np.float64)
    return np.frombuffer(raw, dtype={"F64": "<f8", "F32": "<f4", "F16": "<f2"}[dtype]).astype(np.float64)


@dataclass
class Remote:
    repo: str
    commit: str
    tensors: dict = field(default_factory=dict)   # name -> (url, data_start, spec)

    @classmethod
    def open(cls, spec: str) -> "Remote":
        """spec: hf:org/name[@revision]. Only safetensors weights are read."""
        res = hub.resolve(spec if spec.startswith("hf:") else "hf:" + spec)
        files = [f for f in res.files if f.endswith(".safetensors")]
        if not files:
            raise hub.HubError(f"{res.repo}: no safetensors weights")
        r = cls(res.repo, res.commit)
        for f in files:
            url = resolve_url(res.repo, f, res.commit)
            start, header = read_header(url)
            for name, s in header.items():
                if name != "__metadata__" and isinstance(s, dict):
                    r.tensors[name] = (url, start, s)
        return r

    def float_shapes(self) -> dict[str, tuple]:
        return {n: ("float", tuple(s["shape"])) for n, (_, _, s) in self.tensors.items() if s["dtype"] in FLOATS}

    def sample(self, name: str, max_bytes: int = SAMPLE_BYTES) -> np.ndarray:
        url, start, s = self.tensors[name]
        # read the same number of *elements* whatever the stored width, so two
        # models stored in different float types are compared element for element
        elements = max_bytes // 4
        return decode(s["dtype"], tensor_bytes(url, start, s, elements * ITEMSIZE[s["dtype"]]))


def _spaced(items: list, k: int) -> list:
    if len(items) <= k:
        return list(items)
    idx = np.linspace(0, len(items) - 1, k).round().astype(int)
    return [items[i] for i in sorted(set(idx))]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    m = min(a.size, b.size)
    a, b = a[:m], b[:m]
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 0 else float("nan")


def compare(model: Remote, other: Remote, n_tensors: int = N_TENSORS) -> dict:
    """Name/shape alignment and sampled weight similarity of two models."""
    ms, os_ = model.float_shapes(), other.float_shapes()
    pairing, dm, do = align(ms, os_)
    match = len(pairing) / max(len(ms), 1)
    big = [n for n in sorted(pairing) if int(np.prod(ms[n][1])) >= MIN_ELEMENTS]
    names = _spaced(big, n_tensors)
    cos = {n: cosine(model.sample(n), other.sample(pairing[n])) for n in names}
    vals = [c for c in cos.values() if c == c]
    return {"model": model.repo, "model_commit": model.commit, "other": other.repo, "other_commit": other.commit,
            "float_tensors": len(ms), "matched": len(pairing), "match_share": match,
            "dropped_prefix": [dm, do], "compared": len(vals), "cosines": cos,
            "median_cosine": float(np.median(vals)) if vals else None}


@dataclass
class Local:
    """A safetensors model on disk (one file, or every *.safetensors in a folder), read
    with the same interface as Remote, so ground-truth derivatives are compared exactly
    like Hub models."""
    repo: str
    commit: str = "local"
    tensors: dict = field(default_factory=dict)   # name -> (path, data_start, spec)

    @classmethod
    def open(cls, path) -> "Local":
        import json
        import struct
        from pathlib import Path
        path = Path(path)
        files = sorted(path.glob("*.safetensors")) if path.is_dir() else [path]
        r = cls(str(path))
        for f in files:
            with open(f, "rb") as fh:
                (n,) = struct.unpack("<Q", fh.read(8))
                header = json.loads(fh.read(n))
            for name, s in header.items():
                if name != "__metadata__" and isinstance(s, dict):
                    r.tensors[name] = (f, 8 + n, s)
        return r

    def float_shapes(self) -> dict[str, tuple]:
        return {n: ("float", tuple(s["shape"])) for n, (_, _, s) in self.tensors.items() if s["dtype"] in FLOATS}

    def sample(self, name: str, max_bytes: int = SAMPLE_BYTES) -> np.ndarray:
        f, start, s = self.tensors[name]
        elements = max_bytes // 4
        b, e = s["data_offsets"]
        length = min(e - b, elements * ITEMSIZE[s["dtype"]]) // ITEMSIZE[s["dtype"]] * ITEMSIZE[s["dtype"]]
        with open(f, "rb") as fh:
            fh.seek(start + b)
            return decode(s["dtype"], fh.read(length))
