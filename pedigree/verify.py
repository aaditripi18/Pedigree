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

from . import hub, torchzip
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
        """spec: hf:org/name[@revision]. Safetensors weights, or PyTorch zip checkpoints
        (read without unpickling code, see torchzip) where a model has no safetensors."""
        res = hub.resolve(spec if spec.startswith("hf:") else "hf:" + spec)
        files = [f for f in res.files if f.endswith((".safetensors", ".bin"))]
        if not files:
            raise hub.HubError(f"{res.repo}: no safetensors or PyTorch weights")
        r = cls(res.repo, res.commit)
        for f in files:
            url = resolve_url(res.repo, f, res.commit)
            if f.endswith(".bin"):
                start, header = torchzip.read_header(url, res.sizes[f])
            else:
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


# --- permutation-invariant variant (E3) -----------------------------------------------------
# A function-preserving permutation reorders one axis of a weight matrix (the rows of a
# feed-forward up-projection, the columns of its down-projection). Gram matrices over the
# other axis do not change: (PW)'(PW) = W'W. Their diagonal (squared norms) is large and
# alike for any two trained models, so only off-diagonal entries are compared. Both axes
# are tried and the larger cosine kept, since which axis was permuted is unknown.

INVARIANT_TENSORS = 12
INVARIANT_MAX_SIDE = 4096


def full(m, name: str) -> np.ndarray:
    """Every value of one float tensor, with its shape."""
    shape = m.float_shapes()[name][1]
    n = int(np.prod(shape))
    return m.sample(name, max_bytes=n * 4)[:n].reshape(shape)


def _offdiag_gram(w: np.ndarray) -> np.ndarray:
    g = w.T @ w
    return g[~np.eye(g.shape[0], dtype=bool)]


def gram_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Permutation-invariant similarity of two same-shape matrices (see above)."""
    a, b = a.reshape(a.shape[0], -1), b.reshape(b.shape[0], -1)
    best = float("nan")
    for x, y in ((a, b), (a.T, b.T)):
        if x.shape[1] > INVARIANT_MAX_SIDE:
            continue
        c = cosine(_offdiag_gram(x), _offdiag_gram(y))
        best = c if best != best else max(best, c)
    return best


def compare_invariant(model, other, n_tensors: int = INVARIANT_TENSORS) -> dict:
    """Median permutation-invariant similarity over spaced matrices both models share."""
    ms, os_ = model.float_shapes(), other.float_shapes()
    pairing, _, _ = align(ms, os_)
    mats = [n for n in sorted(pairing) if len(ms[n][1]) >= 2 and int(np.prod(ms[n][1])) >= MIN_ELEMENTS]
    names = _spaced(mats, n_tensors)
    cos = {n: gram_cosine(full(model, n), full(other, pairing[n])) for n in names}
    vals = [c for c in cos.values() if c == c]
    return {"compared": len(vals), "cosines": cos, "median_invariant": float(np.median(vals)) if vals else None}
