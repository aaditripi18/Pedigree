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

from . import ggufremote, hub, torchzip
from .align import align
from .remote import ITEMSIZE, read_header, resolve_url, tensor_bytes

FLOATS = ("F64", "F32", "F16", "BF16")
FP8 = ("F8_E4M3", "F8_E5M2")


def _fp8_table(exp_bits: int, man_bits: int, bias: int, e4m3: bool) -> np.ndarray:
    """All 256 values of an 8-bit float format (OCP FP8: E4M3 has no infinities)."""
    out = np.empty(256)
    for b in range(256):
        sign = -1.0 if b & 0x80 else 1.0
        e, m = (b >> man_bits) & ((1 << exp_bits) - 1), b & ((1 << man_bits) - 1)
        if e4m3 and e == 15 and m == 7:
            out[b] = np.nan
        elif not e4m3 and e == 31:
            out[b] = np.nan if m else sign * np.inf
        elif e == 0:
            out[b] = sign * m / (1 << man_bits) * 2.0 ** (1 - bias)
        else:
            out[b] = sign * (1 + m / (1 << man_bits)) * 2.0 ** (e - bias)
    return out


FP8_TABLES = {"F8_E4M3": _fp8_table(4, 3, 7, True), "F8_E5M2": _fp8_table(5, 2, 15, False),
              # OCP Microscaling (MX) v1.0 scale format: 8 exponent bits, no sign or mantissa,
              # value 2^(e - 127), 0xFF is NaN. Stored as the block scales of FP8 weights (DeepSeek-V4).
              "F8_E8M0": np.where(np.arange(256) == 255, np.nan, np.exp2(np.arange(256, dtype=np.float64) - 127))}
SAMPLE_BYTES = 32 * 1024
N_TENSORS = 24
MIN_ELEMENTS = 4096


def decode(dtype: str, raw: bytes) -> np.ndarray:
    """Float values (float64) of a little-endian tensor slice."""
    if dtype == "BF16":
        u = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
        return u.view(np.float32).astype(np.float64)
    if dtype in FP8_TABLES:
        return FP8_TABLES[dtype][np.frombuffer(raw, dtype=np.uint8)]
    return np.frombuffer(raw, dtype={"F64": "<f8", "F32": "<f4", "F16": "<f2"}[dtype]).astype(np.float64)


@dataclass
class Remote:
    repo: str
    commit: str
    tensors: dict = field(default_factory=dict)   # name -> (url, data_start, spec)
    files: list = field(default_factory=list)
    gguf_meta: dict | None = None
    quant: dict = field(default_factory=dict)     # weight name -> how to dequantise it (MLX, FP8)

    @classmethod
    def open(cls, spec: str) -> "Remote":
        """spec: hf:org/name[@revision]. Safetensors weights, or PyTorch zip checkpoints
        (read without unpickling code, see torchzip) where a model has no safetensors."""
        res = hub.resolve(spec if spec.startswith("hf:") else "hf:" + spec)
        return cls.from_files(res.repo, res.commit, res.files, res.sizes)

    @classmethod
    def from_files(cls, repo: str, commit: str, names: list[str], sizes: dict[str, int]) -> "Remote":
        """A model made of the given weight files of a repository at a commit."""
        from types import SimpleNamespace
        res = SimpleNamespace(repo=repo, commit=commit, files=names, sizes=sizes)
        files = [f for f in res.files if f.endswith((".safetensors", ".bin"))]
        gguf = [] if files else ggufremote.choose_variant(res.files)
        if not files and not gguf:
            raise hub.HubError(f"{res.repo}: no safetensors, PyTorch or GGUF weights")
        r = cls(res.repo, res.commit)
        for f in gguf:
            url = resolve_url(res.repo, f, res.commit)
            meta, header = ggufremote.read_header(url, res.sizes[f])
            r.gguf_meta = meta
            r.files.append(f)
            for name, s in header.items():
                r.tensors[name] = (url, 0, s)
        for f in files:
            url = resolve_url(res.repo, f, res.commit)
            r.files.append(f)
            if f.endswith(".bin"):
                start, header = torchzip.read_header(url, res.sizes[f])
            else:
                start, header = read_header(url)
            for name, s in header.items():
                if name != "__metadata__" and isinstance(s, dict):
                    r.tensors[name] = (url, start, s)
        r._find_quantised()
        return r

    def _config(self) -> dict:
        try:
            with hub._open(f"{hub.HOST}/{self.repo}/resolve/{self.commit}/config.json") as f:
                import json
                return json.loads(f.read().decode("utf-8"))
        except Exception:
            return {}

    def _find_quantised(self) -> None:
        """MLX affine-quantised weights (packed U32 + .scales/.biases) and FP8 weights with
        their scales are read as the float values they stand for (see _dequantised)."""
        t = self.tensors
        mlx = [n for n, (_, _, s) in t.items() if s["dtype"] == "U32" and n.endswith(".weight")
               and n[:-7] + ".scales" in t and n[:-7] + ".biases" in t]
        fp8 = [n for n, (_, _, s) in t.items() if s["dtype"] in FP8]
        if not mlx and not fp8:
            return
        cfg = self._config()
        q = cfg.get("quantization") or cfg.get("quantization_config") or {}
        def plain_path(k: str) -> tuple:   # vision-language exports reorder "model" and "language_model"
            return tuple(x for x in k.split(".") if x not in ("model", "language_model"))
        layers = {plain_path(k): v for k, v in q.items() if isinstance(v, dict)}
        for n in mlx:
            # mixed precision: MLX writes per-layer exceptions under the layer's path
            own = q.get(n[:-7]) if isinstance(q.get(n[:-7]), dict) else layers.get(plain_path(n[:-7]), {})
            bits, gs = int(own.get("bits", q.get("bits", 4))), int(own.get("group_size", q.get("group_size", 64)))
            shape = list(t[n][2]["shape"])           # (..., packed columns); MoE experts add axes
            groups = t[n[:-7] + ".scales"][2]["shape"][-1]
            if shape[-1] * 32 != groups * gs * bits and (shape[-1] * 32) % (groups * gs) == 0:
                bits = shape[-1] * 32 // (groups * gs)   # the stored shapes decide when the config disagrees
            self.quant[n] = {"kind": "mlx", "bits": bits, "group": gs,
                             "shape": tuple(shape[:-1] + [shape[-1] * 32 // bits]),
                             "scales": n[:-7] + ".scales", "biases": n[:-7] + ".biases"}
        block = q.get("weight_block_size")
        for n in fp8:
            shape = tuple(t[n][2]["shape"])
            stem = n[:-6] if n.endswith("weight") else n + "_"
            for sname in (stem + "weight_scale_inv", stem + "weight_scale", stem + "scale"):
                if sname in t:
                    sshape = tuple(t[sname][2]["shape"])
                    self.quant[n] = {"kind": "fp8", "shape": shape, "scale": sname, "scale_shape": sshape,
                                     "block": tuple(block) if block and len(sshape) == 2 and len(shape) == 2
                                     and sshape != (shape[0], 1) else None}
                    break
            else:
                self.quant[n] = {"kind": "fp8", "shape": shape, "scale": None}

    def _helpers(self) -> set:
        h = set()
        for q in self.quant.values():
            h.update(x for x in (q.get("scales"), q.get("biases"), q.get("scale")) if x)
        return h

    def float_shapes(self) -> dict[str, tuple]:
        helpers = self._helpers()
        out = {n: ("float", tuple(s["shape"])) for n, (_, _, s) in self.tensors.items()
               if (s["dtype"] in FLOATS or s["dtype"] == "GGUF") and n not in helpers}
        out.update({n: ("float", tuple(q["shape"])) for n, q in self.quant.items()})
        return out

    def _raw(self, name: str, elements: int) -> np.ndarray:
        url, start, s = self.tensors[name]
        return decode(s["dtype"], tensor_bytes(url, start, s, elements * ITEMSIZE[s["dtype"]]))

    def _dequantised(self, name: str, elements: int) -> np.ndarray:
        q = self.quant[name]
        if q["kind"] == "mlx":
            rows_total, cols = int(np.prod(q["shape"][:-1])), q["shape"][-1]
        else:
            rows_total, cols = q["shape"][0], int(np.prod(q["shape"][1:]))
        rows = min(rows_total, -(-elements // cols))
        if q["kind"] == "mlx":
            url, start, s = self.tensors[name]
            words_per_row = s["shape"][-1]
            raw = tensor_bytes(url, start, s, rows * words_per_row * 4)
            # each row is one little-endian bit stream: value k holds bits k*b .. k*b+b-1, so 3- and
            # 6-bit values may straddle two words (4- and 8-bit ones never do)
            b = q["bits"]
            bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8).reshape(rows, words_per_row * 4), axis=1,
                                 bitorder="little")[:, :cols * b].reshape(rows, cols, b)
            vals = bits.astype(np.int64) @ (1 << np.arange(b, dtype=np.int64))
            ng = cols // q["group"]
            sc = self._raw(q["scales"], rows * ng).reshape(rows, ng)
            bi = self._raw(q["biases"], rows * ng).reshape(rows, ng)
            g = np.arange(cols) // q["group"]
            return (vals * sc[:, g] + bi[:, g]).reshape(-1)[:elements]
        x = self._raw(name, rows * cols).reshape(rows, cols)      # fp8
        if q["scale"] is None:
            return x.reshape(-1)[:elements]
        ss = q["scale_shape"]
        if q.get("block"):
            b0, b1 = q["block"]
            nb = ss[1]
            sc = self._raw(q["scale"], (-(-rows // b0)) * nb).reshape(-1, nb)
            x = x * sc[np.arange(rows) // b0][:, np.arange(cols) // b1]
        elif len(ss) >= 1 and int(np.prod(ss)) == rows_total:              # one scale per row
            x = x * self._raw(q["scale"], rows).reshape(rows, 1)
        # a single scale per tensor leaves every cosine unchanged
        return x.reshape(-1)[:elements]

    def sample(self, name: str, max_bytes: int = SAMPLE_BYTES) -> np.ndarray:
        if name in self.quant:
            return self._dequantised(name, max_bytes // 4)
        url, start, s = self.tensors[name]
        if s["dtype"] == "GGUF":
            return ggufremote.sample(url, s, max_bytes // 4)
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


class Renamed:
    """A model seen under other tensor names (a GGUF file under Hugging Face names)."""

    def __init__(self, model, mapping: dict[str, str]):
        self.model, self.repo, self.commit = model, model.repo, model.commit
        self.to_old = {new: old for old, new in mapping.items()}

    def float_shapes(self):
        fs = self.model.float_shapes()
        return {new: fs[old] for new, old in self.to_old.items() if old in fs}

    def sample(self, name: str, max_bytes: int = SAMPLE_BYTES) -> np.ndarray:
        return self.model.sample(self.to_old[name], max_bytes)


STRUCT_BELOW = 8     # structural alignment is tried when names pair fewer tensors (H17)


def _gguf_mapping(gm, hm, alignment: str) -> dict[str, str]:
    """{gguf name: hf name}. alignment: "names" (llama.cpp's table, as E2 ran), "auto" (names,
    else structural alignment when names pair fewer than STRUCT_BELOW tensors and it pairs more),
    "struct" (structural alignment only, names ignored: H17a's check), "struct2" (v2, stacks of
    unequal block counts: H20's check) or "auto2" (names, else v2)."""
    from .structalign import struct_names, struct_names_v2
    if alignment == "struct":
        return struct_names(gm.float_shapes(), hm.float_shapes())
    if alignment == "struct2":
        return struct_names_v2(gm.float_shapes(), hm.float_shapes())
    if alignment == "auto2":
        names = ggufremote.hf_names(gm.gguf_meta, gm.float_shapes(), hm.float_shapes())
        if len(names) < STRUCT_BELOW:
            st = struct_names_v2(gm.float_shapes(), hm.float_shapes())
            if len(st) > len(names):
                return st
        return names
    names = ggufremote.hf_names(gm.gguf_meta, gm.float_shapes(), hm.float_shapes())
    if alignment == "auto" and len(names) < STRUCT_BELOW:
        st = struct_names(gm.float_shapes(), hm.float_shapes())
        if len(st) > len(names):
            return st
    return names


def harmonise(model, other, alignment: str = "names"):
    """Give a GGUF model the tensor names of the other (non-GGUF) model."""
    def g(m):
        return getattr(m, "gguf_meta", None)
    if g(model) and not g(other):
        model = Renamed(model, _gguf_mapping(model, other, alignment))
    elif g(other) and not g(model):
        other = Renamed(other, _gguf_mapping(other, model, alignment))
    return model, other


def compare(model: Remote, other: Remote, n_tensors: int = N_TENSORS, alignment: str = "names") -> dict:
    """Name/shape alignment and sampled weight similarity of two models."""
    model, other = harmonise(model, other, alignment)
    ms, os_ = model.float_shapes(), other.float_shapes()
    pairing, dm, do = align(ms, os_)
    match = len(pairing) / max(len(ms), 1)
    big = [n for n in sorted(pairing) if int(np.prod(ms[n][1])) >= MIN_ELEMENTS]
    names = _spaced(big, n_tensors)
    cos, kurt, identical = {}, {}, bool(names)
    for n in names:
        a, b = model.sample(n), other.sample(pairing[n])
        cos[n] = cosine(a, b)
        kurt[n] = [excess_kurtosis(a), excess_kurtosis(b)]
        identical &= a.shape == b.shape and bool(np.array_equal(a, b))
    vals = [c for c in cos.values() if c == c]
    return {"model": model.repo, "model_commit": model.commit, "other": other.repo, "other_commit": other.commit,
            "float_tensors": len(ms), "matched": len(pairing), "match_share": match,
            "dropped_prefix": [dm, do], "compared": len(vals), "cosines": cos, "kurtosis": kurt, "identical": identical,
            "median_cosine": float(np.median(vals)) if vals else None}


def excess_kurtosis(x: np.ndarray) -> float:
    """Excess kurtosis of a tensor's values (H8's direction statistic, as in MoTHer:
    training adds outlying weights, so the child is expected to have the higher value)."""
    x = x[np.isfinite(x)]
    if x.size < 4:
        return float("nan")
    x = x - x.mean()
    v = (x ** 2).mean()
    return float((x ** 4).mean() / v ** 2 - 3) if v > 0 else float("nan")


def kurtosis_gap(kurt: dict) -> float | None:
    """Median over compared tensors of kurtosis(model) - kurtosis(other): positive means
    the model looks like the child (D2 in PLAN.md, amendment 2026-11-07)."""
    d = [a - b for a, b in kurt.values() if a == a and b == b]
    return float(np.median(d)) if d else None


def compare_whole(model, other) -> dict:
    """H6: the plain score over every aligned tensor with >= MIN_ELEMENTS values, each
    read in full (one tensor at a time, so memory stays at two tensors)."""
    model, other = harmonise(model, other)
    ms, os_ = model.float_shapes(), other.float_shapes()
    pairing, _, _ = align(ms, os_)
    big = [n for n in sorted(pairing) if int(np.prod(ms[n][1])) >= MIN_ELEMENTS]
    cos = {n: cosine(full(model, n).ravel(), full(other, pairing[n]).ravel()) for n in big}
    vals = [c for c in cos.values() if c == c]
    return {"compared": len(vals), "cosines": cos, "median_whole": float(np.median(vals)) if vals else None}


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
        if not files:
            raise FileNotFoundError(f"no safetensors weights in {path}")
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


def compare_invariant(model, other, n_tensors: int = INVARIANT_TENSORS, max_elements: int | None = None) -> dict:
    """Median permutation-invariant similarity over spaced matrices both models share
    (only matrices of at most max_elements values, when given: each is read in full)."""
    model, other = harmonise(model, other)
    ms, os_ = model.float_shapes(), other.float_shapes()
    pairing, _, _ = align(ms, os_)
    mats = [n for n in sorted(pairing) if len(ms[n][1]) >= 2 and MIN_ELEMENTS <= int(np.prod(ms[n][1]))
            and (max_elements is None or int(np.prod(ms[n][1])) <= max_elements)]
    names = _spaced(mats, n_tensors)
    cos = {n: gram_cosine(full(model, n), full(other, pairing[n])) for n in names}
    vals = [c for c in cos.values() if c == c]
    return {"compared": len(vals), "cosines": cos, "median_invariant": float(np.median(vals)) if vals else None}


class InMemory:
    """A model held as a state dict (torch tensors or arrays), read like Remote/Local."""

    def __init__(self, sd: dict, repo: str = "memory", commit: str = "memory"):
        self.repo, self.commit = repo, commit
        self.arrays = {}
        for k, v in sd.items():
            a = v.detach().float().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
            if np.issubdtype(a.dtype, np.floating):
                self.arrays[k] = a.astype(np.float64).ravel()
        self.shapes = {k: tuple(v.shape) for k, v in sd.items() if k in self.arrays}

    def float_shapes(self) -> dict[str, tuple]:
        return {k: ("float", s) for k, s in self.shapes.items()}

    def sample(self, name: str, max_bytes: int = SAMPLE_BYTES) -> np.ndarray:
        return self.arrays[name][: max_bytes // 4]
