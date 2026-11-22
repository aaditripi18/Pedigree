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

from . import formats, ggufremote, hub, torchzip
from .align import align
from .remote import ITEMSIZE, read_header, resolve_url, tensor_bytes

FLOATS = ("F64", "F32", "F16", "BF16")
OTHER_FORMATS = (".onnx", ".npz", ".nemo", ".pt", ".pth", ".ckpt", ".tflite", ".litertlm",
                 ".pdiparams")   # amendment 2026-11-20 (R1)
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
    return np.frombuffer(raw, dtype=NP_DTYPE[dtype]).astype(np.float64)


NP_DTYPE = {"F64": "<f8", "F32": "<f4", "F16": "<f2", "I64": "<i8", "I32": "<i4", "I16": "<i2", "I8": "i1",
            "U64": "<u8", "U32": "<u4", "U16": "<u2", "U8": "u1", "BOOL": "u1"}


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
        other = [] if files or gguf else [f for f in res.files if f.lower().endswith(OTHER_FORMATS)]
        if not files and not gguf and not other:
            raise hub.HubError(f"{res.repo}: no safetensors, PyTorch or GGUF weights")
        r = cls(res.repo, res.commit)
        for f in other:                  # amendment 2026-11-20 (R1): public formats beyond the three
            r._add_other(f, res.sizes[f])
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
                if formats.is_ct2(url):          # a CTranslate2 model.bin (faster-whisper and similar)
                    for name, (u, s) in formats.ct2_header(url, res.sizes[f]).items():
                        r.tensors[name] = (u, 0, s)
                    continue
                start, header = torchzip.read_header(url, res.sizes[f])
            else:
                start, header = read_header(url)
            for name, s in header.items():
                if name != "__metadata__" and isinstance(s, dict):
                    r.tensors[name] = (url, start, s)
        r._find_quantised()
        return r

    def _add_other(self, f: str, size: int) -> None:
        url = resolve_url(self.repo, f, self.commit)
        self.files.append(f)
        low = f.lower()
        if low.endswith(".onnx"):
            enc = formats.READERS_V5 and f.rsplit("/", 1)[-1].startswith("encoder_model")
            header, deq = formats.onnx_header(self.repo, self.commit, f, size, prefix="encoder." if enc else "")
            for name, (u, s) in header.items():
                self.tensors[name] = (u, 0, s)
            for x, (scale, zero, axis) in deq.items():
                if isinstance(axis, tuple) and axis[0] == "nbits" and scale in header:
                    _, k, n, bits, block = axis           # onnxruntime MatMulNBits: B is [N, K/block, blob]
                    self.quant[x] = {"kind": "nbits", "shape": [n, k], "scale": scale,
                                     "zero": zero if zero in header else None, "bits": bits, "block": block}
                elif scale in header:
                    self.quant[x] = {"kind": "affine", "shape": header[x][1]["shape"], "scale": scale,
                                     "zero": zero if zero in header else None, "axis": axis}
            return
        if low.endswith((".npz", ".tflite", ".litertlm", ".pdiparams")):
            read = {".npz": formats.npz_header, ".tflite": formats.tflite_header,
                    ".litertlm": formats.litertlm_header, ".pdiparams": formats.pdiparams_header}
            for name, (u, s) in read["." + low.rsplit(".", 1)[-1]](url, size).items():
                self.tensors[name] = (u, 0, s)
                if s.get("affine"):            # TFLite integer weights with their scales
                    self.quant[name] = {"kind": "inline", "shape": s["shape"], **s["affine"]}
            return
        if low.endswith(".nemo"):
            from .remote import fetch_range
            if fetch_range(url, 0, 1) == b"\x1f\x8b":          # gzip-compressed (older NeMo)
                local = formats.gz_member(url, "model_weights.ckpt")
                import os
                _, header = torchzip.read_header(local, os.path.getsize(local))
                url = local
            else:
                members = formats.tar_members(url, size)
                ckpt = [m for m in members if m.endswith("model_weights.ckpt")]
                if not ckpt:
                    raise ValueError("a .nemo archive without model_weights.ckpt")
                off, n = members[ckpt[0]]
                _, header = torchzip.read_header(url, n, base=off)
        else:                            # .pt / .pth / .ckpt: PyTorch, zip or legacy
            _, header = torchzip.read_header(url, size)
        for name, s in header.items():
            if isinstance(s, dict):
                self.tensors[name] = (url, 0, s)

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
        for n, (_, _, s) in t.items():   # CTranslate2 int8: weight * scale per row was rounded
            if s["dtype"] == "I8" and n.endswith("/weight") and n + "_scale" in t and n not in self.quant:
                self.quant[n] = {"kind": "ct2", "shape": s["shape"], "scale": n + "_scale"}
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
            h.update(x for x in (q.get("scales"), q.get("biases"), q.get("scale"), q.get("zero"))
                     if isinstance(x, str))      # inline TFLite scales are values, not tensor names
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
        if q["kind"] == "nbits":       # amendment 2026-11-21, reader (viii): onnxruntime's MatMulNBits
            n, k = q["shape"]
            bits, block = q["bits"], q["block"]
            if bits not in (4, 8):
                raise ValueError(f"MatMulNBits with {bits} bits not read")
            nb, blob = -(-k // block), block * bits // 8
            rows = min(n, -(-elements // k))
            b = self._raw(name, rows * nb * blob).astype(np.uint8).reshape(rows, nb, blob)
            v = np.stack([b & 15, b >> 4], -1).reshape(rows, nb, block) if bits == 4 else b
            sc = self._raw(q["scale"], rows * nb).reshape(rows, nb, 1)
            if q["zero"] is None:
                zp = float(2 ** (bits - 1))
            elif self.tensors[q["zero"]][2]["dtype"] == "U8":
                per = -(-nb * bits // 8)                   # packed, low nibble first, per row
                z = self._raw(q["zero"], rows * per).astype(np.uint8).reshape(rows, per)
                if bits == 4:
                    z = np.stack([z & 15, z >> 4], -1).reshape(rows, -1)
                zp = z[:, :nb].astype(np.float64).reshape(rows, nb, 1)
            else:
                zp = self._raw(q["zero"], rows * nb).reshape(rows, nb, 1)
            w = (v.astype(np.float64) - zp) * sc
            return w.reshape(rows, nb * block)[:, :k].reshape(-1)[:elements]
        if q["kind"] == "inline":
            shape = q["shape"]
            rows_total = shape[0] if shape else 1
            cols = int(np.prod(shape[1:])) if len(shape) > 1 else 1
            rows = min(rows_total, -(-elements // cols))
            x = self._raw(name, rows * cols).astype(np.float64)
            sc, zp = np.asarray(q["scale"], np.float64), np.asarray(q["zero"], np.float64)
            if sc.size == 1:
                return ((x - zp.reshape(-1)[0]) * sc[0])[:elements]
            if q["axis"] == 0 and sc.size == rows_total:
                x = x.reshape(rows, cols)
                z = zp[:rows].reshape(rows, 1) if zp.size == sc.size else 0.0
                return ((x - z) * sc[:rows].reshape(rows, 1)).reshape(-1)[:elements]
            if q["axis"] == len(shape) - 1 and sc.size == shape[-1]:
                x = x.reshape(-1, shape[-1])
                z = zp.reshape(1, -1) if zp.size == sc.size else 0.0
                return ((x - z) * sc.reshape(1, -1)).reshape(-1)[:elements]
            raise ValueError(f"TFLite quantisation layout of {name} not read")
        if q["kind"] in ("affine", "ct2"):
            shape = q["shape"]
            rows_total = shape[0] if shape else 1
            cols = int(np.prod(shape[1:])) if len(shape) > 1 else 1
            rows = min(rows_total, -(-elements // cols))
            x = self._raw(name, rows * cols).astype(np.float64).reshape(rows, cols)
            sshape = self.tensors[q["scale"]][2]["shape"]
            nsc = int(np.prod(sshape)) if sshape else 1
            if q["kind"] == "ct2":                       # one scale per output row: w = q / scale
                sc = self._raw(q["scale"], rows).reshape(rows, 1) if nsc == rows_total else self._raw(q["scale"], 1)
                return (x / sc).reshape(-1)[:elements]
            zp = 0.0
            if nsc == 1:                                 # per tensor (ONNX DequantizeLinear)
                sc = self._raw(q["scale"], 1)[0]
                if q["zero"]:
                    zp = self._raw(q["zero"], 1)[0]
            elif q["axis"] in (0, -len(shape)) and nsc == rows_total:
                sc = self._raw(q["scale"], rows).reshape(rows, 1)
                if q["zero"]:
                    zp = self._raw(q["zero"], rows).reshape(rows, 1)
            elif len(shape) == 2 and nsc == shape[1]:    # per column
                sc = self._raw(q["scale"], cols).reshape(1, cols)
                if q["zero"]:
                    zp = self._raw(q["zero"], cols).reshape(1, cols)
            else:
                raise ValueError(f"DequantizeLinear layout of {name} not read")
            return ((x - zp) * sc).reshape(-1)[:elements]
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

    def sample_transposed(self, name: str, max_bytes: int = SAMPLE_BYTES) -> np.ndarray:
        """The leading values of a 2-D tensor's transpose (the whole tensor is read)."""
        shape = self.float_shapes()[name][1]
        rows, cols = shape
        if name in self.quant:
            full = self._dequantised(name, rows * cols)
        else:
            url, start, s = self.tensors[name]
            if s["dtype"] == "GGUF":
                raise ValueError("a GGUF tensor is not read transposed")
            full = decode(s["dtype"], tensor_bytes(url, start, s, TRANSPOSE_MAX_BYTES))
            if full.size < rows * cols:
                raise ValueError(f"{name} is larger than a transposed read allows")
        return full.reshape(rows, cols).T.reshape(-1)[:max_bytes // 4]

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
    """A model seen under other tensor names (a GGUF file under Hugging Face names); tensors in
    `transposed` are seen as their transposes (alignment v3 (d))."""

    def __init__(self, model, mapping: dict[str, str], transposed: set | None = None, shapes: dict | None = None):
        self.model, self.repo, self.commit = model, model.repo, model.commit
        self.to_old = {new: old for old, new in mapping.items()}
        self.transposed = transposed or set()
        self.shapes = shapes or {}      # v4 (e): a tensor seen with its partner's shape (size-1 dims only)

    def float_shapes(self):
        fs = self.model.float_shapes()
        out = {}
        for new, old in self.to_old.items():
            if old in fs:
                kind, shape = fs[old]
                if new in self.shapes:
                    out[new] = (kind, tuple(self.shapes[new]))
                else:
                    out[new] = (kind, tuple(shape)[::-1]) if old in self.transposed else fs[old]
        return out

    def sample(self, name: str, max_bytes: int = SAMPLE_BYTES) -> np.ndarray:
        old = self.to_old[name]
        if old in self.transposed:
            return self.model.sample_transposed(old, max_bytes)
        return self.model.sample(old, max_bytes)

    def sample_transposed(self, name: str, max_bytes: int = SAMPLE_BYTES) -> np.ndarray:
        old = self.to_old[name]
        if old in self.transposed:          # the transpose of a transposed view: the tensor as stored
            return self.model.sample(old, max_bytes)
        return self.model.sample_transposed(old, max_bytes)


TRANSPOSE_MAX_BYTES = 64 * 1024 * 1024     # a transposed tensor is read whole; larger ones are left out


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


def _big(shapes: dict):
    return lambda names: sum(int(np.prod(shapes[n][1])) >= MIN_ELEMENTS for n in names if n in shapes)


def candidates_v4(model, other, alignment: str) -> list[tuple]:
    """[(model, other, label)] under alignment v4 (amendment 2026-11-20 (2), H25): names if they
    pair >= STRUCT_BELOW large tensors ("auto4"), else v4's candidates, one per depth map (c4).
    "struct4" ignores names. Counts are of tensors with >= MIN_ELEMENTS values."""
    from .structalign import names_v4

    def g(m):
        return getattr(m, "gguf_meta", None)
    ms, os_ = model.float_shapes(), other.float_shapes()
    if g(model) and not g(other):
        names = {} if alignment == "struct4" else ggufremote.hf_names(model.gguf_meta, ms, os_)
        flip, base = False, 0
    elif g(other) and not g(model):
        names = {} if alignment == "struct4" else ggufremote.hf_names(other.gguf_meta, os_, ms)
        flip, base = True, 0
    else:
        pairing, _, _ = align(ms, os_)
        if alignment == "auto4" and _big(ms)(pairing) >= STRUCT_BELOW:
            return [(model, other, "names")]
        names, flip, base = {}, False, (0 if alignment == "struct4" else _big(ms)(pairing))
    a, b = (other, model) if flip else (model, other)
    sa, sb = a.float_shapes(), b.float_shapes()
    if names and _big(sa)(names) >= STRUCT_BELOW:
        return [(model, Renamed(other, names), "names") if flip else (Renamed(model, names), other, "names")]
    out = []
    for label, mp, tr in names_v4(sa, sb, _big(sa)):
        if _big(sa)(mp) <= max(_big(sa)(names), base):
            continue
        shapes = {y: sb[y][1] for x, y in mp.items() if x not in tr and tuple(sa[x][1]) != tuple(sb[y][1])}
        r = Renamed(a, mp, tr, shapes)
        out.append((model, r, label) if flip else (r, other, label))
    if out:
        return out
    if names:
        return [(model, Renamed(other, names), "names") if flip else (Renamed(model, names), other, "names")]
    return [(model, other, "names")]


def harmonise(model, other, alignment: str = "names"):
    """Give a GGUF model the tensor names of the other (non-GGUF) model. With alignment "auto3"
    (names, else v3) or "struct3" (v3, names ignored), any two models whose names pair fewer than
    STRUCT_BELOW tensors are aligned by v3 (amendment 2026-11-20, H23)."""
    def g(m):
        return getattr(m, "gguf_meta", None)
    if alignment in ("auto3", "struct3"):
        from .structalign import names_v3
        ms, os_ = model.float_shapes(), other.float_shapes()
        if g(model) and not g(other):
            names = {} if alignment == "struct3" else ggufremote.hf_names(model.gguf_meta, ms, os_)
            flip = False
        elif g(other) and not g(model):
            names = {} if alignment == "struct3" else ggufremote.hf_names(other.gguf_meta, os_, ms)
            flip = True
        else:
            pairing, _, _ = align(ms, os_)
            if alignment == "auto3" and len(pairing) >= STRUCT_BELOW:
                return model, other
            names, flip = {}, False
        if len(names) >= STRUCT_BELOW:
            return (model, Renamed(other, names)) if flip else (Renamed(model, names), other)
        a, b = (other, model) if flip else (model, other)
        v3, tr = names_v3(a.float_shapes(), b.float_shapes())
        if len(v3) > len(names) and len(v3) > (0 if (g(model) or g(other)) else len(align(ms, os_)[0])):
            names, tr = v3, tr
        else:
            tr = set()
        if not names:
            return model, other
        return (model, Renamed(other, names, tr)) if flip else (Renamed(model, names, tr), other)
    if g(model) and not g(other):
        model = Renamed(model, _gguf_mapping(model, other, alignment))
    elif g(other) and not g(model):
        other = Renamed(other, _gguf_mapping(other, model, alignment))
    return model, other


ORIENT_MIN = 2      # (j): non-square evidence needed before a side's (v) weights are read untransposed


def orient(model, other):
    """(j) (amendment 2026-11-22) Rule (v) reads an ONNX MatMul weight as its transpose, the layout
    of a PyTorch Linear. A model built from Conv1D layers (GPT-2) stores [in, out] already, so its
    weights were turned the wrong way, and square ones then compared transposed. For each side with
    (v)-transposed tensors: among the non-square ones whose name the other side holds, if more match
    the other side's shape untransposed than transposed (at least ORIENT_MIN), all of that side's
    (v) tensors are read untransposed."""
    if not formats.READERS_V5:
        return model, other
    from .structalign import _strip

    def flip(x, y):
        t = {n for n, v in (getattr(x, "tensors", None) or {}).items() if v[2].get("v5_transposed")}
        if not t:
            return x
        sx, sy = x.float_shapes(), y.float_shapes()
        by = {_strip(n): n for n in sy}
        rev = ok = 0
        for n in t:
            sh = tuple(sx[n][1])
            if len(sh) != 2 or sh[0] == sh[1]:
                continue
            m = n if n in sy else by.get(_strip(n))
            if m is None:
                continue
            rev += tuple(sy[m][1]) == sh[::-1]
            ok += tuple(sy[m][1]) == sh
        return Renamed(x, {n: n for n in sx}, t) if rev > ok and rev >= ORIENT_MIN else x
    model, other = flip(model, other), flip(other, model)
    return orient_gguf(model, other)


def orient_gguf(model, other):
    """(k) (amendment 2026-11-22 (3)) llama.cpp writes GPT-2's Conv1D weights as a PyTorch Linear
    stores them, [out, in]; the Hugging Face file keeps [in, out]. Names aside, an MLP's up and down
    projections then have each other's shapes, and square weights compare transposed. When exactly
    one side is GGUF: of the other side's non-square 2-D tensors, count those whose shape the GGUF
    side holds as stored and those it holds reversed. If reversed wins (at least ORIENT_MIN), that
    side is seen in Linear layout: its square 2-D tensors and its non-square ones held reversed are
    read transposed. Shapes alone decide, so the rule is the same with names used or ignored."""
    from . import units
    if not units.READERS_V6:
        return model, other

    def g(m):
        return getattr(m, "gguf_meta", None)
    if bool(g(model)) == bool(g(other)):
        return model, other
    gg, h = (model, other) if g(model) else (other, model)
    gs = {tuple(v[1]) for v in gg.float_shapes().values() if len(v[1]) == 2}
    hs = {n: tuple(v[1]) for n, v in h.float_shapes().items() if len(v[1]) == 2}
    ok = sum(sh in gs for sh in hs.values() if sh[0] != sh[1])
    rev = sum(sh[::-1] in gs for sh in hs.values() if sh[0] != sh[1])
    if not (rev > ok and rev >= ORIENT_MIN):
        return model, other
    t = {n for n, sh in hs.items() if sh[0] == sh[1] or sh[::-1] in gs}
    r = Renamed(h, {n: n for n in h.float_shapes()}, t)
    return (model, r) if g(model) else (r, other)


def compare(model: Remote, other: Remote, n_tensors: int = N_TENSORS, alignment: str = "names") -> dict:
    """Name/shape alignment and sampled weight similarity of two models. Under v4 ("auto4",
    "struct4") each depth-map candidate is scored and the highest median kept (c4)."""
    model, other = orient(model, other)
    if alignment in ("auto4", "struct4"):
        best, scores = None, {}
        for m, o, label in candidates_v4(model, other, alignment):
            r = _compare_aligned(m, o, n_tensors)
            r["alignment_map"] = label
            scores[label] = r["median_cosine"]
            if best is None or (r["median_cosine"] is not None and
                                (best["median_cosine"] is None or r["median_cosine"] > best["median_cosine"])):
                best = r
        best["map_scores"] = scores
        return best
    model, other = harmonise(model, other, alignment)
    return _compare_aligned(model, other, n_tensors)


def _compare_aligned(model, other, n_tensors: int) -> dict:
    ms, os_ = model.float_shapes(), other.float_shapes()
    pairing, dm, do = align(ms, os_)
    match = len(pairing) / max(len(ms), 1)
    big = [n for n in sorted(pairing) if int(np.prod(ms[n][1])) >= MIN_ELEMENTS]
    names = _spaced(big, n_tensors)
    cos, kurt, identical = {}, {}, bool(names)
    for n in names:
        try:
            a, b = model.sample(n), other.sample(pairing[n])
        except ValueError as e:
            if "transposed" not in str(e):        # only a v3 transposed read may be left out
                raise
            continue
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
