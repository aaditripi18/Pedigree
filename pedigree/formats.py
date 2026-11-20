"""Remote readers for the weight formats beyond safetensors, PyTorch and GGUF (PLAN.md, amendment
2026-11-20, R1). Each returns {tensor name: (url, spec)} with spec = {dtype, shape, data_offsets}
holding absolute byte offsets, so a tensor is sampled by HTTP range requests like any other.
Every layout here is taken from the format's public specification or reference writer, and
tests/test_formats.py checks each reader against that reference library on the same file.

  ONNX         onnx.proto (ModelProto.graph -> GraphProto.initializer, Constant nodes), external
               data files; DequantizeLinear inputs are recorded so int8 weights are read as values
  NumPy .npz   a zip of .npy members (numpy's format spec, NEP 1); stored members only
  CTranslate2  model.bin as ctranslate2.specs.model_spec._serialize writes it (binary version 6)
  tar          member offsets of an uncompressed tar (POSIX ustar), for NeMo's .nemo archives
  TFLite       schema.fbs (the `tflite` package's generated code gives the vtable slots); int8
               weights read with their scales and zero points
  LiteRT-LM    the .litertlm container (litert-lm-builder's peek tool and header schema): every
               TFLite section inside it, read in place
  Paddle       .pdiparams as save_combine writes it (LoDTensor records)

Nothing here runs code from the file: ONNX and CTranslate2 are walked as bytes, .npy headers
are parsed with ast.literal_eval, and archives are only indexed.
"""
from __future__ import annotations

import ast
import os
import struct
import zipfile

from .remote import fetch_range, resolve_url
from .torchzip import RangeFile

ONNX_DTYPE = {1: "F32", 2: "U8", 3: "I8", 4: "U16", 5: "I16", 6: "I32", 7: "I64", 9: "BOOL", 10: "F16",
              11: "F64", 12: "U32", 13: "U64", 16: "BF16", 17: "F8_E4M3", 19: "F8_E5M2"}
CT2_DTYPE = ("F32", "I8", "I16", "I32", "F16", "BF16")
NPY_DTYPE = {"<f4": "F32", "<f2": "F16", "<f8": "F64", "<i8": "I64", "<i4": "I32", "<i2": "I16", "|i1": "I8",
             "|u1": "U8", "<u4": "U32", "<u2": "U16", "<u8": "U64", "|b1": "BOOL"}


# --- ONNX ----------------------------------------------------------------------------------------

class _PB:
    """A protobuf wire-format walker over a RangeFile: payloads are located, never read unless asked."""

    def __init__(self, fh: RangeFile):
        self.fh = fh

    def varint(self) -> int:
        out = shift = 0
        while True:
            b = self.fh.read(1)
            if not b:
                raise ValueError("truncated protobuf")
            out |= (b[0] & 0x7F) << shift
            shift += 7
            if b[0] < 0x80:
                return out

    def fields(self, start: int, end: int):
        """(field number, wire type, value) for each field in [start, end); value is the varint,
        or (offset, length) of a length-delimited payload, or the offset of a fixed one."""
        pos = start
        while pos < end:
            self.fh.seek(pos)
            key = self.varint()
            f, wt = key >> 3, key & 7
            if wt == 0:
                v = self.varint()
                pos = self.fh.tell()
            elif wt == 2:
                n = self.varint()
                v = (self.fh.tell(), n)
                pos = v[0] + n
            elif wt == 1:
                v = self.fh.tell()
                pos = v + 8
            elif wt == 5:
                v = self.fh.tell()
                pos = v + 4
            else:
                raise ValueError(f"unsupported protobuf wire type {wt}")
            yield f, wt, v

    def bytes_at(self, span) -> bytes:
        self.fh.seek(span[0])
        return self.fh.read(span[1])

    def packed_varints(self, span) -> list[int]:
        out, end = [], span[0] + span[1]
        self.fh.seek(span[0])
        while self.fh.tell() < end:
            out.append(self.varint())
        return out


def _onnx_tensor(pb: _PB, span) -> dict:
    t = {"dims": [], "dtype": None, "name": "", "data": None, "external": {}, "location": 0}
    for f, wt, v in pb.fields(span[0], span[0] + span[1]):
        if f == 1:
            t["dims"] += pb.packed_varints(v) if wt == 2 else [v]
        elif f == 2:
            t["dtype"] = v
        elif f == 8:
            t["name"] = pb.bytes_at(v).decode("utf-8", "replace")
        elif f == 9:
            t["data"] = v
        elif f == 4 and wt == 2:           # float_data, packed: little-endian float32s
            t["data"], t["packed_float"] = v, True
        elif f == 13:
            kv = {}
            for g, _, w in pb.fields(v[0], v[0] + v[1]):
                kv[g] = pb.bytes_at(w).decode("utf-8", "replace")
            t["external"][kv.get(1, "")] = kv.get(2, "")
        elif f == 14:
            t["location"] = v
    return t


NODE_NAMES = os.environ.get("PEDIGREE_READERS_V4", "1") != "0"   # amendment 2026-11-20 (2), reader (ii)
ONNX_TRANSPOSE_MAX = 64 * 1024 * 1024


def _node_path(node: str) -> str | None:
    """'/encoder/layer.0/attention/query/MatMul' -> 'encoder.layer.0.attention.query.weight'."""
    parts = [p for p in node.strip("/").split("/") if p]
    if len(parts) < 2:
        return None
    return ".".join(parts[:-1]) + ".weight"


def _node_named(out: dict, uses: dict) -> dict:
    """Anonymous `onnx::` initializers used as the weight (input 1) of exactly one MatMul or Gemm
    node take that node's path; MatMul and Gemm without transB store [in, out], seen transposed."""
    new: dict = {}
    for name, use in uses.items():
        if not name.startswith("onnx::") or name not in out or len(use) != 1:
            continue
        op, node, idx, trans_b = use[0]
        path = _node_path(node) if idx == 1 else None
        u, spec = out[name]
        if path is None or path in out or len(spec["shape"]) != 2:
            continue
        if op == "MatMul" or (op == "Gemm" and not trans_b):
            nbytes = spec["data_offsets"][1] - spec["data_offsets"][0]
            if nbytes > ONNX_TRANSPOSE_MAX:
                continue
            spec = {**spec, "shape": list(spec["shape"])[::-1], "order": "F"}
        new.setdefault(path, []).append((name, (u, spec)))
    for path, items in new.items():
        if len(items) == 1:                        # a path two weights would share is left anonymous
            old, val = items[0]
            del out[old]
            out[path] = val
    return out


def onnx_header(repo: str, commit: str, path: str, size: int) -> tuple[dict, dict]:
    """({name: (url, spec)}, {quantised name: (scale name, zero-point name or None, axis)})."""
    url = resolve_url(repo, path, commit)
    pb = _PB(RangeFile(url, size))
    folder = path.rsplit("/", 1)[0] + "/" if "/" in path else ""
    tensors, consts, deq, uses = [], [], {}, {}
    for f, wt, v in pb.fields(0, size):
        if f != 7 or wt != 2:                      # ModelProto.graph
            continue
        for g, gwt, w in pb.fields(v[0], v[0] + v[1]):
            if g == 5 and gwt == 2:                # GraphProto.initializer
                tensors.append(_onnx_tensor(pb, w))
            elif g == 1 and gwt == 2:              # GraphProto.node
                ins, outs, op, attrs, node = [], [], "", [], ""
                for h, hwt, x in pb.fields(w[0], w[0] + w[1]):
                    if h == 1:
                        ins.append(pb.bytes_at(x).decode("utf-8", "replace"))
                    elif h == 2:
                        outs.append(pb.bytes_at(x).decode("utf-8", "replace"))
                    elif h == 3 and hwt == 2:
                        node = pb.bytes_at(x).decode("utf-8", "replace")
                    elif h == 4:
                        op = pb.bytes_at(x).decode("utf-8", "replace")
                    elif h == 5 and hwt == 2:
                        attrs.append(x)
                if NODE_NAMES and op in ("MatMul", "Gemm"):
                    trans_b = False
                    if op == "Gemm":
                        for a in attrs:
                            aname, aval = "", None
                            for k, kwt, y in pb.fields(a[0], a[0] + a[1]):
                                if k == 1:
                                    aname = pb.bytes_at(y).decode("utf-8", "replace")
                                elif k == 3 and kwt == 0:
                                    aval = y
                            if aname == "transB" and aval:
                                trans_b = True
                    for i, nm in enumerate(ins):
                        uses.setdefault(nm, []).append((op, node, i, trans_b))
                elif NODE_NAMES:
                    for nm in ins:
                        if nm.startswith("onnx::"):
                            uses.setdefault(nm, []).append((op, node, -1, False))
                if op == "Constant" and outs:
                    for a in attrs:
                        for k, kwt, y in pb.fields(a[0], a[0] + a[1]):
                            if k == 5 and kwt == 2:     # AttributeProto.t
                                c = _onnx_tensor(pb, y)
                                c["name"] = outs[0]
                                consts.append(c)
                elif op == "DequantizeLinear" and len(ins) >= 2:
                    axis = 1
                    for a in attrs:
                        aname, aval = "", None
                        for k, kwt, y in pb.fields(a[0], a[0] + a[1]):
                            if k == 1:
                                aname = pb.bytes_at(y).decode("utf-8", "replace")
                            elif k == 3 and kwt == 0:
                                aval = y if y < 2 ** 63 else y - 2 ** 64
                        if aname == "axis" and aval is not None:
                            axis = aval
                    deq[ins[0]] = (ins[1], ins[2] if len(ins) > 2 and ins[2] else None, axis)
    out = {}
    for t in tensors + consts:
        dtype = ONNX_DTYPE.get(t["dtype"])
        if not t["name"] or dtype is None:
            continue
        n = 1
        for d in t["dims"]:
            n *= d
        from .remote import ITEMSIZE
        nbytes = n * ITEMSIZE[dtype if not t.get("packed_float") else "F32"]
        if t["location"] == 1 or t["external"]:
            ext = t["external"]
            off, length = int(ext.get("offset", 0) or 0), int(ext.get("length", nbytes) or nbytes)
            u = resolve_url(repo, folder + ext["location"], commit)
            out[t["name"]] = (u, {"dtype": dtype, "shape": t["dims"], "data_offsets": [off, off + length]})
        elif t["data"] is not None:
            b, length = t["data"]
            if length != nbytes:
                continue                           # not the dense little-endian layout (e.g. strings)
            out[t["name"]] = (url, {"dtype": "F32" if t.get("packed_float") else dtype, "shape": t["dims"],
                                    "data_offsets": [b, b + length]})
    if NODE_NAMES:
        out = _node_named(out, uses)
    return out, {k: v for k, v in deq.items() if k in out}


# --- NumPy .npz ----------------------------------------------------------------------------------

def npz_header(url: str, size: int) -> dict:
    zf = zipfile.ZipFile(RangeFile(url, size))
    out = {}
    for info in zf.infolist():
        if not info.filename.endswith(".npy") or info.compress_type != zipfile.ZIP_STORED:
            continue
        local = fetch_range(url, info.header_offset, info.header_offset + 29)
        nlen, xlen = struct.unpack("<HH", local[26:30])
        start = info.header_offset + 30 + nlen + xlen
        head = fetch_range(url, start, start + 11)
        if head[:6] != b"\x93NUMPY":
            continue
        if head[6] == 1:
            (hlen,) = struct.unpack("<H", head[8:10])
            pre = 10
        else:
            (hlen,) = struct.unpack("<I", head[8:12])
            pre = 12
        d = ast.literal_eval(fetch_range(url, start + pre, start + pre + hlen - 1).decode("latin1"))
        dtype = NPY_DTYPE.get(d["descr"])
        if dtype is None:
            continue
        shape = list(d["shape"])
        from .remote import ITEMSIZE
        n = 1
        for s in shape:
            n *= s
        b = start + pre + hlen
        spec = {"dtype": dtype, "shape": shape, "data_offsets": [b, b + n * ITEMSIZE[dtype]]}
        if d.get("fortran_order") and len(shape) == 2:
            spec["order"] = "F"
        elif d.get("fortran_order"):
            continue
        out[info.filename[:-4]] = (url, spec)
    return out


# --- CTranslate2 ---------------------------------------------------------------------------------

def is_ct2(url: str) -> bool:
    """A CTranslate2 file starts with its binary version (5 or 6) and a NUL-terminated spec name;
    a PyTorch file starts with a zip signature or a pickle opcode (0x80)."""
    head = fetch_range(url, 0, 7)
    (version,) = struct.unpack("<I", head[:4])
    return version in (5, 6) and head[:2] != b"PK"


def ct2_header(url: str, size: int) -> dict:
    fh = RangeFile(url, size)

    def take(fmt):
        return struct.unpack("<" + fmt, fh.read(struct.calcsize("<" + fmt)))

    def string():
        (n,) = take("H")
        return fh.read(n)[:-1].decode("utf-8", "replace")
    (version,) = take("I")
    if version < 5 or version > 6:
        raise ValueError(f"CTranslate2 binary version {version} not read")
    string()                                   # spec name
    take("I")                                  # revision
    (count,) = take("I")
    out = {}
    for _ in range(count):
        name = string()
        (rank,) = take("B")
        shape = list(take("I" * rank)) if rank else []
        (tid,) = take("B")
        (nbytes,) = take("I")
        b = fh.tell()
        out[name] = (url, {"dtype": CT2_DTYPE[tid], "shape": shape, "data_offsets": [b, b + nbytes]})
        fh.seek(b + nbytes)
    return out


# --- tar (NeMo) ----------------------------------------------------------------------------------

GZ_MAX_BYTES = 8 * 1024 ** 3


def gz_member(url: str, suffix: str) -> str:
    """A gzip-compressed tar (older .nemo files) cannot be read by range: it is streamed once and
    the one member ending in `suffix` is written to the cache; returns its local path."""
    import hashlib
    import tarfile
    from .hub import _open, cache_root
    dest = cache_root() / "members" / (hashlib.sha256(f"{url}|{suffix}".encode()).hexdigest() + ".bin")
    if dest.exists():
        return str(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with _open(url, timeout=600) as r, tarfile.open(fileobj=r, mode="r|gz") as tf:
        seen = 0
        for m in tf:
            seen += m.size
            if seen > GZ_MAX_BYTES:
                raise ValueError("compressed archive larger than the reader allows")
            if m.isfile() and m.name.endswith(suffix):
                part = dest.with_suffix(".part")
                with tf.extractfile(m) as src, open(part, "wb") as out:
                    while True:
                        chunk = src.read(1 << 22)
                        if not chunk:
                            break
                        out.write(chunk)
                part.replace(dest)
                return str(dest)
    raise ValueError(f"no member ending in {suffix}")


def tar_members(url: str, size: int) -> dict[str, tuple[int, int]]:
    """{member name: (data offset, size)} of an uncompressed tar archive (ustar headers)."""
    out, pos, long_name = {}, 0, None
    while pos + 512 <= size:
        h = fetch_range(url, pos, pos + 511)
        if h == b"\0" * 512:
            break
        name = h[:100].rstrip(b"\0").decode("utf-8", "replace")
        prefix = h[345:500].rstrip(b"\0").decode("utf-8", "replace")
        n = int(h[124:136].rstrip(b"\0 ").decode() or "0", 8)
        kind = h[156:157]
        data = pos + 512
        if kind == b"L":                       # GNU long name: the next member's name
            long_name = fetch_range(url, data, data + n - 1).rstrip(b"\0").decode("utf-8", "replace")
        else:
            full = long_name or (f"{prefix}/{name}" if prefix else name)
            long_name = None
            if kind in (b"0", b"\0"):
                out[full.lstrip("./")] = (data, n)
        pos = data + (n + 511) // 512 * 512
    return out


# --- FlatBuffers (TFLite, LiteRT-LM) -------------------------------------------------------------

class _FB:
    """A FlatBuffers table walker over a RangeFile (little-endian, the standard layout: a table
    starts with an int32 offset back to its vtable; offsets to tables, vectors and strings are
    uint32 relative to where they are stored). Slot numbers are the generated code's vtable offsets
    (tflite/Model.py, litert_lm_builder/litertlm_header_schema_py_generated.py)."""

    def __init__(self, fh: RangeFile, base: int = 0):
        self.fh, self.base = fh, base

    def u(self, pos: int, fmt: str):
        self.fh.seek(self.base + pos)
        return struct.unpack("<" + fmt, self.fh.read(struct.calcsize("<" + fmt)))[0]

    def root(self) -> int:
        return self.u(0, "I")

    def field(self, table: int, slot: int) -> int:
        """Position of a field's value, or 0 when absent."""
        vt = table - self.u(table, "i")
        vsize = self.u(vt, "H")
        if slot >= vsize:
            return 0
        off = self.u(vt + slot, "H")
        return table + off if off else 0

    def scalar(self, table: int, slot: int, fmt: str, default=0):
        p = self.field(table, slot)
        return self.u(p, fmt) if p else default

    def ref(self, pos: int) -> int:
        return pos + self.u(pos, "I")

    def table(self, table: int, slot: int) -> int:
        p = self.field(table, slot)
        return self.ref(p) if p else 0

    def vector(self, table: int, slot: int) -> tuple[int, int]:
        """(position of the first element, length), or (0, 0)."""
        p = self.field(table, slot)
        if not p:
            return 0, 0
        v = self.ref(p)
        return v + 4, self.u(v, "I")

    def tables(self, table: int, slot: int) -> list[int]:
        start, n = self.vector(table, slot)
        return [self.ref(start + 4 * i) for i in range(n)]

    def numbers(self, table: int, slot: int, fmt: str) -> list:
        start, n = self.vector(table, slot)
        if not n:
            return []
        self.fh.seek(self.base + start)
        size = struct.calcsize("<" + fmt)
        return list(struct.unpack(f"<{n}{fmt}", self.fh.read(n * size)))

    def string(self, table: int, slot: int) -> str:
        start, n = self.vector(table, slot)
        if not n:
            return ""
        self.fh.seek(self.base + start)
        return self.fh.read(n).decode("utf-8", "replace")


TFLITE_DTYPE = {0: "F32", 1: "F16", 2: "I32", 3: "U8", 4: "I64", 6: "BOOL", 7: "I16", 9: "I8", 10: "F64",
                12: "U64", 15: "U32", 16: "U16", 18: "BF16"}


def tflite_header(url: str, size: int, base: int = 0) -> dict:
    """{name: (url, spec)} of a TFLite model (schema.fbs: Model.subgraphs 8, Model.buffers 12;
    Tensor.shape 4, type 6, buffer 8, name 10, quantization 12; Buffer.data 4, offset 6, size 8;
    QuantizationParameters.scale 8, zero_point 10, quantized_dimension 16). Integer weights with
    scales get spec["affine"] = {scale, zero, axis}, read as (q - zero) * scale. `base`: where the
    model starts inside a container (LiteRT-LM)."""
    from .remote import ITEMSIZE
    fb = _FB(RangeFile(url, base + size), base)
    model = fb.root()
    buffers = fb.tables(model, 12)
    out = {}
    for sg in fb.tables(model, 8):
        for t in fb.tables(sg, 4):
            dtype = TFLITE_DTYPE.get(fb.scalar(t, 6, "b"))
            bi = fb.scalar(t, 8, "I")
            if dtype is None or not bi or bi >= len(buffers):
                continue
            buf = buffers[bi]
            start, n = fb.vector(buf, 4)
            if n:
                b, e = base + start, base + start + n
            else:
                off, ln = fb.scalar(buf, 6, "Q"), fb.scalar(buf, 8, "Q")
                if not ln or off <= 1:
                    continue                     # an activation or an empty buffer: no weights
                b, e = base + off, base + off + ln
            shape = fb.numbers(t, 4, "i")
            count = 1
            for d in shape:
                count *= d
            if count * ITEMSIZE[dtype] != e - b:
                continue
            spec = {"dtype": dtype, "shape": shape, "data_offsets": [b, e]}
            q = fb.table(t, 12)
            if q and dtype in ("I8", "U8", "I16", "I32"):
                sc = fb.numbers(q, 8, "f")
                if sc:
                    spec["affine"] = {"scale": sc, "zero": fb.numbers(q, 10, "q") or [0],
                                      "axis": fb.scalar(q, 16, "i")}
            name = fb.string(t, 10) or f"tensor_{bi}"
            out[name if name not in out else f"{name}#{bi}"] = (url, spec)
    return out


LITERTLM_TFLITE = 3          # AnySectionDataType.TFLiteModel


def litertlm_sections(url: str, size: int) -> list[tuple[int, int, int]]:
    """[(data type, begin, end)] of a .litertlm container (litertlm_peek.read_litertlm_header:
    magic "LITERTLM", header end at byte 24, FlatBuffer header from byte 32; LiteRTLMMetaData.
    section_metadata 6 -> SectionMetadata.objects 4 -> SectionObject begin 6, end 8, type 10)."""
    head = fetch_range(url, 0, 31)
    if head[:8] != b"LITERTLM":
        raise ValueError("not a LiteRT-LM container")
    (end,) = struct.unpack("<Q", head[24:32])
    fb = _FB(RangeFile(url, end), 32)
    meta = fb.root()
    sm = fb.table(meta, 6)
    out = []
    for o in fb.tables(sm, 4) if sm else []:
        out.append((fb.scalar(o, 10, "B"), fb.scalar(o, 6, "Q"), fb.scalar(o, 8, "Q")))
    return out


def litertlm_header(url: str, size: int) -> dict:
    """The tensors of every TFLite model section, names prefixed by the section's index."""
    out = {}
    for k, (kind, b, e) in enumerate(litertlm_sections(url, size)):
        if kind != LITERTLM_TFLITE:
            continue
        for name, v in tflite_header(url, e - b, base=b).items():
            out[f"s{k}/{name}"] = v
    return out


# --- PaddlePaddle .pdiparams ---------------------------------------------------------------------

PADDLE_DTYPE = {0: "BOOL", 1: "I16", 2: "I32", 3: "I64", 4: "F16", 5: "F32", 6: "F64", 20: "U8", 21: "I8",
                22: "BF16"}


def pdiparams_header(url: str, size: int, names: list[str] | None = None) -> dict:
    """{name: (url, spec)} of a combined Paddle parameter file, written by save_combine as one
    record per parameter (paddle/fluid/framework/lod_tensor.cc SerializeToStream, tensor_util.cc
    TensorToStream): uint32 version, uint64 LoD level count and each level (uint64 byte length and
    data), uint32 tensor version, int32 TensorDesc length, the TensorDesc protobuf (data_type 1,
    dims 2), the raw data. Records carry no names; `names` (the program's persistable parameters in
    saved order) labels them, else they are param_<i> and aligned by structure."""
    from .remote import ITEMSIZE
    fh = RangeFile(url, size)
    pb = _PB(fh)
    out, pos, i = {}, 0, 0
    while pos < size:
        fh.seek(pos)
        (ver,) = struct.unpack("<I", fh.read(4))
        if ver != 0:
            raise ValueError(f"Paddle LoDTensor version {ver} not read")
        (levels,) = struct.unpack("<Q", fh.read(8))
        for _ in range(levels):
            (nb,) = struct.unpack("<Q", fh.read(8))
            fh.seek(fh.tell() + nb)
        (tver,) = struct.unpack("<I", fh.read(4))
        (dlen,) = struct.unpack("<i", fh.read(4))
        dstart = fh.tell()
        dtype_id, dims = None, []
        for f, wt, v in pb.fields(dstart, dstart + dlen):
            if f == 1 and wt == 0:
                dtype_id = v
            elif f == 2:
                dims += pb.packed_varints(v) if wt == 2 else [v if v < 2 ** 63 else v - 2 ** 64]
        dtype = PADDLE_DTYPE.get(dtype_id)
        if dtype is None:
            raise ValueError(f"Paddle data type {dtype_id} not read")
        count = 1
        for d in dims:
            count *= d
        b = dstart + dlen
        e = b + count * ITEMSIZE[dtype]
        name = names[i] if names and i < len(names) else f"param_{i}"
        out[name] = (url, {"dtype": dtype, "shape": dims, "data_offsets": [b, e]})
        pos, i = e, i + 1
    return out
