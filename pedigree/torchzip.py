"""Read PyTorch checkpoints (pytorch_model.bin) remotely, without running their pickle.

Many Hub models (and every intermediate checkpoint of some training suites) exist only
as PyTorch files. Since torch 1.6 such a file is an uncompressed zip: `*/data.pkl`
describes the tensors, and each storage is a separate entry `*/data/<key>`. This
module reads the zip directory and data.pkl with HTTP range requests and turns them
into a safetensors-style header {name: {dtype, shape, data_offsets}} with absolute
byte offsets, so the rest of Pedigree reads .bin and .safetensors files the same way.

Safety: data.pkl is unpickled by a restricted Unpickler that knows only the records
PyTorch uses to describe tensors (OrderedDict, the tensor rebuild functions, storage
types). They are replaced by inert stand-ins that record layout and execute nothing;
functions and other torch/numpy globals make the file unreadable (reported, never silently
skipped); any other class becomes an inert placeholder that keeps its pickled state, so tensors
held inside a framework's objects are still found (amendment 2026-11-20).
"""
from __future__ import annotations

import io
import pickle
import struct
import zipfile
from collections import OrderedDict
from dataclasses import dataclass

from .remote import fetch_range

BLOCK = 64 * 1024

STORAGE_DTYPE = {"FloatStorage": "F32", "HalfStorage": "F16", "BFloat16Storage": "BF16",
                 "DoubleStorage": "F64", "LongStorage": "I64", "IntStorage": "I32", "ShortStorage": "I16",
                 "CharStorage": "I8", "ByteStorage": "U8", "BoolStorage": "BOOL"}
ITEM = {"F32": 4, "F16": 2, "BF16": 2, "F64": 8, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}


class RangeFile(io.RawIOBase):
    """A read-only, seekable file over HTTP range requests (cached in blocks)."""

    def __init__(self, url: str, size: int, base: int = 0):
        self.url, self.size, self.pos, self.base = url, size, 0, base
        self._block: tuple[int, bytes] | None = None   # the last block read, kept in memory

    def _get(self, b0: int) -> bytes:
        if self._block is None or self._block[0] != b0:
            self._block = (b0, fetch_range(self.url, self.base + b0, self.base + min(b0 + BLOCK, self.size) - 1))
        return self._block[1]

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos))
        out = bytearray()
        while len(out) < n:
            b0 = (self.pos // BLOCK) * BLOCK
            block = self._get(b0)
            take = block[self.pos - b0: self.pos - b0 + n - len(out)]
            out += take
            self.pos += len(take)
        return bytes(out)

    def readinto(self, b):
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


@dataclass(frozen=True)
class _Storage:
    dtype: str
    key: str


@dataclass(frozen=True)
class _Tensor:
    storage: _Storage
    offset: int
    shape: tuple
    stride: tuple


def _rebuild_tensor_v2(storage, storage_offset, size, stride, *rest):
    return _Tensor(storage, int(storage_offset), tuple(int(s) for s in size), tuple(int(s) for s in stride))


def _rebuild_parameter(data, *rest):
    return data


class _Marker(str):
    pass


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ("collections", "OrderedDict"):
            return OrderedDict
        if module in ("builtins", "__builtin__") and name in ("set", "frozenset", "slice", "complex"):
            return {"set": set, "frozenset": frozenset, "slice": slice, "complex": complex}[name]
        if (module, name) in (("torch._utils", "_rebuild_tensor_v2"), ("torch._utils", "_rebuild_tensor")):
            return _rebuild_tensor_v2
        if (module, name) in (("torch._utils", "_rebuild_parameter"),
                              ("torch._utils", "_rebuild_parameter_with_state")):
            return _rebuild_parameter
        if module == "torch" and name in STORAGE_DTYPE:
            return _Marker(STORAGE_DTYPE[name])
        internal = module.split(".")[0] in ("torch", "numpy") and not module.startswith("torch.nn.modules.")
        if internal or not name[:1].isupper():
            # functions, and torch/numpy internals other than the records above: never stood in for
            raise pickle.UnpicklingError(f"refusing global {module}.{name}")
        # any other class (a training framework's config or vocabulary object, e.g. flair's
        # Dictionary, Lightning's hyper-parameter records): an inert stand-in that keeps its state
        # so tensors inside it are still found; no code of the file runs (amendment 2026-11-20, R1)
        return _opaque(module, name)

    def persistent_load(self, pid):
        # ('storage', storage_type, key, location, numel)
        if not (isinstance(pid, tuple) and pid and pid[0] == "storage" and isinstance(pid[1], _Marker)):
            raise pickle.UnpicklingError(f"unexpected persistent id {pid!r}")
        return _Storage(str(pid[1]), str(pid[2]))


class _LazySpec(dict):
    """A header entry whose data_offsets are resolved on first use."""

    def __init__(self, fields: dict, offsets):
        super().__init__(fields)
        self._offsets = offsets

    def __missing__(self, key):
        if key != "data_offsets":
            raise KeyError(key)
        self["data_offsets"] = value = self._offsets()
        return value


def _contiguous(shape: tuple, stride: tuple) -> bool:
    expect = 1
    for n, s in zip(reversed(shape), reversed(stride)):
        if n != 1 and s != expect:
            return False
        expect *= n
    return True


class _Opaque:
    """Stand-in for a class the restricted unpickler does not know."""
    __slots__ = ("_state", "_args")

    def __init__(self, *args, **kwargs):
        self._args, self._state = args, kwargs or None

    def __setstate__(self, state):
        self._state = state

    def __setitem__(self, k, v):            # SETITEM(S) on a dict subclass
        if not isinstance(self._state, dict):
            self._state = {}
        self._state[k] = v

    def append(self, v):                     # APPEND(S) on a list subclass
        self._args = tuple(self._args) + (v,)

    def extend(self, vs):
        self._args = tuple(self._args) + tuple(vs)


_OPAQUE: dict = {}


def _opaque(module: str, name: str):
    key = (module, name)
    if key not in _OPAQUE:
        _OPAQUE[key] = type(name, (_Opaque,), {"__slots__": (), "__module__": "pedigree.torchzip.opaque"})
    return _OPAQUE[key]


MODULE_SLOTS = ("_parameters", "_buffers", "_modules")    # a whole pickled torch.nn.Module


def _flatten(obj, prefix=""):
    if isinstance(obj, _Tensor):
        yield prefix, obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if k in MODULE_SLOTS:
                yield from _flatten(v, prefix)
            else:
                yield from _flatten(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, _Opaque):
        if isinstance(obj._state, (dict, _Opaque)):
            yield from _flatten(obj._state, prefix)
        elif isinstance(obj._state, tuple):              # (dict state, slot state)
            for part in obj._state:
                yield from _flatten(part, prefix)


LEGACY_MAGIC = 0x1950A86A20F9469CFC6C


class _KeyedUnpickler(_Unpickler):
    """The restricted unpickler, also recording each storage's dtype by key."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.dtypes: dict[str, str] = {}

    def persistent_load(self, pid):
        s = super().persistent_load(pid)
        self.dtypes[s.key] = s.dtype
        return s


def _legacy_header(fh: RangeFile, url: str) -> tuple[int, dict]:
    """A pre-1.6 PyTorch file: pickled magic number, protocol, system info and the object, then
    the storage keys, then each storage as an 8-byte element count and its bytes, in key order.
    Every pickle goes through the restricted unpickler; the storages are located by walking the
    counts (8 bytes read per storage, on first use)."""
    buf = io.BufferedReader(fh, BLOCK)
    if _Unpickler(buf).load() != LEGACY_MAGIC:
        raise ValueError("not a PyTorch file (neither zip nor legacy magic number)")
    _Unpickler(buf).load()                      # protocol version
    _Unpickler(buf).load()                      # system info (endianness, type sizes)
    up = _KeyedUnpickler(buf)
    state = up.load()
    keys = _Unpickler(buf).load()
    if not isinstance(keys, list):
        raise ValueError("legacy PyTorch file without a storage key list")
    first = buf.tell()
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    starts: dict[str, int] = {}

    def start_of(key: str) -> int:
        if not starts:
            pos = first
            for k in keys:
                (numel,) = struct.unpack("<q", fetch_range(url, fh.base + pos, fh.base + pos + 7))
                starts[str(k)] = fh.base + pos + 8
                pos += 8 + numel * ITEM[up.dtypes.get(str(k), "U8")]
        return starts[key]

    return 0, _header(state, start_of)


def _header(state, start_of) -> dict:
    header = {}
    for name, t in _flatten(state):
        fields = {"dtype": t.storage.dtype, "shape": list(t.shape)}
        if not _contiguous(t.shape, t.stride):
            # a plain transpose of a 2-D matrix is read whole and reordered (remote.tensor_bytes);
            # any other layout is left out
            if len(t.shape) == 2 and t.stride == (1, t.shape[0]):
                fields["order"] = "F"
            else:
                continue
        item = ITEM[t.storage.dtype]
        n = 1
        for d in t.shape:
            n *= d
        header[name] = _LazySpec(fields,
                                 lambda k=t.storage.key, o=t.offset * item, m=n * item:
                                 [start_of(k) + o, start_of(k) + o + m])
    return header


def read_header(url: str, size: int, base: int = 0) -> tuple[int, dict]:
    """(0, safetensors-style header with absolute offsets) of a remote PyTorch file: the zip
    format (torch >= 1.6) or the legacy pickle format before it. Non-contiguous tensors are left
    out (their leading bytes are not their leading values)."""
    fh = RangeFile(url, size, base)
    if fh.read(4) != b"PK\x03\x04":
        fh.seek(0)
        return _legacy_header(fh, url)
    zf = zipfile.ZipFile(fh)
    pkl = [n for n in zf.namelist() if n.endswith("/data.pkl") or n == "data.pkl"]
    if len(pkl) != 1:
        raise ValueError("no unique data.pkl in the archive")
    root = pkl[0][: -len("data.pkl")]
    state = _Unpickler(io.BytesIO(zf.read(pkl[0]))).load()
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    starts: dict[str, int] = {}

    def start_of(key: str) -> int:
        # the local header carries alignment padding the central directory does not
        # record; read its 30 bytes only for tensors that are actually sampled
        if key not in starts:
            info = zf.getinfo(f"{root}data/{key}")
            if info.compress_type != zipfile.ZIP_STORED:
                raise ValueError("compressed tensor storage")
            local = fetch_range(url, base + info.header_offset, base + info.header_offset + 29)
            nlen, xlen = struct.unpack("<HH", local[26:30])
            starts[key] = base + info.header_offset + 30 + nlen + xlen
        return starts[key]

    return 0, _header(state, start_of)
