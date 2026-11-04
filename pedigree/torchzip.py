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
any other global makes the file unreadable (reported, never silently skipped).
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

    def __init__(self, url: str, size: int):
        self.url, self.size, self.pos = url, size, 0

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
            block = fetch_range(self.url, b0, min(b0 + BLOCK, self.size) - 1)
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
        if (module, name) in (("torch._utils", "_rebuild_tensor_v2"), ("torch._utils", "_rebuild_tensor")):
            return _rebuild_tensor_v2
        if (module, name) in (("torch._utils", "_rebuild_parameter"),
                              ("torch._utils", "_rebuild_parameter_with_state")):
            return _rebuild_parameter
        if module == "torch" and name in STORAGE_DTYPE:
            return _Marker(STORAGE_DTYPE[name])
        raise pickle.UnpicklingError(f"refusing global {module}.{name}")

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


def _flatten(obj, prefix=""):
    if isinstance(obj, _Tensor):
        yield prefix, obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _flatten(v, f"{prefix}.{k}" if prefix else str(k))


def read_header(url: str, size: int) -> tuple[int, dict]:
    """(0, safetensors-style header with absolute offsets) of a remote PyTorch zip file.
    Non-contiguous tensors are left out (their leading bytes are not their leading values)."""
    fh = RangeFile(url, size)
    if fh.read(4) != b"PK\x03\x04":
        raise ValueError("not a zip-format PyTorch file (legacy pickle files are not read)")
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
            local = fetch_range(url, info.header_offset, info.header_offset + 29)
            nlen, xlen = struct.unpack("<HH", local[26:30])
            starts[key] = info.header_offset + 30 + nlen + xlen
        return starts[key]

    header = {}
    for name, t in _flatten(state):
        if not _contiguous(t.shape, t.stride):
            continue
        item = ITEM[t.storage.dtype]
        n = 1
        for d in t.shape:
            n *= d
        header[name] = _LazySpec({"dtype": t.storage.dtype, "shape": list(t.shape)},
                                 lambda k=t.storage.key, o=t.offset * item, m=n * item:
                                 [start_of(k) + o, start_of(k) + o + m])
    return 0, header
