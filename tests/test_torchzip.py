"""PyTorch zip checkpoints are read exactly, and nothing in their pickle is executed."""
import pickle
import zipfile

import numpy as np
import pytest
import torch

from pedigree import torchzip
from pedigree.verify import decode


@pytest.fixture
def local_ranges(monkeypatch):
    def fake(url, start, end, retries=0):
        with open(url, "rb") as fh:
            fh.seek(start)
            return fh.read(end - start + 1)
    monkeypatch.setattr(torchzip, "fetch_range", fake)


def _read(path, spec):
    b, e = spec["data_offsets"]
    with open(path, "rb") as fh:
        fh.seek(b)
        return fh.read(e - b)


def test_header_matches_the_tensors(tmp_path, local_ranges):
    g = torch.Generator().manual_seed(0)
    base = torch.randn(64, 32, generator=g)
    sd = {"emb.weight": base, "head.weight": base,                 # tied: one storage
          "half": torch.randn(300, generator=g).half(),
          "bf": torch.randn(17, 5, generator=g).bfloat16(),
          "view": base.view(-1)[100:612].view(16, 32),              # storage offset
          "t": torch.randn(8, 4, generator=g).t(),                  # non-contiguous
          "steps": torch.arange(10)}
    path = tmp_path / "pytorch_model.bin"
    torch.save(sd, path)
    start, header = torchzip.read_header(str(path), path.stat().st_size)
    assert start == 0 and "t" not in header                         # left out, never misread
    for name in ["emb.weight", "head.weight", "half", "bf", "view"]:
        spec = header[name]
        assert spec["shape"] == list(sd[name].shape)
        got = decode(spec["dtype"], _read(path, spec))
        np.testing.assert_array_equal(got, sd[name].float().numpy().ravel())
    assert header["steps"]["dtype"] == "I64"


def test_nested_state_dict(tmp_path, local_ranges):
    path = tmp_path / "ckpt.bin"
    torch.save({"state_dict": {"w": torch.ones(3)}, "epoch": 3}, path)
    _, header = torchzip.read_header(str(path), path.stat().st_size)
    assert list(header) == ["w"]


class _Evil:
    def __reduce__(self):
        import os
        return (os.system, ("echo pwned",))


def test_refuses_code_in_the_pickle(tmp_path, local_ranges):
    path = tmp_path / "evil.bin"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as z:
        z.writestr("archive/data.pkl", pickle.dumps({"w": _Evil()}, protocol=2))
    with pytest.raises(pickle.UnpicklingError, match="refusing global"):
        torchzip.read_header(str(path), path.stat().st_size)


def test_legacy_files_are_read_exactly(tmp_path, local_ranges):
    g = torch.Generator().manual_seed(1)
    base = torch.randn(64, 32, generator=g)
    sd = {"emb.weight": base, "head.weight": base, "half": torch.randn(300, generator=g).half(),
          "view": base.view(-1)[100:612].view(16, 32), "steps": torch.arange(10),
          "t": torch.randn(8, 4, generator=g).t()}
    path = tmp_path / "pytorch_model.bin"
    torch.save(sd, path, _use_new_zipfile_serialization=False)
    assert path.read_bytes()[:2] != b"PK"
    start, header = torchzip.read_header(str(path), path.stat().st_size)
    assert start == 0 and "t" not in header
    for name in ["emb.weight", "head.weight", "half", "view"]:
        spec = header[name]
        assert spec["shape"] == list(sd[name].shape)
        got = decode(spec["dtype"], _read(path, spec))
        np.testing.assert_array_equal(got, sd[name].float().numpy().ravel())
    assert header["steps"]["dtype"] == "I64"


def test_refuses_code_in_a_legacy_pickle(tmp_path, local_ranges):
    path = tmp_path / "evil_old.bin"
    with open(path, "wb") as f:
        for obj in (torchzip.LEGACY_MAGIC, 1001, {"little_endian": True}):
            pickle.dump(obj, f, protocol=2)
        pickle.dump({"w": _Evil()}, f, protocol=2)
    with pytest.raises(pickle.UnpicklingError, match="refusing global"):
        torchzip.read_header(str(path), path.stat().st_size)


def test_rejects_a_plain_pickle(tmp_path, local_ranges):
    path = tmp_path / "old.bin"
    path.write_bytes(pickle.dumps({"w": 1}))
    with pytest.raises(ValueError, match="magic number"):
        torchzip.read_header(str(path), path.stat().st_size)
