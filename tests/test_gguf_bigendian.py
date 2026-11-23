"""Big-endian GGUF (correction to readers v6, 2026-11-23): a file llama.cpp's own converter
(`gguf_convert_endian.py`) turned big-endian reads as the same values as the little-endian file."""
import os
import types

import numpy as np
import pytest

from tests.test_formats import local, remote_model  # noqa: F401  (fixture)

gguf = pytest.importorskip("gguf")
from gguf.scripts import gguf_convert_endian as conv  # noqa: E402  llama.cpp's reference byte swaps

Q = gguf.GGMLQuantizationType
RNG = np.random.RandomState(7)
WEIGHTS = {"blk.0.f32.weight": (Q.F32, RNG.randn(8, 64).astype(np.float32)),
           "blk.0.f16.weight": (Q.F16, RNG.randn(8, 64).astype(np.float16)),
           "blk.0.q8.weight": (Q.Q8_0, RNG.randn(8, 64).astype(np.float32)),
           "blk.0.q40.weight": (Q.Q4_0, RNG.randn(8, 64).astype(np.float32)),
           "blk.0.q4k.weight": (Q.Q4_K, RNG.randn(8, 256).astype(np.float32)),
           "blk.0.q6k.weight": (Q.Q6_K, RNG.randn(8, 256).astype(np.float32))}


def _write(path, big):
    w = gguf.GGUFWriter(str(path), "llama", endianess=gguf.GGUFEndian.BIG if big else gguf.GGUFEndian.LITTLE)
    w.add_string("general.type", "model")
    w.add_block_count(1)
    for name, (qt, x) in WEIGHTS.items():
        if qt in (Q.F32, Q.F16):
            w.add_tensor(name, x)                     # the writer swaps whole elements itself
            continue
        if qt in (Q.Q4_K, Q.Q6_K):                    # gguf-py has no quantiser for these: any bytes
            block, size = gguf.GGML_QUANT_SIZES[qt]
            q = np.random.RandomState(int(qt)).randint(0, 256, (x.shape[0], x.shape[1] // block * size), np.uint8)
            q[:, ::size] = 0                          # keep the fp16 scales finite
        else:
            q = gguf.quants.quantize(x, qt)
        if big:                                       # quantised blocks: swapped as the converter does
            flat = q.reshape(-1).copy()
            t = types.SimpleNamespace(data=flat)
            size = gguf.GGML_QUANT_SIZES[qt][1]
            for b in range(len(flat) // size):
                conv.byteswap_tensors[qt](t, b * size)
            q = flat.reshape(q.shape)
        w.add_tensor(name, q, raw_dtype=qt)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def test_a_big_endian_gguf_reads_as_the_little_endian_one(tmp_path, local):  # noqa: F811
    from pedigree import ggufremote
    (tmp_path / "le").mkdir(), (tmp_path / "be").mkdir()
    _write(tmp_path / "le" / "m-Q8_0.gguf", big=False)
    _write(tmp_path / "be" / "m-Q8_0.gguf", big=True)
    with open(tmp_path / "be" / "m-Q8_0.gguf", "rb") as fh:
        assert fh.read(8)[4:] == b"\x00\x00\x00\x03"     # it is big-endian
    le, be = remote_model(tmp_path / "le", ["m-Q8_0.gguf"]), remote_model(tmp_path / "be", ["m-Q8_0.gguf"])
    assert le.float_shapes() == be.float_shapes() and len(le.float_shapes()) == len(WEIGHTS)
    for name in WEIGHTS:
        np.testing.assert_array_equal(be.sample(name, 10 ** 6), le.sample(name, 10 ** 6))
    f32 = WEIGHTS["blk.0.f32.weight"][1]
    np.testing.assert_array_equal(be.sample("blk.0.f32.weight", 10 ** 6), f32.ravel())
    url = str(tmp_path / "be" / "m-Q8_0.gguf")
    size = os.path.getsize(url)
    assert ggufremote.file_type(url, size) == "model"
    assert ggufremote.tensor_count(url, size) == (len(WEIGHTS), None)


def test_a_big_endian_header_no_longer_reads_as_garbage(tmp_path, local):  # noqa: F811
    """Read as little-endian, version 3 big-endian is 50331648 and passed the version check; the
    first string length then read as ~1.4e18 (H28's 12-hour stall)."""
    from pedigree import ggufremote, remote
    _write(tmp_path / "m.gguf", big=True)
    url = str(tmp_path / "m.gguf")
    assert ggufremote.byte_order(b"\x00\x00\x00\x03") == ">"
    assert ggufremote.byte_order(b"\x03\x00\x00\x00") == "<"
    with pytest.raises(ValueError, match="not plausible"):
        ggufremote.byte_order(b"\x00\x01\x00\x01")
    calls = []
    real = remote.fetch_range
    remote.fetch_range = lambda u, s, e, retries=0: calls.append((s, e)) or real(u, s, e)
    try:
        assert ggufremote.n_tensors(url) == len(WEIGHTS)
    finally:
        remote.fetch_range = real
    assert calls == [(0, 23)]


def test_a_big_endian_type_with_no_published_byte_order_is_refused(tmp_path):
    from pedigree import ggufremote
    with pytest.raises(ValueError, match="defines no byte order"):
        ggufremote.to_little(b"\x00" * 34, "Q5_K", 34)
