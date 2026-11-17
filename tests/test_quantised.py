"""MLX 4-bit and FP8 weights are read as the float values they stand for (E2 reads both)."""
import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from pedigree import remote, verify


@pytest.fixture
def local_ranges(monkeypatch):
    def fake(url, start, end, retries=0):
        with open(url, "rb") as fh:
            fh.seek(start)
            return fh.read(end - start + 1)
    monkeypatch.setattr(remote, "fetch_range", fake)


def _open_local(path, config):
    start, header = remote.read_header(str(path))
    r = verify.Remote("local/test", "c0")
    for name, s in header.items():
        if name != "__metadata__":
            r.tensors[name] = (str(path), start, s)
    r._config = lambda: config
    r._find_quantised()
    return r


def test_mlx_affine_4bit(tmp_path, local_ranges):
    rng = np.random.default_rng(0)
    out, cols, gs = 6, 128, 64
    q = rng.integers(0, 16, size=(out, cols)).astype(np.uint32)
    words = np.zeros((out, cols // 8), dtype=np.uint32)
    for k in range(8):
        words |= q[:, k::8] << (4 * k)
    scales = rng.normal(size=(out, cols // gs)).astype(np.float16)
    biases = rng.normal(size=(out, cols // gs)).astype(np.float16)
    path = tmp_path / "model.safetensors"
    save_file({"l.weight": torch.from_numpy(words.view(np.int32)).view(torch.uint32),
               "l.scales": torch.from_numpy(scales), "l.biases": torch.from_numpy(biases)}, str(path))
    r = _open_local(path, {"quantization": {"bits": 4, "group_size": gs}})
    assert r.float_shapes() == {"l.weight": ("float", (out, cols))}
    g = np.arange(cols) // gs
    want = q * scales.astype(np.float64)[:, g] + biases.astype(np.float64)[:, g]
    np.testing.assert_allclose(r.sample("l.weight", max_bytes=out * cols * 4), want.ravel())
    np.testing.assert_allclose(r.sample("l.weight", max_bytes=200 * 4), want.ravel()[:200])


def test_fp8_block_scaled(tmp_path, local_ranges):
    g = torch.Generator().manual_seed(0)
    w = torch.randn(256, 256, generator=g)
    s = torch.rand(2, 2, generator=g) + 0.5
    full_s = s.repeat_interleave(128, 0).repeat_interleave(128, 1)
    w8 = (w / full_s).to(torch.float8_e4m3fn)
    path = tmp_path / "model.safetensors"
    save_file({"l.weight": w8, "l.weight_scale_inv": s}, str(path))
    r = _open_local(path, {"quantization_config": {"weight_block_size": [128, 128]}})
    assert set(r.float_shapes()) == {"l.weight"}
    want = (w8.float() * full_s).numpy().astype(np.float64).ravel()
    np.testing.assert_allclose(r.sample("l.weight", max_bytes=256 * 256 * 4), want, rtol=1e-6)
    assert verify.cosine(r.sample("l.weight"), w.numpy().ravel().astype(np.float64)) > 0.99


def test_fp8_per_tensor_scale(tmp_path, local_ranges):
    w = torch.randn(64, 128, generator=torch.Generator().manual_seed(1))
    path = tmp_path / "model.safetensors"
    save_file({"l.weight": (w / 3).to(torch.float8_e4m3fn), "l.weight_scale": torch.tensor(3.0)}, str(path))
    r = _open_local(path, {})
    assert verify.cosine(r.sample("l.weight"), w.numpy().ravel().astype(np.float64)) > 0.99


def test_kurtosis_gap_signs_the_heavier_tail():
    rng = np.random.default_rng(0)
    light, heavy = rng.normal(size=20000), rng.standard_t(4, size=20000)
    assert verify.excess_kurtosis(light) == pytest.approx(0, abs=0.15)
    assert verify.kurtosis_gap({"a": [verify.excess_kurtosis(heavy), verify.excess_kurtosis(light)]}) > 1


def test_mlx_experts_axis(tmp_path, local_ranges):
    rng = np.random.default_rng(1)
    e, out, cols, gs = 3, 4, 64, 32
    q = rng.integers(0, 16, size=(e, out, cols)).astype(np.uint32)
    words = np.zeros((e, out, cols // 8), dtype=np.uint32)
    for k in range(8):
        words |= q[..., k::8] << (4 * k)
    scales = rng.normal(size=(e, out, cols // gs)).astype(np.float16)
    biases = rng.normal(size=(e, out, cols // gs)).astype(np.float16)
    path = tmp_path / "model.safetensors"
    save_file({"x.weight": torch.from_numpy(words.view(np.int32)).view(torch.uint32),
               "x.scales": torch.from_numpy(scales), "x.biases": torch.from_numpy(biases)}, str(path))
    r = _open_local(path, {"quantization": {"bits": 4, "group_size": gs}})
    assert r.float_shapes() == {"x.weight": ("float", (e, out, cols))}
    g = np.arange(cols) // gs
    want = q * scales.astype(np.float64)[..., g] + biases.astype(np.float64)[..., g]
    np.testing.assert_allclose(r.sample("x.weight", max_bytes=e * out * cols * 4), want.ravel())


@pytest.mark.parametrize("bits", [3, 6])
def test_mlx_values_straddling_words(tmp_path, local_ranges, bits):
    """MLX 3- and 6-bit rows are bit streams: a value may start in one 32-bit word and end in the
    next (E2 errored on 49 such comparisons before this was read correctly)."""
    rng = np.random.default_rng(1)
    out, cols, gs = 4, 160, 32       # 160 x 3 = 480 and 160 x 6 = 960 bits: whole words, values straddle
    q = rng.integers(0, 1 << bits, size=(out, cols)).astype(np.uint8)
    stream = ((q[:, :, None] >> np.arange(bits)) & 1).astype(np.uint8).reshape(out, cols * bits)
    words = np.packbits(stream, axis=1, bitorder="little").view("<u4")
    scales = rng.normal(size=(out, cols // gs)).astype(np.float16)
    biases = rng.normal(size=(out, cols // gs)).astype(np.float16)
    path = tmp_path / "model.safetensors"
    save_file({"l.weight": torch.from_numpy(words.view(np.int32).copy()).view(torch.uint32),
               "l.scales": torch.from_numpy(scales), "l.biases": torch.from_numpy(biases)}, str(path))
    r = _open_local(path, {"quantization": {"bits": bits, "group_size": gs}})
    assert r.float_shapes() == {"l.weight": ("float", (out, cols))}
    g = np.arange(cols) // gs
    want = q * scales.astype(np.float64)[:, g] + biases.astype(np.float64)[:, g]
    np.testing.assert_allclose(r.sample("l.weight", max_bytes=out * cols * 4), want.ravel())


def test_gguf_plain_integer_tensor(monkeypatch):
    """A GGUF int8 tensor is stored as it is; the reader casts it instead of dequantising."""
    from pedigree import ggufremote
    data = np.arange(-50, 50, dtype=np.int8).tobytes()
    monkeypatch.setattr(remote, "fetch_range", lambda url, a, b, retries=0: data[a:b + 1])
    import gguf
    spec = {"qtype": int(gguf.GGMLQuantizationType.I8), "shape": [100], "start": 0}
    np.testing.assert_array_equal(ggufremote.sample("u", spec, 30), np.arange(-50, -20, dtype=np.float64))


def test_gguf_fork_type_is_named(monkeypatch):
    from pedigree import ggufremote
    with pytest.raises(ValueError, match="not in llama.cpp's published list"):
        ggufremote.sample("u", {"qtype": 101, "shape": [10], "start": 0}, 10)


def test_mlx_mixed_precision_layer(tmp_path, local_ranges):
    """A layer listed under its own path in the quantisation config (8-bit inside a 4-bit model)
    is read with its own width."""
    rng = np.random.default_rng(2)
    out, cols, gs = 2, 64, 64
    q = rng.integers(0, 256, size=(out, cols)).astype(np.uint8)
    words = q.view("<u4")
    scales = rng.normal(size=(out, 1)).astype(np.float16)
    biases = rng.normal(size=(out, 1)).astype(np.float16)
    path = tmp_path / "model.safetensors"
    save_file({"mlp.gate.weight": torch.from_numpy(words.view(np.int32).copy()).view(torch.uint32),
               "mlp.gate.scales": torch.from_numpy(scales), "mlp.gate.biases": torch.from_numpy(biases)}, str(path))
    r = _open_local(path, {"quantization": {"bits": 4, "group_size": 64, "mlp.gate": {"bits": 8, "group_size": 64}}})
    assert r.float_shapes() == {"mlp.gate.weight": ("float", (out, cols))}
    want = q * scales.astype(np.float64) + biases.astype(np.float64)
    np.testing.assert_allclose(r.sample("mlp.gate.weight", max_bytes=out * cols * 4), want.ravel())


def test_mlx_override_under_reordered_path(tmp_path, local_ranges):
    """A vision-language export names the layer model.language_model.X while its config says
    language_model.model.X; the 8-bit override still applies."""
    rng = np.random.default_rng(3)
    out, cols = 2, 64
    q = rng.integers(0, 256, size=(out, cols)).astype(np.uint8)
    scales = rng.normal(size=(out, 1)).astype(np.float16)
    biases = rng.normal(size=(out, 1)).astype(np.float16)
    path = tmp_path / "model.safetensors"
    stem = "model.language_model.layers.0.mlp.gate"
    save_file({stem + ".weight": torch.from_numpy(q.view("<u4").view(np.int32).copy()).view(torch.uint32),
               stem + ".scales": torch.from_numpy(scales), stem + ".biases": torch.from_numpy(biases)}, str(path))
    cfg = {"quantization": {"bits": 4, "group_size": 64, "language_model.model.layers.0.mlp.gate": {"bits": 8}}}
    r = _open_local(path, cfg)
    want = q * scales.astype(np.float64) + biases.astype(np.float64)
    np.testing.assert_allclose(r.sample(stem + ".weight", max_bytes=out * cols * 4), want.ravel())
