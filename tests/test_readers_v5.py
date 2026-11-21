"""Readers v5 (PLAN.md, amendment 2026-11-21): each rule on a file its reference library wrote, the
values read checked against the values written, and the rule off (PEDIGREE_READERS_V5=0) reproducing
E2x and H25 as they ran."""
import collections
import io
import os
import pickle

import numpy as np
import pytest
import torch

from pedigree import formats, hub, torchzip
from tests.test_formats import local, remote_model, values  # noqa: F401  (fixture)

onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper, numpy_helper  # noqa: E402


def _save(tmp_path, name, nodes, inits, outputs=("y",)):
    g = helper.make_graph(nodes, "g", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 16])],
                          [helper.make_tensor_value_info(o, TensorProto.FLOAT, None) for o in outputs], inits)
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(helper.make_model(g), str(path))
    return path


def _header(tmp_path, name, **kw):
    return formats.onnx_header(str(tmp_path), "local", name, os.path.getsize(tmp_path / name), **kw)


def test_named_matmul_weights_are_seen_transposed(tmp_path, local, monkeypatch):
    """(v) A named initializer fed to MatMul holds [in, out] (onnxruntime-genai and similar exporters
    name them `<module>.MatMul.weight`); Gemm with transB already holds [out, in]."""
    rng = np.random.RandomState(1)
    w = rng.randn(16, 24).astype(np.float32)
    g = rng.randn(8, 24).astype(np.float32)
    inits = [numpy_helper.from_array(w, "model.layers.0.mlp.up_proj.MatMul.weight"),
             numpy_helper.from_array(g, "model.head.weight")]
    nodes = [helper.make_node("MatMul", ["x", "model.layers.0.mlp.up_proj.MatMul.weight"], ["h"],
                              name="/model/layers.0/mlp/up_proj/MatMul"),
             helper.make_node("Gemm", ["h", "model.head.weight"], ["y"], name="/model/head/Gemm", transB=1)]
    _save(tmp_path, "model.onnx", nodes, inits)
    h, _ = _header(tmp_path, "model.onnx")
    assert set(h) == {"model.layers.0.mlp.up_proj.weight", "model.head.weight"}
    assert h["model.layers.0.mlp.up_proj.weight"][1]["shape"] == [24, 16]
    r = remote_model(tmp_path, ["model.onnx"])
    np.testing.assert_array_equal(r.sample("model.layers.0.mlp.up_proj.weight", 10 ** 6), w.T.ravel())
    np.testing.assert_array_equal(r.sample("model.head.weight", 10 ** 6), g.ravel())
    monkeypatch.setattr(formats, "READERS_V5", False)        # E2x and H25 as they ran
    h, _ = _header(tmp_path, "model.onnx")
    assert set(h) == {"model.layers.0.mlp.up_proj.MatMul.weight", "model.head.weight"}
    assert h["model.layers.0.mlp.up_proj.MatMul.weight"][1]["shape"] == [16, 24]


def test_dynamic_quantisation_read_as_values(tmp_path, local, monkeypatch):
    """(vi) onnxruntime's quantize_dynamic: `X_quantized` (int8, [in, out]) with a per-column
    `X_scale` and `X_zero_point` feeding MatMulInteger, and a per-tensor uint8 embedding."""
    rng = np.random.RandomState(2)
    q = rng.randint(-127, 128, size=(16, 12)).astype(np.int8)
    scale = rng.rand(12).astype(np.float32) / 50 + 0.001
    zero = rng.randint(-3, 4, size=12).astype(np.int8)
    eq = rng.randint(0, 256, size=(40, 16)).astype(np.uint8)
    es, ez = np.array(0.02, np.float32), np.array(128, np.uint8)
    inits = [numpy_helper.from_array(q, "onnx::MatMul_5_quantized"),
             numpy_helper.from_array(scale, "onnx::MatMul_5_scale"),
             numpy_helper.from_array(zero, "onnx::MatMul_5_zero_point"),
             numpy_helper.from_array(eq, "model.embed_tokens.weight_quantized"),
             numpy_helper.from_array(es, "model.embed_tokens.weight_scale"),
             numpy_helper.from_array(ez, "model.embed_tokens.weight_zero_point")]
    nodes = [helper.make_node("DynamicQuantizeLinear", ["x"], ["xq", "xs", "xz"]),
             helper.make_node("MatMulInteger", ["xq", "onnx::MatMul_5_quantized", "xz", "onnx::MatMul_5_zero_point"],
                              ["y"], name="/model/layers.0/self_attn/q_proj/MatMul_quant"),
             helper.make_node("DequantizeLinear", ["model.embed_tokens.weight_quantized",
                                                   "model.embed_tokens.weight_scale",
                                                   "model.embed_tokens.weight_zero_point"], ["emb"])]
    _save(tmp_path, "model_quantized.onnx", nodes, inits)
    r = remote_model(tmp_path, ["model_quantized.onnx"])
    want = ((q.astype(np.float64) - zero.reshape(1, -1)) * scale.reshape(1, -1)).T
    name = "model.layers.0.self_attn.q_proj.weight"
    assert r.float_shapes()[name][1] == (12, 16)
    np.testing.assert_allclose(r.sample(name, 10 ** 6), want.ravel(), rtol=1e-6)
    np.testing.assert_allclose(r.sample("model.embed_tokens.weight", 10 ** 6),
                               ((eq.astype(np.float64) - 128) * np.float32(0.02)).ravel(), rtol=1e-6)
    monkeypatch.setattr(formats, "READERS_V5", False)
    h, deq = _header(tmp_path, "model_quantized.onnx")
    assert "onnx::MatMul_5_quantized" in h and "onnx::MatMul_5_quantized" not in deq


def test_merged_decoder_nodes_inside_if_branches(tmp_path, local):
    """(iv) A merged decoder keeps its nodes in If branches and suffixes its initializers `_merged_0`."""
    rng = np.random.RandomState(3)
    w = rng.randn(16, 16).astype(np.float32)
    e = rng.randn(30, 16).astype(np.float32)
    branch = helper.make_graph([helper.make_node("MatMul", ["x", "onnx::MatMul_9_merged_0"], ["b"],
                                                 name="/model/decoder/layers.0/fc1/MatMul")],
                               "then", [], [helper.make_tensor_value_info("b", TensorProto.FLOAT, None)])
    other = helper.make_graph([helper.make_node("Identity", ["x"], ["c"])], "else", [],
                              [helper.make_tensor_value_info("c", TensorProto.FLOAT, None)])
    inits = [numpy_helper.from_array(w, "onnx::MatMul_9_merged_0"),
             numpy_helper.from_array(e, "model.decoder.embed_tokens.weight_merged_0"),
             numpy_helper.from_array(np.array(True), "use_cache")]
    nodes = [helper.make_node("If", ["use_cache"], ["y"], then_branch=branch, else_branch=other),
             helper.make_node("Gather", ["model.decoder.embed_tokens.weight_merged_0", "x"], ["g"])]
    _save(tmp_path, "decoder_model_merged.onnx", nodes, inits)
    r = remote_model(tmp_path, ["decoder_model_merged.onnx"])
    assert {"model.decoder.layers.0.fc1.weight", "model.decoder.embed_tokens.weight"} <= set(r.float_shapes())
    np.testing.assert_array_equal(r.sample("model.decoder.layers.0.fc1.weight", 10 ** 6), w.T.ravel())


def test_encoder_decoder_exports_read_together(tmp_path, local):
    """(iii) encoder_model.onnx and decoder_model.onnx of one precision are one model; the encoder's
    names (relative to the encoder) take `encoder.`."""
    from experiments.e2_audit import other_unit
    rng = np.random.RandomState(4)
    we, wd = rng.randn(16, 16).astype(np.float32), rng.randn(16, 16).astype(np.float32)
    _save(tmp_path, "onnx/encoder_model.onnx",
          [helper.make_node("MatMul", ["x", "onnx::MatMul_1"], ["y"], name="/layers.0/fc1/MatMul")],
          [numpy_helper.from_array(we, "onnx::MatMul_1")])
    _save(tmp_path, "onnx/decoder_model.onnx",
          [helper.make_node("MatMul", ["x", "onnx::MatMul_2"], ["y"], name="/model/decoder/layers.0/fc1/MatMul")],
          [numpy_helper.from_array(wd, "onnx::MatMul_2")])
    size = {"onnx/encoder_model.onnx": 5, "onnx/decoder_model.onnx": 9, "onnx/decoder_model_merged.onnx": 10,
            "onnx/decoder_with_past_model.onnx": 8, "onnx/encoder_model_fp16.onnx": 3,
            "onnx/decoder_model_merged_fp16.onnx": 4}
    files = other_unit(list(size), size)
    assert files == ["onnx/encoder_model.onnx", "onnx/decoder_model.onnx"]
    r = remote_model(tmp_path, files)
    assert set(r.float_shapes()) == {"encoder.layers.0.fc1.weight", "model.decoder.layers.0.fc1.weight"}
    np.testing.assert_array_equal(r.sample("encoder.layers.0.fc1.weight", 10 ** 6), we.T.ravel())
    only = {"onnx/model.onnx": 4, "onnx/model_quantized.onnx": 2}
    assert other_unit(list(only), only) == ["onnx/model.onnx"]       # a single-file export: unchanged


def test_projector_gguf_is_not_the_model(monkeypatch):
    """A top-level mmproj-*.gguf (vision projector) is set aside, so GGUF sets in subfolders are found."""
    names = ["mmproj-BF16.gguf", "Q8_0/m-Q8_0.gguf", "README.md"]
    with pytest.raises(hub.HubError):
        hub.choose_files(names, lambda n: {})
    monkeypatch.setattr(hub, "READERS_V5", False)
    assert hub.choose_files(names, lambda n: {}) == ["mmproj-BF16.gguf"]   # E2x and H25 as they ran


def test_framework_records_in_pickles(tmp_path, local):
    """collections.defaultdict, torch.Size, torch.__version__ and optimizer classes, as flair and
    Ultralytics pickle them, are read as inert values; their tensors are found, an optimizer's
    state is not taken for weights, and functions are still refused."""
    obj = {"state_dict": {"w": torch.randn(3, 4)}, "counts": collections.defaultdict(int, {"a": 2}),
           "shape": torch.Size([3, 4]), "torch": torch.__version__, "optimizer": torch.optim.SGD}
    opt = torch.optim.SGD([torch.nn.Parameter(torch.randn(2))], lr=0.1, momentum=0.9)
    obj["optimizer_instance"] = opt
    torch.save(obj, tmp_path / "m.pt")
    _, h = torchzip.read_header(str(tmp_path / "m.pt"), os.path.getsize(tmp_path / "m.pt"))
    assert set(h) == {"w"}                                  # the optimizer's parameter is not a weight
    for bad in ((os.system, ("echo pwned",)), (torch.load, ("x",)), (collections.Counter.fromkeys, ("ab",))):
        class Evil:
            def __reduce__(self, bad=bad):
                return bad
        with pytest.raises(pickle.UnpicklingError):
            torchzip._Unpickler(io.BytesIO(pickle.dumps({"x": Evil()}, protocol=2))).load()


def test_framework_records_refused_without_v5(tmp_path, local, monkeypatch):
    monkeypatch.setattr(torchzip, "READERS_V5", False)
    torch.save({"w": torch.randn(2, 2), "counts": collections.defaultdict(int)}, tmp_path / "m.pt")
    with pytest.raises(pickle.UnpicklingError):
        torchzip.read_header(str(tmp_path / "m.pt"), os.path.getsize(tmp_path / "m.pt"))


def test_verify_reads_what_the_audit_reads(monkeypatch):
    """`pedigree verify` (hub.resolve) falls back to the audit's file rule (pedigree/units.py) where
    its own finds no safetensors, pytorch_model.bin or top-level GGUF."""
    info = {"sha": "abc", "siblings": [{"rfilename": "onnx/model.onnx", "size": 5},
                                       {"rfilename": "onnx/model_quantized.onnx", "size": 2},
                                       {"rfilename": "README.md", "size": 1}]}
    monkeypatch.setattr(hub, "_json", lambda url: info)
    res = hub.resolve("hf:someone/model-onnx")
    assert res.files == ["onnx/model.onnx"] and res.sizes == {"onnx/model.onnx": 5}
    info["siblings"] = [{"rfilename": "README.md", "size": 1}]
    with pytest.raises(hub.HubError, match="no weight files"):
        hub.resolve("hf:someone/empty")
    monkeypatch.setattr(hub, "READERS_V5", False)
    info["siblings"] = [{"rfilename": "onnx/model.onnx", "size": 5}]
    with pytest.raises(hub.HubError):
        hub.resolve("hf:someone/model-onnx")


def _gguf(path, n, arch="llama"):
    import gguf
    w = gguf.GGUFWriter(str(path), arch)
    for i in range(n):
        w.add_tensor(f"blk.{i}.w", np.full((4, 4), i, np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def test_the_unit_with_the_most_tensors_is_the_model(tmp_path, local, monkeypatch):
    """(vii) A draft head, a projector, a VAE or a classifier head beside the model holds fewer
    tensors; one model's quantisation variants tie and are chosen among as before."""
    from safetensors.numpy import save_file
    from pedigree import remote, units
    monkeypatch.setattr(remote, "resolve_url", lambda repo, f, rev="main": os.path.join(repo, f))
    _gguf(tmp_path / "m-Q8_0.gguf", 30)
    _gguf(tmp_path / "m-Q4_K_M.gguf", 30)
    _gguf(tmp_path / "m-mtp-draft.gguf", 3)
    _gguf(tmp_path / "mmproj-m-f16.gguf", 50)          # a projector never counts, however large
    save_file({"head.weight": np.ones((2, 2), np.float32)}, str(tmp_path / "heads.safetensors"))
    names = sorted(p.name for p in tmp_path.iterdir())
    d = {"sha": "local", "siblings": [{"rfilename": n, "size": os.path.getsize(tmp_path / n)} for n in names]}
    assert units._most_tensors(str(tmp_path), d, names, ["heads.safetensors", "m-mtp-draft.gguf"]) == ["m-Q8_0.gguf"]
    assert units._most_tensors(str(tmp_path), d, names, ["m-Q4_K_M.gguf", "m-Q8_0.gguf"]) == ["m-Q8_0.gguf"]
    fmt, files, _, _ = units.weight_files(str(tmp_path), d)
    assert (fmt, files) == ("gguf", ["m-Q8_0.gguf"])
    monkeypatch.setattr(formats, "READERS_V5", False)         # E2x and H25 as they ran
    fmt, files, _, _ = units.weight_files(str(tmp_path), d)
    assert files[0] == "heads.safetensors"


@pytest.mark.parametrize("with_zero", [False, True])
def test_matmulnbits_4bit_blocks(tmp_path, local, with_zero):
    """(viii) onnxruntime's MatMulNBits (com.microsoft): B is [N, K/block, block/2] bytes of 4-bit
    values (low nibble first), one scale per block, zero points 4-bit packed per row (default 8);
    the weight it stands for is [N, K]."""
    rng = np.random.RandomState(5)
    n, k, block = 6, 40, 16                                   # K not a multiple of the block
    nb = -(-k // block)
    q = rng.randint(0, 16, size=(n, nb * block)).astype(np.uint8)
    scale = (rng.rand(n, nb).astype(np.float32) + 0.1) / 10
    zero = rng.randint(0, 16, size=(n, nb)).astype(np.uint8) if with_zero else np.full((n, nb), 8, np.uint8)
    packed = (q[:, 0::2] | (q[:, 1::2] << 4)).reshape(n, nb, block // 2)
    want = ((q.astype(np.float64).reshape(n, nb, block) - zero[..., None]) * scale[..., None]).reshape(n, -1)[:, :k]
    inits = [numpy_helper.from_array(packed, "model.layers.0.mlp.down_proj.MatMul.weight_Q4"),
             numpy_helper.from_array(scale.reshape(-1), "model.layers.0.mlp.down_proj.MatMul.weight_scales")]
    ins = ["x", "model.layers.0.mlp.down_proj.MatMul.weight_Q4", "model.layers.0.mlp.down_proj.MatMul.weight_scales"]
    if with_zero:
        zp = np.zeros((n, -(-nb // 2)), np.uint8)
        for j in range(nb):
            zp[:, j // 2] |= zero[:, j] << (4 * (j % 2))
        inits.append(numpy_helper.from_array(zp.reshape(-1), "model.layers.0.mlp.down_proj.MatMul.weight_zp"))
        ins.append("model.layers.0.mlp.down_proj.MatMul.weight_zp")
    node = helper.make_node("MatMulNBits", ins, ["y"], domain="com.microsoft", K=k, N=n, bits=4, block_size=block,
                            name="/model/layers.0/mlp/down_proj/MatMul_Q4")
    _save(tmp_path, "model_q4.onnx", [node], inits)
    r = remote_model(tmp_path, ["model_q4.onnx"])
    name = "model.layers.0.mlp.down_proj.weight"
    assert r.float_shapes() == {name: ("float", (n, k))}
    np.testing.assert_allclose(r.sample(name, 10 ** 6), want.ravel(), rtol=1e-6)
    np.testing.assert_allclose(r.sample(name, 4 * 50), want.ravel()[:50], rtol=1e-6)   # a partial read


def _float_matmul_model(path, w, name="model.layers.0.mlp.up_proj.MatMul.weight", node="/model/layers.0/mlp/up_proj/MatMul"):
    k = w.shape[0]
    g = helper.make_graph([helper.make_node("MatMul", ["x", name], ["y"], name=node)], "g",
                          [helper.make_tensor_value_info("x", TensorProto.FLOAT, [k, k])],
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT, [k, w.shape[1]])],
                          [numpy_helper.from_array(w, name)])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 9
    onnx.save_model(m, str(path))


def _run_identity(path, k):
    ort = pytest.importorskip("onnxruntime")
    s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return s.run(None, {"x": np.eye(k, dtype=np.float32)})[0]          # = W as the runtime uses it


def test_dynamic_quantisation_matches_onnxruntime(tmp_path, local):
    """Reference: onnxruntime's quantize_dynamic writes the file, onnxruntime runs it."""
    qz = pytest.importorskip("onnxruntime.quantization")
    rng = np.random.RandomState(6)
    w = rng.randn(32, 48).astype(np.float32)
    _float_matmul_model(tmp_path / "model.onnx", w)
    qz.quantize_dynamic(str(tmp_path / "model.onnx"), str(tmp_path / "model_quantized.onnx"),
                        weight_type=qz.QuantType.QInt8, per_channel=True)
    r = remote_model(tmp_path, ["model_quantized.onnx"])
    name = "model.layers.0.mlp.up_proj.weight"
    got = r.sample(name, 10 ** 6).reshape(48, 32)
    np.testing.assert_allclose(got, _run_identity(tmp_path / "model_quantized.onnx", 32).T, atol=1e-5)
    assert np.abs(got - w.T).max() < np.abs(w).max() / 100                 # 8-bit error of the float weight


def test_matmulnbits_matches_onnxruntime(tmp_path, local):
    """Reference: onnxruntime's MatMulNBitsQuantizer writes the file, onnxruntime runs it."""
    pytest.importorskip("onnxruntime")
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
    rng = np.random.RandomState(7)
    w = rng.randn(64, 40).astype(np.float32)
    _float_matmul_model(tmp_path / "model.onnx", w)
    q = MatMulNBitsQuantizer(onnx.load(str(tmp_path / "model.onnx")), block_size=32, is_symmetric=False)
    q.process()
    q.model.save_model_to_file(str(tmp_path / "model_q4.onnx"))
    ops = {n.op_type for n in onnx.load(str(tmp_path / "model_q4.onnx")).graph.node}
    assert "MatMulNBits" in ops
    r = remote_model(tmp_path, ["model_q4.onnx"])
    name = "model.layers.0.mlp.up_proj.weight"
    assert r.float_shapes()[name][1] == (40, 64)
    got = r.sample(name, 10 ** 6).reshape(40, 64)
    np.testing.assert_allclose(got, _run_identity(tmp_path / "model_q4.onnx", 64).T, atol=1e-5)


def test_token_goes_to_huggingface_only_and_not_across_redirects(monkeypatch):
    """HF_TOKEN (for gated repositories) is attached only to https://huggingface.co requests and is
    dropped when a redirect leaves that host (file bytes come from a CDN)."""
    import urllib.request
    from pedigree import remote
    monkeypatch.setenv("HF_TOKEN", "hf_test_not_a_real_token")
    assert remote._token_for("https://huggingface.co/org/m/resolve/abc/model.safetensors")
    for url in ("http://huggingface.co/x", "https://huggingface.co.evil.example/x", "https://cdn-lfs.hf.co/x",
                "https://cas-bridge.xethub.hf.co/x", "https://example.com/huggingface.co"):
        assert remote._token_for(url) is None, url
    h = remote._TokenStaysHome()
    req = urllib.request.Request("https://huggingface.co/org/m/resolve/abc/f", headers={"Range": "bytes=0-7"})
    req.add_header("Authorization", "Bearer hf_test_not_a_real_token")
    out = h.redirect_request(req, None, 302, "Found", {}, "https://cas-bridge.xethub.hf.co/signed/f")
    assert not out.has_header("Authorization") and out.get_header("Range") == "bytes=0-7"
    back = h.redirect_request(req, None, 307, "Temporary Redirect", {}, "https://huggingface.co/org/m2/resolve/abc/f")
    assert back.has_header("Authorization")                  # a rename within the Hub keeps it
    monkeypatch.delenv("HF_TOKEN")
    assert remote._token_for("https://huggingface.co/x") is None
