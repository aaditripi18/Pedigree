"""R1 readers (PLAN.md, amendment 2026-11-20): every format is read exactly as its reference library
writes it, on a file that library wrote; quantised weights are read as the values they stand for."""
import io
import os
import pickle
import struct
import tarfile

import numpy as np
import pytest
import torch

from pedigree import formats, remote, torchzip
from pedigree.verify import Remote, decode


@pytest.fixture
def local(monkeypatch):
    def fake(url, start, end, retries=0):
        with open(url, "rb") as fh:
            fh.seek(start)
            return fh.read(end - start + 1)
    for m in (formats, torchzip, remote):
        monkeypatch.setattr(m, "fetch_range", fake)
    import pedigree.verify as verify
    monkeypatch.setattr(formats, "resolve_url", lambda repo, f, rev="main": os.path.join(repo, f))
    monkeypatch.setattr(verify, "resolve_url", lambda repo, f, rev="main": os.path.join(repo, f))


def values(url, spec):
    b, e = spec["data_offsets"]
    with open(url, "rb") as fh:
        fh.seek(b)
        return decode(spec["dtype"], fh.read(e - b)).reshape(spec["shape"])


def remote_model(folder, files):
    sizes = {f: os.path.getsize(os.path.join(folder, f)) for f in files}
    return Remote.from_files(str(folder), "local", files, sizes)


# --- ONNX ----------------------------------------------------------------------------------------

def test_onnx_initializers_constants_external_and_dequantize(tmp_path, local):
    onnx = pytest.importorskip("onnx")
    from onnx import helper, numpy_helper, TensorProto
    rng = np.random.RandomState(0)
    w = rng.randn(8, 16).astype(np.float32)
    big = rng.randn(64, 64).astype(np.float32)
    half = rng.randn(4, 4).astype(np.float16)
    c = rng.randn(5, 3).astype(np.float32)
    q = rng.randint(-120, 120, size=(6, 10)).astype(np.int8)
    scale, zp = np.array(0.05, np.float32), np.array(3, np.int8)
    packed = helper.make_tensor("packed", TensorProto.FLOAT, [2, 3], rng.randn(6).astype(np.float32).tolist())
    inits = [numpy_helper.from_array(w, "w"), numpy_helper.from_array(big, "big"),
             numpy_helper.from_array(half, "half"), numpy_helper.from_array(q, "q"),
             numpy_helper.from_array(scale, "q_scale"), numpy_helper.from_array(zp, "q_zp"), packed]
    nodes = [helper.make_node("Constant", [], ["cst"], value=numpy_helper.from_array(c)),
             helper.make_node("DequantizeLinear", ["q", "q_scale", "q_zp"], ["qf"]),
             helper.make_node("Identity", ["w"], ["out"])]
    g = helper.make_graph(nodes, "g", [], [helper.make_tensor_value_info("out", TensorProto.FLOAT, [8, 16])], inits)
    m = helper.make_model(g)
    path = tmp_path / "model.onnx"
    onnx.save_model(m, str(path), save_as_external_data=True, all_tensors_to_one_file=True,
                    location="model.onnx_data", size_threshold=1024)
    ref = {t.name: numpy_helper.to_array(t, base_dir=str(tmp_path)) for t in onnx.load(str(path)).graph.initializer}
    header, deq = formats.onnx_header(str(tmp_path), "local", "model.onnx", os.path.getsize(path))
    for name in ("w", "big", "half", "q", "packed"):
        np.testing.assert_array_equal(values(*header[name]), ref[name])
    assert header["big"][0].endswith("model.onnx_data")          # read from the external data file
    np.testing.assert_array_equal(values(*header["cst"]), c)
    assert deq == {"q": ("q_scale", "q_zp", 1)}
    r = remote_model(tmp_path, ["model.onnx"])
    assert "q" in r.float_shapes() and "q_scale" not in r.float_shapes()
    np.testing.assert_allclose(r.sample("q", 10 ** 6), ((q.astype(np.float64) - 3) * 0.05).ravel(), rtol=1e-6)


# --- NumPy .npz ----------------------------------------------------------------------------------

def test_npz_members_and_fortran_order(tmp_path, local):
    rng = np.random.RandomState(1)
    a = rng.randn(7, 9).astype(np.float32)
    b = rng.randn(12).astype(np.float16)
    f = np.asfortranarray(rng.randn(5, 4).astype(np.float32))
    u = rng.randint(0, 2 ** 31, size=(3, 4)).astype(np.uint32)
    np.savez(tmp_path / "weights.npz", **{"enc.a": a, "b": b, "f": f, "u": u})
    h = formats.npz_header(str(tmp_path / "weights.npz"), os.path.getsize(tmp_path / "weights.npz"))
    np.testing.assert_array_equal(values(*h["enc.a"]), a)
    np.testing.assert_array_equal(values(*h["b"]), b)
    np.testing.assert_array_equal(values(*h["u"]), u)
    assert h["f"][1]["order"] == "F"
    got = decode("F32", remote.tensor_bytes(h["f"][0], 0, h["f"][1], f.size * 4))
    np.testing.assert_array_equal(got, np.ascontiguousarray(f).ravel())


# --- CTranslate2 ---------------------------------------------------------------------------------

def _ct2(tmp_path, quantization):
    from ctranslate2.specs import common_spec, model_spec as ms

    class Tiny(ms.ModelSpec):
        def __init__(self):
            super().__init__()
            self.dense = common_spec.LinearSpec()

        @property
        def name(self):
            return "Tiny"

        def get_default_config(self):
            return None

        def get_vocabulary_size(self):
            return 0
    t = Tiny()
    w = np.random.RandomState(2).randn(16, 32).astype(np.float32)
    t.dense.weight, t.dense.bias = w.copy(), np.arange(16, dtype=np.float32)
    t.validate()
    t.optimize(quantization=quantization)
    tmp_path.mkdir(parents=True, exist_ok=True)
    t.save(str(tmp_path))
    return w


def test_ctranslate2_float_and_int8(tmp_path, local):
    pytest.importorskip("ctranslate2")
    w = _ct2(tmp_path / "f", None)
    h = formats.ct2_header(str(tmp_path / "f" / "model.bin"), os.path.getsize(tmp_path / "f" / "model.bin"))
    np.testing.assert_array_equal(values(*h["dense/weight"]), w)
    np.testing.assert_array_equal(values(*h["dense/bias"]), np.arange(16, dtype=np.float32))
    w = _ct2(tmp_path / "q", "int8")
    r = remote_model(tmp_path / "q", ["model.bin"])             # not PyTorch: read as CTranslate2
    assert r.quant["dense/weight"]["kind"] == "ct2" and "dense/weight_scale" not in r.float_shapes()
    got = r.sample("dense/weight", 10 ** 6)
    assert np.abs(got - w.ravel()).max() < np.abs(w).max(axis=1).max() / 100     # int8 rounding only


# --- NeMo (tar) and PyTorch with framework objects -----------------------------------------------

def test_nemo_checkpoint_read_in_place(tmp_path, local):
    sd = {"encoder.layers.0.weight": torch.randn(6, 4), "decoder.bias": torch.randn(3)}
    ck = tmp_path / "model_weights.ckpt"
    torch.save(sd, ck)
    (tmp_path / "model_config.yaml").write_text("name: tiny\n")
    with tarfile.open(tmp_path / "tiny.nemo", "w", format=tarfile.USTAR_FORMAT) as tf:
        tf.add(tmp_path / "model_config.yaml", arcname="./model_config.yaml")
        tf.add(ck, arcname="./model_weights.ckpt")
    r = remote_model(tmp_path, ["tiny.nemo"])
    for k, t in sd.items():
        np.testing.assert_array_equal(r.sample(k, 10 ** 6), t.numpy().ravel())


class Vocabulary:                     # stands for a framework object pickled beside the weights
    def __init__(self):
        self.items = {"a": 1}


def test_unknown_classes_become_placeholders_but_functions_are_refused(tmp_path, local):
    obj = {"state_dict": {"w": torch.randn(4, 4)}, "dictionary": Vocabulary(), "hparams": {"lr": 0.1}}
    torch.save(obj, tmp_path / "flair.pt")
    _, h = torchzip.read_header(str(tmp_path / "flair.pt"), os.path.getsize(tmp_path / "flair.pt"))
    assert set(h) == {"w"}

    class Evil:
        def __reduce__(self):
            return (os.system, ("echo pwned",))
    data = pickle.dumps({"x": Evil()}, protocol=2)
    with pytest.raises(pickle.UnpicklingError):
        torchzip._Unpickler(io.BytesIO(data)).load()


def test_pickled_whole_module(tmp_path, local):
    m = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.ReLU(), torch.nn.Linear(3, 2))
    torch.save(m, tmp_path / "m.pt")
    _, h = torchzip.read_header(str(tmp_path / "m.pt"), os.path.getsize(tmp_path / "m.pt"))
    assert set(h) == set(m.state_dict())
    for k, t in m.state_dict().items():
        np.testing.assert_array_equal(values(str(tmp_path / "m.pt"), h[k]), t.numpy())


# --- TFLite and LiteRT-LM ------------------------------------------------------------------------

def _tflite_bytes(w: np.ndarray, q: np.ndarray, scale: list, zero: list) -> bytes:
    import flatbuffers
    import importlib
    B, M, Q, S, T = (importlib.import_module("tflite." + n) for n in ("Buffer", "Model", "QuantizationParameters", "SubGraph", "Tensor"))
    b = flatbuffers.Builder(1024)

    def vec(start, items, put):
        start(b, len(items))
        for x in reversed(items):
            put(x)
        return b.EndVector()

    def buffer(raw):
        d = b.CreateByteVector(raw) if raw is not None else None
        B.BufferStart(b)
        if d is not None:
            B.BufferAddData(b, d)
        return B.BufferEnd(b)
    bufs = [buffer(None), buffer(w.tobytes()), buffer(q.tobytes())]

    def tensor(name, shape, ttype, bi, quant=None):
        n = b.CreateString(name)
        s = vec(lambda bb, k: T.TensorStartShapeVector(bb, k), list(shape), b.PrependInt32)
        T.TensorStart(b)
        T.TensorAddShape(b, s)
        T.TensorAddType(b, ttype)
        T.TensorAddBuffer(b, bi)
        T.TensorAddName(b, n)
        if quant is not None:
            T.TensorAddQuantization(b, quant)
        return T.TensorEnd(b)
    sc = vec(lambda bb, k: Q.QuantizationParametersStartScaleVector(bb, k), scale, b.PrependFloat32)
    zp = vec(lambda bb, k: Q.QuantizationParametersStartZeroPointVector(bb, k), zero, b.PrependInt64)
    Q.QuantizationParametersStart(b)
    Q.QuantizationParametersAddScale(b, sc)
    Q.QuantizationParametersAddZeroPoint(b, zp)
    Q.QuantizationParametersAddQuantizedDimension(b, 0)
    qp = Q.QuantizationParametersEnd(b)
    tensors = [tensor("dense/w", w.shape, 0, 1), tensor("conv/q", q.shape, 9, 2, qp), tensor("act", [1, 4], 0, 0)]
    tv = vec(lambda bb, k: S.SubGraphStartTensorsVector(bb, k), tensors, b.PrependUOffsetTRelative)
    S.SubGraphStart(b)
    S.SubGraphAddTensors(b, tv)
    sg = S.SubGraphEnd(b)
    sgs = vec(lambda bb, k: M.ModelStartSubgraphsVector(bb, k), [sg], b.PrependUOffsetTRelative)
    bv = vec(lambda bb, k: M.ModelStartBuffersVector(bb, k), bufs, b.PrependUOffsetTRelative)
    M.ModelStart(b)
    M.ModelAddVersion(b, 3)
    M.ModelAddSubgraphs(b, sgs)
    M.ModelAddBuffers(b, bv)
    b.Finish(M.ModelEnd(b), file_identifier=b"TFL3")
    return bytes(b.Output())


def _tflite_case():
    rng = np.random.RandomState(3)
    w = rng.randn(4, 6).astype(np.float32)
    q = rng.randint(-100, 100, size=(3, 5)).astype(np.int8)
    return w, q, [0.1, 0.2, 0.3], [0, 1, -2]


def test_tflite_float_and_per_channel_int8(tmp_path, local):
    import tflite
    w, q, scale, zero = _tflite_case()
    (tmp_path / "m.tflite").write_bytes(_tflite_bytes(w, q, scale, zero))
    ref = tflite.Model.GetRootAsModel((tmp_path / "m.tflite").read_bytes(), 0)
    t0 = ref.Subgraphs(0).Tensors(0)
    np.testing.assert_array_equal(ref.Buffers(t0.Buffer()).DataAsNumpy().view(np.float32).reshape(4, 6), w)
    h = formats.tflite_header(str(tmp_path / "m.tflite"), os.path.getsize(tmp_path / "m.tflite"))
    assert set(h) == {"dense/w", "conv/q"}                          # the activation has no data
    np.testing.assert_array_equal(values(*h["dense/w"]), w)
    r = remote_model(tmp_path, ["m.tflite"])
    want = (q.astype(np.float64) - np.array(zero).reshape(3, 1)) * np.array(scale).reshape(3, 1)
    np.testing.assert_allclose(r.sample("conv/q", 10 ** 6), want.ravel(), rtol=1e-6)


def test_litertlm_container_sections(tmp_path, local):
    import flatbuffers
    from litert_lm_builder import litertlm_header_schema_py_generated as H
    from litert_lm_builder import litertlm_peek
    w, q, scale, zero = _tflite_case()
    model = _tflite_bytes(w, q, scale, zero)
    begin = 4096
    b = flatbuffers.Builder(256)
    H.SectionObjectStart(b)
    H.SectionObjectAddBeginOffset(b, begin)
    H.SectionObjectAddEndOffset(b, begin + len(model))
    H.SectionObjectAddDataType(b, H.AnySectionDataType.TFLiteModel)
    so = H.SectionObjectEnd(b)
    H.SectionMetadataStartObjectsVector(b, 1)
    b.PrependUOffsetTRelative(so)
    objs = b.EndVector()
    H.SectionMetadataStart(b)
    H.SectionMetadataAddObjects(b, objs)
    sm = H.SectionMetadataEnd(b)
    H.LiteRTLMMetaDataStart(b)
    H.LiteRTLMMetaDataAddSectionMetadata(b, sm)
    b.Finish(H.LiteRTLMMetaDataEnd(b))
    head = bytes(b.Output())
    raw = bytearray(begin + len(model))
    raw[0:8] = b"LITERTLM"
    raw[8:20] = struct.pack("<III", 1, 0, 0)
    raw[24:32] = struct.pack("<Q", 32 + len(head))
    raw[32:32 + len(head)] = head
    raw[begin:] = model
    path = tmp_path / "m.litertlm"
    path.write_bytes(bytes(raw))
    meta = litertlm_peek.read_litertlm_header(str(path), io.StringIO())       # the reference reader
    o = meta.SectionMetadata().Objects(0)
    assert formats.litertlm_sections(str(path), len(raw)) == [(o.DataType(), o.BeginOffset(), o.EndOffset())]
    h = formats.litertlm_header(str(path), len(raw))
    np.testing.assert_array_equal(values(*h["s0/dense/w"]), w)


# --- PaddlePaddle --------------------------------------------------------------------------------

def test_paddle_pdiparams(tmp_path, local):
    paddle = pytest.importorskip("paddle")
    paddle.seed(0)
    layer = paddle.nn.Sequential(paddle.nn.Linear(6, 4), paddle.nn.Linear(4, 3))
    paddle.jit.save(layer, str(tmp_path / "inference"),
                    input_spec=[paddle.static.InputSpec(shape=[None, 6], dtype="float32")])
    path = tmp_path / "inference.pdiparams"
    h = formats.pdiparams_header(str(path), os.path.getsize(path))
    got = sorted((tuple(s["shape"]), values(u, s).astype(np.float32).tobytes()) for u, s in h.values())
    want = sorted((tuple(p.shape), p.numpy().astype(np.float32).tobytes()) for p in layer.state_dict().values())
    assert all(s["dtype"] == "F32" for _, s in h.values())
    assert got == want


# --- file choice ---------------------------------------------------------------------------------

def test_other_unit_choice():
    from experiments.e2_audit import other_unit
    size = {"IQ3_S/m-IQ3_S-00001-of-00002.gguf": 9, "IQ3_S/m-IQ3_S-00002-of-00002.gguf": 9,
            "Q8_0/m-Q8_0-00001-of-00002.gguf": 9, "Q8_0/m-Q8_0-00002-of-00002.gguf": 9}
    assert other_unit(list(size), size) == ["Q8_0/m-Q8_0-00001-of-00002.gguf", "Q8_0/m-Q8_0-00002-of-00002.gguf"]
    size = {"bigvgan_discriminator_optimizer.pt": 99, "bigvgan_generator.pt": 5}
    assert other_unit(list(size), size) == ["bigvgan_generator.pt"]
    size = {"onnx/model.onnx": 4, "onnx/model_q4.onnx": 9, "onnx/model_fp16.onnx": 2}
    assert other_unit(list(size), size) == ["onnx/model.onnx"]
    assert other_unit(["model.bin", "vocabulary.txt"], {"model.bin": 1}) == ["model.bin"]


def test_gzip_nemo_member_extracted_once(tmp_path, local, monkeypatch):
    from pedigree import hub
    sd = {"encoder.w": torch.randn(5, 7)}
    torch.save(sd, tmp_path / "model_weights.ckpt")
    with tarfile.open(tmp_path / "old.nemo", "w:gz") as tf:
        tf.add(tmp_path / "model_weights.ckpt", arcname="./model_weights.ckpt")
    monkeypatch.setattr(hub, "_open", lambda url, headers=None, timeout=60: open(url, "rb"))
    monkeypatch.setenv("PEDIGREE_CACHE", str(tmp_path / "cache"))
    r = remote_model(tmp_path, ["old.nemo"])
    np.testing.assert_array_equal(r.sample("encoder.w", 10 ** 6), sd["encoder.w"].numpy().ravel())
