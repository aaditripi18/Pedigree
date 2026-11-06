"""Decoding, alignment and similarity, on synthetic tensors (no network)."""
import numpy as np

from pedigree.align import align
from pedigree.verify import cosine, decode


def test_decode_matches_numpy_for_every_float_type():
    x = np.random.default_rng(0).standard_normal(1000)
    assert np.allclose(decode("F32", x.astype("<f4").tobytes()), x.astype(np.float32))
    assert np.allclose(decode("F16", x.astype("<f2").tobytes()), x.astype(np.float16))
    assert np.array_equal(decode("F64", x.astype("<f8").tobytes()), x)
    # bfloat16 = the top 16 bits of float32
    bf = (x.astype(np.float32).view(np.uint32) >> 16).astype("<u2")
    expect = (bf.astype(np.uint32) << 16).view(np.float32)
    assert np.array_equal(decode("BF16", bf.tobytes()), expect.astype(np.float64))


def test_cosine_separates_a_finetune_from_an_unrelated_init():
    rng = np.random.default_rng(1)
    base = rng.standard_normal(8192) * 0.02
    finetune = base + rng.standard_normal(8192) * 0.002          # small update
    unrelated = rng.standard_normal(8192) * 0.02                  # same shape, own init
    assert cosine(base, finetune) > 0.99
    assert abs(cosine(base, unrelated)) < 0.05
    # a float16 copy of the fine-tune still matches
    assert cosine(base, decode("F16", finetune.astype("<f2").tobytes())) > 0.99


def test_cosine_ignores_non_finite_values_and_handles_zero_vectors():
    a = np.array([1.0, np.nan, 2.0, np.inf])
    b = np.array([1.0, 5.0, 2.0, 1.0])
    assert abs(cosine(a, b) - 1.0) < 1e-12
    assert np.isnan(cosine(np.zeros(4), np.ones(4)))


def test_align_pairs_wrapped_names_and_legacy_layernorm():
    shapes = {"encoder.layer.0.weight": ("float", (4, 4)), "encoder.ln.weight": ("float", (4,)),
              "pooler.weight": ("float", (4, 4))}
    model = {"bert." + k.replace("ln.weight", "ln.gamma"): v for k, v in shapes.items()}
    pairs, dm, db = align(model, shapes)
    assert pairs == {"bert.encoder.layer.0.weight": "encoder.layer.0.weight",
                     "bert.encoder.ln.gamma": "encoder.ln.weight", "bert.pooler.weight": "pooler.weight"}
    assert (dm, db) == (1, 0)


def test_align_refuses_shape_mismatch():
    pairs, _, _ = align({"w": ("float", (4, 4))}, {"w": ("float", (4, 5))})
    assert pairs == {}


def test_align_pairs_old_and_new_transformers_names():
    """A ViT saved by transformers 5 and its base saved by transformers 4 pair up."""
    old = {"vit.encoder.layer.0.attention.attention.query.weight": ("float", (8, 8)),
           "vit.encoder.layer.0.attention.output.dense.weight": ("float", (8, 8)),
           "vit.encoder.layer.0.intermediate.dense.weight": ("float", (32, 8)),
           "vit.encoder.layer.0.output.dense.weight": ("float", (8, 32))}
    new = {"vit.layers.0.attention.q_proj.weight": ("float", (8, 8)),
           "vit.layers.0.attention.o_proj.weight": ("float", (8, 8)),
           "vit.layers.0.mlp.fc1.weight": ("float", (32, 8)),
           "vit.layers.0.mlp.fc2.weight": ("float", (8, 32))}
    pairs, _, _ = align(new, old)
    assert pairs == {"vit.layers.0.attention.q_proj.weight": "vit.encoder.layer.0.attention.attention.query.weight",
                     "vit.layers.0.attention.o_proj.weight": "vit.encoder.layer.0.attention.output.dense.weight",
                     "vit.layers.0.mlp.fc1.weight": "vit.encoder.layer.0.intermediate.dense.weight",
                     "vit.layers.0.mlp.fc2.weight": "vit.encoder.layer.0.output.dense.weight"}
