from pedigree.structalign import struct_names


def sh(d):
    return {k: ("float", v) for k, v in d.items()}


def test_pairs_unique_shapes_per_block_and_skips_ambiguous():
    g = sh({"token_embd.weight": (100, 8), "blk.0.ffn_up.weight": (32, 8), "blk.0.ffn_down.weight": (8, 32),
            "blk.0.attn_q.weight": (8, 8), "blk.0.attn_output.weight": (8, 8),
            "blk.1.ffn_up.weight": (32, 8), "blk.1.ffn_down.weight": (8, 32)})
    h = sh({"model.embed.weight": (100, 8), "model.layers.0.mlp.up.weight": (32, 8),
            "model.layers.0.mlp.down.weight": (8, 32), "model.layers.0.attn.q.weight": (8, 8),
            "model.layers.0.attn.o.weight": (8, 8), "model.layers.1.mlp.up.weight": (32, 8),
            "model.layers.1.mlp.down.weight": (8, 32)})
    m = struct_names(g, h)
    assert m["blk.1.ffn_down.weight"] == "model.layers.1.mlp.down.weight"
    assert m["token_embd.weight"] == "model.embed.weight"
    assert "blk.0.attn_q.weight" not in m and "blk.0.attn_output.weight" not in m   # same shape: ambiguous
    assert len(m) == 5


def test_block_count_must_match():
    g = sh({"blk.0.a": (4, 4), "blk.1.a": (4, 4)})
    h = sh({"layers.0.a": (4, 4)})
    assert struct_names(g, h) == {}


def test_v2_aligns_the_language_model_beside_a_vision_tower_and_an_extra_block():
    from pedigree.structalign import struct_names, struct_names_v2
    g, h = {}, {}
    for i in range(4):                                     # GGUF: 4 LM blocks + 1 appended (MTP)
        g[f"blk.{i}.ffn_up.weight"] = ("float", (32, 8))
        g[f"blk.{i}.ffn_down.weight"] = ("float", (8, 32))
        h[f"model.language_model.layers.{i}.mlp.up.weight"] = ("float", (32, 8))
        h[f"model.language_model.layers.{i}.mlp.down.weight"] = ("float", (8, 32))
    g["blk.4.ffn_up.weight"] = ("float", (32, 8))
    for i in range(2):                                     # HF: a 2-block vision tower and an MTP layer
        h[f"model.visual.blocks.{i}.mlp.fc1.weight"] = ("float", (16, 4))
    h["mtp.layers.0.mlp.up.weight"] = ("float", (32, 8))
    assert struct_names(g, h) == {}                        # v1: block counts differ, nothing aligns
    m = struct_names_v2(g, h)
    assert m["blk.3.ffn_down.weight"] == "model.language_model.layers.3.mlp.down.weight"
    assert "blk.4.ffn_up.weight" not in m and len(m) == 8


def test_v2_reads_clip_stacks():
    from pedigree.structalign import struct_names_v2
    g = {f"v.blk.{i}.ffn_up.weight": ("float", (64, 16)) for i in range(3)}
    g.update({f"t.blk.{i}.ffn_up.weight": ("float", (48, 12)) for i in range(2)})
    h = {f"vision_model.encoder.layers.{i}.mlp.fc1.weight": ("float", (64, 16)) for i in range(3)}
    h.update({f"text_model.encoder.layers.{i}.mlp.fc1.weight": ("float", (48, 12)) for i in range(2)})
    m = struct_names_v2(g, h)
    assert m["v.blk.2.ffn_up.weight"] == "vision_model.encoder.layers.2.mlp.fc1.weight"
    assert m["t.blk.1.ffn_up.weight"] == "text_model.encoder.layers.1.mlp.fc1.weight"


# --- v3 (amendment 2026-11-20, H23) ---------------------------------------------------------------

def test_v3_pairs_stacks_by_role_not_size():
    from pedigree.structalign import names_v3
    g = {f"enc.blk.{i}.attn_q.weight": ("f", (64, 64)) for i in range(3)}
    h = {}
    for side, extra in (("encoder", 0), ("decoder", 1)):
        for i in range(3):
            h[f"{side}.block.{i}.layer.0.SelfAttention.q.weight"] = ("f", (64, 64))
            if extra:
                h[f"{side}.block.{i}.layer.1.EncDecAttention.q.weight"] = ("f", (64, 32))
    m, tr = names_v3(g, h)
    assert all(v.startswith("encoder.") for v in m.values()) and len(m) == 3 and not tr


def test_v3_unequal_depth_is_order_preserving():
    from pedigree.structalign import struct_names_v3
    g = {f"blk.{i}.w": ("f", (8, 8)) for i in range(4)}
    h = {f"model.layers.{i}.w": ("f", (8, 8)) for i in range(7)}
    m, _ = struct_names_v3(g, h)
    assert m == {"blk.0.w": "model.layers.0.w", "blk.1.w": "model.layers.2.w",
                 "blk.2.w": "model.layers.4.w", "blk.3.w": "model.layers.6.w"}


def test_v3_direct_names_after_prefix():
    from pedigree.structalign import names_v3
    g = {f"model.diffusion_model.blocks.{i}.attn.qkv.weight": ("f", (96, 32)) for i in range(3)}
    g["model.diffusion_model.patch_embed.weight"] = ("f", (32, 4))
    h = {k.replace("model.diffusion_model.", ""): v for k, v in g.items()}
    m, tr = names_v3(g, h)
    assert m == {k: k.replace("model.diffusion_model.", "") for k in g} and not tr


def test_v3_transposed_2d_tensors_are_read_as_transposes(tmp_path, monkeypatch):
    import numpy as np
    import json, struct
    from pedigree import remote, structalign
    from pedigree.verify import Remote, Renamed
    rng = np.random.RandomState(0)
    w = rng.randn(6, 10).astype(np.float32)
    head = json.dumps({"onnx::MatMul_7": {"dtype": "F32", "shape": [10, 6], "data_offsets": [0, 240]}}).encode()
    (tmp_path / "a.safetensors").write_bytes(struct.pack("<Q", len(head)) + head + np.ascontiguousarray(w.T).tobytes())

    def fake(url, a, b, retries=0):
        with open(url, "rb") as fh:
            fh.seek(a)
            return fh.read(b - a + 1)
    monkeypatch.setattr(remote, "fetch_range", fake)
    import pedigree.verify as verify
    monkeypatch.setattr(verify, "resolve_url", lambda repo, f, rev="main": str(tmp_path / f))
    r = Remote.from_files("x", "local", ["a.safetensors"], {"a.safetensors": 0})
    m, tr = structalign.struct_names_v3(r.float_shapes(), {"layer.dense.weight": ("f", (6, 10))})
    assert m == {"onnx::MatMul_7": "layer.dense.weight"} and tr == {"onnx::MatMul_7"}
    view = Renamed(r, m, tr)
    assert view.float_shapes()["layer.dense.weight"][1] == (6, 10)
    np.testing.assert_array_equal(view.sample("layer.dense.weight"), w.ravel())
