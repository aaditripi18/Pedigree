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


# --- v4 (amendment 2026-11-20 (2), H25) ------------------------------------------------------------

def test_v4_repeated_shapes_pair_by_leaf_name_and_size1_dims_are_ignored():
    from pedigree.structalign import struct_names_v4
    g, h = {}, {}
    for i in range(2):
        for k in ("q", "k", "v", "out"):
            g[f"encoder.layers.{i}.attn.{k}.weight"] = ("f", (16, 16))
        g[f"encoder.layers.{i}.ff1.linear1.weight"] = ("f", (64, 16))
        g[f"encoder.layers.{i}.ff2.linear1.weight"] = ("f", (64, 16))
        g[f"encoder.layers.{i}.conv.pw1.weight"] = ("f", (32, 16))
        for k in ("q", "k", "v", "out"):
            h[f"encoder.layers.{i}.self_attn.linear_{k}.weight"] = ("f", (16, 16))
        h[f"encoder.layers.{i}.feed_forward1.linear1.weight"] = ("f", (64, 16))
        h[f"encoder.layers.{i}.feed_forward2.linear1.weight"] = ("f", (64, 16))
        h[f"encoder.layers.{i}.conv.pointwise_conv1.weight"] = ("f", (32, 16, 1))
    [(label, m, tr)] = struct_names_v4(g, h)
    assert label == "equal" and not tr and len(m) == len(g)
    assert m["encoder.layers.1.attn.k.weight"] == "encoder.layers.1.self_attn.linear_k.weight"
    assert m["encoder.layers.0.ff2.linear1.weight"] == "encoder.layers.0.feed_forward2.linear1.weight"
    assert m["encoder.layers.0.conv.pw1.weight"] == "encoder.layers.0.conv.pointwise_conv1.weight"


def test_v4_leaves_equally_similar_names_unpaired():
    from pedigree.structalign import struct_names_v4
    g = {f"blk.0.{k}": ("f", (8, 8)) for k in ("xa", "xb")}
    h = {f"model.layers.0.{k}": ("f", (8, 8)) for k in ("ya", "yb")}
    g["blk.0.w"], h["model.layers.0.u"] = ("f", (8, 4)), ("f", (8, 4))
    [(_, m, _)] = struct_names_v4(g, h)
    assert m == {"blk.0.w": "model.layers.0.u", "blk.0.xa": "model.layers.0.ya", "blk.0.xb": "model.layers.0.yb"}
    h2 = {f"model.layers.0.{k}": ("f", (8, 8)) for k in ("p", "q")}
    [(_, m2, _)] = struct_names_v4({k: v for k, v in g.items() if k != "blk.0.w"}, h2)
    assert m2 == {}                                        # no single best match: left out


def test_v4_unequal_depth_gives_three_maps():
    from pedigree.structalign import struct_names_v4
    g = {f"blk.{i}.w": ("f", (8, 8)) for i in range(5)}
    h = {f"model.layers.{i}.w": ("f", (8, 8)) for i in range(3)}
    maps = {label: m for label, m, _ in struct_names_v4(g, h)}
    assert maps["first"] == {f"blk.{i}.w": f"model.layers.{i}.w" for i in range(3)}
    assert maps["last"] == {f"blk.{i}.w": f"model.layers.{i - 2}.w" for i in range(2, 5)}
    # round(i * 2 / 4): blocks 0, 1 -> 0 and 3, 4 -> 2; a base tensor is never paired twice
    assert maps["proportional"] == {"blk.0.w": "model.layers.0.w", "blk.2.w": "model.layers.1.w",
                                    "blk.3.w": "model.layers.2.w"}


class _Fake:
    def __init__(self, repo, tensors):
        self.repo, self.commit, self.gguf_meta, self.t = repo, "x", None, tensors

    def float_shapes(self):
        return {n: ("float", a.shape) for n, a in self.t.items()}

    def sample(self, name, max_bytes=32 * 1024):
        return self.t[name].ravel()[:max_bytes // 4].astype("float64")


def test_v4_depth_maps_find_kept_blocks_and_stay_low_for_independent_models():
    """The upscaled model keeps the base's blocks 0-7 at their own indices (as H23's opposite pair
    did); the best of three maps finds it. Independent weights stay far below every threshold."""
    import numpy as np
    from pedigree.verify import compare
    rng = np.random.RandomState(3)

    def model(nb, seed_blocks):
        return {f"blk.{i}.{k}": seed_blocks(i, k) for i in range(nb) for k in ("attn_q.weight", "ffn_up.weight")}
    shape = {"attn_q.weight": (64, 128), "ffn_up.weight": (256, 64)}
    base = {f"model.layers.{i}.{k2}": rng.randn(*shape[k]).astype("float32")
            for i in range(8) for k, k2 in (("attn_q.weight", "self_attn.q_proj.weight"), ("ffn_up.weight", "mlp.up_proj.weight"))}
    up = model(11, lambda i, k: base[f"model.layers.{i}." + ("self_attn.q_proj.weight" if k == "attn_q.weight" else "mlp.up_proj.weight")]
               if i < 8 else rng.randn(*shape[k]).astype("float32"))
    c = compare(_Fake("up", up), _Fake("base", base), alignment="struct4")
    assert c["alignment_map"] == "first" and c["median_cosine"] > 0.99
    worst = 0.0
    for s in range(20):
        r = np.random.RandomState(100 + s)
        other = model(11, lambda i, k: r.randn(*shape[k]).astype("float32"))
        worst = max(worst, compare(_Fake("o", other), _Fake("base", base), alignment="struct4")["median_cosine"])
    assert worst < 0.0255 / 2


# --- amendment 2026-11-21: (g) positional and (h) prefix pairings (with readers v5) ---------------

def test_v5_positional_pairs_numbered_tensors_only_when_every_shape_matches():
    """(g) A converter that numbered its tensors (t0000...) in the base's name order."""
    from pedigree.structalign import positional_names
    b = {"dec.a.weight": ("float", (8, 4)), "dec.b.weight": ("float", (4, 8)), "dec.c.weight": ("float", (8, 8))}
    a = {"t0002": ("float", (8, 8)), "t0000": ("float", (8, 4)), "t0001": ("float", (4, 8, 1))}
    want = {"t0000": "dec.a.weight", "t0001": "dec.b.weight", "t0002": "dec.c.weight"}
    assert positional_names(a, b) == [("positional-header", want)]          # header and name order agree
    assert positional_names(b, a) == [("positional-header", {v: k for k, v in want.items()})]
    a["t0001"] = ("float", (8, 4))                         # one shape out of step: nothing is paired
    assert positional_names(a, b) == []
    assert positional_names({"x.w": ("float", (8, 4)), "y.w": ("float", (4, 8))}, b) == []   # names carry meaning


def test_v5_prefix_and_peft_base_layer():
    """(h) The base inside a larger model under `llm.base_model.model.`, LoRA left unmerged."""
    from pedigree.structalign import prefix_names
    base = {f"model.layers.{i}.self_attn.{p}.weight": ("float", (64, 64)) for i in range(4) for p in ("q_proj", "k_proj")}
    big = {f"llm.base_model.model.model.layers.{i}.self_attn.{p}." + ("base_layer.weight" if p == "q_proj" else "weight"):
           ("float", (64, 64)) for i in range(4) for p in ("q_proj", "k_proj")}
    big.update({"perception.encoder.layers.0.conv.weight": ("float", (64, 64)),
                f"llm.base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight": ("float", (8, 64))})
    m = prefix_names(big, base)
    assert len(m) == 8 and m["llm.base_model.model.model.layers.2.self_attn.q_proj.base_layer.weight"] == \
        "model.layers.2.self_attn.q_proj.weight"
    assert prefix_names(base, big) == {v: k for k, v in m.items()}


def test_v5_rules_compare_end_to_end_and_independent_models_stay_low(monkeypatch):
    import numpy as np
    from pedigree import structalign
    from pedigree.verify import compare
    rng = np.random.RandomState(9)
    base = {f"model.layers.{i}.mlp.{p}.weight": rng.randn(64, 128).astype("float32") for i in range(6) for p in ("up", "gate")}
    numbered = {f"t{j:04d}": base[n] for j, n in enumerate(sorted(base))}
    nested = {"llm.base_model.model." + n.replace("up.weight", "up.base_layer.weight"): w for n, w in base.items()}
    for other, label in ((numbered, "positional-name"), (nested, "prefix")):
        c = compare(_Fake("m", other), _Fake("base", base), alignment="auto4")
        assert c["alignment_map"] == label and c["compared"] >= 8 and c["median_cosine"] > 0.999
    # every shape equal: both orders qualify and are scored; the name order (the true one) wins
    assert set(compare(_Fake("m", numbered), _Fake("base", base), alignment="auto4")["map_scores"]) ==         {"positional-header", "positional-name"}
    worst = 0.0
    for s in range(20):                     # the best of two orders, on 20 independent models
        r = np.random.RandomState(200 + s)
        indep = {f"t{j:04d}": r.randn(64, 128).astype("float32") for j in range(12)}
        c = compare(_Fake("m", indep), _Fake("base", base), alignment="auto4")
        assert c["alignment_map"].startswith("positional")
        worst = max(worst, c["median_cosine"])
    assert worst < 0.0255 / 2
    monkeypatch.setattr(structalign, "V5", False)                 # H25 as it ran
    c = compare(_Fake("m", numbered), _Fake("base", base), alignment="auto4")
    assert not c["alignment_map"].startswith("positional")


def test_v5_stack_partner_by_shape_overlap(monkeypatch):
    """(i) A multimodal base (an audio stack of the text stack's depth with other shapes): the GGUF
    text stack pairs with the stack whose shapes it shares (LiquidAI/d1-omni-600M in H25)."""
    from pedigree import structalign
    from pedigree.structalign import struct_names_v4
    gguf = {f"blk.{i}.{k}": ("float", s) for i in range(4) for k, s in
            (("ffn_up.weight", (96, 32)), ("ffn_down.weight", (32, 96)), ("attn_norm.weight", (32,)))}
    base = {f"encoder.layers.{i}.{k}": ("float", s) for i in range(4) for k, s in
            (("mlp.up.weight", (96, 32)), ("mlp.down.weight", (32, 96)), ("norm.weight", (32,)))}
    base.update({f"audio.encoder.layers.{i}.conv.bias": ("float", (32,)) for i in range(4)})
    base.update({f"audio.encoder.layers.{i}.conv.{k}.weight": ("float", (64, 8)) for i in range(4) for k in "abcd"})
    (_, mp, _), = struct_names_v4(gguf, base)
    assert mp["blk.2.ffn_up.weight"] == "encoder.layers.2.mlp.up.weight" and len(mp) == 12
    monkeypatch.setattr(structalign, "V5", False)          # H25 as it ran: the audio stack (larger) won
    (_, mp, _), = struct_names_v4(gguf, base)
    assert "blk.2.ffn_up.weight" not in mp


TWO_TOWERS = r'''
import json
from pedigree.structalign import struct_names_v3, struct_names_v4
leaf = (("mlp.fc1.weight", (96, 32)), ("mlp.fc2.weight", (32, 96)), ("layer_norm1.weight", (32,)))
onnx = {f"vision_model.encoder.layers.{i}.{k}": ("float", s) for i in range(3) for k, s in leaf}
base = {f"{t}_model.encoder.layers.{i}.{k}": ("float", s) for t in ("text", "vision") for i in range(3) for k, s in leaf}
(_, mp, _), = struct_names_v4(onnx, base)
v3, _ = struct_names_v3(onnx, base)
print(json.dumps([mp["vision_model.encoder.layers.1.mlp.fc1.weight"], v3["vision_model.encoder.layers.1.mlp.fc1.weight"]]))
'''


def test_stack_partner_ties_are_settled_by_name_the_same_in_every_run():
    """Two towers of one shape (SigLIP2 so400m, H26a): every tie-break but the name ties, so the
    partner was a set's iteration order, which changes with Python's hash seed. Now the stack whose
    name is closest wins, in every run."""
    import os
    import subprocess
    import sys
    out = set()
    for seed in ("0", "1", "2", "3", "4", "5", "6", "7"):
        r = subprocess.run([sys.executable, "-c", TWO_TOWERS], capture_output=True, text=True, check=True,
                           env={**os.environ, "PYTHONHASHSEED": seed})
        out.add(r.stdout.strip())
    assert out == {'["vision_model.encoder.layers.1.mlp.fc1.weight", "vision_model.encoder.layers.1.mlp.fc1.weight"]'}
