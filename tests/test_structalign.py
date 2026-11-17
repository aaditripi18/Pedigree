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
