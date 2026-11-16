"""ggufremote.hf_names: exact names as before; nested multimodal names only by unique shape."""
from pedigree import ggufremote

META = {"general.architecture": "llama", "llama.block_count": 2}
G = {"blk.0.attn_q.weight": ("float", (8, 8)), "blk.0.ffn_down.weight": ("float", (8, 16)),
     "token_embd.weight": ("float", (100, 8))}


def test_exact_names_map_as_before():
    hf = {"model.layers.0.self_attn.q_proj.weight": ("float", (8, 8)), "model.embed_tokens.weight": ("float", (100, 8))}
    out = ggufremote.hf_names(META, G, hf)
    assert out == {"blk.0.attn_q.weight": "model.layers.0.self_attn.q_proj.weight",
                   "token_embd.weight": "model.embed_tokens.weight"}
    assert ggufremote.hf_names(META, list(G), list(hf)) == out  # plain lists: exact names only


def test_nested_language_model_names_map_by_shape():
    hf = {"model.language_model.layers.0.self_attn.q_proj.weight": ("float", (8, 8)),
          "model.language_model.layers.0.mlp.down_proj.weight": ("float", (8, 16)),
          "model.vision_tower.layers.0.mlp.down_proj.weight": ("float", (4, 4))}  # other shape: not taken
    out = ggufremote.hf_names(META, G, hf)
    assert out == {"blk.0.attn_q.weight": "model.language_model.layers.0.self_attn.q_proj.weight",
                   "blk.0.ffn_down.weight": "model.language_model.layers.0.mlp.down_proj.weight"}


def test_two_nested_candidates_with_the_same_shape_are_left_unmapped():
    hf = {"model.language_model.layers.0.self_attn.q_proj.weight": ("float", (8, 8)),
          "model.vision_tower.layers.0.self_attn.q_proj.weight": ("float", (8, 8))}
    assert ggufremote.hf_names(META, G, hf) == {}
