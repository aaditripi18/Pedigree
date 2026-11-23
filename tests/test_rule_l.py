"""Rule (l), readers v7 (PLAN.md, amendment 2026-11-23): an underscore separates a block index too, as
in onnxruntime-genai's names, so such a file's repeated shapes pair block by block."""
from pedigree import structalign


def _genai_and_hf(layers=4, d=2048, kv=512, ff=8192):
    a, b = {}, {}
    for i in range(layers):
        for proj, shape in (("q_proj", (d, d)), ("k_proj", (kv, d)), ("v_proj", (kv, d)), ("o_proj", (d, d))):
            a[f"model_layers_{i}_attn_{proj}_MatMul_weight_quant"] = ("float", shape)
            b[f"model.layers.{i}.self_attn.{proj}.weight"] = ("float", shape)
        for proj, shape in (("gate_proj", (ff, d)), ("up_proj", (ff, d)), ("down_proj", (d, ff))):
            a[f"model_layers_{i}_mlp_{proj}_MatMul_weight_quant"] = ("float", shape)
            b[f"model.layers.{i}.mlp.{proj}.weight"] = ("float", shape)
        a[f"model.layers.{i}.input_layernorm.weight"] = ("float", (d,))
        b[f"model.layers.{i}.input_layernorm.weight"] = ("float", (d,))
    return a, b


def _expected(a_name: str) -> str:
    i = a_name.split("_")[2]
    proj = a_name.split("_MatMul")[0].split(f"_{i}_", 1)[1]
    part, leaf = proj.split("_", 1)
    return f"model.layers.{i}.{'self_attn' if part == 'attn' else 'mlp'}.{leaf}.weight"


def test_underscore_block_index(monkeypatch):
    monkeypatch.setattr(structalign, "READERS_V7", True)
    a, b = _genai_and_hf()
    (kind, mp, tr), = structalign.struct_names_v4(a, b)
    quant = {k: v for k, v in mp.items() if k.endswith("_quant")}
    assert kind == "equal" and len(quant) == 28 and not tr
    assert all(v == _expected(k) for k, v in quant.items())


def test_without_rule_l_nothing_repeated_pairs(monkeypatch):
    monkeypatch.setattr(structalign, "READERS_V7", False)
    a, b = _genai_and_hf()
    pairs = [m for _, m, _ in structalign.struct_names_v4(a, b)]
    assert all(not any(k.endswith("_quant") for k in m) for m in pairs)


def test_dotted_names_unchanged():
    for name in ("model.layers.3.mlp.up_proj.weight", "transformer/h_7/attn/c_attn/kernel", "blk.2.attn_q.weight",
                 "encoder.layer.11.output.dense.weight"):
        a, b = structalign._ANY_STACK.search(name), structalign._ANY_STACK_L.search(name)
        assert a and b and a.groups() == b.groups() and name[:a.end(1)] == name[:b.end(1)]
    for name in ("lm_head.weight", "model.embed_tokens.weight", "conv_1x1.weight"):
        assert structalign._ANY_STACK_L.search(name) is None
