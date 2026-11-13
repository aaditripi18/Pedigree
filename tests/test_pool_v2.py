"""Pool v2's rules (pedigree/pool_v2.py), each on the case that motivated it."""
import gzip
import json

from pedigree import pool
from pedigree.pool_v2 import PoolV2, float_params, name_match


def test_name_match_drops_owner_and_quantisation_words():
    assert name_match("mradermacher/Logics-SWE-Qwen3.6-27B-GGUF", "Logics-MLLM/Logics-SWE-Qwen3.6-27B") == 1.0
    assert name_match("Qwen/Qwen3-14B", "Qwen/Qwen3-14B-Base") == 2 / 3
    assert name_match("a/unrelated-name", "b/Llama-3-8B") == 0.0


def test_float_params_leaves_out_buffers_but_not_packed_weights():
    assert float_params({"params": {"F16": 162_323_040, "U8": 50_331_648}, "params_total": 212_654_688}) == 162_323_040
    awq = {"params": {"F16": 1_557_976_064, "I32": 13_212_057_600}, "params_total": 14_770_033_664}
    assert float_params(awq) == 14_770_033_664


def _frame(tmp_path, models, bases):
    fm = tmp_path / "e2_frame_meta.jsonl.gz"
    with gzip.open(fm, "wt", encoding="utf-8") as f:
        f.writelines(json.dumps(m) + "\n" for m in models)
    bm = tmp_path / "e2_bases_meta.jsonl.gz"
    with gzip.open(bm, "wt", encoding="utf-8") as f:
        f.writelines(json.dumps(b) + "\n" for b in bases)
    return PoolV2(pool.Frame(fm, configs=tmp_path / "none.jsonl.gz"), bm)


def _m(i, **kw):
    return {"id": i, "model_type": "llama", "params_total": 1000, **kw}


def test_outside_base_needs_another_declarer(tmp_path):
    models = [_m("a/model-x", declared_base=["o/outside"]), _m("b/other")]
    P = _frame(tmp_path, models, [{"id": "o/outside", "canonical": "o/outside", "model_type": "llama",
                                   "params_total": 1000}])
    assert "o/outside" not in P.candidates("a/model-x", "type:llama", 1000, None, False)
    models.append(_m("c/also-from-outside", declared_base=["o/outside"]))
    P = _frame(tmp_path, models, [{"id": "o/outside", "canonical": "o/outside", "model_type": "llama",
                                   "params_total": 1000}])
    assert "o/outside" in P.candidates("a/model-x", "type:llama", 1000, None, False)


def test_renamed_base_is_matched_by_its_current_name(tmp_path):
    models = [_m("org/New-Name"), _m("a/m", declared_base=["org/Old-Name"]), _m("b/n", declared_base=["org/Old-Name"])]
    P = _frame(tmp_path, models, [{"id": "org/Old-Name", "canonical": "org/New-Name", "model_type": "llama",
                                   "params_total": 1000}])
    assert P.name("org/Old-Name") == "org/New-Name"
    assert "org/New-Name" in P.candidates("a/m", "type:llama", 1000, None, False)


def test_union_keeps_both_orders(tmp_path):
    many = [_m(f"q/Base-7B-copy{i}", declared_base=["z/Popular"]) for i in range(5)]
    models = [_m("z/Popular"), _m("q/Base-7B"), *many]
    P = _frame(tmp_path, models, [])
    got = P.candidates("me/Base-7B-GGUF", "type:llama", 1000, None, True, limit=2)
    assert "z/Popular" in got and "q/Base-7B" in got
