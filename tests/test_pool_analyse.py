"""The candidate pool's matching rules and the analysis's decision rules."""
import gzip
import json

import pytest

from pedigree import pool
from scripts import analyse


def test_dims_reads_the_common_spellings():
    assert pool.dims({"model_type": "gpt2", "n_embd": 768, "n_layer": 12, "n_inner": None}) == {"hidden": 768, "layers": 12}
    assert pool.dims({"architecture": "resnet18", "num_classes": 10}) == {"architecture": "resnet18"}
    assert pool.dims({"model_type": "x"}) is None
    assert pool.same_dims({"hidden": 768, "layers": 12, "ffn": 3072}, {"hidden": 768, "layers": 12})
    assert not pool.same_dims({"hidden": 768, "layers": 12}, {"hidden": 768, "layers": 6})


def test_pool_matches_dimensions_not_head_size(tmp_path, monkeypatch):
    meta = [{"id": "org/base", "model_type": "resnet", "params_total": 11_689_512, "declared_base": []},
            {"id": "a/child", "model_type": "resnet", "params_total": 11_200_000, "declared_base": ["org/base"]},
            {"id": "b/other", "model_type": "resnet", "params_total": 11_600_000, "declared_base": []},
            {"id": "c/huge", "model_type": "resnet", "params_total": 60_000_000, "declared_base": []}]
    cfg = [{"id": "org/base", "dims": {"hidden": 512, "layers": 18}}, {"id": "a/child", "dims": {"hidden": 512, "layers": 18}},
           {"id": "b/other", "dims": {"hidden": 512, "layers": 34}}, {"id": "c/huge", "dims": {"hidden": 512, "layers": 18}}]
    m, c = tmp_path / "e2_frame_meta.jsonl.gz", tmp_path / "e2_frame_configs.jsonl.gz"
    with gzip.open(m, "wt") as f:
        f.writelines(json.dumps(x) + "\n" for x in meta)
    with gzip.open(c, "wt") as f:
        f.writelines(json.dumps(x) + "\n" for x in cfg)
    fr = pool.Frame(m)
    # a 10-class fine-tune: 4% fewer parameters than its base, same dimensions
    got = fr.candidates("type:resnet", 11_180_000, {"hidden": 512, "layers": 18}, exclude={"a/child"})
    assert got == ["org/base"]          # b/other has other dimensions, c/huge is 5x larger


def test_threshold_rule_and_verdicts():
    e1 = [{"split": "dev", "label": "negative", "kind": "scratch", "median_cosine": x} for x in (0.0, 0.004, -0.003)]
    e1 += [{"split": "dev", "kind": "ft_long", "median_cosine": 0.99}]
    T = analyse.thresholds(e1)
    sd = analyse.np.std([0.0, 0.004, -0.003], ddof=1)
    assert T["plain"]["t"] == pytest.approx(0.004 + 5 * sd)
    assert T["plain"]["midpoint"] == pytest.approx((0.004 + 0.99) / 2)
    assert analyse.related({"median_cosine": 0.5, "compared": 24}, T) == "related"
    assert analyse.related({"median_cosine": 0.001, "compared": 24}, T) == "unrelated"
    assert analyse.related({"median_cosine": 0.001, "compared": 3}, T) == "not comparable"


def test_wilson_and_auc():
    assert analyse.wilson(0, 20)[0] == 0.0 and analyse.wilson(20, 20)[1] == 1.0
    assert analyse.auc([0.9, 0.8], [0.1, 0.0]) == 1.0


def test_open_text_reads_every_form_kaggle_produces(tmp_path):
    rows = '{"id": "a"}\n'
    gz = tmp_path / "x.jsonl.gz"
    with gzip.open(gz, "wt") as f:
        f.write(rows)
    assert pool.open_text(gz).read() == rows                      # as committed
    d = tmp_path / "plain"
    d.mkdir()
    (d / "x.jsonl").write_text(rows)
    assert pool.open_text(d / "x.jsonl.gz").read() == rows       # decompressed beside it
    f2 = tmp_path / "folder"
    (f2 / "x.jsonl").mkdir(parents=True)
    (f2 / "x.jsonl" / "x.jsonl").write_text(rows)
    assert pool.open_text(f2 / "x.jsonl.gz").read() == rows      # unpacked into a folder
