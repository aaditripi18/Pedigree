"""H16c (PLAN.md amendment 2026-11-23 (2)): the dev gate and the judgement, on made-up scores."""
from scripts import analyse

TEACHER = 0.5   # the floor is half of it


def row(kind, seed, dark_js, acc):
    return {"family": "text_encoder", "model": f"{kind}_s{seed}", "dark_js": dark_js,
            "accuracy": acc, "teacher_accuracy": TEACHER}


def batch(seeds, distill_js=0.2, indep_js=0.4, acc=0.3):
    return [row("distill", s, distill_js, acc) for s in seeds] + [row("indep", s, indep_js, acc) for s in seeds]


def test_pending_before_any_student():
    assert analyse.h16c([], [])["verdict"] == "pending"


def test_pending_while_the_dev_batch_is_incomplete():
    assert analyse.h16c(batch([70, 71]), [])["verdict"] == "pending"


def test_dev_gate_stops_when_a_dev_independent_misses_the_floor():
    dev = batch([70, 71, 72])
    dev[-1]["accuracy"] = 0.2           # under 0.25
    h = analyse.h16c(dev, [])
    assert h["verdict"] == "not decidable" and "dev gate" in h["evidence"]


def test_dev_gate_passes_and_waits_for_the_fresh_batch():
    h = analyse.h16c(batch([70, 71, 72]), [])
    assert h["verdict"] == "pending" and "dev gate passed" in h["evidence"]


def test_judged_on_the_full_batch():
    full = batch(range(70, 81))
    for r in full:                      # small spread among independents so the dev gap is positive
        if r["model"].startswith("indep"):
            r["dark_js"] += (int(r["model"].rsplit("_s", 1)[1]) % 3) * 0.001
    h = analyse.h16c([], full)
    assert h["verdict"] == "supported", h
    assert h["numbers"]["tpr"] == [5, 5] and h["numbers"]["fpr"][0] == 0


def test_a_fresh_independent_under_the_floor_makes_it_not_decidable():
    full = batch(range(70, 81))
    full[-1]["accuracy"] = 0.1
    assert analyse.h16c([], full)["verdict"] == "not decidable"
