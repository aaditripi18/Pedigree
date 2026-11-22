"""H26a/H26b, fresh data (PLAN.md, amendment 2026-11-21): readers v5 with alignment v4 on models
stored as GGUF, ONNX, CTranslate2 or NeMo, drawn after H25 by H25's rule.

  python -m experiments.h26_fresh --sample     # draw results/h26_sample.json (committed before any reading)
  python -m experiments.h26_fresh [--out PATH] [--resume PATH]
                                               # read the fresh sample (ledger H26); --resume starts
                                               # from records an earlier, cut-off run wrote
  python -m experiments.h26_fresh --design [--out PATH]
                                               # H25's pairs v5 was designed on, under v5 (ledger H26d)

Each sampled model is aligned with each declared base and the two top-ranked pool candidates exactly
as in H25 (experiments/h25_fresh.py); only the readers differ (v5, PEDIGREE_READERS_V5=1).
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from experiments import h25_fresh as h25
from experiments.h23_fresh import LIBRARIES
from pedigree import pool

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "results" / "h26_sample.json"
SEED = 20261122
PER_TIER = 150
EARLIER = ("h13_sample.json", "h15_sample.json", "h19_sample.json", "h23_sample.json", "h25_sample.json")


def draw(frame: pool.Frame) -> None:
    e2 = json.loads((ROOT / "results" / "e2_sample.json").read_text(encoding="utf-8"))
    seen = {e["id"] for rows in e2["sample"].values() for e in rows}
    for f in EARLIER:
        s = json.loads((ROOT / "results" / f).read_text(encoding="utf-8"))
        seen |= {i for ids in s["sample"].values() for i in ids}
    libs = set().union(*LIBRARIES.values())
    rng = random.Random(SEED)
    out = {"seed": SEED, "per_tier": PER_TIER, "tiers": e2["tiers"], "snapshot": e2["snapshot"],
           "libraries": {k: sorted(v) for k, v in LIBRARIES.items()},
           "rule": "frame models with a declared base whose library is one of the listed ones, in none of E2's, "
                   "H13's, H15's, H19's, H23's or H25's samples; uniform within each download tier; all where "
                   "fewer than 150 remain",
           "sample": {}}
    for tier, (lo, hi) in e2["tiers"].items():
        pool_ = [m["id"] for m in frame.models[lo - 1:hi]
                 if m["id"] not in seen and not m.get("missing") and m.get("declared_base")
                 and frame.by_id.get(m["id"], {}).get("library") in libs]
        k = min(PER_TIER, len(pool_))
        out["sample"][tier] = sorted(rng.sample(pool_, k))
        print(f"{tier}: {len(pool_)} eligible, {k} drawn", flush=True)
    SAMPLE.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")


def design_pairs() -> list[dict]:
    """H25's pairs that v5 was designed on (the committed H25 ledger): its declared pairs names
    aligned < 8 tensors in, and its pairs where v4 with names ignored gave a verdict other than names'
    (the opposite verdict and the abstention of H25a)."""
    rows = [json.loads(x) for x in (ROOT / "results" / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
            if x.strip()]
    h = [r for r in rows if r.get("experiment") == "H25"]
    last = max(r["run_id"] for r in h)
    out, seen = [], set()

    def add(i, other, role):
        if (i, other) not in seen:
            seen.add((i, other))
            out.append({"id": i, "other": other, "role": role})
    for r in (x for x in h if x["run_id"] == last):
        for d in r.get("declared", []):
            a = d.get("alignments") or {}
            nm, s4 = a.get("names"), a.get("struct4") or {}
            if nm is None:
                continue
            low = (nm.get("compared") or 0) < h25.MIN_COMPARED
            differs = not low and (s4.get("compared") or 0) >= h25.MIN_COMPARED and \
                (s4.get("median_cosine") or 0) <= 0.0255 < (nm.get("median_cosine") or 0)
            abstains = not low and (s4.get("compared") or 0) < h25.MIN_COMPARED
            if low or differs or abstains:
                add(r["id"], d["base"], "declared")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--design", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--resume", default=None, help="raw records of an earlier run, relative to the repository")
    a = ap.parse_args()
    frame = pool.Frame(ROOT / "results" / "e2_frame_meta.jsonl.gz")
    if a.sample:
        draw(frame)
        return
    out = Path(a.out or ROOT / "results" / ("h26_design_raw.jsonl" if a.design else "h26_raw.jsonl"))
    if a.resume and not out.exists():
        import shutil
        shutil.copy(ROOT / a.resume, out)
    if a.design:
        todo = design_pairs()
        h25.run(todo, lambda r: (r["id"], r["other"], r["role"]), lambda p: h25.read_design(p, frame),
                out, "H26d", None, None)
        return
    sample = json.loads(SAMPLE.read_text(encoding="utf-8"))
    todo = [{"id": i, "tier": t} for t, ids in sample["sample"].items() for i in ids]
    h25.run(todo, lambda r: r["id"], lambda p: h25.read_one(p["id"], p["tier"], frame),
            out, "H26", sample["seed"], sample["snapshot"])


if __name__ == "__main__":
    main()
