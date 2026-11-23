"""H29a-d, fresh data (PLAN.md, amendment 2026-11-23): readers v7 (rule (l), an underscore separates a
block index) with everything of H28, on models drawn after H28 by H25's rule.

  python -m experiments.h29_fresh --sample     # draw results/h29_sample.json (committed before any reading)
  python -m experiments.h29_fresh [--out PATH] # read the fresh sample (ledger H29)
  python -m experiments.h29_fresh --design [--out PATH]
                                               # H28's pairs (l) was designed on (ledger H29-design)

Each sampled model is read and aligned exactly as in H25 and H26 (experiments/h25_fresh.py).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from experiments import h25_fresh as h25
from experiments.h23_fresh import LIBRARIES
from pedigree import pool

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "results" / "h29_sample.json"
SEED = 20261125
PER_TIER = 150
EARLIER = ("h13_sample.json", "h15_sample.json", "h19_sample.json", "h23_sample.json", "h25_sample.json",
           "h26_sample.json", "h27_sample.json", "h28_sample.json")


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
                   "H13's, H15's, H19's, H23's, H25's, H26's, H27's or H28's samples; uniform within each download tier; all "
                   "where fewer than 150 remain",
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
    """H28's pairs (l) was designed on (the committed H28 and H28-design ledgers): H28's declared pairs
    names aligned < 8 tensors in that auto4 left not comparable, and any pair, declared, pool or design,
    where v4 with names ignored gave a verdict other than names'."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import analyse as A
    rows = A.load()
    T = A.thresholds(A.latest_runs(rows, "E1", lambda r: r.get("family")))
    out, seen = [], set()

    def add(i, other, role):
        if (i, other) not in seen:
            seen.add((i, other))
            out.append({"id": i, "other": other, "role": role})

    def differs(a):
        nm = a.get("names") or {}
        if nm.get("error") or (nm.get("compared") or 0) < h25.MIN_COMPARED:
            return False
        return A._verdict_plain(nm, T, "t") != A._verdict_plain(a.get("struct4"), T, "t")
    for m in A.latest_runs(rows, "H28", lambda r: 0):
        for d in m.get("declared", []):
            a = d.get("alignments") or {}
            nm = a.get("names")
            if nm is None or nm.get("error"):
                continue
            au = a.get("auto4") or {}
            if (nm.get("compared") or 0) < h25.MIN_COMPARED and (au.get("error") or (au.get("compared") or 0) < h25.MIN_COMPARED):
                add(m["id"], d["base"], "declared")
            elif differs(a):
                add(m["id"], d["base"], "declared")
        for p in m.get("pool_alignments", []):
            if differs(p):
                add(m["id"], p["candidate"], "pool")
    for r in A.latest_runs(rows, "H28-design", lambda r: 0):
        if differs(r.get("alignments") or {}):
            add(r["id"], r["other"], r["role"])
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
    out = Path(a.out or ROOT / "results" / ("h29_design_raw.jsonl" if a.design else "h29_raw.jsonl"))
    if a.resume and not out.exists():
        import shutil
        shutil.copy(ROOT / a.resume, out)
    if a.design:
        todo = design_pairs()
        h25.run(todo, lambda r: (r["id"], r["other"], r["role"]), lambda p: h25.read_design(p, frame),
                out, "H29-design", None, None)
        return
    sample = json.loads(SAMPLE.read_text(encoding="utf-8"))
    todo = [{"id": i, "tier": t} for t, ids in sample["sample"].items() for i in ids]
    h25.run(todo, lambda r: r["id"], lambda p: h25.read_one(p["id"], p["tier"], frame),
            out, "H29", sample["seed"], sample["snapshot"])


if __name__ == "__main__":
    main()
