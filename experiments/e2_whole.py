"""H6 on the Hub (PLAN.md, amendment 2026-11-07): 50 pairs E2 compared, drawn at random
(seed 20261107) from those whose two models are each <= 2 GB of weights, scored again on
whole files (every aligned tensor of >= 4096 values, read in full).

Reads E2's records from the ledger (the latest E2 run). Each pair's downloaded ranges are
deleted after it is scored, so disk use stays at one pair.

    python -m experiments.e2_whole
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from experiments import ledger
from experiments.e2_audit import info, open_repo, weight_files
from pedigree import hub, pool
from pedigree.verify import compare, compare_whole

ROOT = Path(__file__).resolve().parent.parent
SEED = 20261107
N = 50
MAX_BYTES = 2 * 1024 ** 3


def size_of(repo: str) -> int | None:
    d = info(repo)
    if d.get("_http"):
        return None
    if d.get("gated"):
        return None            # read through a copy in E2; left out here to keep the rule simple
    _, files, sizes, _ = weight_files(repo, d)
    return sum(sizes.values()) if files else None


def main() -> None:
    rows = [json.loads(x) for x in (ROOT / "results" / "ledger.jsonl").read_text(encoding="utf-8").splitlines() if x]
    e2 = [r for r in rows if r.get("experiment") == "E2"]
    last = max(r["run_id"] for r in e2)
    e2 = [r for r in e2 if r["run_id"] == last]
    pairs = []
    for m in e2:
        for d in m.get("declared", []):
            if d.get("median_cosine") is not None and not d.get("via"):
                pairs.append((m["id"], d["base"]))
        for p in m.get("pool", []):
            if p.get("median_cosine") is not None and not p.get("via"):
                pairs.append((m["id"], p["candidate"]))
    pairs = sorted(set(pairs))
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(pairs))
    frame = pool.Frame(ROOT / "results" / "e2_frame_meta.jsonl.gz")
    run = ledger.Run("E2W", seed=SEED, e2_run=last, eligible_rule="each model <= 2 GB, not gated")
    taken = 0
    for i in order:
        a, b = pairs[i]
        sa, sb = size_of(a), size_of(b)
        if sa is None or sb is None or sa > MAX_BYTES or sb > MAX_BYTES:
            continue
        ma, _ = open_repo(a, frame)
        mb, _ = open_repo(b, frame)
        if ma is None or mb is None:
            continue
        try:
            c, w = compare(ma, mb), compare_whole(ma, mb)
            run.record(model=a, base=b, bytes=[sa, sb], median_cosine=c["median_cosine"], compared=c["compared"],
                       median_whole=w["median_whole"], whole_compared=w["compared"])
            print(f"{taken + 1:>2} {a} vs {b}: sampled {c['median_cosine']:.4f} whole {w['median_whole']:.4f}",
                  flush=True)
            taken += 1
        except Exception as e:
            run.record(model=a, base=b, error=f"{type(e).__name__}: {e}")
            print(f"   {a} vs {b}: ERROR {e}", flush=True)
        shutil.rmtree(hub.cache_root() / "ranges", ignore_errors=True)
        if taken == N:
            break
    print(f"{taken} pairs scored of {len(pairs)} compared by E2")


if __name__ == "__main__":
    main()
