"""E2 sampling frame (PLAN.md): frozen before any weight is read.

The Hub's models sorted by downloads (one snapshot), cut into three tiers: ranks 1-1,000,
1,001-10,000 and 10,001-100,000. PER_TIER models are drawn at random from each tier with
a recorded seed. Every sampled model stays in the audit: one without readable weights is
reported as unverifiable, never dropped.

Writes results/e2_frame.txt.gz (every id in the frame with rank and downloads) and
results/e2_sample.json (the sample, the seed, the snapshot time).

    python -m experiments.e2_sample
"""
from __future__ import annotations

import gzip
import json
import time
import urllib.request
from pathlib import Path

import numpy as np

from pedigree.hub import UA, _open

ROOT = Path(__file__).resolve().parent.parent
TIERS = {"top_1k": (1, 1_000), "1k_10k": (1_001, 10_000), "10k_100k": (10_001, 100_000)}
PER_TIER = 200
SEED = 20261105
URL = "https://huggingface.co/api/models?sort=downloads&direction=-1&limit=1000"


def frame() -> list[dict]:
    rows, url = [], URL
    while url and len(rows) < 100_000:
        with _open(url) as r:
            page = json.loads(r.read().decode("utf-8"))
            link = r.headers.get("Link", "")
        for m in page:
            rows.append({"rank": len(rows) + 1, "id": m["id"], "downloads": m.get("downloads", 0),
                         "library": m.get("library_name"), "pipeline": m.get("pipeline_tag")})
        url = link.split(";")[0].strip("<> ") if 'rel="next"' in link else None
        print(f"{len(rows)} models", flush=True)
    return rows[:100_000]


def main() -> None:
    snapshot = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    rows = frame()
    if len(rows) < 100_000:
        raise SystemExit(f"frame incomplete: {len(rows)} models")
    with gzip.open(ROOT / "results" / "e2_frame.txt.gz", "wt", encoding="utf-8") as f:
        f.write(f"# Hub models by downloads, snapshot {snapshot}\n# rank\tid\tdownloads\n")
        f.writelines(f"{r['rank']}\t{r['id']}\t{r['downloads']}\n" for r in rows)
    rng = np.random.default_rng(SEED)
    sample = {}
    for tier, (lo, hi) in TIERS.items():
        idx = sorted(rng.choice(np.arange(lo - 1, hi), PER_TIER, replace=False).tolist())
        sample[tier] = [rows[i] for i in idx]
    (ROOT / "results" / "e2_sample.json").write_text(json.dumps(
        {"snapshot": snapshot, "seed": SEED, "per_tier": PER_TIER, "tiers": TIERS,
         "rule": "uniform random within each download tier; every sampled model is audited",
         "sample": sample}, indent=1), encoding="utf-8")
    print({t: len(v) for t, v in sample.items()})


if __name__ == "__main__":
    main()
