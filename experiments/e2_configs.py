"""E2 candidate pool, frozen dimensions: every frame model's config.json (main branch),
reduced to what the pool uses (model_type, dimensions), fetched once so that the pool is
reproducible and needs no live lookups. The Hub's listing leaves model_type out for about
a third of transformers models, so the listing alone would hide them from every pool.

Writes results/e2_frame_configs.jsonl.gz: {"id", "model_type", "dims", "config": bool}.

    python -m experiments.e2_configs
"""
from __future__ import annotations

import gzip
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pedigree import pool

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "e2_frame_configs.jsonl.gz"


def one(repo: str) -> dict:
    for attempt in range(4):
        try:
            cfg = pool.config(repo)
            break
        except Exception:
            if attempt == 3:
                return {"id": repo, "error": "network"}
    mt = cfg.get("model_type") if isinstance(cfg.get("model_type"), str) else None
    return {"id": repo, "config": bool(cfg), "model_type": mt, "dims": pool.dims(cfg)}


def main() -> None:
    with gzip.open(ROOT / "results" / "e2_frame_meta.jsonl.gz", "rt", encoding="utf-8") as f:
        ids = [json.loads(line)["id"] for line in f]
    rows = []
    with ThreadPoolExecutor(24) as ex:
        for k, r in enumerate(ex.map(one, ids), 1):
            rows.append(r)
            if k % 5000 == 0:
                print(f"{k}/{len(ids)}", flush=True)
    lost = [r["id"] for r in rows if r.get("error")]
    if lost:
        raise SystemExit(f"{len(lost)} configs lost to the network; run again (finished ones are cached)")
    tmp = OUT.with_suffix(".part")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        f.writelines(json.dumps(r, sort_keys=True) + "\n" for r in rows)
    tmp.replace(OUT)
    print(f"wrote {OUT.name}: {sum(r['config'] for r in rows)} of {len(rows)} have a config.json")


if __name__ == "__main__":
    main()
