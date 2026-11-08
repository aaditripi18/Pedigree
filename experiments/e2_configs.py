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
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1", help="k/n: only every n-th frame model, starting at k")
    ap.add_argument("--rows-out", help="write this shard's rows here (jsonl) instead of the frozen file")
    ap.add_argument("--merge", nargs="*", help="shard row files to combine into the frozen file")
    a = ap.parse_args()
    with gzip.open(ROOT / "results" / "e2_frame_meta.jsonl.gz", "rt", encoding="utf-8") as f:
        ids = [json.loads(line)["id"] for line in f]
    if a.merge is not None:
        got = {}
        for path in a.merge:
            for line in Path(path).read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                got[r["id"]] = r
        missing = [i for i in ids if i not in got or got[i].get("error")]
        if missing:
            raise SystemExit(f"{len(missing)} frame models missing from the shards")
        with gzip.open(OUT, "wt", encoding="utf-8") as f:
            f.writelines(json.dumps(got[i], sort_keys=True) + "\n" for i in ids)
        print(f"wrote {OUT.name}: {sum(got[i]['config'] for i in ids)} of {len(ids)} have a config.json")
        return
    k, n = map(int, a.shard.split("/"))
    ids = ids[k::n]
    rows = []
    with ThreadPoolExecutor(24) as ex:
        for k, r in enumerate(ex.map(one, ids), 1):
            rows.append(r)
            if k % 5000 == 0:
                print(f"{k}/{len(ids)}", flush=True)
    lost = [r["id"] for r in rows if r.get("error")]
    if lost and not a.rows_out:
        raise SystemExit(f"{len(lost)} configs lost to the network; run again (finished ones are cached)")
    if a.rows_out:
        Path(a.rows_out).write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
        print(f"wrote {len(rows)} rows to {a.rows_out}")
        return
    tmp = OUT.with_suffix(".part")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        f.writelines(json.dumps(r, sort_keys=True) + "\n" for r in rows)
    tmp.replace(OUT)
    print(f"wrote {OUT.name}: {sum(r['config'] for r in rows)} of {len(rows)} have a config.json")


if __name__ == "__main__":
    main()
