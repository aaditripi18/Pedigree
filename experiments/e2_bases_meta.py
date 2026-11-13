"""Pool v2, frozen metadata of declared bases outside the frame (PLAN.md, amendment 2026-11-13).

E2 found that the declared parent of 59 of 199 weight-verified Hub models lies outside the
frame (the 100k most downloaded models), so no pool drawn from the frame could hold it.
Pool v2 also draws from every model that some frame model declares as its base. This fetches,
once, for each such base not in the frame: the same listing metadata as experiments/e2_meta.py
and the same config.json reduction as experiments/e2_configs.py.

Resumable: rows are appended to results/e2_bases_meta.part.jsonl; the finished file is
results/e2_bases_meta.jsonl.gz, sorted by id. A base the Hub no longer has is kept with
"missing": true; a renamed one keeps the asked-for id with its current name in "canonical".
A network failure is retried, never written.

    python -m experiments.e2_bases_meta [--shard 1/2 --part PATH]
"""
from __future__ import annotations

import gzip
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from experiments import e2_configs, e2_meta
from pedigree import pool

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "e2_bases_meta.jsonl.gz"
PART_DEFAULT = ROOT / "results" / "e2_bases_meta.part.jsonl"


def wanted() -> list[str]:
    with pool.open_text(ROOT / "results" / "e2_frame_meta.jsonl.gz") as f:
        frame = [json.loads(line) for line in f]
    ids = {m["id"] for m in frame}
    return sorted({b for m in frame for b in (m.get("declared_base") or []) if b not in ids})


def one(repo: str) -> dict | None:
    for attempt in range(6):   # the Hub rate-limits its API: back off, then leave it for the next run
        try:
            meta = e2_meta.one(repo)
            break
        except Exception:
            if attempt == 5:
                return None
            time.sleep(30 * (attempt + 1))
    meta["canonical"], meta["id"] = meta["id"], repo   # the Hub redirects renamed repositories
    if meta.get("missing"):
        return meta
    for attempt in range(6):
        c = e2_configs.one(repo)
        if not c.get("error"):
            break
        if attempt == 5:
            return None
        time.sleep(30 * (attempt + 1))
    meta["config_model_type"], meta["dims"] = c["model_type"], c["dims"]
    return meta


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1", help="k/n: only every n-th base, starting at k (the Hub limits each address)")
    ap.add_argument("--part", default=str(PART_DEFAULT), help="this shard's row file")
    a = ap.parse_args()
    part = Path(a.part)
    k, n = map(int, a.shard.split("/"))
    todo = wanted()
    mine = set(todo[k::n])
    done = set()
    if part.exists():
        done = {json.loads(x)["id"] for x in part.read_text(encoding="utf-8").splitlines() if x}
    rest = [b for b in todo if b not in done and b in mine]
    print(f"{len(todo)} declared bases outside the frame; {len(done)} fetched, {len(rest)} to fetch", flush=True)
    failed = 0
    with part.open("a", encoding="utf-8") as out, ThreadPoolExecutor(4) as ex:
        for k, r in enumerate(ex.map(one, rest)):
            if r is None:
                failed += 1
            else:
                out.write(json.dumps(r, sort_keys=True) + "\n")
            if k % 500 == 0:
                out.flush()
                print(f"  {k}/{len(rest)}, {failed} to retry", flush=True)
    if failed:
        raise SystemExit(f"{failed} bases failed on the network; run again to fetch them")
    if n > 1:
        return                      # shards are merged by appending their row files, then one run without --shard
    rows = {json.loads(x)["id"]: json.loads(x) for x in part.read_text(encoding="utf-8").splitlines() if x}
    with gzip.open(OUT, "wt", encoding="utf-8") as f:
        f.writelines(json.dumps(rows[b], sort_keys=True) + "\n" for b in todo)
    print(f"wrote {OUT.name}: {len(todo)} bases, {sum(1 for b in todo if rows[b].get('missing'))} missing")


if __name__ == "__main__":
    main()
