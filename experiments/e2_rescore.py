"""E2r: E2's comparisons that ended in a reader error, scored again with the fixed readers
(PLAN.md, amendment 2026-11-13). Three reader gaps caused them: MLX 3- and 6-bit weights were
unpacked as if no value straddled two 32-bit words; GGUF int8 tensors were sent to the
dequantiser, which has no int8 case; GGUF tensor types outside llama.cpp's published list
(fork formats) raised a bare enum error. The first two are fixed; the third now says what it is.

E2's own records are untouched. scripts/analyse.py reports E2 as recorded and, beside it, E2 with
these pairs replaced by their new scores.

    python -m experiments.e2_rescore
    python -m experiments.e2_rescore --match "legacy pickle" --name E2p   (amendment 2026-11-17)
    python -m experiments.e2_rescore --source E2r,E2g --match "tensor type" --name E2f
        (amendment 2026-11-17 (4): GGUF types decoded from public sources, pedigree/ggufextra.py)
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from experiments import ledger
from experiments.e2_audit import NetworkError, WORKERS, _pair, open_repo
from pedigree import pool

ROOT = Path(__file__).resolve().parent.parent


def errored(e2: list[dict], match: str = "") -> list[tuple[str, str, str]]:
    """Comparisons that ended in a reader error; with `match`, also those whose base was recorded
    as unreadable (`reason`) for that cause (E2p: legacy PyTorch files)."""
    def hit(d):
        if d.get("error"):
            return match in d["error"]
        return bool(match) and match in (d.get("reason") or "")
    out = []
    for m in e2:
        out += [(m["id"], d["base"], "declared") for d in m.get("declared", []) if hit(d)]
        out += [(m["id"], p["candidate"], "pool") for p in m.get("pool", []) if hit(p)]
    return out


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", default="", help="only errors containing this text")
    ap.add_argument("--name", default="E2r", help="ledger experiment name")
    ap.add_argument("--source", default="E2", help="E2, or rescoring runs (E2r,E2g) whose latest errors to read again")
    a = ap.parse_args()
    rows = [json.loads(x) for x in (ROOT / "results" / "ledger.jsonl").read_text(encoding="utf-8").splitlines() if x]
    e2 = [r for r in rows if r.get("experiment") == "E2"]
    last = max(r["run_id"] for r in e2)
    if a.source == "E2":
        pairs = errored([r for r in e2 if r["run_id"] == last], a.match)
    else:   # each pair's latest record across the named rescoring runs
        latest: dict = {}
        for r in rows:
            if r.get("experiment") in a.source.split(",") and "lost" not in r:
                k = (r["model"], r["other"], r["role"])
                # by time: run ids of different experiments do not sort in time order
                if k not in latest or str(r.get("ts", "")) >= str(latest[k].get("ts", "")):
                    latest[k] = r
        pairs = sorted(k for k, r in latest.items() if a.match in str(r.get("error") or ""))
    print(f"{len(pairs)} errored comparisons in {last}", flush=True)
    frame = pool.Frame(ROOT / "results" / "e2_frame_meta.jsonl.gz")

    def one(p):
        model_id, other_id, role = p
        for attempt in range(3):
            try:
                a, _ = open_repo(model_id, frame)
                b, _ = open_repo(other_id, frame)
                if a is None or b is None:
                    return p, {"error": "not readable on re-reading"}
                return p, _pair(a, b, invariant=role == "declared")
            except NetworkError as e:
                err = str(e)
        return p, {"lost": err}

    run = ledger.Run(a.name, e2_run=last, match=a.match or None)
    lost = 0
    with ThreadPoolExecutor(WORKERS) as ex:
        for (model_id, other_id, role), r in ex.map(one, pairs):
            r.pop("cosines", None) if role == "pool" else None
            lost += "lost" in r
            run.record(model=model_id, other=other_id, role=role, **r)
            print(f"{model_id} vs {other_id} ({role}): {r.get('median_cosine', r.get('error', r.get('lost')))}",
                  flush=True)
    print(f"done; {lost} lost to the network", flush=True)


if __name__ == "__main__":
    main()
