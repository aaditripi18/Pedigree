"""E2r: E2's comparisons that ended in a reader error, scored again with the fixed readers
(PLAN.md, amendment 2026-11-13). Three reader gaps caused them: MLX 3- and 6-bit weights were
unpacked as if no value straddled two 32-bit words; GGUF int8 tensors were sent to the
dequantiser, which has no int8 case; GGUF tensor types outside llama.cpp's published list
(fork formats) raised a bare enum error. The first two are fixed; the third now says what it is.

E2's own records are untouched. scripts/analyse.py reports E2 as recorded and, beside it, E2 with
these pairs replaced by their new scores.

    python -m experiments.e2_rescore
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from experiments import ledger
from experiments.e2_audit import NetworkError, WORKERS, _pair, open_repo
from pedigree import pool

ROOT = Path(__file__).resolve().parent.parent


def errored(e2: list[dict]) -> list[tuple[str, str, str]]:
    out = []
    for m in e2:
        out += [(m["id"], d["base"], "declared") for d in m.get("declared", []) if d.get("error")]
        out += [(m["id"], p["candidate"], "pool") for p in m.get("pool", []) if p.get("error")]
    return out


def main() -> None:
    rows = [json.loads(x) for x in (ROOT / "results" / "ledger.jsonl").read_text(encoding="utf-8").splitlines() if x]
    e2 = [r for r in rows if r.get("experiment") == "E2"]
    last = max(r["run_id"] for r in e2)
    pairs = errored([r for r in e2 if r["run_id"] == last])
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

    run = ledger.Run("E2r", e2_run=last)
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
