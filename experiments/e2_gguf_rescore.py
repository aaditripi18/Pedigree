"""E2g: E2's comparisons that involve a GGUF file and compared too few tensors to judge, scored again
with the GGUF name fix for multimodal checkpoints (PLAN.md, amendment 2026-11-17).

Selected from E2's last run: comparisons without an error (reader errors are E2r's) whose model or
other side is a GGUF file and that compared fewer than MIN_COMPARED tensors. E2's own records are
untouched; scripts/analyse.py reports E2 as recorded and, beside it, E2 with E2r and E2g applied.

    python -m experiments.e2_gguf_rescore
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from experiments import ledger
from experiments.e2_audit import NetworkError, WORKERS, _pair, open_repo
from pedigree import pool

ROOT = Path(__file__).resolve().parent.parent
MIN_COMPARED = 8


def selected(e2: list[dict]) -> list[tuple[str, str, str]]:
    out = []
    for m in e2:
        model_gguf = m.get("format") == "gguf"
        for d in m.get("declared", []):
            if (model_gguf or d.get("format") == "gguf") and not d.get("error") and (d.get("compared") or 0) < MIN_COMPARED:
                out.append((m["id"], d["base"], "declared"))
        if model_gguf:
            out += [(m["id"], p["candidate"], "pool") for p in m.get("pool", [])
                    if not p.get("error") and (p.get("compared") or 0) < MIN_COMPARED]
    return out


def main() -> None:
    rows = [json.loads(x) for x in (ROOT / "results" / "ledger.jsonl").read_text(encoding="utf-8").splitlines() if x]
    e2 = [r for r in rows if r.get("experiment") == "E2"]
    last = max(r["run_id"] for r in e2)
    pairs = selected([r for r in e2 if r["run_id"] == last])
    # resumable: a pair already scored by an earlier E2g run (not lost) is not read again
    done = {(r["model"], r["other"], r["role"]) for r in rows if r.get("experiment") == "E2g" and "lost" not in r}
    print(f"{len(pairs)} GGUF comparisons with < {MIN_COMPARED} tensors in {last}; {len(done & set(pairs))} "
          "already scored", flush=True)
    pairs = [p for p in pairs if p not in done]
    frame = pool.Frame(ROOT / "results" / "e2_frame_meta.jsonl.gz")

    def one(p):
        model_id, other_id, role = p
        err = ""
        for _ in range(3):
            try:
                a, _ = open_repo(model_id, frame)
                b, _ = open_repo(other_id, frame)
                if a is None or b is None:
                    return p, {"error": "not readable on re-reading"}
                return p, _pair(a, b, invariant=role == "declared")
            except NetworkError as e:
                err = str(e)
            except Exception as e:  # noqa: BLE001  (an empty Hub reply once stopped the whole run)
                err = f"{type(e).__name__}: {e}"[:300]
        return p, {"lost": err}

    run = ledger.Run("E2g", e2_run=last)
    lost = 0
    with ThreadPoolExecutor(WORKERS) as ex:
        for (model_id, other_id, role), r in ex.map(one, pairs):
            if role == "pool":
                r.pop("cosines", None)
            lost += "lost" in r
            run.record(model=model_id, other=other_id, role=role, **r)
            print(f"{model_id} vs {other_id} ({role}): compared {r.get('compared')}, "
                  f"{r.get('median_cosine', r.get('error', r.get('lost')))}", flush=True)
    print(f"done; {lost} lost to the network", flush=True)


if __name__ == "__main__":
    main()
