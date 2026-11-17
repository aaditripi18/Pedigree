"""E2s: E2's GGUF comparisons that still align too few tensors, scored again with structural
alignment (PLAN.md, amendment 2026-11-17 (4), H17b).

Selected from E2's last run with E2r, E2g and E2p applied (scripts/analyse.py's with_rescores):
comparisons whose model or other side is a GGUF file that compared fewer than MIN_COMPARED
tensors without an error, plus those E2g could not read again ("not readable on re-reading").
Each is scored with alignment "auto": llama.cpp's names, else structural alignment when names
pair fewer than 8 tensors. E2's own records are untouched.

    python -m experiments.e2_struct_rescore
"""
from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from experiments import ledger
from experiments.e2_audit import NetworkError, WORKERS, _pair, open_repo
from pedigree import pool

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
MIN_COMPARED = 8


def selected(e2f: list[dict]) -> list[tuple[str, str, str]]:
    out = []
    for m in e2f:
        model_gguf = m.get("format") == "gguf"
        for role, key in (("declared", "base"), ("pool", "candidate")):
            for d in m.get(role, []):
                if not (model_gguf or d.get("format") == "gguf"):
                    continue
                if role == "pool" and not model_gguf:
                    continue
                err = d.get("error")
                if (not err and (d.get("compared") or 0) < MIN_COMPARED) or err == "not readable on re-reading":
                    out.append((m["id"], d[key], role))
    return out


def main() -> None:
    import analyse as A
    rows = A.load()
    e2 = A.latest_runs(rows, "E2", lambda r: 0)
    e2r = A.latest_runs(rows, "E2r", lambda r: 0) + A.latest_runs(rows, "E2p", lambda r: 0)
    e2g = A.latest_runs(rows, "E2g", lambda r: (r["model"], r["other"], r["role"]))
    e2f = A.with_rescores(e2, e2r, e2g)
    # E2g's re-read failures replace the pair's record in with_rescores only when E2 had an error;
    # mark them here so they are read again
    gfail = {(r["model"], r["other"], r["role"]) for r in e2g if r.get("error") == "not readable on re-reading"}
    for m in e2f:
        for role, key in (("declared", "base"), ("pool", "candidate")):
            for d in m.get(role, []):
                if (m["id"], d[key], role) in gfail:
                    d["error"] = "not readable on re-reading"
    pairs = selected(e2f)
    done = {(r["model"], r["other"], r["role"]) for r in rows if r.get("experiment") == "E2s" and "lost" not in r}
    print(f"{len(pairs)} GGUF comparisons to score ({sum(p[2] == 'declared' for p in pairs)} declared); "
          f"{len(done & set(pairs))} already scored", flush=True)
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
                return p, _pair(a, b, invariant=role == "declared", alignment="auto")
            except NetworkError as e:
                err = str(e)
            except Exception as e:  # noqa: BLE001
                err = f"{type(e).__name__}: {e}"[:300]
        return p, {"lost": err}

    run = ledger.Run("E2s", e2_run=max(r["run_id"] for r in e2))
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
