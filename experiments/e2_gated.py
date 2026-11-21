"""E2gate (PLAN.md, amendment 2026-11-20, R2 rule 4): the declared pairs R2 puts under "gated", read
with an account that accepted each base's terms. Runs locally with HF_TOKEN set (never on Kaggle);
the token goes only to huggingface.co (pedigree/remote.py) and is not logged or recorded.

  python -m experiments.e2_gated --targets   # freeze results/e2_gated_targets.json from the ledger
  python -m experiments.e2_gated --access    # which gated bases this account can read (no weights)
  python -m experiments.e2_gated             # read them (ledger E2gate)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
from pathlib import Path

from experiments import ledger
from experiments.e2x_reaudit import v3_row
from pedigree import hub, pool

ROOT = Path(__file__).resolve().parent.parent
TARGETS = ROOT / "results" / "e2_gated_targets.json"


def freeze() -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    import analyse as A
    rows = [json.loads(x) for x in (ROOT / "results" / "ledger.jsonl").read_text(encoding="utf-8").splitlines() if x]
    L = A.latest_runs
    key = lambda r: (r["model"], r["other"], r["role"])  # noqa: E731
    allm = A.with_rescores(L(rows, "E2", lambda r: 0), L(rows, "E2r", lambda r: 0) + L(rows, "E2p", lambda r: 0),
                           L(rows, "E2g", key), L(rows, "E2s", key), L(rows, "E2f", key))
    allm = A.with_e2x(allm, L(rows, "E2x", lambda r: 0), L(rows, "E2xp", lambda r: 0), L(rows, "E2v3", lambda r: 0))
    T = A.thresholds(L(rows, "E1", lambda r: r.get("family")))
    gated = A.r2_categories(allm, T, "t")["categories"].get("gated", [])
    TARGETS.write_text(json.dumps({"rule": "R2 rule 4 (gated), E2 with E2x applied; frozen before any reading",
                                   "pairs": gated}, indent=1) + "\n", encoding="utf-8")
    print(f"{len(gated)} gated pairs")


def access(repo: str) -> str:
    """Whether this account may read the repository's files: the Hub's answer for its file list
    and for one small file (no weights are read)."""
    q = urllib.parse.quote(repo, safe="/")
    try:
        with hub._open(f"{hub.HOST}/api/models/{q}/revision/main") as r:
            files = [s["rfilename"] for s in json.loads(r.read().decode("utf-8")).get("siblings", [])]
        small = next((f for f in ("config.json", "model_index.json", "README.md") if f in files), files[0])
        with hub._open(f"{hub.HOST}/{q}/resolve/main/{urllib.parse.quote(small)}"):
            return "readable"
    except urllib.error.HTTPError as e:
        return f"HTTP {e.code}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", action="store_true")
    ap.add_argument("--access", action="store_true")
    a = ap.parse_args()
    if a.targets:
        freeze()
        return
    if not os.environ.get("HF_TOKEN"):
        sys.exit("HF_TOKEN is not set: gated repositories cannot be read")
    pairs = json.loads(TARGETS.read_text(encoding="utf-8"))["pairs"]
    if a.access:
        for b in sorted({p[1] for p in pairs}):
            print(f"{b}: {access(b)}", flush=True)
        return
    frame = pool.Frame(ROOT / "results" / "e2_frame_meta.jsonl.gz")
    rows = []
    for m, b in pairs:
        st = access(b)
        if st != "readable":
            rows.append({"model": m, "other": b, "role": "declared", "access": st,
                         "error": "not readable on re-reading", "reasons": [None, f"gated: {st}"]})
        else:
            r = v3_row(m, b, frame, "auto4")
            r["access"] = st
            rows.append(r)
        r = rows[-1]
        print(f"E2gate {m} vs {b}: {st}; compared {r.get('compared')}, median {r.get('median_cosine')}", flush=True)
    run = ledger.Run("E2gate", pairs=len(rows))
    for r in rows:
        run.record(**r)
    print("recorded in the ledger", flush=True)


if __name__ == "__main__":
    main()
