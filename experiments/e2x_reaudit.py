"""E2x, E2xp and E2v3 (PLAN.md, amendment 2026-11-20): every part of E2 that a reader could not
read, read again with the R1 readers; alignment v3 and the R2 facts for every declared pair that is
still not comparable.

  python -m experiments.e2x_reaudit --targets   # freeze results/e2x_targets.json from the ledger (no reading)
  python -m experiments.e2x_reaudit [--out DIR]   # read
  python -m experiments.e2x_reaudit --round y [--targets]
      # E2y, E2yp, E2v4 (amendment 2026-11-21): what is still unread or not comparable after E2x,
      # read again with readers v5 and aligned by v4 (auto4); targets frozen from the ledger first

E2x   each E2 model recorded "not readable" (adapters aside), audited exactly as E2 audits a model
      (experiments/e2_audit.audit_one: declared bases, the E2 pool, the invariant score).
E2xp  each declared or pool comparison of a readable E2 model whose other side was unreadable
      (a recorded reason, or "not readable on re-reading"), read again (names alignment, as E2).
E2v3  each declared pair not comparable under names after all re-readings (the frozen list, and
      those of E2x's models): alignment `auto3` (names, else v3), the number of tensor shapes the
      two share, and the Hub's anonymous answer for the base (R2: status code, redirect target).
E2 and its earlier re-readings stay as recorded; scripts/analyse.py reports E2 with these applied
beside E2 as recorded.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from experiments import ledger
from experiments.e2_audit import WORKERS, NetworkError, _pair, audit_one, open_repo
from pedigree import hub, pool

ROOT = Path(__file__).resolve().parent.parent
TARGETS = ROOT / "results" / "e2x_targets.json"
ROUNDS = {"x": {"targets": TARGETS, "ledger": ("E2x", "E2xp", "E2v3"), "alignment": "auto3",
                "rule": "PLAN.md amendment 2026-11-20 (R1, R2, R3)"},
          "y": {"targets": ROOT / "results" / "e2y_targets.json", "ledger": ("E2y", "E2yp", "E2v4"),
                "alignment": "auto4",
                "rule": "PLAN.md amendment 2026-11-21 (readers v5, alignment v4): E2 with E2x applied"}}


def freeze(round_: str = "x") -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    import analyse as A
    rows = [json.loads(x) for x in (ROOT / "results" / "ledger.jsonl").read_text(encoding="utf-8").splitlines() if x]
    L = A.latest_runs
    e2 = L(rows, "E2", lambda r: 0)
    key = lambda r: (r["model"], r["other"], r["role"])  # noqa: E731
    allm = A.with_rescores(e2, L(rows, "E2r", lambda r: 0) + L(rows, "E2p", lambda r: 0), L(rows, "E2g", key),
                           L(rows, "E2s", key), L(rows, "E2f", key))
    T = A.thresholds(L(rows, "E1", lambda r: r.get("family")))
    if round_ == "y":                     # E2 with E2x applied, as scripts/analyse.py reports it
        allm = A.with_e2x(allm, L(rows, "E2x", lambda r: 0), L(rows, "E2xp", lambda r: 0),
                          L(rows, "E2v3", lambda r: 0))
    sample =json.loads((ROOT / "results" / "e2_sample.json").read_text(encoding="utf-8"))
    entry = {e["id"]: (e, t) for t, es in sample["sample"].items() for e in es}
    models = sorted(m["id"] for m in allm if m.get("reason") and m["reason"] != "adapter")
    pairs, notcomp = [], []
    for m in allm:
        if m.get("reason"):
            continue
        for role, k, rs in (("declared", "base", m.get("declared", [])), ("pool", "candidate", m.get("pool", []))):
            for d in rs:
                unread = (str(d.get("reason") or "").startswith("not readable")
                          or d.get("error") == "not readable on re-reading")
                if unread:
                    pairs.append([m["id"], d[k], role])
                elif role == "declared" and A.related(d, T, "t") == "not comparable":
                    notcomp.append([m["id"], d[k]])
    out = {"rule": ROUNDS[round_]["rule"] + "; frozen from the ledger before any re-reading",
           "models": [[i, entry[i][1], entry[i][0]] for i in models], "pairs": pairs, "not_comparable": notcomp}
    ROUNDS[round_]["targets"].write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"{len(models)} models, {len(pairs)} unread pairs, {len(notcomp)} declared pairs not comparable")


def hub_status(repo: str) -> dict:
    """R2: the Hub's anonymous answer for a declared base (no token is sent)."""
    if repo.count("/") != 1 or not all(repo.split("/")):
        return {"status": "not a repository id"}
    url = f"{hub.HOST}/api/models/{urllib.parse.quote(repo, safe='/')}"
    req = urllib.request.Request(url, headers={"User-Agent": hub.UA})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                d = json.loads(r.read().decode("utf-8"))
            return {"status": 200, "id": d.get("id"), "gated": d.get("gated"),
                    "renamed_to": d.get("id") if d.get("id") and d["id"].lower() != repo.lower() else None}
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < 3:
                time.sleep(10 * 2 ** attempt)
                continue
            return {"status": e.code}
        except Exception as e:  # noqa: BLE001
            if attempt == 3:
                return {"status": f"{type(e).__name__}"}
            time.sleep(10 * 2 ** attempt)


def shape_overlap(a, b) -> int:
    ca = Counter(tuple(s) for _, s in a.float_shapes().values())
    cb = Counter(tuple(s) for _, s in b.float_shapes().values())
    return sum(min(n, cb[s]) for s, n in ca.items())


def v3_row(model_id: str, base: str, frame, alignment: str = "auto3") -> dict:
    a, fa = open_repo(model_id, frame)
    b, fb = open_repo(base, frame)
    row = {"model": model_id, "other": base, "role": "declared", "hub": hub_status(base)}
    if a is None or b is None:
        row["error"] = "not readable on re-reading"
        row["reasons"] = [fa.get("reason"), fb.get("reason")]
        return row
    row["shared_shapes"] = shape_overlap(a, b)
    r = _pair(a, b, invariant=True, alignment=alignment)
    r.pop("cosines", None)
    row.update(r)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--round", choices=sorted(ROUNDS), default="x")
    a = ap.parse_args()
    R = ROUNDS[a.round]
    if a.targets:
        freeze(a.round)
        return
    t = json.loads(R["targets"].read_text(encoding="utf-8"))
    tm, tp, tv = R["ledger"]
    frame = pool.Frame(ROOT / "results" / "e2_frame_meta.jsonl.gz")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    def retry(fn, *args):
        last = None
        for _ in range(3):
            try:
                return fn(*args)
            except NetworkError as e:
                last = e
                time.sleep(30)
        return {"lost": str(last)}

    # E2x: the models
    recs = {}
    with ThreadPoolExecutor(WORKERS) as ex:
        futs = {ex.submit(retry, audit_one, e, tier, frame): i for i, tier, e in t["models"]}
        for k, f in enumerate(as_completed(futs), 1):
            i = futs[f]
            try:
                recs[i] = f.result()
            except Exception as err:  # noqa: BLE001  a bug: recorded as lost, never as a category
                import traceback
                traceback.print_exc()
                recs[i] = {"id": i, "lost": f"unexpected {type(err).__name__}: {err}"}
            r = recs[i]
            dec = ", ".join(f"{x['base']}={x.get('median_cosine', x.get('reason'))}" for x in r.get("declared", []))
            print(f"{tm} [{k}/{len(futs)}] {i}: {r.get('lost') or r.get('reason') or r.get('format')}; {dec or '-'}",
                  flush=True)
    (out / f"{tm.lower()}_raw.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs.values()), encoding="utf-8")

    # E2xp: the pairs whose other side was unreadable
    def pair(p):
        m, o, role = p
        x, _ = open_repo(m, frame)
        y, fy = open_repo(o, frame)
        if x is None or y is None:
            return {"error": "not readable on re-reading", "reason": fy.get("reason")}
        r = _pair(x, y, invariant=role == "declared")
        r.pop("cosines", None)
        return r
    prows = []
    with ThreadPoolExecutor(WORKERS) as ex:
        futs = {ex.submit(retry, pair, p): p for p in t["pairs"]}
        for f in as_completed(futs):
            m, o, role = futs[f]
            try:
                r = f.result()
            except Exception as err:  # noqa: BLE001
                r = {"lost": f"unexpected {type(err).__name__}: {err}"}
            prows.append({"model": m, "other": o, "role": role, **r})
            print(f"{tp} {m} vs {o} ({role}): {r.get('median_cosine', r.get('error') or r.get('lost'))}", flush=True)

    # E2v3: declared pairs still not comparable under names
    todo = [tuple(p) for p in t["not_comparable"]]
    for i, r in recs.items():
        for d in r.get("declared", []):
            if (d.get("compared") or 0) < 8:
                todo.append((i, d["base"]))
    for r in prows:
        if r["role"] == "declared" and (r.get("compared") or 0) < 8:
            todo.append((r["model"], r["other"]))
    todo = sorted(set(todo))
    vrows = []
    with ThreadPoolExecutor(WORKERS) as ex:
        futs = {ex.submit(retry, v3_row, m, b, frame, R["alignment"]): (m, b) for m, b in todo}
        for f in as_completed(futs):
            m, b = futs[f]
            try:
                r = f.result()
            except Exception as err:  # noqa: BLE001
                r = {"model": m, "other": b, "role": "declared", "lost": f"unexpected {type(err).__name__}: {err}"}
            r.setdefault("model", m)
            r.setdefault("other", b)
            r.setdefault("role", "declared")
            vrows.append(r)
            print(f"{tv} {m} vs {b}: compared {r.get('compared')}, {r.get('median_cosine', r.get('error') or r.get('lost'))};"
                  f" hub {r.get('hub')}", flush=True)

    lost = [i for i, r in recs.items() if "lost" in r] + [r for r in prows + vrows if "lost" in r]
    print(f"{tm} {len(recs)} models, {tp} {len(prows)} pairs, {tv} {len(vrows)} pairs; {len(lost)} lost", flush=True)
    if not lost:
        run = ledger.Run(tm, models=len(recs))
        for r in recs.values():
            run.record(**r)
        run = ledger.Run(tp, pairs=len(prows))
        for r in prows:
            run.record(**r)
        run = ledger.Run(tv, pairs=len(vrows))
        for r in vrows:
            run.record(**r)
        print("recorded in the ledger", flush=True)


if __name__ == "__main__":
    main()
