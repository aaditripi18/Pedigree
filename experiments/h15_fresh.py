"""H15 and H17a, fresh data (PLAN.md, amendment 2026-11-17 (4)).

  python -m experiments.h15_fresh --sample     # draw results/h15_sample.json (committed before any reading)
  python -m experiments.h15_fresh [--out PATH] # read each sampled model and its declared bases

The sample: frame models that declare a base and are in neither E2's nor H13's sample, 120 per
download tier, uniform within the tier, seed 20261118. Each is compared by weights with each
declared base exactly as H13 does, recording both creation dates and the kurtosis gap (H15's
direction rules are applied afterwards by scripts/analyse.py). A GGUF model is also compared
with its declared bases and the two top-ranked E2-pool candidates under llama.cpp's names and
under structural alignment with names ignored (H17a).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from experiments import ledger
from experiments.e2_audit import BUDGET, WORKERS, NetworkError, _network, _pair, info, logical_params, open_repo
from pedigree.verify import compare
from pedigree import pool

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "results" / "h15_sample.json"
SEED = 20261118
PER_TIER = 120


def draw(frame: pool.Frame) -> None:
    e2 = json.loads((ROOT / "results" / "e2_sample.json").read_text(encoding="utf-8"))
    seen = {e["id"] for rows in e2["sample"].values() for e in rows}
    h13 = json.loads((ROOT / "results" / "h13_sample.json").read_text(encoding="utf-8"))
    seen |= {i for ids in h13["sample"].values() for i in ids}
    rng = random.Random(SEED)
    out = {"seed": SEED, "per_tier": PER_TIER, "tiers": e2["tiers"], "snapshot": e2["snapshot"],
           "rule": "frame models with a declared base, in neither E2's nor H13's sample; uniform within each download tier",
           "sample": {}}
    for tier, (lo, hi) in e2["tiers"].items():
        pool_ = [m["id"] for m in frame.models[lo - 1:hi]
                 if m["id"] not in seen and not m.get("missing") and m.get("declared_base")]
        out["sample"][tier] = sorted(rng.sample(pool_, PER_TIER))
        print(f"{tier}: {len(pool_)} eligible, {PER_TIER} drawn", flush=True)
    SAMPLE.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")


def read_one(repo: str, tier: str, frame: pool.Frame) -> dict:
    meta = frame.by_id.get(repo, {})
    rec = {"id": repo, "tier": tier, "created": meta.get("createdAt"), "declared_base": meta.get("declared_base") or [],
           "family_key": pool.family_key(meta)}
    model, facts = open_repo(repo, frame)
    rec.update(facts)
    if model is None:
        return rec
    rec.update(params=(None if model.quant else pool.meta_params(meta)) or logical_params(model),
               dims=(frame.frozen_dims[repo] if repo in frame.frozen_dims and not facts.get("via")
                     else pool.dims(pool.config(facts.get("via") or repo))),
               quantised=bool(model.quant) or facts.get("format") == "gguf")
    rec["declared"] = []
    for b in rec["declared_base"]:
        other, bf = open_repo(b, frame)
        row = {"base": b, **{k: v for k, v in bf.items() if k in ("via", "reason", "commit", "format")},
               "created": frame.by_id.get(b, {}).get("createdAt") or info(b).get("createdAt")}
        if other is not None:
            row.update(_pair(model, other, invariant=True))
            if rec.get("format") == "gguf":
                row["alignments"] = alignments(model, other)
        rec["declared"].append(row)
    if rec.get("format") == "gguf":       # H17a's extra pairs: the two top-ranked E2-pool candidates
        rec["pool_alignments"] = []
        cands = [c for c in frame.candidates(rec["family_key"], rec.get("params"), rec.get("dims"), exclude={repo})
                 if c not in rec["declared_base"]][:2]
        for c in cands:
            other, _ = open_repo(c, frame)
            if other is not None:
                rec["pool_alignments"].append({"candidate": c, **alignments(model, other)})
    return rec


def alignments(model, other) -> dict:
    """H17a: the plain score under llama.cpp's names and under structural alignment alone."""
    out = {}
    if bool(getattr(model, "gguf_meta", None)) == bool(getattr(other, "gguf_meta", None)):
        return {"applicable": False}
    for mode in ("names", "struct"):
        try:
            c = compare(model, other, alignment=mode)
            out[mode] = {"median_cosine": c["median_cosine"], "compared": c["compared"], "matched": c["matched"]}
        except Exception as e:
            if _network(e):
                raise NetworkError(f"{model.repo} vs {other.repo}: {type(e).__name__}: {e}") from None
            out[mode] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "results" / "h15_raw.jsonl"))
    a = ap.parse_args()
    frame = pool.Frame(ROOT / "results" / "e2_frame_meta.jsonl.gz")
    if a.sample:
        draw(frame)
        return
    sample = json.loads(SAMPLE.read_text(encoding="utf-8"))
    todo = [(i, t) for t, ids in sample["sample"].items() for i in ids]
    out = Path(a.out)
    done = {}
    sources = [out] if out.exists() else []
    prev = os.environ.get("PEDIGREE_H15_PREVIOUS")
    if prev:
        sources += [p for p in Path(prev).rglob(out.name) if "Pedigree" not in p.parts and p != out]
    for src in sources:
        for line in src.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done.setdefault(r["id"], r)
    if prev and done:
        out.write_text("".join(json.dumps(r) + "\n" for r in done.values()), encoding="utf-8")
    pending = [(i, t) for i, t in todo if i not in done]
    print(f"{len(done)} done, {len(pending)} to read", flush=True)
    t0, lost, lock = time.time(), [], threading.Lock()

    def work(i, t):
        if time.time() - t0 > BUDGET:
            return i, None, "budget"
        try:
            return i, read_one(i, t, frame), None
        except NetworkError as err:
            return i, None, str(err)
        except Exception as err:      # a bug or an unforeseen format: lost (retried), never categorised
            import traceback
            traceback.print_exc()
            return i, None, f"unexpected {type(err).__name__}: {err}"

    with ThreadPoolExecutor(WORKERS) as ex:
        futs = [ex.submit(work, i, t) for i, t in pending]
        for k, f in enumerate(as_completed(futs), 1):
            repo, rec, err = f.result()
            if rec is None:
                lost.append((repo, err))
                print(f"[{k}/{len(pending)}] {repo}: LOST ({err})", flush=True)
                continue
            with lock, open(out, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
            done[repo] = rec
            dec = ", ".join(f"{x['base']}={x.get('median_cosine', x.get('reason'))}" for x in rec.get("declared", []))
            print(f"[{k}/{len(pending)}] {repo}: {rec.get('reason') or rec.get('format')}; {dec or '-'} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    print(f"{len(done)} of {len(todo)} done; {len(lost)} lost to the network or the time budget", flush=True)
    if len(done) == len(todo) and not lost:
        run = ledger.Run("H15", sample_seed=sample["seed"], snapshot=sample["snapshot"])
        for i, t in todo:
            run.record(**done[i])
        print("recorded in the ledger", flush=True)


if __name__ == "__main__":
    main()
