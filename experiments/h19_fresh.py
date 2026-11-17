"""H19 and H20, fresh data (PLAN.md, amendment 2026-11-18). The last follow-ups on direction and
alignment.

  python -m experiments.h19_fresh --sample     # draw results/h19_sample.json (committed before any reading)
  python -m experiments.h19_fresh [--out PATH] # read each sampled model and its declared bases

The sample: frame models that declare a base and are in none of E2's, H13's or H15's samples, 120
per download tier, uniform within the tier, seed 20261119. Each is compared by weights with each
declared base exactly as H15 does (experiments/h15_fresh.py), and both repositories' content dates
(H19's rule D4: the earliest last-commit date among the current weight files, from the Hub's
tree?expand=true listing at the commit read) are recorded. A GGUF model is also aligned with its
declared bases and the two top-ranked E2-pool candidates under llama.cpp's names, structural
alignment v1 and v2 (H20). The 14 frozen E2 pairs of results/h20_e2_pairs.json are scored with
names-else-v2 alignment (ledger H20e2). Verdicts are computed afterwards by scripts/analyse.py.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import threading
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from experiments import ledger
from experiments.e2_audit import BUDGET, WORKERS, NetworkError, _network, _pair, info, logical_params, open_repo
from pedigree import hub, pool
from pedigree.verify import compare

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "results" / "h19_sample.json"
E2_PAIRS = ROOT / "results" / "h20_e2_pairs.json"
SEED = 20261119
PER_TIER = 120
WEIGHT_EXT = (".safetensors", ".gguf", ".bin", ".pt", ".pth", ".ckpt")
ALIGNMENTS = ("names", "struct", "struct2")


def draw(frame: pool.Frame) -> None:
    e2 = json.loads((ROOT / "results" / "e2_sample.json").read_text(encoding="utf-8"))
    seen = {e["id"] for rows in e2["sample"].values() for e in rows}
    for f in ("h13_sample.json", "h15_sample.json"):
        s = json.loads((ROOT / "results" / f).read_text(encoding="utf-8"))
        seen |= {i for ids in s["sample"].values() for i in ids}
    rng = random.Random(SEED)
    out = {"seed": SEED, "per_tier": PER_TIER, "tiers": e2["tiers"], "snapshot": e2["snapshot"],
           "rule": "frame models with a declared base, in none of E2's, H13's or H15's samples; uniform within each "
                   "download tier; all of them where fewer than 120 remain", "sample": {}}
    for tier, (lo, hi) in e2["tiers"].items():
        pool_ = [m["id"] for m in frame.models[lo - 1:hi]
                 if m["id"] not in seen and not m.get("missing") and m.get("declared_base")]
        k = min(PER_TIER, len(pool_))       # all that remain where a tier has fewer (PLAN.md, 2026-11-18)
        out["sample"][tier] = sorted(rng.sample(pool_, k))
        print(f"{tier}: {len(pool_)} eligible, {k} drawn", flush=True)
    SAMPLE.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")


def content_date(repo: str, commit: str | None) -> dict:
    """H19's D4 date: the earliest last-commit date among the current weight files at `commit`."""
    rev = commit or "main"
    url = (f"{hub.HOST}/api/models/{urllib.parse.quote(repo, safe='/')}/tree/{rev}"
           "?recursive=true&expand=true")
    dates, files, pages = [], 0, 0
    try:
        while url and pages < 50:
            with hub._open(url) as r:
                items = json.loads(r.read().decode("utf-8"))
                link = r.headers.get("Link") or ""
            pages += 1
            for it in items:
                if it.get("type") == "file" and it.get("path", "").lower().endswith(WEIGHT_EXT):
                    files += 1
                    d = (it.get("lastCommit") or {}).get("date")
                    if d:
                        dates.append(d)
            url = None
            for part in link.split(","):
                if 'rel="next"' in part:
                    url = part.split(";")[0].strip().strip("<>")
    except Exception as e:  # noqa: BLE001
        if _network(e):
            raise NetworkError(f"{repo} tree: {type(e).__name__}: {e}") from None
        return {"content_date": None, "content_error": f"{type(e).__name__}: {str(e)[:120]}"}
    return {"content_date": min(dates) if dates else None, "weight_files_dated": len(dates), "weight_files": files}


def alignments(model, other) -> dict:
    if bool(getattr(model, "gguf_meta", None)) == bool(getattr(other, "gguf_meta", None)):
        return {"applicable": False}
    out = {}
    for mode in ALIGNMENTS:
        try:
            c = compare(model, other, alignment=mode)
            out[mode] = {"median_cosine": c["median_cosine"], "compared": c["compared"], "matched": c["matched"]}
        except Exception as e:  # noqa: BLE001
            if _network(e):
                raise NetworkError(f"{model.repo} vs {other.repo}: {type(e).__name__}: {e}") from None
            out[mode] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
    return out


def read_one(repo: str, tier: str, frame: pool.Frame) -> dict:
    meta = frame.by_id.get(repo, {})
    rec = {"id": repo, "tier": tier, "created": meta.get("createdAt"), "declared_base": meta.get("declared_base") or [],
           "family_key": pool.family_key(meta)}
    model, facts = open_repo(repo, frame)
    rec.update(facts)
    if model is None:
        return rec
    rec.update(content_date(repo, facts.get("commit")))
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
            row.update({"base_" + k: v for k, v in content_date(b, bf.get("commit")).items()})
            if rec.get("format") == "gguf":
                row["alignments"] = alignments(model, other)
        rec["declared"].append(row)
    if rec.get("format") == "gguf":       # H20's extra pairs: the two top-ranked E2-pool candidates
        rec["pool_alignments"] = []
        cands = [c for c in frame.candidates(rec["family_key"], rec.get("params"), rec.get("dims"), exclude={repo})
                 if c not in rec["declared_base"]][:2]
        for c in cands:
            other, _ = open_repo(c, frame)
            if other is not None:
                rec["pool_alignments"].append({"candidate": c, **alignments(model, other)})
    return rec


def e2_pairs(frame: pool.Frame) -> None:
    """H20's measurement: the 14 frozen E2 pairs, names else v2 (ledger H20e2)."""
    pairs = json.loads(E2_PAIRS.read_text(encoding="utf-8"))["pairs"]
    run = ledger.Run("H20e2", pairs=len(pairs))
    for model_id, base in pairs:
        r = {"lost": "not read"}
        for _ in range(3):
            try:
                a, _ = open_repo(model_id, frame)
                b, _ = open_repo(base, frame)
                r = ({"error": "not readable on re-reading"} if a is None or b is None
                     else _pair(a, b, invariant=True, alignment="auto2"))
                break
            except NetworkError as e:
                r = {"lost": str(e)}
        r.pop("cosines", None)
        run.record(model=model_id, other=base, role="declared", **r)
        print(f"H20e2 {model_id} vs {base}: compared {r.get('compared')}, {r.get('median_cosine', r.get('error'))}",
              flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "results" / "h19_raw.jsonl"))
    a = ap.parse_args()
    frame = pool.Frame(ROOT / "results" / "e2_frame_meta.jsonl.gz")
    if a.sample:
        draw(frame)
        return
    e2_pairs(frame)
    sample = json.loads(SAMPLE.read_text(encoding="utf-8"))
    todo = [(i, t) for t, ids in sample["sample"].items() for i in ids]
    out = Path(a.out)
    done = {}
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done.setdefault(r["id"], r)
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
            print(f"[{k}/{len(pending)}] {repo}: {rec.get('reason') or rec.get('format')} "
                  f"content {rec.get('content_date')}; {dec or '-'} ({time.time() - t0:.0f}s)", flush=True)
    # a model lost to the network is retried once here before the run is judged complete
    for repo, _ in list(lost):
        tier = next(t for i, t in todo if i == repo)
        try:
            rec = read_one(repo, tier, frame)
        except Exception as err:  # noqa: BLE001
            print(f"retry {repo}: LOST again ({err})", flush=True)
            continue
        with open(out, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        done[repo] = rec
        lost.remove((repo, _))
    print(f"{len(done)} of {len(todo)} done; {len(lost)} lost to the network or the time budget", flush=True)
    if len(done) == len(todo) and not lost:
        run = ledger.Run("H19", sample_seed=sample["seed"], snapshot=sample["snapshot"])
        for i, t in todo:
            run.record(**done[i])
        print("recorded in the ledger", flush=True)


if __name__ == "__main__":
    main()
