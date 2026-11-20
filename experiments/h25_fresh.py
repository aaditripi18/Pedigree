"""H25a/H25b, fresh data (PLAN.md, amendment 2026-11-20 (2)): alignment v4 on models stored as GGUF,
ONNX, CTranslate2 or NeMo, drawn after H23 by the same rule.

  python -m experiments.h25_fresh --sample     # draw results/h25_sample.json (committed before any reading)
  python -m experiments.h25_fresh [--out PATH] # read the fresh sample (ledger H25)
  python -m experiments.h25_fresh --design [--out PATH]
                                               # H23's pairs v4 was designed on, under v4 (ledger H25d)

Each sampled model is aligned with each declared base under names, v3 and v4 (`struct3`, `auto3`,
`struct4`, `auto4`), and with the two top-ranked pool candidates under names, v3 and v4 with names
ignored. Verdicts are computed afterwards by scripts/analyse.py.
"""
from __future__ import annotations

import argparse
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from experiments import ledger
from experiments.e2_audit import BUDGET, WORKERS, NetworkError, _network, logical_params, open_repo
from experiments.h23_fresh import LIBRARIES
from pedigree import pool
from pedigree.verify import compare

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "results" / "h25_sample.json"
SEED = 20261121
PER_TIER = 150
DECLARED_MODES = ("names", "struct3", "auto3", "struct4", "auto4")
POOL_MODES = ("names", "struct3", "struct4")
MIN_COMPARED = 8


def draw(frame: pool.Frame) -> None:
    e2 = json.loads((ROOT / "results" / "e2_sample.json").read_text(encoding="utf-8"))
    seen = {e["id"] for rows in e2["sample"].values() for e in rows}
    for f in ("h13_sample.json", "h15_sample.json", "h19_sample.json", "h23_sample.json"):
        s = json.loads((ROOT / "results" / f).read_text(encoding="utf-8"))
        seen |= {i for ids in s["sample"].values() for i in ids}
    libs = set().union(*LIBRARIES.values())
    rng = random.Random(SEED)
    out = {"seed": SEED, "per_tier": PER_TIER, "tiers": e2["tiers"], "snapshot": e2["snapshot"],
           "libraries": {k: sorted(v) for k, v in LIBRARIES.items()},
           "rule": "frame models with a declared base whose library is one of the listed ones, in none of E2's, "
                   "H13's, H15's, H19's or H23's samples; uniform within each download tier; all where fewer "
                   "than 150 remain",
           "sample": {}}
    for tier, (lo, hi) in e2["tiers"].items():
        pool_ = [m["id"] for m in frame.models[lo - 1:hi]
                 if m["id"] not in seen and not m.get("missing") and m.get("declared_base")
                 and frame.by_id.get(m["id"], {}).get("library") in libs]
        k = min(PER_TIER, len(pool_))
        out["sample"][tier] = sorted(rng.sample(pool_, k))
        print(f"{tier}: {len(pool_)} eligible, {k} drawn", flush=True)
    SAMPLE.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")


def alignments(model, other, modes) -> dict:
    out = {}
    for mode in modes:
        try:
            c = compare(model, other, alignment=mode)
            out[mode] = {"median_cosine": c["median_cosine"], "compared": c["compared"], "matched": c["matched"]}
            if "alignment_map" in c:
                out[mode].update(map=c["alignment_map"], map_scores=c["map_scores"])
        except Exception as e:  # noqa: BLE001
            if _network(e):
                raise NetworkError(f"{model.repo} vs {other.repo}: {type(e).__name__}: {e}") from None
            out[mode] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
    return out


def read_one(repo: str, tier: str, frame: pool.Frame) -> dict:
    meta = frame.by_id.get(repo, {})
    rec = {"id": repo, "tier": tier, "library": meta.get("library"), "declared_base": meta.get("declared_base") or [],
           "family_key": pool.family_key(meta)}
    model, facts = open_repo(repo, frame)
    rec.update(facts)
    if model is None:
        return rec
    rec.update(params=(None if model.quant else pool.meta_params(meta)) or logical_params(model),
               dims=pool.dims(pool.config(facts.get("via") or repo)))
    rec["declared"] = []
    for b in rec["declared_base"]:
        other, bf = open_repo(b, frame)
        row = {"base": b, **{k: v for k, v in bf.items() if k in ("via", "reason", "format")}}
        if other is not None:
            row["alignments"] = alignments(model, other, DECLARED_MODES)
        rec["declared"].append(row)
    rec["pool_alignments"] = []
    cands = [c for c in frame.candidates(rec["family_key"], rec.get("params"), rec.get("dims"), exclude={repo})
             if c not in rec["declared_base"]][:2]
    for c in cands:
        other, _ = open_repo(c, frame)
        if other is not None:
            rec["pool_alignments"].append({"candidate": c, **alignments(model, other, POOL_MODES)})
    return rec


def design_pairs() -> list[dict]:
    """H23's pairs that v4 was designed on: declared pairs names aligned < 8 tensors in (both read),
    and the pool pairs where v3 gave the opposite of names' verdict (found by scripts/analyse.py's rule:
    any pool pair named here is one of them; the list is fixed from the committed H23 ledger)."""
    rows = [json.loads(x) for x in (ROOT / "results" / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
            if x.strip()]
    h23 = [r for r in rows if r.get("experiment") == "H23"]
    last = max(r["run_id"] for r in h23)
    out = []
    for r in (x for x in h23 if x["run_id"] == last):
        for d in r.get("declared", []):
            a = d.get("alignments") or {}
            if a.get("names") is not None and (a["names"].get("compared") or 0) < MIN_COMPARED:
                out.append({"id": r["id"], "other": d["base"], "role": "declared"})
    for repo, cand in DESIGN_OPPOSITE:
        out.append({"id": repo, "other": cand, "role": "pool"})
    return out


# H23a's two opposite verdicts (PLAN.md, outcome of H23)
DESIGN_OPPOSITE = [("SalahALHaismawi/Nuqta-Arabic-OCR-4B-GGUF", "huihui-ai/Huihui-Qwen3.5-4B-abliterated"),
                   ("SalahALHaismawi/Nuqta-Arabic-OCR-4B-GGUF", "InternScience/Agents-A1-4B")]


def read_design(p: dict, frame: pool.Frame) -> dict:
    rec = dict(p)
    model, mf = open_repo(p["id"], frame)
    other, of = open_repo(p["other"], frame)
    if model is None or other is None:
        rec["reason"] = mf.get("reason") if model is None else of.get("reason")
        return rec
    rec["alignments"] = alignments(model, other, ("names", "struct3", "auto3", "struct4", "auto4"))
    return rec


def run(todo: list, key, reader, out: Path, experiment: str, sample_seed, snapshot) -> None:
    done = {}
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done.setdefault(key(r), r)
    pending = [t for t in todo if key(t) not in done]
    print(f"{len(done)} done, {len(pending)} to read", flush=True)
    t0, lost, lock = time.time(), [], threading.Lock()

    def work(t):
        if time.time() - t0 > BUDGET:
            return t, None, "budget"
        try:
            return t, reader(t), None
        except NetworkError as err:
            return t, None, str(err)
        except Exception as err:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            return t, None, f"unexpected {type(err).__name__}: {err}"

    def note(k, rec):
        rows = rec.get("declared") or ([{"base": rec.get("other"), "alignments": rec.get("alignments")}]
                                       if "other" in rec else [])
        dec = "; ".join(f"{x['base']}: " + ", ".join(f"{m}={(v or {}).get('compared')}"
                                                    for m, v in (x.get("alignments") or {}).items()) for x in rows)
        print(f"[{k}/{len(pending)}] {rec['id']}: {rec.get('reason') or rec.get('format') or ''}; {dec or '-'} "
              f"({time.time() - t0:.0f}s)", flush=True)

    with ThreadPoolExecutor(WORKERS) as ex:
        futs = [ex.submit(work, t) for t in pending]
        for k, f in enumerate(as_completed(futs), 1):
            t, rec, err = f.result()
            if rec is None:
                lost.append((t, err))
                print(f"[{k}/{len(pending)}] {key(t)}: LOST ({err})", flush=True)
                continue
            with lock, open(out, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
            done[key(t)] = rec
            note(k, rec)
    for t, err in list(lost):
        try:
            rec = reader(t)
        except Exception as e:  # noqa: BLE001
            print(f"retry {key(t)}: LOST again ({e})", flush=True)
            continue
        with open(out, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        done[key(t)] = rec
        lost.remove((t, err))
    print(f"{len(done)} of {len(todo)} done; {len(lost)} lost to the network or the time budget", flush=True)
    if len(done) == len(todo) and not lost:
        r = ledger.Run(experiment, sample_seed=sample_seed, snapshot=snapshot)
        for t in todo:
            r.record(**done[key(t)])
        print("recorded in the ledger", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--design", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    frame = pool.Frame(ROOT / "results" / "e2_frame_meta.jsonl.gz")
    if a.sample:
        draw(frame)
        return
    if a.design:
        todo = design_pairs()
        run(todo, lambda r: (r["id"], r["other"], r["role"]), lambda p: read_design(p, frame),
            Path(a.out or ROOT / "results" / "h25_design_raw.jsonl"), "H25d", None, None)
        return
    sample = json.loads(SAMPLE.read_text(encoding="utf-8"))
    todo = [{"id": i, "tier": t} for t, ids in sample["sample"].items() for i in ids]
    run(todo, lambda r: r["id"], lambda p: read_one(p["id"], p["tier"], frame),
        Path(a.out or ROOT / "results" / "h25_raw.jsonl"), "H25", sample["seed"], sample["snapshot"])


if __name__ == "__main__":
    main()
