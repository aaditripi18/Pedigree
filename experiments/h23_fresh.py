"""H23a/H23b, fresh data (PLAN.md, amendment 2026-11-20): alignment v3 on models stored as GGUF,
ONNX, CTranslate2 or NeMo.

  python -m experiments.h23_fresh --sample     # draw results/h23_sample.json (committed before any reading)
  python -m experiments.h23_fresh [--out PATH] # read

Each sampled model is aligned with each declared base and the two top-ranked E2-pool candidates
under names (E2's alignment), v2 (GGUF only) and v3 with names ignored (`struct3`), and the declared
pairs also under `auto3`. Verdicts are computed afterwards by scripts/analyse.py.
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
from pedigree import pool
from pedigree.verify import compare

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "results" / "h23_sample.json"
SEED = 20261120
PER_TIER = 150
LIBRARIES = {
    "gguf": {"gguf", "GGUF", "llama.cpp", "llama-cpp", "llamacpp", "llama_cpp", "llama-cpp-python", "node-llama-cpp",
             "ggml", "ik_llama.cpp", "comfyui-gguf-reboot", "wllama", "ollama"},
    "onnx": {"onnx", "onnxruntime", "onnxruntime-web", "onnxruntime-genai", "transformers.js", "gliner2-onnx", "onnx-asr"},
    "ctranslate2": {"ctranslate2", "faster-whisper"},
    "nemo": {"nemo"},
}
MODES = ("names", "struct2", "struct3", "auto3")


def draw(frame: pool.Frame) -> None:
    e2 = json.loads((ROOT / "results" / "e2_sample.json").read_text(encoding="utf-8"))
    seen = {e["id"] for rows in e2["sample"].values() for e in rows}
    for f in ("h13_sample.json", "h15_sample.json", "h19_sample.json"):
        s = json.loads((ROOT / "results" / f).read_text(encoding="utf-8"))
        seen |= {i for ids in s["sample"].values() for i in ids}
    libs = set().union(*LIBRARIES.values())
    rng = random.Random(SEED)
    out = {"seed": SEED, "per_tier": PER_TIER, "tiers": e2["tiers"], "snapshot": e2["snapshot"],
           "libraries": {k: sorted(v) for k, v in LIBRARIES.items()},
           "rule": "frame models with a declared base whose library is one of the listed ones, in none of E2's, "
                   "H13's, H15's or H19's samples; uniform within each download tier; all where fewer than 150 remain",
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
    gg = bool(getattr(model, "gguf_meta", None))
    rec["declared"] = []
    for b in rec["declared_base"]:
        other, bf = open_repo(b, frame)
        row = {"base": b, **{k: v for k, v in bf.items() if k in ("via", "reason", "format")}}
        if other is not None:
            modes = MODES if gg != bool(getattr(other, "gguf_meta", None)) else ("names", "struct3", "auto3")
            row["alignments"] = alignments(model, other, modes)
        rec["declared"].append(row)
    rec["pool_alignments"] = []
    cands = [c for c in frame.candidates(rec["family_key"], rec.get("params"), rec.get("dims"), exclude={repo})
             if c not in rec["declared_base"]][:2]
    for c in cands:
        other, _ = open_repo(c, frame)
        if other is not None:
            modes = ("names", "struct2", "struct3") if gg != bool(getattr(other, "gguf_meta", None)) else ("names", "struct3")
            rec["pool_alignments"].append({"candidate": c, **alignments(model, other, modes)})
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "results" / "h23_raw.jsonl"))
    a = ap.parse_args()
    frame = pool.Frame(ROOT / "results" / "e2_frame_meta.jsonl.gz")
    if a.sample:
        draw(frame)
        return
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
        except Exception as err:  # noqa: BLE001
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
            dec = "; ".join(f"{x['base']}: " + ", ".join(f"{m}={(v or {}).get('compared')}"
                                                        for m, v in (x.get("alignments") or {}).items())
                            for x in rec.get("declared", []))
            print(f"[{k}/{len(pending)}] {repo}: {rec.get('reason') or rec.get('format')}; {dec or '-'} "
                  f"({time.time() - t0:.0f}s)", flush=True)
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
        run = ledger.Run("H23", sample_seed=sample["seed"], snapshot=sample["snapshot"])
        for i, t in todo:
            run.record(**done[i])
        print("recorded in the ledger", flush=True)


if __name__ == "__main__":
    main()
