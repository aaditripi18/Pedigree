"""E6 and E7 (PLAN.md, amendment 2026-11-05): public checkpoints only, no training.

E6  What does weight similarity detect? pythia-160m's final weights against runs that
    share its initialisation (data-seed*, deduped: byte-identical step0, see
    results/e1_init_identity.json) and runs that do not (seed*, weight-seed*-corrected).
E7  How far back along a training run is lineage visible? Each Stanford Mistral GPT-2
    small run's final weights against its own checkpoints (ancestors by construction) and
    against the same steps of the other four runs (matched negatives).

Each pair: plain verifier and permutation-invariant check, appended to the ledger.

    python -m experiments.e6_e7 [e6] [e7]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from experiments import ledger
from pedigree.verify import Remote, compare, compare_invariant

ROOT = Path(__file__).resolve().parent.parent


def _open(spec: str, cache: dict):
    if spec not in cache:
        cache[spec] = Remote.open(spec)
    return cache[spec]


def _pair(run, cache, a: str, b: str, **label) -> None:
    t0 = time.time()
    try:
        x, y = _open(a, cache), _open(b, cache)
        plain, inv = compare(x, y), compare_invariant(x, y)
        run.record(**label, model=a, model_commit=x.commit, base=b, base_commit=y.commit,
                   matched=plain["matched"], compared=plain["compared"],
                   median_cosine=plain["median_cosine"], cosines=plain["cosines"],
                   median_invariant=inv["median_invariant"], seconds=round(time.time() - t0, 1))
        print(f"  {a:<55} vs {b:<45} plain {plain['median_cosine']:.4f} inv {inv['median_invariant']:.4f}",
              flush=True)
    except Exception as e:  # recorded, never dropped
        run.record(**label, model=a, base=b, error=f"{type(e).__name__}: {e}")
        print(f"  {a} vs {b}: ERROR {e}", flush=True)


def e6() -> None:
    spec = json.loads((ROOT / "results" / "e1_bases.json").read_text(encoding="utf-8"))
    labels = json.loads((ROOT / "results" / "e1_init_identity.json").read_text(encoding="utf-8"))["models"]
    ref = spec["shared_origin"]["reference"]
    others = spec["shared_origin"]["models"] + spec["families"]["text_decoder"]["fresh"]["independent"]
    run = ledger.Run("E6", reference=ref)
    cache: dict = {}
    print(f"E6: {ref}", flush=True)
    for m in others:
        lab = labels.get(m, {})
        if "shared_init_with_reference" not in lab:
            run.record(model=m, base=ref, error=f"no initialisation label: {lab.get('error', 'missing')}")
            continue
        relation = ("shared_init_different_data" if lab["shared_init_with_reference"]
                    else "different_init_same_data" if "weight-seed" in m else "different_init_different_data")
        _pair(run, cache, "hf:" + m, "hf:" + ref, relation=relation,
              shared_init=lab["shared_init_with_reference"])


def e7() -> None:
    spec = json.loads((ROOT / "results" / "e1_bases.json").read_text(encoding="utf-8"))["trajectory"]
    runs, steps, final = spec["runs"], spec["checkpoint_steps"], spec["final"]
    run = ledger.Run("E7", runs=runs, steps=steps, final=final)
    cache: dict = {}
    for r in runs:
        print(f"E7: {r}@checkpoint-{final}", flush=True)
        for step in steps:
            for other in runs:
                _pair(run, cache, f"hf:{other}@checkpoint-{step}", f"hf:{r}@checkpoint-{final}",
                      run_final=r, checkpoint_run=other, step=step, distance=final - step,
                      same_run=other == r)
        # keep memory bounded: tensors of finished runs are cached on disk anyway
        cache.clear()


if __name__ == "__main__":
    which = sys.argv[1:] or ["e6", "e7"]
    for w in which:
        {"e6": e6, "e7": e7}[w]()
