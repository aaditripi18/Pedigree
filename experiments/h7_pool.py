"""H7 (a) (PLAN.md, amendments 2026-11-07): is the true parent in the candidate pool of a
fresh E1 derivative? The pool (pedigree/pool.py) is built from the derivative itself: its
parameter count from its own safetensors header, its family key and dimensions from its
architecture (fine-tuning never changes them, so they are the base's config.json). The
base's identity is used only to score the answer.

    python -m experiments.h7_pool [family ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from experiments import ledger
from experiments.e1_verify import TRAINED
from experiments.finetune import OUT, bases
from pedigree import pool
from pedigree.verify import Local

ROOT = Path(__file__).resolve().parent.parent


def main(families: list[str]) -> None:
    frame = pool.Frame(ROOT / "results" / "e2_frame_meta.jsonl.gz")
    run = ledger.Run("H7a")
    for family in families:
        base = bases()[family]["fresh"]["base"]
        meta = frame.by_id.get(base) or {}
        key = pool.family_key(meta)
        mdims = pool.dims(pool.config(base))
        for kind in TRAINED:
            d = OUT / family / "fresh" / kind
            if not (d / "model.safetensors").exists():
                run.record(family=family, kind=kind, base=base, error="derivative not built")
                continue
            m = Local.open(d)
            params = int(sum(np.prod(s[1]) for s in m.float_shapes().values()))
            cands = frame.candidates(key, params, mdims)
            hit = base in cands
            run.record(family=family, kind=kind, base=base, base_in_frame=base in frame.by_id, family_key=key,
                       params=params, dims=mdims, pool=cands, parent_in_pool=hit,
                       parent_rank=cands.index(base) + 1 if hit else None)
            print(f"{family}/{kind}: key {key} params {params} pool {len(cands)} parent in pool {hit}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or list(bases()))
