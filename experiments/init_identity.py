"""Ground-truth labels for shared initialisation (PLAN.md amendment 2026-11-05).

PolyPythias publishes each run's initial weights (revision `step0`). Two runs share an
initialisation exactly when their step0 tensors are byte-identical. This is a label,
fixed by equality, not by the similarity measure E1 evaluates: it reads only step0,
never the trained weights.

    python -m experiments.init_identity
"""
from __future__ import annotations

import json
import sys

import numpy as np
from pathlib import Path

from pedigree.align import align
from pedigree.verify import Remote

ROOT = Path(__file__).resolve().parent.parent
REFERENCE = "EleutherAI/pythia-160m"
VARIANTS = ([f"EleutherAI/pythia-160m-seed{i}" for i in range(1, 10)]
            + [f"EleutherAI/pythia-160m-data-seed{i}" for i in range(1, 4)]
            + [f"EleutherAI/pythia-160m-weight-seed{i}-corrected" for i in range(1, 4)]
            + ["EleutherAI/pythia-160m-deduped"])
N_TENSORS = 12  # evenly spaced over the model; every byte sampled must match


def identical(a: Remote, b: Remote) -> tuple[bool, int, int]:
    pairs, _, _ = align(a.float_shapes(), b.float_shapes())
    shapes = a.float_shapes()
    # only randomly initialised matrices: LayerNorm gains (all 1) and biases (all 0) are
    # identical at step 0 in every run whatever its seed, so they say nothing
    names = sorted(n for n in pairs if len(shapes[n][1]) >= 2 and int(np.prod(shapes[n][1])) >= 4096)
    step = max(1, len(names) // N_TENSORS)
    checked = equal = 0
    for name in names[::step][:N_TENSORS]:
        x, y = a.sample(name), b.sample(pairs[name])
        checked += 1
        equal += int(x.shape == y.shape and x.tobytes() == y.tobytes())
    return equal == checked and checked > 0, equal, checked


def main() -> None:
    ref = Remote.open(f"hf:{REFERENCE}@step0")
    out = {"reference": REFERENCE + "@step0", "rule": "byte-identical step0 tensors on all sampled tensors",
           "models": {}}
    for repo in VARIANTS:
        try:
            other = Remote.open(f"hf:{repo}@step0")
            same, eq, n = identical(ref, other)
            out["models"][repo] = {"shared_init_with_reference": same, "equal_tensors": eq, "checked": n}
            print(f"{repo}: {'SAME' if same else 'different'} init ({eq}/{n} tensors identical)", flush=True)
        except BaseException as e:  # reported, never silently dropped
            out["models"][repo] = {"error": f"{type(e).__name__}: {e}"}
            print(f"{repo}: ERROR {e}", flush=True)
    (ROOT / "results" / "e1_init_identity.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    sys.exit(1 if any("error" in v for v in out["models"].values()) else 0)


if __name__ == "__main__":
    main()
