"""E1b (PLAN.md, amendment 2026-11-07): whole files (H6) and direction by weights (H8, D2).

The same ground-truth pairs as E1 (experiments/e1_verify.py), rebuilt the same way:

  fresh, every pair      sampled score (as E1) and whole-file score (every aligned
                         tensor of >= 4096 values, read in full)                     H6
  dev and fresh,         the median kurtosis gap (derivative minus base) of the
  trained derivatives    sampled tensors; dev fixes the margin, fresh judges         H8 D2

    python -m experiments.e1b_whole [family ...]
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

from experiments import ledger, transforms
from experiments.e1_verify import FULL_ATTACK, TRAINED, _attack_full, _transformed
from experiments.finetune import OUT, bases
from pedigree.verify import Local, Remote, compare, compare_whole, kurtosis_gap

ROOT = Path(__file__).resolve().parent.parent


def _pair(run, mdl, base, whole: bool, **label) -> None:
    t0 = time.time()
    try:
        c = compare(mdl, base)
        rec = {"median_cosine": c["median_cosine"], "compared": c["compared"],
               "kurtosis_gap": kurtosis_gap(c["kurtosis"]), "kurtosis": c["kurtosis"]}
        if whole:
            w = compare_whole(mdl, base)
            rec.update(median_whole=w["median_whole"], whole_compared=w["compared"])
        run.record(**label, base=base.repo, base_commit=base.commit, **rec, seconds=round(time.time() - t0, 1))
        print(f"  {label['kind']:<16} {label.get('model', '')[-38:]:<38} sampled {rec['median_cosine']}"
              f"  whole {rec.get('median_whole')}  kurtosis gap {rec['kurtosis_gap']}", flush=True)
    except Exception as e:
        run.record(**label, base=base.repo, error=f"{type(e).__name__}: {e}")
        print(f"  {label['kind']:<16} ERROR {e}", flush=True)


def main(families: list[str]) -> None:
    spec = json.loads((ROOT / "results" / "e1_bases.json").read_text(encoding="utf-8"))
    run = ledger.Run("E1b", derivatives=str(OUT))
    for family in families:
        for split in ("dev", "fresh"):
            entry = bases()[family][split]
            base = Remote.open("hf:" + entry["base"])
            whole = split == "fresh"
            common = {"family": family, "split": split}
            print(f"{family}/{split}: {base.repo}@{base.commit[:10]}", flush=True)
            for kind in TRAINED + ["scratch"]:
                d = OUT / family / split / kind
                if not (d / "meta.json").exists():
                    run.record(**common, kind=kind, error="derivative not built")
                    continue
                if split == "dev" and kind == "scratch":
                    continue
                _pair(run, Local.open(d), base, whole, **common, kind=kind, model=f"local:{family}/{split}/{kind}",
                      label="negative" if kind == "scratch" else "positive")
            if not whole:
                continue
            src = OUT / family / split / "ft_long"
            if (src / "meta.json").exists():
                with tempfile.TemporaryDirectory(dir=os.environ.get("PEDIGREE_TMP")) as tmp:
                    jobs = [("q8_0", transforms.quantize_q8_0), ("q4_0", transforms.quantize_q4_0),
                            ("prune50", lambda sd: transforms.prune(sd, 0.5)),
                            ("prune90", lambda sd: transforms.prune(sd, 0.9)),
                            ("permuted", transforms.permute_hidden)]
                    if family in FULL_ATTACK:
                        jobs.append(("permuted_full", _attack_full(family, entry["base"])))
                    for name, fn in jobs:
                        try:
                            m, _ = _transformed(src, fn, Path(tmp), name)
                        except Exception as e:
                            run.record(**common, kind=name, error=f"{type(e).__name__}: {e}")
                            continue
                        _pair(run, m, base, True, **common, kind=name, model=f"local:{family}/{split}/ft_long+{name}",
                              label="positive")
            for repo in entry["independent"]:
                try:
                    _pair(run, Remote.open("hf:" + repo), base, True, **common, kind="independent", model=repo,
                          label="negative")
                except Exception as e:
                    run.record(**common, kind="independent", model=repo, error=f"{type(e).__name__}: {e}")
            for repo in spec.get("public_derivatives", {}).get(entry["base"], []):
                try:
                    _pair(run, Remote.open("hf:" + repo), base, True, **common, kind="public_derivative", model=repo,
                          label="positive")
                except Exception as e:
                    run.record(**common, kind="public_derivative", model=repo, error=f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    main(sys.argv[1:] or list(bases()))
