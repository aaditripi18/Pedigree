"""E1/E3/E4: the verifier on ground truth (PLAN.md).

For every family and split in results/e1_bases.json, the base (read from the Hub,
pinned to its commit) is compared with:

  trained derivatives   ft_short, ft_long, lora_r8, lora_r64          positive (H1)
  weight transforms     of ft_long: q8_0, q4_0, prune50, prune90       positive (E4, H5)
  permuted              ft_long with every hidden layer permuted       positive (E3, H4)
  scratch               same architecture, fresh initialisation        negative (H1)
  independent models    public, trained separately, same shapes        negative (H1)
  public derivatives    documented distillations (partial shapes)      reported (E4)

Each pair gets the plain verifier (median cosine of sampled tensors) and the
permutation-invariant one (median off-diagonal Gram cosine), appended to the ledger.
Thresholds are not chosen here: scripts/analyse_e1.py picks them on dev only.

    python -m experiments.e1_verify [family ...]
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

from experiments import ledger, transforms
from experiments.finetune import OUT, bases
from pedigree.verify import Local, Remote, compare, compare_invariant

ROOT = Path(__file__).resolve().parent.parent
TRAINED = ["ft_short", "ft_long", "lora_r8", "lora_r64"]


def _pair(run, mdl, base, **label) -> None:
    t0 = time.time()
    try:
        plain = compare(mdl, base)
        inv = compare_invariant(mdl, base)
        run.record(**label, base=base.repo, base_commit=base.commit, model_commit=mdl.commit,
                   matched=plain["matched"], match_share=round(plain["match_share"], 4),
                   compared=plain["compared"], median_cosine=plain["median_cosine"],
                   cosines=plain["cosines"], invariant_compared=inv["compared"],
                   median_invariant=inv["median_invariant"], seconds=round(time.time() - t0, 1))
        print(f"  {label['kind']:<12} {label.get('model', '')[:40]:<40} plain {plain['median_cosine']}"
              f"  invariant {inv['median_invariant']}", flush=True)
    except Exception as e:  # reported in the ledger, never dropped
        run.record(**label, base=base.repo, error=f"{type(e).__name__}: {e}")
        print(f"  {label['kind']:<12} ERROR {e}", flush=True)


def _transformed(src: Path, fn, tmp: Path, name: str) -> tuple[Local, dict]:
    from safetensors.torch import load_file, save_file
    sd = load_file(str(src / "model.safetensors"))
    out = fn(sd)
    report = {}
    if isinstance(out, tuple):
        out, report = out
    d = tmp / name
    d.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in out.items()}, str(d / "model.safetensors"))
    return Local.open(d), report


def main(families: list[str]) -> None:
    spec = json.loads((ROOT / "results" / "e1_bases.json").read_text(encoding="utf-8"))
    run = ledger.Run("E1", derivatives=str(OUT))
    for family in families:
        for split in ("dev", "fresh"):
            entry = bases()[family][split]
            base = Remote.open("hf:" + entry["base"])
            print(f"{family}/{split}: {base.repo}@{base.commit[:10]}", flush=True)
            common = {"family": family, "split": split}
            for kind in TRAINED + ["scratch"]:
                d = OUT / family / split / kind
                if not (d / "meta.json").exists():
                    run.record(**common, kind=kind, label="missing", error="derivative not built")
                    print(f"  {kind}: MISSING", flush=True)
                    continue
                _pair(run, Local.open(d), base, **common, kind=kind, model=f"local:{family}/{split}/{kind}",
                      label="negative" if kind == "scratch" else "positive", experiment_part="E1")
            src = OUT / family / split / "ft_long"
            if (src / "meta.json").exists():
                with tempfile.TemporaryDirectory(dir=os.environ.get("PEDIGREE_TMP")) as tmp:
                    tmp = Path(tmp)
                    for name, fn, part in [("q8_0", transforms.quantize_q8_0, "E4"),
                                           ("q4_0", transforms.quantize_q4_0, "E4"),
                                           ("prune50", lambda sd: transforms.prune(sd, 0.5), "E4"),
                                           ("prune90", lambda sd: transforms.prune(sd, 0.9), "E4"),
                                           ("permuted", transforms.permute_hidden, "E3")]:
                        m, report = _transformed(src, fn, tmp, name)
                        _pair(run, m, base, **common, kind=name, model=f"local:{family}/{split}/ft_long+{name}",
                              label="positive", experiment_part=part, transform_report=report)
            for repo in entry["independent"]:
                try:
                    other = Remote.open("hf:" + repo)
                except Exception as e:
                    run.record(**common, kind="independent", model=repo, label="negative",
                               error=f"{type(e).__name__}: {e}")
                    print(f"  independent {repo}: ERROR {e}", flush=True)
                    continue
                _pair(run, other, base, **common, kind="independent", model=repo, label="negative",
                      experiment_part="E1")
            for repo in spec.get("public_derivatives", {}).get(entry["base"], []):
                try:
                    _pair(run, Remote.open("hf:" + repo), base, **common, kind="public_derivative", model=repo,
                          label="positive", experiment_part="E4")
                except Exception as e:
                    run.record(**common, kind="public_derivative", model=repo, error=f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    main(sys.argv[1:] or list(bases()))
