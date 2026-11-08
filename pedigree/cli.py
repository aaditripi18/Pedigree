"""pedigree verify: does a model come from the model it claims, by its weights?

    pedigree verify hf:org/model                      # against its declared base_model
    pedigree verify hf:org/model --base hf:org/base   # against a base you name
    pedigree verify hf:org/model --base hf:a --base hf:b --json

Reads only file headers and small slices of tensors (HTTP range requests), never a
whole model unless the permutation-invariant check needs full matrices (only when the
plain score is low, and only matrices up to --max-elements values). Nothing downloaded
is executed: PyTorch checkpoints are read by a restricted unpickler.

Verdicts use the thresholds fixed on ground truth (pedigree/thresholds.json, written by
scripts/analyse.py from E1 dev negatives): related, unrelated (while enough tensors were
compared), or not comparable. "Unrelated" means the weights show no lineage, not that
the claim is false for every reason; read the README's limitations.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import hub
from .verify import Remote, compare, compare_invariant, kurtosis_gap

THRESHOLDS = Path(__file__).with_name("thresholds.json")
MIN_COMPARED = 8


def load_thresholds() -> dict:
    return json.loads(THRESHOLDS.read_text(encoding="utf-8"))


def check(model: Remote, base: Remote, T: dict, max_elements: int) -> dict:
    c = compare(model, base)
    out = {"base": base.repo, "base_commit": base.commit, "matched_tensors": c["matched"],
           "compared": c["compared"], "plain": c["median_cosine"], "identical_samples": c["identical"],
           "kurtosis_gap": kurtosis_gap(c["kurtosis"])}
    if c["compared"] and (c["median_cosine"] is None or c["median_cosine"] <= T["plain"]):
        inv = compare_invariant(model, base, max_elements=max_elements)
        out["invariant"] = inv["median_invariant"]
    scores = {k: out.get(k) for k in ("plain", "invariant") if out.get(k) is not None}
    if any(v > T[k] for k, v in scores.items()):
        out["verdict"] = "identical copy" if c["identical"] else "related"
    elif scores and c["compared"] >= MIN_COMPARED:
        out["verdict"] = "unrelated"
    else:
        out["verdict"] = "not comparable"
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pedigree", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify", help="compare a model with its declared or named base")
    v.add_argument("model", help="hf:org/name[@revision]")
    v.add_argument("--base", action="append", default=[], help="hf:org/name (repeatable); default: declared base_model")
    v.add_argument("--json", action="store_true")
    v.add_argument("--max-elements", type=int, default=32 * 2 ** 20)
    a = ap.parse_args(argv)
    T = load_thresholds()
    spec = a.model if a.model.startswith("hf:") else "hf:" + a.model
    res = hub.resolve(spec)
    bases = [b if b.startswith("hf:") else "hf:" + b for b in a.base] or ["hf:" + b for b in res.base_model]
    report = {"model": res.repo, "commit": res.commit, "declared_base": res.base_model,
              "thresholds": T, "results": []}
    if not bases:
        report["note"] = "the model card declares no base_model; name one with --base"
    else:
        model = Remote.from_files(res.repo, res.commit, res.files, res.sizes)
        for b in bases:
            try:
                report["results"].append(check(model, Remote.open(b), T, a.max_elements))
            except Exception as e:
                report["results"].append({"base": b, "verdict": "not comparable", "error": f"{type(e).__name__}: {e}"})
    if a.json:
        print(json.dumps(report, indent=1))
    else:
        print(f"{report['model']} @ {report['commit'][:10]}")
        if "note" in report:
            print("  " + report["note"])
        for r in report["results"]:
            if "error" in r:
                print(f"  {r['base']}: not comparable ({r['error']})")
                continue
            inv = f", invariant {r['invariant']:.3f}" if r.get("invariant") is not None else ""
            plain = f"{r['plain']:.3f}" if r.get("plain") is not None else "n/a"
            print(f"  vs {r['base']}: {r['verdict'].upper()}  (plain {plain}{inv}; "
                  f"{r['compared']} tensors compared, {r['matched_tensors']} matched by name and shape)")
    return 0 if all(r.get("verdict") in ("related", "identical copy") for r in report["results"]) else 1


if __name__ == "__main__":
    sys.exit(main())
