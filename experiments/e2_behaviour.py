"""E2, second instrument (PLAN.md, amendment 2026-11-07): where a declared base cannot
be compared by weights, compare behaviour instead.

Eligible pairs (model, declared base) from the latest E2 run: the weight verdict is "not
comparable" (no score above its threshold, fewer than 8 tensors compared, or a weight
read failed), and both have <= 2B parameters. Then:

  text   both load as the same kind of language model (causal or masked) with every
         weight of the language-model head present, and the two tokenizers have the same
         vocabulary: E5's WikiText-2 test probes, E5's statistics
  image  both load as image classifiers with the same labels: E5's CIFAR-10 test probes

Anything else is recorded as "behaviour not applicable" with the reason. Statistics only;
scripts/analyse.py applies H9's rule (chosen on E5 dev) to them.

    python -m experiments.e2_behaviour
"""
from __future__ import annotations

import json
from pathlib import Path

from experiments import e5_behaviour as e5
from experiments import ledger

ROOT = Path(__file__).resolve().parent.parent
MAX_PARAMS = 2_000_000_000
MIN_COMPARED = 8


def eligible(e2: list[dict], T: dict) -> list[tuple[dict, dict]]:
    out = []
    for m in e2:
        if m.get("reason"):
            continue
        for d in m.get("declared", []):
            scores = [d.get("median_cosine"), d.get("median_invariant")]
            related = any(s is not None and s > T[k] for s, k in zip(scores, ("plain", "invariant")))
            comparable = (d.get("compared") or 0) >= MIN_COMPARED
            if not related and not comparable:
                out.append((m, d))
    return out


def _load_text(repo: str):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer
    cfg = AutoConfig.from_pretrained(repo)
    causal = not getattr(cfg, "is_encoder_decoder", False) and any(
        a.endswith(("ForCausalLM", "LMHeadModel")) for a in (cfg.architectures or []))
    masked = any(a.endswith("ForMaskedLM") for a in (cfg.architectures or []))
    if not causal and not masked:
        raise ValueError(f"not a language model ({cfg.architectures})")
    cls = AutoModelForCausalLM if causal else AutoModelForMaskedLM
    model, info = cls.from_pretrained(repo, output_loading_info=True)
    head_missing = [k for k in info["missing_keys"] if "lm_head" in k or "cls." in k or "pred" in k]
    if head_missing:
        raise ValueError(f"language-model head not in the checkpoint: {head_missing[:2]}")
    tok = AutoTokenizer.from_pretrained(repo)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model.float().eval(), tok, causal


def _load_image(repo: str):
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    model = AutoModelForImageClassification.from_pretrained(repo).float().eval()
    proc = AutoImageProcessor.from_pretrained(repo)
    size = proc.size.get("height") or proc.size.get("shortest_edge") or 224
    return model, (size, tuple(proc.image_mean), tuple(proc.image_std))


def pair(model_id: str, base_id: str) -> dict:
    try:
        a, ta, ca = _load_text(model_id)
        b, tb, cb = _load_text(base_id)
        if ca != cb:
            return {"applicable": False, "reason": "one causal, one masked language model"}
        if ta.get_vocab() != tb.get_vocab():
            return {"applicable": False, "reason": "different tokenizers: no shared output space"}
        lines, digest = e5.text_probes()
        r = e5.compare_behaviour(b, a, lambda m: e5._text_run(m, tb, lines, cb))
        return {"applicable": True, "modality": "text", "probes": digest, **r}
    except Exception as text_err:
        try:
            a, (size, mean, std) = _load_image(model_id)
            b, _ = _load_image(base_id)
            if a.config.id2label != b.config.id2label:
                return {"applicable": False, "reason": f"text: {text_err}; image: different label sets"}
            x, digest = e5.image_probes(size, mean, std)
            r = e5.compare_behaviour(b, a, lambda m: e5._image_run(m, x))
            return {"applicable": True, "modality": "image", "probes": digest, **r}
        except Exception as img_err:
            return {"applicable": False, "reason": f"text: {str(text_err)[:150]}; image: {str(img_err)[:150]}"}


def main() -> None:
    rows = [json.loads(x) for x in (ROOT / "results" / "ledger.jsonl").read_text(encoding="utf-8").splitlines() if x]
    e2 = [r for r in rows if r.get("experiment") == "E2"]
    last = max(r["run_id"] for r in e2)
    e2 = [r for r in e2 if r["run_id"] == last]
    T = json.loads((ROOT / "pedigree" / "thresholds.json").read_text(encoding="utf-8"))
    run = ledger.Run("E2B", e2_run=last)
    pairs = eligible(e2, T)
    print(f"{len(pairs)} declared pairs not comparable by weights", flush=True)
    for m, d in pairs:
        if (m.get("params") or 0) > MAX_PARAMS:
            r = {"applicable": False, "reason": "model above 2B parameters"}
        else:
            r = pair(m["id"], d["base"])
        run.record(model=m["id"], base=d["base"], tier=m["tier"], **r)
        print(f"  {m['id']} vs {d['base']}: {r.get('modality') or r['reason'][:90]} "
              f"{'agree %.3f' % r['agreement'] if r.get('applicable') else ''}", flush=True)


if __name__ == "__main__":
    main()
