"""H18's scope (PLAN.md, amendment 2026-11-17 (4)): the declared pairs still not comparable after
H17, each put in exactly one category by the rules below, in order. Only pairs in the last category
are run by behaviour; the others are counted by reason.

  python -m experiments.h18_scope      # writes results/h18_scope.json (no weights are read)

1. not readable: the model or its base could not be read (gated, 401/403/404, no weight files)
2. more than 14B parameters: either side
3. not a text language model: a side whose config.json names an architecture that is not
   *ForCausalLM, or that holds a vision, audio or speech sub-model; a side with no config.json
   (GGUF, NeMo) is judged by its base's
4. runs only through the repository's own code: an `auto_map` in config.json, or a model_type
   transformers does not define (Pedigree never runs code from a repository)
5. parameter count unknown: either side (it cannot be shown to be <= 14B)
6. different tokenizer: the two tokenizer.json vocabularies are not identical
7. in scope
"""
from __future__ import annotations

import hashlib
import json
import sys
import urllib.parse
from collections import Counter
from pathlib import Path

from pedigree import hub, pool

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "h18_scope.json"
MAX_PARAMS = 14e9
SUB_MODELS = ("vision_config", "audio_config", "speech_config", "encoder_config", "vision_tower", "audio_tower")


def pairs_after_h17() -> list[tuple[dict, dict]]:
    sys.path.insert(0, str(ROOT / "scripts"))
    import analyse as A
    rows = A.load()
    L = A.latest_runs
    key = lambda r: (r["model"], r["other"], r["role"])  # noqa: E731
    T = A.thresholds(L(rows, "E1", lambda r: r.get("family")))
    allm = A.with_rescores(L(rows, "E2", lambda r: 0), L(rows, "E2r", lambda r: 0) + L(rows, "E2p", lambda r: 0),
                           L(rows, "E2g", key), L(rows, "E2s", key), L(rows, "E2f", key))
    return [(m, d) for m in allm if not m.get("reason") for d in m.get("declared", [])
            if A.related(d, T, "t") == "not comparable"]


def _unreadable(d: dict) -> str | None:
    r = str(d.get("reason") or d.get("error") or "")
    return r if r.startswith("not readable") or "HTTP" in r or "gated" in r else None


def _hub_params(repo: str) -> int:
    """The Hub's count of a repository's safetensors parameters (0 if it gives none)."""
    from experiments.e2_audit import info
    return int((info(repo).get("safetensors") or {}).get("total") or 0)


def _text_lm(cfg: dict) -> bool:
    arch = cfg.get("architectures") or []
    return bool(arch) and all(a.endswith("ForCausalLM") for a in arch) and not any(k in cfg for k in SUB_MODELS)


def _own_code(cfg: dict) -> bool:
    from transformers import CONFIG_MAPPING
    return bool(cfg.get("auto_map")) or (bool(cfg.get("model_type")) and cfg["model_type"] not in CONFIG_MAPPING)


def _vocab(repo: str) -> str | None:
    q = urllib.parse.quote(repo, safe="/")
    try:
        with hub._open(f"{hub.HOST}/{q}/resolve/main/tokenizer.json") as r:
            v = json.loads(r.read().decode("utf-8")).get("model", {}).get("vocab")
    except Exception:  # noqa: BLE001
        return None
    return hashlib.sha256(json.dumps(v, sort_keys=True).encode()).hexdigest() if v else None


def classify(m: dict, d: dict, frame: pool.Frame) -> tuple[str, str]:
    base = d["base"]
    why = _unreadable(d)
    if why:
        return "not readable", why
    pm = m.get("params") or 0
    pb = pool.meta_params(frame.by_id.get(base, {})) or _hub_params(base)
    if max(pm, pb) > MAX_PARAMS:
        return "more than 14B parameters", f"model {pm:.3g}, base {pb:.3g}"
    cm, cb = pool.config(m["id"]), pool.config(base)
    for side, cfg in (("base", cb), ("model", cm)):
        if cfg and not _text_lm(cfg):
            return "not a text language model", f"{side}: {cfg.get('architectures')} {cfg.get('model_type')}"
    if not cb:
        return "not a text language model", "base: no config.json"
    for side, cfg in (("model", cm), ("base", cb)):
        if cfg and _own_code(cfg):
            return "runs only through the repository's own code", f"{side}: model_type {cfg.get('model_type')}, " \
                                                                   f"auto_map {sorted(cfg.get('auto_map') or {})}"
    if not pm or not pb:
        return "parameter count unknown", f"model {pm or 'unknown'}, base {pb or 'unknown'}"
    if m.get("format") != "safetensors":
        raise NotImplementedError(f"{m['id']}: tokenizer comparison for {m.get('format')} is not written")
    va, vb = _vocab(m["id"]), _vocab(base)
    if va is None or va != vb:
        return "different tokenizer", f"vocab {va and va[:12]} vs {vb and vb[:12]}"
    return "in scope", ""


def main() -> None:
    frame = pool.Frame(ROOT / "results" / "e2_frame_meta.jsonl.gz")
    rows = []
    for m, d in pairs_after_h17():
        cat, why = classify(m, d, frame)
        rows.append({"model": m["id"], "base": d["base"], "category": cat, "evidence": why})
        print(f"{cat:45} {m['id']} vs {d['base']}: {why}", flush=True)
    counts = Counter(r["category"] for r in rows)
    OUT.write_text(json.dumps({"rule": "PLAN.md H18 scope; experiments/h18_scope.py rules 1-7 in order; declared pairs "
                                       "not comparable after H17 (E2 with E2r, E2p, E2g, E2s, E2f)",
                               "counts": dict(counts), "pairs": rows}, indent=1) + "\n", encoding="utf-8")
    print(dict(counts))


if __name__ == "__main__":
    main()
