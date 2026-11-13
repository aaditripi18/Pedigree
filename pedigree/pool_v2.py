"""Candidate pool, version 2 (PLAN.md, amendment 2026-11-13). E2's pool (pedigree/pool.py)
held the weight-verified declared parent for 76 of 199 Hub models; every miss has a measured
cause (results/h7_misses.json). Each change below answers one cause; none uses the model's
own declaration.

1. Universe. The frame plus every model that some *other* frame model declares as its base
   (results/e2_bases_meta.jsonl.gz). Renamed repositories are matched by their current name.
2. Ranking. The pool is the union of the first POOL candidates by children (v1's order) and
   the first POOL by how much of the candidate's name (owner and quantisation words dropped)
   appears in the model's own name, ties broken by downloads; at most 2 x POOL are compared.
3. Quantised models (GGUF, packed 4/8-bit) report packed parameter counts: for them the size
   filter is dropped wherever dimensions decide, and is 0.1x-1.3x otherwise (GGUF files store
   tied embeddings twice). An unquantised model without dimensions keeps the 25% size window
   for a candidate whose name matches (>= 0.5), 2% otherwise.
4. A candidate without a parameter count is kept when dimensions decide. Counts leave out
   stored buffers (integer tensors), which are not weights.
5. Weak family keys (a library rather than an architecture, such as lib:gguf) and known aliases
   (timm_wrapper / timm) do not exclude a candidate whose name matches (>= 0.5); nor does any
   family key for a quantised model whose name matches closely (>= 0.8), since repackagers'
   configs often name another architecture (a Gemma 3 GGUF whose config says llama).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from . import pool

POOL = pool.POOL
QUANT_WORDS = {"gguf", "i1", "imatrix", "nvfp4", "mxfp4", "fp4", "fp8", "int4", "int8", "awq", "gptq", "bnb", "4bit",
               "8bit", "mlx", "exl2", "exl3", "dynamic", "w4a16", "w8a8", "w4a4", "nf4", "onnx", "ov", "openvino",
               "quantized", "quant", "quants", "static", "nvfp4a16", "fp8a16", "gs128", "q4", "q8", "q4km", "q5km",
               "q6k", "q8_0", "q4_k_m", "q5_k_m", "q6_k", "f16", "bf16", "fp16", "fp32", "f32", "safetensors"}
ALIASES = {"type:timmwrapper": "lib:timm"}


def words(repo: str) -> set[str]:
    name = repo.split("/", 1)[-1].lower()
    return {w for w in re.split(r"[-_ ]+", name) if w and w not in QUANT_WORDS}


def name_match(model: str, cand: str) -> float:
    """Share of the candidate's name words that appear in the model's name."""
    c = words(cand)
    return len(c & words(model)) / len(c) if c else 0.0


FLOAT_DTYPES = {"F64", "F32", "F16", "BF16", "F8_E4M3", "F8_E5M2"}


def float_params(m: dict) -> int | None:
    """The Hub's parameter count without stored buffers (uint8 attention masks, int64 position
    ids), which a model's weights do not include: pythia-160m counts 212.7M with them, 162.3M
    without. Integer tensors that are most of a checkpoint are packed weights (AWQ, GPTQ, FP4),
    not buffers, and stay counted."""
    by = m.get("params") or {}
    total = pool.meta_params(m)
    f = sum(v for k, v in by.items() if k in FLOAT_DTYPES) if isinstance(by, dict) else 0
    return f if f and total and f >= total / 2 else total


def weak(key: str | None) -> bool:
    return key is None or key.startswith("lib:")


class PoolV2:
    def __init__(self, frame: pool.Frame, bases_path: Path):
        self.frame = frame
        self.meta = dict(frame.by_id)
        self.dims = dict(frame.frozen_dims)
        self.canonical: dict[str, str] = {}
        with pool.open_text(bases_path) as f:
            for r in map(json.loads, f):
                if r.get("missing"):
                    continue
                if not r.get("model_type") and r.get("config_model_type"):
                    r["model_type"] = r["config_model_type"]
                if r.get("canonical") and r["canonical"] != r["id"]:
                    self.canonical[r["id"]] = r["canonical"]
                    if r["canonical"] in self.meta:      # a renamed frame model: already a candidate
                        continue
                cid = r.get("canonical") or r["id"]
                self.meta[cid] = r
                self.dims[cid] = r.get("dims")
        # who declares whom, by current names: a base enters a model's pool only through others
        self.declarers: dict[str, set[str]] = {}
        for m in frame.models:
            for b in m.get("declared_base") or []:
                self.declarers.setdefault(self.canonical.get(b, b), set()).add(m["id"])
        self.children = {k: len(v) for k, v in self.declarers.items()}
        self.ids = [i for i, m in self.meta.items() if not m.get("missing")]

    def name(self, repo: str) -> str:
        return self.canonical.get(repo, repo)

    def candidates(self, model: str, key: str | None, params: int | None, model_dims: dict | None,
                   quantised: bool, limit: int = POOL) -> list[str]:
        out = []
        for c in self.ids:
            if c == model:
                continue
            m = self.meta[c]
            if c not in self.frame.by_id and not (self.declarers.get(c, set()) - {model}):
                continue                                 # outside the frame and declared only by this model
            nm = name_match(model, c)
            ck = pool.family_key(m)
            k1, k2 = ALIASES.get(key, key), ALIASES.get(ck, ck)
            if k1 != k2 and not ((weak(k1) or weak(k2)) and nm >= 0.5) and not (quantised and nm >= 0.8):
                continue
            cd = self.dims.get(c)
            p = float_params(m)
            by_dims = bool(model_dims and cd)
            if by_dims and not pool.same_dims(model_dims, cd):
                continue
            if not by_dims:
                if not p or not params:
                    if nm < 0.5:
                        continue
                elif quantised:
                    if not 0.1 * p <= params <= 1.3 * p:
                        continue
                elif abs(p - params) > (pool.COARSE if nm >= 0.5 else pool.FINE) * params:
                    continue
            elif p and params and not quantised and abs(p - params) > pool.COARSE * params:
                continue
            out.append((nm, self.children.get(c, 0), self.frame.rank.get(c, 10 ** 7), c))
        # v1's order (children) and the name order each fill a pool; v2 is their union
        by_children = [c for *_, c in sorted(out, key=lambda x: (-x[1], x[2]))[:limit]]
        by_name = [c for *_, c in sorted(out, key=lambda x: (-x[0], x[2]))[:limit]]   # ties: most downloaded
        return by_children + [c for c in by_name if c not in set(by_children)]
