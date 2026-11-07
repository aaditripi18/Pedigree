"""Candidate parents of a model, without using its own declaration (PLAN.md, amendment
2026-11-07 (2)).

From the frame's metadata (results/e2_frame_meta.jsonl.gz): models of the same family key
(`model_type`; else GGUF architecture; else library), parameter count within 25%, and the
same dimensions in `config.json` (hidden size, layers, feed-forward size; timm's
`architecture`). Without dimensions on either side, parameter count within 2% instead.
Ranked by how many frame models declare the candidate as their base, then by downloads;
the first POOL are kept.
"""
from __future__ import annotations

import collections
import gzip
import json
import threading
from pathlib import Path

from . import hub

POOL = 30
COARSE, FINE = 0.25, 0.02
DIMS = {"hidden": ("hidden_size", "d_model", "n_embd", "dim", "embed_dim", "hidden_dim"),
        "layers": ("num_hidden_layers", "n_layer", "num_layers", "n_layers", "depth", "encoder_layers"),
        "ffn": ("intermediate_size", "n_inner", "ffn_dim", "encoder_ffn_dim", "mlp_dim")}


def dims(cfg: dict) -> dict | None:
    """The dimensions that must agree for two models' weights to align, or None."""
    if not isinstance(cfg, dict) or not cfg:
        return None
    if isinstance(cfg.get("architecture"), str) and "model_type" not in cfg:     # timm
        return {"architecture": cfg["architecture"]}
    out = {}
    for k, names in DIMS.items():
        for n in names:
            v = cfg.get(n)
            if isinstance(v, int) and not isinstance(v, bool):
                out[k] = v
                break
    return out if "hidden" in out and "layers" in out else None


def same_dims(a: dict, b: dict) -> bool:
    shared = set(a) & set(b)
    return bool(shared) and ("architecture" in shared or {"hidden", "layers"} <= shared) and \
        all(a[k] == b[k] for k in shared)


_config_lock = threading.Lock()


def config(repo: str) -> dict:
    """A repository's config.json (main branch), cached on disk; {} if it has none."""
    path = hub.cache_root() / "configs" / (repo.replace("/", "--") + ".json")
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    try:
        with hub._open(f"{hub.HOST}/{repo}/resolve/main/config.json") as r:
            cfg = json.loads(r.read().decode("utf-8"))
        if not isinstance(cfg, dict):
            cfg = {}
    except Exception as e:  # 401/403/404 or not JSON: no readable dimensions
        if not _definitive(e):
            raise
        cfg = {}
    with _config_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cfg), encoding="utf-8")
    return cfg


def _definitive(e: Exception) -> bool:
    import urllib.error
    return (isinstance(e, urllib.error.HTTPError) and e.code in (401, 403, 404)) or isinstance(e, ValueError)


def family_key(m: dict) -> str | None:
    if m.get("model_type"):
        return "type:" + m["model_type"]
    if m.get("gguf_architecture"):
        return "type:" + m["gguf_architecture"]
    if m.get("library"):
        return "lib:" + m["library"]
    return None


def meta_params(m: dict) -> int | None:
    return m.get("params_total") or m.get("gguf_total") or None


class Frame:
    def __init__(self, path: Path):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            self.models = [json.loads(line) for line in f]
        self.by_id = {m["id"]: m for m in self.models}
        self.rank = {m["id"]: i + 1 for i, m in enumerate(self.models)}
        self.children = collections.Counter(b for m in self.models for b in set(m.get("declared_base") or []))
        self._by_key = collections.defaultdict(list)
        for m in self.models:
            if not m.get("missing") and family_key(m) and meta_params(m):
                self._by_key[family_key(m)].append(m["id"])
        for ids in self._by_key.values():
            ids.sort(key=lambda i: (-self.children[i], self.rank[i]))

    def candidates(self, key: str | None, params: int | None, model_dims: dict | None, exclude=(),
                   limit: int = POOL) -> list[str]:
        """The pool for a model with this family key, parameter count and dimensions."""
        if not key or not params:
            return []
        out = []
        for c in self._by_key.get(key, []):
            if c in exclude:
                continue
            p = meta_params(self.by_id[c])
            if abs(p - params) > COARSE * params:
                continue
            cd = dims(config(c)) if model_dims else None
            if model_dims and cd:
                ok = same_dims(model_dims, cd)
            else:
                ok = abs(p - params) <= FINE * params
            if ok:
                out.append(c)
                if len(out) == limit:
                    break
        return out
