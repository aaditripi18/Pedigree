"""E2 candidate pool metadata (PLAN.md, amendment 2026-11-07): for every model in the
frozen frame (results/e2_frame.txt.gz), the Hub's listing metadata that the candidate
pool is built from: model_type, architectures, parameter count by dtype, creation date,
declared base models and their relation, library. No weights are read.

The listing is paged in the frame's own order (downloads). Frame ids the listing no
longer returns in the first 100k (rank drift since the snapshot, or deleted) are asked
for one by one; an id the Hub no longer has is kept with "missing": true.

Writes results/e2_frame_meta.jsonl.gz, one line per frame model, in frame order.

    python -m experiments.e2_meta
"""
from __future__ import annotations

import gzip
import json
import time
import urllib.error
import urllib.parse
from pathlib import Path

from pedigree.hub import _open

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "e2_frame_meta.jsonl.gz"
EXPAND = ["config", "safetensors", "createdAt", "cardData", "baseModels", "library_name", "pipeline_tag",
          "downloads", "gguf", "tags"]
LIST = ("https://huggingface.co/api/models?sort=downloads&direction=-1&limit=1000&"
        + "&".join(f"expand[]={e}" for e in EXPAND))


def frame_ids() -> list[str]:
    with gzip.open(ROOT / "results" / "e2_frame.txt.gz", "rt", encoding="utf-8") as f:
        return [line.split("\t")[1] for line in f if not line.startswith("#")]


def slim(m: dict) -> dict:
    cfg = m.get("config") or {}
    card = m.get("cardData") or {}
    base = card.get("base_model") or []
    if isinstance(base, str):
        base = [base]
    bm = m.get("baseModels") or {}
    st = m.get("safetensors") or {}
    gg = m.get("gguf") or {}
    return {"id": m["id"], "createdAt": m.get("createdAt"), "library": m.get("library_name"),
            "pipeline": m.get("pipeline_tag"), "model_type": cfg.get("model_type"),
            "architectures": cfg.get("architectures"),
            "special_tokens": sorted(f"{k}={v if isinstance(v, str) else json.dumps(v, sort_keys=True)}"
                                     for k, v in (cfg.get("tokenizer_config") or {}).items()
                                     if k.endswith("_token")),
            "params": st.get("parameters"), "params_total": st.get("total"),
            "gguf_total": gg.get("total"), "gguf_architecture": gg.get("architecture"),
            "declared_base": [str(b) for b in base], "base_relation": bm.get("relation"),
            "base_models": [x.get("id") for x in bm.get("models", []) if x.get("id")],
            "tags": [t for t in m.get("tags", []) if t.startswith(("base_model:", "license:"))]}


def one(repo: str) -> dict:
    url = (f"https://huggingface.co/api/models/{urllib.parse.quote(repo, safe='/')}?"
           + "&".join(f"expand[]={e}" for e in EXPAND))
    try:
        with _open(url) as r:
            return slim(json.loads(r.read().decode("utf-8")))
    except urllib.error.HTTPError as e:
        return {"id": repo, "missing": True, "http": e.code}


def main() -> None:
    ids = frame_ids()
    want = set(ids)
    got: dict[str, dict] = {}
    url, pages = LIST, 0
    while url and pages < 110:
        with _open(url) as r:
            page = json.loads(r.read().decode("utf-8"))
            link = r.headers.get("Link", "")
        for m in page:
            if m["id"] in want:
                got[m["id"]] = slim(m)
        pages += 1
        url = link.split(";")[0].strip("<> ") if 'rel="next"' in link else None
        print(f"page {pages}: {len(got)} of {len(ids)} frame models", flush=True)
    rest = [i for i in ids if i not in got]
    print(f"{len(rest)} frame models asked for one by one", flush=True)
    for k, repo in enumerate(rest):
        got[repo] = one(repo)
        if k % 200 == 0:
            print(f"  {k}/{len(rest)}", flush=True)
            time.sleep(1)
    tmp = OUT.with_suffix(".part")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        for i in ids:
            f.write(json.dumps(got[i], sort_keys=True) + "\n")
    tmp.replace(OUT)
    print(f"wrote {OUT.name}: {len(ids)} models, {sum(1 for i in ids if got[i].get('missing'))} missing")


if __name__ == "__main__":
    main()
