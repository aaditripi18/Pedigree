"""E2: the Hub audit (PLAN.md; protocol in the amendments of 2026-11-07).

For every model of the frozen sample (results/e2_sample.json): which weights it has, its
declared base(s) compared by weights, and its candidate pool (pedigree/pool.py) compared
by weights. Only scores are recorded here; categories and verdicts are computed by
scripts/analyse.py with the thresholds E1 dev fixes.

Runs in passes: finished models are kept in OUT (one JSON line each) and skipped; a model
lost to the network is tried again in the next pass. When all are done with none lost, the
records go to the ledger in sample order.

    python -m experiments.e2_audit [--limit N] [--ids repo,repo] [--out path]
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
import threading
import time
import urllib.error
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from experiments import ledger
from pedigree import hub, pool
from pedigree.verify import Remote, compare, compare_invariant, kurtosis_gap

ROOT = Path(__file__).resolve().parent.parent
INVARIANT_BELOW = 0.2
INVARIANT_MAX_ELEMENTS = 32 * 2 ** 20
WORKERS = int(os.environ.get("PEDIGREE_WORKERS", "8"))
BUDGET = float(os.environ.get("PEDIGREE_BUDGET_HOURS", "1e6")) * 3600


class NetworkError(Exception):
    pass


def _network(e: BaseException) -> bool:
    if isinstance(e, urllib.error.HTTPError):
        return e.code in (408, 429) or e.code >= 500
    return isinstance(e, (urllib.error.URLError, http.client.HTTPException, TimeoutError, ConnectionError)) or \
        (isinstance(e, OSError) and not isinstance(e, FileNotFoundError))


def _http_code(e: BaseException) -> int | None:
    if isinstance(e, urllib.error.HTTPError):
        return e.code
    if isinstance(e, hub.HubError):
        import re
        m = re.search(r"HTTP (\d{3})", str(e))
        return int(m.group(1)) if m else None
    return None


# --- one repository: metadata, weight files, an opened Remote -----------------------------

_info_cache: dict = {}
_remote_cache: dict = {}
_lock = threading.Lock()


def info(repo: str) -> dict:
    with _lock:
        if repo in _info_cache:
            return _info_cache[repo]
    path = hub.cache_root() / "info" / (repo.replace("/", "--") + ".json")
    if path.exists():
        d = json.loads(path.read_text(encoding="utf-8"))
    else:
        url = f"{hub.HOST}/api/models/{urllib.parse.quote(repo, safe='/')}/revision/main?blobs=true"
        try:
            with hub._open(url) as r:
                d = json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if _network(e):
                raise NetworkError(f"{repo}: HTTP {e.code}") from None
            d = {"_http": e.code}
        except Exception as e:
            if _network(e):
                raise NetworkError(f"{repo}: {type(e).__name__}: {e}") from None
            raise
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(d), encoding="utf-8")
    with _lock:
        _info_cache[repo] = d
    return d


def weight_files(repo: str, d: dict) -> tuple[str, list[str], dict, dict]:
    """(format, files, sizes, sha256) of a repository; format names why there are none."""
    sib = {s["rfilename"]: s for s in d.get("siblings", [])}
    names = [n for n in sib if not n.split("/")[-1].startswith("adapter_model")]
    prefix = ""
    if "model_index.json" in sib:
        for sub in ("unet", "transformer"):
            if any(n.startswith(sub + "/") for n in names):
                prefix = sub + "/"
                break
    inner = [n[len(prefix):] for n in names if n.startswith(prefix)]
    if prefix:   # diffusers names its weights diffusion_pytorch_model.*; read them as model.*
        inner = [n.replace("diffusion_pytorch_model", "model", 1) for n in inner]

    def real(n):
        return prefix + (n.replace("model", "diffusion_pytorch_model", 1) if prefix and n.startswith("model") else n)

    def read_index(idx):
        with hub._open(f"{hub.HOST}/{repo}/resolve/{d['sha']}/{urllib.parse.quote(real(idx))}") as r:
            j = json.loads(r.read().decode("utf-8"))
        if prefix:
            j["weight_map"] = {k: v.replace("diffusion_pytorch_model", "model", 1)
                               for k, v in j.get("weight_map", {}).items()}
        return j
    try:
        files = [real(f) for f in hub.choose_files(inner, read_index)]
    except hub.HubError:
        files = []
    files = [f for f in files if f.endswith((".safetensors", ".bin", ".gguf"))]
    if not files:
        adapters = [n for n in sib if n.split("/")[-1].startswith("adapter_model")]
        top = {n.rsplit(".", 1)[-1] for n in sib}
        if adapters:
            fmt = "adapter"
        elif top & {"onnx"}:
            fmt = "not readable (onnx only)"
        elif top & {"h5", "msgpack"}:
            fmt = "not readable (tf/flax only)"
        elif top & {"pt", "pth", "ckpt", "pkl", "npz", "nemo"}:
            fmt = "not readable (other checkpoint format)"
        else:
            fmt = "not readable (no weight files)"
        return fmt, [], {}, {}
    sizes = {f: int(sib[f].get("size") or (sib[f].get("lfs") or {}).get("size") or 0) for f in files}
    sha = {f: (sib[f].get("lfs") or {}).get("sha256", "") for f in files}
    return ("gguf" if files[0].endswith(".gguf") else "bin" if files[0].endswith(".bin") else "safetensors"),\
        files, sizes, sha


def _search(term: str) -> list[str]:
    url = f"{hub.HOST}/api/models?search={urllib.parse.quote(term)}&sort=downloads&direction=-1&limit=40"
    try:
        with hub._open(url) as r:
            return [m["id"] for m in json.loads(r.read().decode("utf-8"))]
    except Exception as e:
        if _network(e):
            raise NetworkError(str(e)) from None
        return []


def mirror(repo: str, d: dict, frame: pool.Frame) -> str | None:
    """A public repository whose weight files are byte-identical (same SHA-256) to a gated
    repository's."""
    fmt, files, _, sha = weight_files(repo, d)
    want = sorted(v for v in sha.values() if v)
    if not want or len(want) != len(files):
        return None
    name = repo.split("/")[-1]
    tried = set()
    for c in _search(name) + _search(name.replace("Meta-", "")):
        if c == repo or c in tried:
            continue
        tried.add(c)
        dc = info(c)
        if dc.get("_http") or dc.get("gated"):
            continue
        _, cf, _, csha = weight_files(c, dc)
        if sorted(v for v in csha.values() if v) == want:
            return c
    return None


def open_repo(repo: str, frame: pool.Frame) -> tuple[Remote | None, dict]:
    """(Remote or None, facts: format, commit, via, reason)."""
    with _lock:
        if repo in _remote_cache:
            return _remote_cache[repo]
    facts: dict = {}
    d = info(repo)
    if d.get("_http"):
        facts["reason"] = f"not readable (HTTP {d['_http']})"
        out = (None, facts)
    else:
        facts["commit"] = d.get("sha")
        fmt, files, sizes, _ = weight_files(repo, d)
        facts["format"] = fmt
        src, src_d = repo, d
        if files and d.get("gated"):
            m = mirror(repo, d, frame)
            if m is None:
                facts["reason"] = "not readable (gated, no byte-identical public copy)"
                files = []
            else:
                facts["via"] = m
                src, src_d = m, info(m)
                _, files, sizes, _ = weight_files(m, src_d)
        if not files:
            facts.setdefault("reason", fmt)
            out = (None, facts)
        else:
            try:
                r = Remote.from_files(src, src_d["sha"], files, sizes)
                r.repo = repo if src == repo else f"{repo} (via {src})"
                out = (r, facts)
            except Exception as e:
                if _network(e) and _http_code(e) not in (401, 403, 404):
                    raise NetworkError(f"{repo}: {type(e).__name__}: {e}") from None
                code = _http_code(e)
                facts["reason"] = (f"not readable (HTTP {code})" if code else
                                   f"not readable ({type(e).__name__}: {str(e)[:120]})")
                out = (None, facts)
    with _lock:
        _remote_cache[repo] = out
    return out


def logical_params(r: Remote) -> int:
    return int(sum(np.prod(s[1]) for s in r.float_shapes().values()))


def _pair(model: Remote, other: Remote, invariant: bool) -> dict:
    try:
        c = compare(model, other)
        out = {"median_cosine": c["median_cosine"], "compared": c["compared"], "matched": c["matched"],
               "match_share": round(c["match_share"], 4), "identical": c["identical"],
               "kurtosis_gap": kurtosis_gap(c["kurtosis"]), "cosines": c["cosines"]}
        if invariant and c["compared"] and (c["median_cosine"] is None or c["median_cosine"] < INVARIANT_BELOW):
            inv = compare_invariant(model, other, max_elements=INVARIANT_MAX_ELEMENTS)
            out.update(median_invariant=inv["median_invariant"], invariant_compared=inv["compared"])
        return out
    except Exception as e:
        if _network(e):
            raise NetworkError(f"{model.repo} vs {other.repo}: {type(e).__name__}: {e}") from None
        return {"error": f"{type(e).__name__}: {str(e)[:200]}"}


def audit_one(entry: dict, tier: str, frame: pool.Frame) -> dict:
    repo = entry["id"]
    meta = frame.by_id.get(repo, {})
    rec = {"id": repo, "tier": tier, "rank": entry["rank"], "downloads": entry["downloads"],
           "created": meta.get("createdAt"), "declared_base": meta.get("declared_base") or [],
           "base_relation": meta.get("base_relation"), "family_key": pool.family_key(meta)}
    model, facts = open_repo(repo, frame)
    rec.update(facts)
    if model is None:
        return rec
    quantised = bool(model.quant) or facts.get("format") == "gguf"
    params = (None if model.quant else pool.meta_params(meta)) or logical_params(model)
    mdims = pool.dims(pool.config(facts.get("via") or repo))
    rec.update(params=params, dims=mdims, quantised=quantised)
    rec["declared"] = []
    for b in rec["declared_base"]:
        other, bf = open_repo(b, frame)
        row = {"base": b, **{k: v for k, v in bf.items() if k in ("via", "reason", "commit", "format")},
               "created": frame.by_id.get(b, {}).get("createdAt") or info(b).get("createdAt")}
        if other is not None:
            row.update(_pair(model, other, invariant=True))
        rec["declared"].append(row)
    cands = frame.candidates(rec["family_key"], params, mdims, exclude={repo})
    rec["pool"] = []
    for c in cands:
        other, cf = open_repo(c, frame)
        row = {"candidate": c, "created": frame.by_id[c].get("createdAt"), "children": frame.children[c],
               **{k: v for k, v in cf.items() if k in ("via", "reason")}}
        if other is not None:
            row.update(_pair(model, other, invariant=False))
            row.pop("cosines", None)
        rec["pool"].append(row)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--ids")
    ap.add_argument("--out", default=str(ROOT / "results" / "e2_audit_raw.jsonl"))
    ap.add_argument("--no-ledger", action="store_true")
    a = ap.parse_args()
    frame = pool.Frame(ROOT / "results" / "e2_frame_meta.jsonl.gz")
    sample = json.loads((ROOT / "results" / "e2_sample.json").read_text(encoding="utf-8"))
    todo = [(e, t) for t, rows in sample["sample"].items() for e in rows]
    if a.ids:
        want = a.ids.split(",")
        todo = [({"id": i, "rank": frame.rank.get(i, 0), "downloads": frame.by_id.get(i, {}).get("downloads")},
                 "smoke") for i in want]
    out = Path(a.out)
    done = {}
    sources = [out] if out.exists() else []
    prev = os.environ.get("PEDIGREE_E2_PREVIOUS")   # earlier passes' outputs (Kaggle inputs)
    if prev:
        sources += [p for p in Path(prev).rglob(out.name) if "Pedigree" not in p.parts and p != out]
    for src in sources:
        for line in src.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done[r["id"]] = r
    if prev and done:
        out.write_text("".join(json.dumps(r) + "
" for r in done.values()), encoding="utf-8")
    pending = [(e, t) for e, t in todo if e["id"] not in done][: a.limit]
    print(f"{len(done)} done, {len(pending)} to audit", flush=True)
    t0, lost, wlock = time.time(), [], threading.Lock()

    def work(e, t):
        if time.time() - t0 > BUDGET:
            return e["id"], None, "budget"
        try:
            return e["id"], audit_one(e, t, frame), None
        except NetworkError as err:
            return e["id"], None, str(err)

    with ThreadPoolExecutor(WORKERS) as ex:
        futs = [ex.submit(work, e, t) for e, t in pending]
        for k, f in enumerate(as_completed(futs), 1):
            repo, rec, err = f.result()
            if rec is None:
                lost.append((repo, err))
                print(f"[{k}/{len(pending)}] {repo}: LOST ({err})", flush=True)
                continue
            with wlock, open(out, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
            done[repo] = rec
            dec = ", ".join(f"{x['base']}={x.get('median_cosine', x.get('reason'))}" for x in rec.get("declared", []))
            best = max((p.get("median_cosine") or -1 for p in rec.get("pool", [])), default=None)
            print(f"[{k}/{len(pending)}] {repo}: {rec.get('reason') or rec.get('format')}; declared {dec or '-'}; "
                  f"pool {len(rec.get('pool', []))} best {best}  ({time.time() - t0:.0f}s)", flush=True)
    print(f"{len(done)} of {len(todo)} done; {len(lost)} lost to the network or the time budget", flush=True)
    if not a.no_ledger and not a.ids and len(done) == len(todo) and not lost:
        run = ledger.Run("E2", sample_seed=sample["seed"], snapshot=sample["snapshot"])
        for e, t in todo:
            run.record(**done[e["id"]])
        print("recorded in the ledger", flush=True)


if __name__ == "__main__":
    main()
