"""Pool v2 on its design data (PLAN.md, amendment 2026-11-13): E2's 199 Hub models whose declared
base is related by weights, and H7 (a)'s 24 fresh E1 derivatives. Both were seen while v2 was
designed, so these numbers are dev numbers; H13 is judged on the fresh sample only.

Writes results/h7_misses.json: why E2's pool missed each parent, and whether v2 holds it.

    python scripts/pool_v2_dev.py [bases_meta_path]
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pedigree import pool  # noqa: E402
from pedigree.pool_v2 import PoolV2, float_params  # noqa: E402
from scripts import analyse as A  # noqa: E402


def v1_cause(F: pool.Frame, m: dict, b: str) -> str:
    c = F.by_id.get(b)
    if c is None:
        return "base outside the frame"
    if pool.family_key(c) != m["family_key"]:
        return "different family key"
    if not pool.meta_params(c) or not m.get("params"):
        return "no parameter count"
    if abs(pool.meta_params(c) - m["params"]) > pool.COARSE * m["params"]:
        return "parameter count differs by > 25%"
    cd = F.frozen_dims.get(b)
    if m.get("dims") and cd and not pool.same_dims(m["dims"], cd):
        return "config dimensions differ"
    if not (m.get("dims") and cd) and abs(pool.meta_params(c) - m["params"]) > pool.FINE * m["params"]:
        return "no dimensions; parameter count differs by > 2%"
    return "ranked below the top 30"


def main(bases: str) -> None:
    rows = A.load()
    T = A.thresholds(A.latest_runs(rows, "E1", lambda r: r.get("family")))
    e2 = A.latest_runs(rows, "E2", lambda r: 0)
    h7a = [r for r in A.latest_runs(rows, "H7a", lambda r: r.get("family")) if not r.get("error")]
    F = pool.Frame(ROOT / "results" / "e2_frame_meta.jsonl.gz")
    P = PoolV2(F, Path(bases))
    cache = ROOT / "results" / "h7_dev_searches.json"      # one Hub search per dev model, fetched once
    searches = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
    need = sorted({m["id"] for m in e2 if m.get("declared")} - set(searches))
    if need:
        from experiments.e2_audit import _search
        from pedigree.pool_v2 import search_term
        for i in need:
            searches[i] = {"term": search_term(i), "hits": _search(search_term(i))}
        cache.write_text(json.dumps(searches, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    hub, tally = [], collections.Counter()
    for m in e2:
        for d in m.get("declared", []):
            if A.related(d, T, "t") != "related" or d.get("identical"):
                continue
            b = d["base"]
            v1 = b in {p["candidate"] for p in m.get("pool", [])}
            params = m.get("params") if m.get("quantised") else (float_params(F.by_id.get(m["id"], {})) or m.get("params"))
            cands = P.candidates(m["id"], m["family_key"], params, m.get("dims"), bool(m.get("quantised")),
                                 searched=searches[m["id"]]["hits"])
            v2 = P.name(b) in cands
            cause = "in pool" if v1 else v1_cause(F, m, b)
            tally[(cause, v2)] += 1
            hub.append({"model": m["id"], "base": b, "v1": v1, "v1_cause": cause, "v2": v2, "v2_size": len(cands),
                        "v1_size": len(m.get("pool", [])),
                        "v2_rank": cands.index(P.name(b)) + 1 if v2 else None})
    e1 = []
    for r in h7a:
        cands = P.candidates("", r["family_key"], r["params"], r["dims"], False)
        e1.append({"family": r["family"], "kind": r["kind"], "base": r["base"], "v1": r["parent_in_pool"],
                   "v2": r["base"] in cands})
    out = {"bases_meta": Path(bases).name, "hub": {"n": len(hub), "v1": sum(h["v1"] for h in hub),
                                                  "v2": sum(h["v2"] for h in hub),
                                                  "mean_pool_v1": sum(h["v1_size"] for h in hub) / len(hub),
                                                  "mean_pool_v2": sum(h["v2_size"] for h in hub) / len(hub)},
           "e1": {"n": len(e1), "v1": sum(e["v1"] for e in e1), "v2": sum(e["v2"] for e in e1)},
           "by_cause": {f"{c} -> v2 {'hit' if v else 'miss'}": n for (c, v), n in sorted(tally.items())},
           "hub_pairs": hub, "e1_pairs": e1}
    (ROOT / "results" / "h7_misses.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("hub", "e1", "by_cause")}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "results" / "e2_bases_meta.jsonl.gz"))
