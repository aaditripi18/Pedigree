"""Every Pedigree verdict, computed from results/ledger.jsonl under the rules in PLAN.md
(amendments of 2026-11-07 fix the thresholds and H6-H9). Nothing here is tuned: the
thresholds come from E1 dev negatives only, by the rule written before any verdict.

Writes results/verdicts.json and docs/RESULTS.md.

    python scripts/analyse.py
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
LEDGER = ROOT / "results" / "ledger.jsonl"
FAMILIES = ["text_encoder", "text_decoder", "vision_transformer", "vision_cnn", "speech", "diffusion_unet"]
BLIND = {"speech", "diffusion_unet"}       # compared after the threshold rule was written
TRAINED = ("ft_short", "ft_long", "lora_r8", "lora_r64")
SCORES = {"plain": "median_cosine", "invariant": "median_invariant", "aligned": "median_aligned_invariant"}
Z = 5
MIN_COMPARED = 8
TIER_SIZE = {"top_1k": 1_000, "1k_10k": 9_000, "10k_100k": 90_000}


# --- ledger ----------------------------------------------------------------------------------

def load() -> list[dict]:
    invalid = set(json.loads((ROOT / "results" / "ledger_invalid.json").read_text(encoding="utf-8")))
    rows = [json.loads(line) for line in LEDGER.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [r for r in rows if r.get("run_id") not in invalid]


def latest_runs(rows: list[dict], experiment: str, key) -> list[dict]:
    """The rows of the latest run of `experiment` for each value of key(row)."""
    best: dict = {}
    for r in rows:
        if r.get("experiment") != experiment:
            continue
        k = key(r)
        if k not in best or r["run_id"] > best[k]:
            best[k] = r["run_id"]
    keep = set(best.values())
    return [r for r in rows if r.get("experiment") == experiment and r["run_id"] in keep]


def num(x) -> float | None:
    return x if isinstance(x, (int, float)) and x == x else None


def wilson(k: int, n: int, z: float = 1.96) -> list[float] | None:
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def auc(pos: list[float], neg: list[float]) -> float | None:
    if not pos or not neg:
        return None
    wins = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    return wins / (len(pos) * len(neg))


def verdict(ok: bool | None) -> str:
    return "pending" if ok is None else ("supported" if ok else "not supported")


# --- thresholds (amendment 2026-11-07) ---------------------------------------------------------

def thresholds(e1: list[dict]) -> dict:
    out = {}
    for name, field in SCORES.items():
        neg = [num(r.get(field)) for r in e1 if r["split"] == "dev" and r.get("label") == "negative"
               and r.get("kind") in ("scratch", "independent") and not r.get("error")]
        neg = [x for x in neg if x is not None]
        pos = [num(r.get(field)) for r in e1 if r["split"] == "dev" and r.get("kind") in TRAINED and not r.get("error")]
        pos = [x for x in pos if x is not None]
        if len(neg) < 2:
            continue
        t = max(neg) + Z * float(np.std(neg, ddof=1))
        out[name] = {"t": t, "midpoint": (max(neg) + min(pos)) / 2 if pos else None, "n_negatives": len(neg),
                     "max_negative": max(neg), "sd_negatives": float(np.std(neg, ddof=1)),
                     "min_trained_positive": min(pos) if pos else None}
    return out


def related(r: dict, T: dict, rule: str = "t", plain_only: bool = False) -> str:
    """related / unrelated / not comparable, by the weight verifier's verdict rule."""
    scores = {k: num(r.get(f)) for k, f in SCORES.items() if k in T and (not plain_only or k == "plain")}
    scores = {k: v for k, v in scores.items() if v is not None}
    if any(v > T[k][rule] for k, v in scores.items()):
        return "related"
    if scores and (r.get("compared") or 0) >= MIN_COMPARED:
        return "unrelated"
    return "not comparable"


# --- hypotheses -----------------------------------------------------------------------------------

def h1(e1, T, rule):
    fresh = [r for r in e1 if r["split"] == "fresh" and not r.get("error")]
    pos = [r for r in fresh if r.get("kind") in TRAINED]
    neg = [r for r in fresh if r.get("kind") in ("scratch", "independent")]
    tp = sum(related(r, T, rule) == "related" for r in pos)
    fp = sum(related(r, T, rule) == "related" for r in neg)
    a = auc([r["median_cosine"] for r in pos if num(r.get("median_cosine")) is not None],
            [r["median_cosine"] for r in neg if num(r.get("median_cosine")) is not None])
    fams = sorted({r["family"] for r in pos})
    done = set(fams) >= set(FAMILIES)
    ok = (tp / len(pos) >= 0.95 and fp / len(neg) <= 0.01) if pos and neg and done else None
    return {"id": "H1", "statement": "The verifier separates derivatives from independent same-architecture models "
            "on fresh ground truth with TPR >= 95% at FPR <= 1%.", "verdict": verdict(ok),
            "numbers": {"tpr": [tp, len(pos), wilson(tp, len(pos))], "fpr": [fp, len(neg), wilson(fp, len(neg))],
                        "auc_plain": a, "families": fams,
                        # every fresh pair's plain score, for the site's figure
                        "points": {lab: [[r["family"], r["kind"], round(r["median_cosine"], 5)] for r in rs
                                         if num(r.get("median_cosine")) is not None]
                                   for lab, rs in (("derivative", pos), ("independent", neg))}},
            "evidence": f"TPR {tp}/{len(pos)}, FPR {fp}/{len(neg)}, AUC (plain) {a:.3f}" if a is not None else "pending"}


def h4(e1, T, rule):
    rows = [r for r in e1 if r["split"] == "fresh" and r.get("kind") == "permuted_full" and not r.get("error")]
    hidden = sum(num(r.get("median_cosine")) is not None and r["median_cosine"] <= T["plain"][rule] for r in rows)
    restored = sum(num(r.get("median_aligned_invariant")) is not None
                   and r["median_aligned_invariant"] > T["aligned"][rule] for r in rows)
    n = len(rows)
    ok = (hidden / n >= 0.9 and restored / n >= 0.9) if n >= 3 else None
    return {"id": "H4", "statement": "A function-preserving permutation (attack b) hides lineage from the plain "
            "verifier on >= 90% of derivatives, and defence (ii) restores it on >= 90%.", "verdict": verdict(ok),
            "numbers": {"hidden": [hidden, n], "restored": [restored, n],
                        "rows": {r["family"]: [r.get("median_cosine"), r.get("median_aligned_invariant")] for r in rows}},
            "evidence": f"plain hidden on {hidden}/{n}; aligned restores {restored}/{n}"}


def h5(e1, e5, T, rule):
    q4 = [r for r in e1 if r["split"] == "fresh" and r.get("kind") == "q4_0" and not r.get("error")]
    q4_ok = sum(num(r.get("median_cosine")) is not None and r["median_cosine"] > T["plain"][rule] for r in q4)
    st = [r for r in e5 if r.get("split") == "fresh" and r.get("kind") == "distill" and not r.get("error")]
    st_rel = sum(num(r.get("weight_median_cosine")) is not None and (r.get("weight_compared") or 0) >= MIN_COMPARED
                 and r["weight_median_cosine"] > T["plain"][rule] for r in st)
    done = len(q4) >= len(FAMILIES) and len({r.get("family") for r in st}) >= 4
    ok = (q4_ok / len(q4) >= 0.9 and st_rel / len(st) <= 0.1) if done and q4 and st else None
    return {"id": "H5", "statement": "4-bit quantised derivatives are still verified (>= 90%); students distilled "
            "from a random initialisation are not (<= 10%).", "verdict": verdict(ok),
            "numbers": {"q4": [q4_ok, len(q4)], "students_related": [st_rel, len(st)]},
            "evidence": f"q4_0 verified {q4_ok}/{len(q4)}; distilled students related by weights {st_rel}/{len(st)}"}


def h6(e1b, e2w, T, rule):
    rows = [r for r in e1b if r["split"] == "fresh" and num(r.get("median_whole")) is not None
            and num(r.get("median_cosine")) is not None]
    t = T["plain"][rule]
    agree = sum((r["median_cosine"] > t) == (r["median_whole"] > t) for r in rows)
    hub = [r for r in e2w if num(r.get("median_whole")) is not None and num(r.get("median_cosine")) is not None]
    agree_hub = sum((r["median_cosine"] > t) == (r["median_whole"] > t) for r in hub)
    n = len(rows) + len(hub)
    done = len({r["family"] for r in rows}) >= len(FAMILIES) and len(hub) >= 40
    ok = ((agree + agree_hub) / n >= 0.98) if done else None
    return {"id": "H6", "statement": "The sampled verdict equals the whole-file verdict for >= 98% of pairs.",
            "verdict": verdict(ok),
            "numbers": {"e1": [agree, len(rows)], "e2": [agree_hub, len(hub)],
                        "max_abs_difference": max([abs(r["median_cosine"] - r["median_whole"]) for r in rows + hub],
                                                  default=None)},
            "evidence": f"E1 fresh pairs {agree}/{len(rows)}, Hub pairs {agree_hub}/{len(hub)} agree"}


def h7(h7a, e2, T, rule):
    a_hit = sum(bool(r.get("parent_in_pool")) for r in h7a if not r.get("error"))
    a_n = sum(not r.get("error") for r in h7a)
    b = []
    for m in e2:
        for d in m.get("declared", []):
            if related(d, T, rule) == "related" and not d.get("identical"):
                b.append(d["base"] in {p["candidate"] for p in m.get("pool", [])})
    done_a = len({r["family"] for r in h7a if not r.get("error")}) >= len(FAMILIES)
    ok = (a_hit / a_n >= 0.9 and sum(b) / len(b) >= 0.9) if done_a and a_n and b else None
    return {"id": "H7", "statement": "The candidate pool contains the true parent for >= 90% of ground-truth "
            "derivatives (a) and of Hub models whose declared base is verified (b).", "verdict": verdict(ok),
            "numbers": {"a": [a_hit, a_n, wilson(a_hit, a_n)], "b": [sum(b), len(b), wilson(sum(b), len(b))]},
            "evidence": f"(a) {a_hit}/{a_n}; (b) {sum(b)}/{len(b)}"}


def d2_margin(e1b) -> float | None:
    dev = [r["kurtosis_gap"] for r in e1b if r["split"] == "dev" and r.get("kind") in TRAINED
           and num(r.get("kurtosis_gap")) is not None]
    if not dev:
        return None
    wrong = [abs(g) for g in dev if g < 0]
    return max(wrong) if wrong else 0.0


def h8(e1b, e7k, e2, T, rule):
    m = d2_margin(e1b)
    pairs = [(r["kurtosis_gap"], "e1") for r in e1b if r["split"] == "fresh" and r.get("kind") in TRAINED
             and num(r.get("kurtosis_gap")) is not None]
    pairs += [(-r["kurtosis_gap"], "e7") for r in e7k if num(r.get("kurtosis_gap")) is not None]
    decided = [g for g, _ in pairs if m is not None and abs(g) > m]
    right = sum(g > 0 for g in decided)
    d1 = []
    for mo in e2:
        for d in mo.get("declared", []):
            if related(d, T, rule) == "related" and not d.get("identical") and d.get("created") and mo.get("created") \
                    and d["created"] != mo["created"]:
                d1.append(d["created"] < mo["created"])
    ok2 = (right / len(decided) >= 0.9) if decided else (False if pairs and m is not None else None)
    ok1 = (sum(d1) / len(d1) >= 0.9) if d1 else None
    ok = None if ok1 is None or ok2 is None or len(e7k) < 35 else (ok1 and ok2)
    return {"id": "H8", "statement": "Direction is decided correctly for >= 90% of the pairs where it is decided "
            "(D1 Hub dates; D2 weight kurtosis), with the undecided share reported.", "verdict": verdict(ok),
            "numbers": {"d2_margin": m, "d2": [right, len(decided), len(pairs)], "d1": [sum(d1), len(d1)]},
            "evidence": f"D1 (dates) right on {sum(d1)}/{len(d1)}; D2 (kurtosis, margin {m:.2g}) right on {right}/"
                        f"{len(decided)} decided, undecided {len(pairs) - len(decided)}/{len(pairs)}"}


def h9(e5):
    stats = {"agreement": lambda r: r.get("agreement"), "1-js": lambda r: 1 - r["js"] if num(r.get("js")) is not None
             else None, "cka": lambda r: r.get("cka")}

    def split(s):
        return ([r for r in e5 if r.get("split") == s and r.get("kind") == "distill" and not r.get("error")],
                [r for r in e5 if r.get("split") == s and r.get("kind") == "indep" and not r.get("error")])
    dp, dn = split("dev")
    fp_, fn = split("fresh")
    if len({r.get("family") for r in fp_}) < 4 or not dp or not dn:
        return {"id": "H9", "statement": "Behaviour separates distilled students from same-task independent "
                "students with TPR >= 90% at FPR <= 5% (fresh).", "verdict": "pending", "evidence": "needs E5"}
    gaps = {}
    for name, f in stats.items():
        p = [f(r) for r in dp if num(f(r)) is not None]
        q = [f(r) for r in dn if num(f(r)) is not None]
        if p and q:
            gaps[name] = (min(p) - max(q), (min(p) + max(q)) / 2)
    best = max(gaps, key=lambda k: gaps[k][0])
    if gaps[best][0] > 0:
        stat, t = best, gaps[best][1]
    else:                                     # not separable on dev: agreement at dev FPR <= 5%
        stat = "agreement"
        q = sorted(stats[stat](r) for r in dn)
        k = int(math.floor(0.05 * len(q)))
        t = q[-1 - k] if q else None
    f = stats[stat]
    tp = sum(f(r) > t for r in fp_)
    fp = sum(f(r) > t for r in fn)
    ok = tp / len(fp_) >= 0.9 and fp / len(fn) <= 0.05
    return {"id": "H9", "statement": "Behaviour separates distilled students from same-task independent students "
            "with TPR >= 90% at FPR <= 5% (fresh).", "verdict": verdict(ok),
            "numbers": {"statistic": stat, "threshold": t, "tpr": [tp, len(fp_)], "fpr": [fp, len(fn)],
                        "dev_gaps": {k: v[0] for k, v in gaps.items()}},
            "evidence": f"{stat} > {t:.4f} (chosen on dev): TPR {tp}/{len(fp_)}, FPR {fp}/{len(fn)}"}


H12_STATS = {"agreement": lambda r: r.get("agreement"),
             "1-js": lambda r: 1 - r["js"] if num(r.get("js")) is not None else None,
             "cka": lambda r: r.get("cka")}


def _seed(r) -> int:
    return int(str(r.get("model", "")).rsplit("_s", 1)[-1]) if "_s" in str(r.get("model", "")) else -1


def h12_scores(e5o, stat: str, split: str):
    """Margins over the family's reference independents (seeds 0-2) on one split:
    (distilled, independent) lists of (family, model, margin). A reference independent is
    scored against the other two."""
    f = H12_STATS[stat]
    P, N = [], []
    for fam in sorted({r["family"] for r in e5o if r.get("split") == split}):
        rows = [r for r in e5o if r.get("split") == split and r["family"] == fam and not r.get("error")
                and r.get("kind") in ("distill", "indep") and num(f(r)) is not None]
        refs = {r["model"]: f(r) for r in rows if r["kind"] == "indep" and _seed(r) <= 2}
        for r in rows:
            others = [v for m, v in refs.items() if m != r["model"]]
            if len(others) < 2:
                continue
            (P if r["kind"] == "distill" else N).append((fam, r["model"], f(r) - max(others)))
    return P, N


def h12(e5o):
    statement = ("On off-task inputs, a distilled student agrees with its teacher more than the family's "
                 "same-task independent students do (TPR >= 90% at FPR <= 5%, fresh).")
    if not e5o or not any(_seed(r) >= 3 for r in e5o if r.get("split") == "fresh"):
        return {"id": "H12", "statement": statement, "verdict": "pending", "evidence": "needs the new students"}
    dev = {}
    for stat in H12_STATS:
        P, N = h12_scores(e5o, stat, "dev")
        p, n = [m for *_, m in P], [m for *_, m in N]
        dev[stat] = {"gap": min(p) - max(n), "mid": (min(p) + max(n)) / 2, "max_neg": max(n),
                     "tpr_at_fpr0": sum(x > max(n) for x in p) / len(p)}
    best = max(dev, key=lambda s: dev[s]["gap"])
    if dev[best]["gap"] > 0:
        stat, t = best, dev[best]["mid"]
    else:   # none separates dev: highest dev TPR at FPR 0, widest gap breaking ties
        stat = max(dev, key=lambda s: (dev[s]["tpr_at_fpr0"], dev[s]["gap"]))
        t = dev[stat]["max_neg"]
    P, N = h12_scores(e5o, stat, "fresh")
    P = [x for x in P if _seed({"model": x[1]}) >= 3]
    N = [x for x in N if _seed({"model": x[1]}) >= 3]
    tp, fp = sum(m > t for *_, m in P), sum(m > t for *_, m in N)
    done = len({x[0] for x in P}) >= 4
    ok = (tp / len(P) >= 0.9 and fp / len(N) <= 0.05) if done and P and N else None
    per_family = {fam: {"tpr": [sum(m > t for f2, _, m in P if f2 == fam), sum(f2 == fam for f2, *_ in P)],
                        "fpr": [sum(m > t for f2, _, m in N if f2 == fam), sum(f2 == fam for f2, *_ in N)]}
                  for fam in sorted({x[0] for x in P})}
    return {"id": "H12", "statement": statement, "verdict": verdict(ok),
            "numbers": {"statistic": stat, "threshold": t, "tpr": [tp, len(P)], "fpr": [fp, len(N)],
                        "per_family": per_family, "dev": dev},
            "evidence": f"{stat} margin > {t:.4f} (chosen on dev): TPR {tp}/{len(P)}, FPR {fp}/{len(N)}"}


def h10(e6, T, rule):
    sh = [r for r in e6 if r.get("shared_init") and not r.get("error")]
    ds = [r for r in e6 if r.get("relation") == "different_init_same_data" and not r.get("error")]
    a = sum(related(r, T, rule) == "related" for r in sh)
    b = sum(related(r, T, rule) == "related" for r in ds)
    ok = (a / len(sh) >= 2 / 3 and b / len(ds) <= 1 / 3) if sh and ds else None
    return {"id": "H10", "statement": "Weight similarity tracks shared initialisation, not shared data.",
            "verdict": verdict(ok), "numbers": {"shared_init": [a, len(sh)], "different_init_same_data": [b, len(ds)]},
            "evidence": f"shared initialisation related {a}/{len(sh)}; different initialisation, same data {b}/{len(ds)}"}


def h11(e7, T, rule):
    by = defaultdict(list)
    for r in e7:
        if not r.get("error"):
            by[r["run_final"]].append(r)
    own_ok, other_hits, curve, other = True, 0, defaultdict(list), defaultdict(list)
    for run, rs in by.items():
        for r in rs:
            rel = related(r, T, rule) == "related"
            if r["same_run"]:
                curve[r["step"]].append(r["median_cosine"])
                if r["step"] >= 10_000 and not rel:
                    own_ok = False
            else:
                other[r["step"]].append(r["median_cosine"])
                other_hits += rel
    ok = (own_ok and other_hits == 0) if len(by) == 5 else None
    return {"id": "H11", "statement": "A final model is verified against its own checkpoints from step 10,000 on, "
            "and against no checkpoint of another run.", "verdict": verdict(ok),
            "numbers": {"own_from_10k_all_related": own_ok, "other_run_related": other_hits,
                        "curve": {s: [min(v), max(v)] for s, v in sorted(curve.items())},
                        "other_runs": {s: [min(v), max(v)] for s, v in sorted(other.items())}},
            "evidence": f"own checkpoints from 10k all related: {own_ok}; other runs related: {other_hits}"}


# --- E2: categories, H2, H3 -----------------------------------------------------------------------

def audit(e2, T, rule) -> dict:
    cats, h2, h3, relation_tags = [], [], [], []
    for m in e2:
        tier = m["tier"]
        if m.get("reason"):
            cats.append((tier, "adapter" if m["reason"] == "adapter" else "not readable", m["reason"]))
            continue
        dec = [(d, related(d, T, rule)) for d in m.get("declared", [])]
        for d, v in dec:
            if v != "not comparable":
                h3.append((tier, v == "unrelated", m["id"], d["base"], d.get("median_cosine")))
        rel_pool = [p for p in m.get("pool", []) if related(p, T, rule, plain_only=True) == "related"]
        earlier = [p for p in rel_pool if not p.get("identical") and p.get("created") and m.get("created")
                   and p["created"] < m["created"]]
        declared_rel = [d for d, v in dec if v == "related" and not d.get("identical")]
        derived = bool(earlier or declared_rel)
        if derived:
            h2.append((tier, not m.get("declared_base"), m["id"]))
        if dec:
            v = {x for _, x in dec}
            cat = ("declared, related" if "related" in v else "declared, unrelated" if "unrelated" in v
                   else "declared, not comparable")
        else:
            cat = "no declaration, parent found" if earlier else "no declaration, no parent found"
        cats.append((tier, cat, ""))
        if m.get("base_relation") == "quantized":
            relation_tags.append((m["id"], [d.get("median_cosine") for d, _ in dec]))

    def prop(items, flag=1):
        k = sum(x[flag] for x in items)
        n = len(items)
        per = {t: [sum(x[flag] for x in items if x[0] == t), sum(1 for x in items if x[0] == t)] for t in TIER_SIZE}
        w = sum(TIER_SIZE[t] * (a / b) for t, (a, b) in per.items() if b) / sum(TIER_SIZE[t] for t, (a, b) in
                                                                               per.items() if b) if n else None
        return {"k": k, "n": n, "share": k / n if n else None, "wilson": wilson(k, n), "per_tier": per,
                "tier_weighted": w}
    counts = defaultdict(lambda: defaultdict(int))
    for t, c, _ in cats:
        counts[c][t] += 1
    return {"categories": {c: dict(v) for c, v in counts.items()}, "h2": prop(h2), "h3": prop(h3),
            "contradicted": [x[2:] for x in h3 if x[1]], "n_models": len(e2)}


def behaviour_audit(e2b, h9r) -> dict | None:
    """E2's second instrument (PLAN.md, 2026-11-04 and amendment 2026-11-07): declared pairs the
    weights could not compare, judged by H9's rule as chosen on E5 dev. Reported as its own
    category, never moved into H3 (whose denominator is bases compared by weights), and shown
    with H9's fresh error rates, which measure how far a behaviour verdict can be trusted."""
    if not e2b:
        return None
    n = h9r.get("numbers") or {}
    stat, t = n.get("statistic"), n.get("threshold")
    get = {"agreement": lambda r: r.get("agreement"), "1-js": lambda r: 1 - r["js"] if num(r.get("js")) is not None
           else None, "cka": lambda r: r.get("cka")}.get(stat)
    out = {"rule": {"statistic": stat, "threshold": t, "fresh_tpr": n.get("tpr"), "fresh_fpr": n.get("fpr"),
                    "h9": h9r["verdict"]}, "pairs": len(e2b), "verified_by_behaviour": [], "below_rule": [],
           "not_applicable": defaultdict(int), "failed": []}
    for r in e2b:
        key = (r["model"], r["base"])
        if r.get("applicable") is None:
            out["failed"].append([*key, r.get("error")])
        elif not r["applicable"]:
            why = ("neither a shared language-model output nor a shared label set" if r["reason"].startswith("text:")
                   else r["reason"])
            out["not_applicable"][why] += 1
        else:
            v = get(r) if get else None
            (out["verified_by_behaviour"] if num(v) is not None and v > t else out["below_rule"]).append(
                [*key, r.get("modality"), v])
    out["not_applicable"] = dict(out["not_applicable"])
    return out


def with_rescores(e2, e2r, e2g=(), e2s=()):
    """E2 with each comparison that ended in a reader error replaced by its E2r score (the
    fixed readers), and each GGUF comparison that compared too few tensors replaced by its E2g
    score (the name fix of the 2026-11-17 amendment); E2's own records are not changed."""
    import copy
    new = {(r["model"], r["other"], r["role"]): r for r in e2r if "lost" not in r}
    gnew = {(r["model"], r["other"], r["role"]): r for r in e2g if "lost" not in r}
    snew = {(r["model"], r["other"], r["role"]): r for r in e2s if "lost" not in r}
    out = copy.deepcopy(e2)
    keep = ("median_cosine", "compared", "matched", "match_share", "identical", "kurtosis_gap",
            "median_invariant", "invariant_compared", "error")
    for m in out:
        for role, key, rows in (("declared", "base", m.get("declared", [])), ("pool", "candidate", m.get("pool", []))):
            for d in rows:
                r = new.get((m["id"], d[key], role))
                if r and (d.get("error") or "legacy pickle" in (d.get("reason") or "")):
                    d.pop("error", None)
                    d.pop("reason", None)
                    d.update({k: r[k] for k in keep if k in r})
                g = gnew.get((m["id"], d[key], role))
                if g and not d.get("error") and (d.get("compared") or 0) < MIN_COMPARED:
                    d.update({k: g[k] for k in keep if k in g})
                    if g.get("error"):
                        d["error"] = g["error"]
                # E2s (H17b): structural alignment, and the pairs E2g could not read again
                x = snew.get((m["id"], d[key], role))
                if x and (d.get("error") == "not readable on re-reading"
                          or (not d.get("error") and (d.get("compared") or 0) < MIN_COMPARED)):
                    d.pop("error", None)
                    d.update({k: x[k] for k in keep if k in x})
    return out


def h13(h13r, T, rule) -> list[dict]:
    """H13 (PLAN.md, amendment 2026-11-13): pool v2 on the fresh sample. The ground truth is each
    fresh model's declared base where the weights confirm it (as H7 b); both pools are rebuilt
    from the frozen metadata, so the job that read the weights could not tune them."""
    st_a = "Pool v2 holds the weight-verified declared parent for >= 90% of fresh Hub models."
    st_b = "Pool v2 holds more of those parents than E2's pool (exact one-sided McNemar, p < 0.05)."
    if len(h13r) < 360:
        return [{"id": "H13a", "statement": st_a, "verdict": "pending", "evidence": "needs H13"},
                {"id": "H13b", "statement": st_b, "verdict": "pending", "evidence": "needs H13"}]
    from pedigree import pool
    from pedigree.pool_v2 import PoolV2, float_params
    F = pool.Frame(ROOT / "results" / "e2_frame_meta.jsonl.gz")
    P = PoolV2(F, ROOT / "results" / "e2_bases_meta.jsonl.gz")
    pairs = []
    for m in h13r:
        for d in m.get("declared", []):
            if related(d, T, rule) != "related" or d.get("identical"):
                continue
            want = P.name(d["base"])
            v1 = {P.name(c) for c in F.candidates(m["family_key"], m.get("params"), m.get("dims"), exclude={m["id"]})}
            params = m.get("params") if m.get("quantised") else (float_params(F.by_id.get(m["id"], {})) or m.get("params"))
            v2 = P.candidates(m["id"], m["family_key"], params, m.get("dims"), bool(m.get("quantised")),
                              searched=(m.get("search") or {}).get("hits", []))
            pairs.append((m["tier"], want in v1, want in v2, len(v2)))
    n = len(pairs)
    k2, k1 = sum(p[2] for p in pairs), sum(p[1] for p in pairs)
    b = sum(p[2] and not p[1] for p in pairs)
    c = sum(p[1] and not p[2] for p in pairs)
    pval = sum(math.comb(b + c, i) for i in range(b, b + c + 1)) / 2 ** (b + c) if b + c else 1.0
    per = {t: [sum(p[2] for p in pairs if p[0] == t), sum(1 for p in pairs if p[0] == t)] for t in TIER_SIZE}
    nums = {"n": n, "v2": [k2, n, wilson(k2, n)], "v1": [k1, n, wilson(k1, n)], "v2_only": b, "v1_only": c,
            "mcnemar_p": pval, "v2_per_tier": per, "mean_pool_size": sum(p[3] for p in pairs) / n if n else None}
    return [{"id": "H13a", "statement": st_a, "verdict": verdict(k2 / n >= 0.9 if n else None), "numbers": nums,
             "evidence": f"v2 {k2}/{n} (E2's pool {k1}/{n})"},
            {"id": "H13b", "statement": st_b, "verdict": verdict(pval < 0.05 if n else None), "numbers": nums,
             "evidence": f"v2 only {b}, E2's pool only {c}, p = {pval:.2g}"}]


def h2h3(A, done):
    h2 = {"id": "H2", "statement": "At least 30% of Hub models verifiably derived from another public model do not "
          "declare a base.", "verdict": verdict(A["h2"]["share"] >= 0.30 if done and A["h2"]["n"] else None),
          "numbers": A["h2"], "evidence": f"{A['h2']['k']}/{A['h2']['n']} undeclared"}
    h3 = {"id": "H3", "statement": "At least 5% of declared bases are contradicted by the weights.",
          "verdict": verdict(A["h3"]["share"] >= 0.05 if done and A["h3"]["n"] else None),
          "numbers": A["h3"], "evidence": f"{A['h3']['k']}/{A['h3']['n']} contradicted"}
    return [h2, h3]



# --- amendment 2026-11-17 (4): H15-H18 ---------------------------------------------------------

def unverifiable(e2f, T, rule) -> dict:
    """Why each declared pair that weights cannot compare is not comparable (models whose every
    declared base is not comparable), grouped as in PLAN.md amendment 2026-11-17 (4)."""
    out = defaultdict(int)
    models = 0
    for m in e2f:
        if m.get("reason"):
            continue
        dec = [(d, related(d, T, rule)) for d in m.get("declared", [])]
        if not dec or any(v != "not comparable" for _, v in dec):
            continue
        models += 1
        for d, _ in dec:
            err = str(d.get("error") or d.get("reason") or "")
            gguf = m.get("format") == "gguf" or d.get("format") == "gguf"
            if "re-reading" in err:
                k = "read failed again (network)"
            elif "tensor type" in err:
                k = "GGUF tensor type of a fork"
            elif "401" in err or "gated" in err or "403" in err:
                k = "gated or private"
            elif gguf and not err and (d.get("matched") or 0) == 0:
                k = "GGUF: no tensor name maps"
            elif gguf and not err:
                k = "GGUF: too few tensors align"
            else:
                k = "other format (no shared shapes, draft head, repack, no weights)"
            out[k] += 1
    return {"models": models, "pairs": sum(out.values()), "reasons": dict(sorted(out.items(), key=lambda x: -x[1]))}


D3_DAYS = 30
D3_MARGIN = 0.00062         # D2's dev margin (amendment 2026-11-07), as written in amendment 2026-11-17 (4)


def _days(a: str, b: str) -> float:
    from datetime import datetime

    def f(x):
        return datetime.fromisoformat(x.replace("Z", "+00:00"))
    return (f(a) - f(b)).total_seconds() / 86400


def d3(model: str, base: str, created_m: str, created_b: str, kgap) -> str:
    """D3 (PLAN.md, amendment 2026-11-17 (4)) with the declared base as the true parent:
    'right', 'wrong' or 'undecided'. kgap = kurtosis(model) - kurtosis(base)."""
    gap = _days(created_m, created_b)            # > 0: the model is later
    if model.split("/")[0].lower() != base.split("/")[0].lower() or abs(gap) > D3_DAYS:
        return "undecided" if gap == 0 else ("right" if gap > 0 else "wrong")
    a, b = model.split("/")[-1].lower(), base.split("/")[-1].lower()
    if b.endswith("-base") and b[:-5] == a:
        return "right"
    if a.endswith("-base") and a[:-5] == b:
        return "wrong"
    if any(a.startswith(b + c) for c in "-_"):
        return "right"
    if any(b.startswith(a + c) for c in "-_"):
        return "wrong"
    if num(kgap) is not None and abs(kgap) > D3_MARGIN:
        return "right" if kgap > 0 else "wrong"
    return "undecided"


def direction_pairs(rows, T, rule):
    """Related declared pairs (not byte-identical) with both creation dates: (model, base, dates, kgap)."""
    out = []
    for mo in rows:
        for d in mo.get("declared", []):
            if related(d, T, rule) == "related" and not d.get("identical") and d.get("created") and mo.get("created"):
                out.append((mo["id"], d["base"], mo["created"], d["created"], d.get("kurtosis_gap")))
    return out


def direction_scores(pairs) -> dict:
    d1 = [_days(c, cb) > 0 for _, _, c, cb, _ in pairs if c != cb]
    d2 = [k > 0 for *_, k in pairs if num(k) is not None and abs(k) > D3_MARGIN]
    v3 = [d3(*p) for p in pairs]
    dec = [v for v in v3 if v != "undecided"]
    right = sum(v == "right" for v in dec)
    return {"n": len(pairs), "d3": [right, len(dec), wilson(right, len(dec))],
            "d3_undecided": [len(pairs) - len(dec), len(pairs)],
            "d1": [sum(d1), len(d1), wilson(sum(d1), len(d1))], "d2": [sum(d2), len(d2), wilson(sum(d2), len(d2))],
            "d3_wrong": [list(p[:2]) for p, v in zip(pairs, v3) if v == "wrong"]}


def h15(e2, h15r, T, rule) -> dict:
    st = ("Direction by D3 (dates across owners or > 30 days apart; names within one release; kurtosis last) is "
          "right on >= 95% of fresh decided pairs and leaves <= 10% undecided.")
    dev = direction_scores(direction_pairs(e2, T, rule))
    base = {"id": "H15", "statement": st, "follows": "H8", "numbers": {"dev": dev}}
    if len(h15r) < 360:
        return {**base, "verdict": "pending", "evidence": f"fresh sample running; on E2 (dev) D3 is right on "
                f"{dev['d3'][0]}/{dev['d3'][1]} decided, undecided {dev['d3_undecided'][0]}/{dev['d3_undecided'][1]}"}
    f = direction_scores(direction_pairs(h15r, T, rule))
    (r, n, _), (u, nn) = f["d3"], f["d3_undecided"]
    ok = n > 0 and r / n >= 0.95 and u / nn <= 0.10
    return {**base, "verdict": verdict(ok), "numbers": {"dev": dev, "fresh": f},
            "evidence": f"D3 right on {r}/{n} decided, undecided {u}/{nn} (D1 {f['d1'][0]}/{f['d1'][1]}, "
                        f"D2 {f['d2'][0]}/{f['d2'][1]} decided)"}


H16_FAMILIES = ("text_decoder", "vision_cnn")


def _student(r) -> tuple[str, int] | None:
    m = str(r.get("model", ""))
    for kind in ("distill", "indep"):
        if m.startswith(kind + "_s"):
            return kind, int(m.rsplit("_s", 1)[1])
    return None


def drops(rows, refs_of) -> list[tuple[str, str, str, float]]:
    """(family, model, kind, drop) per student; refs_of(row, students of its family) gives its references."""
    out = []
    for fam in sorted({r["family"] for r in rows}):
        fr = [r for r in rows if r["family"] == fam and _student(r)]
        for r in fr:
            refs = refs_of(r, fr)
            if not refs:
                continue
            lo = min(x["dark_js"] for x in refs)
            out.append((fam, r["model"], _student(r)[0], (lo - r["dark_js"]) / lo))
    return out


def h16_rule(e5e_dev, e5e_fresh) -> dict:
    """r* from the students already scored (two batches), each against its batch's other independents."""
    def loo(r, fr):
        return [x for x in fr if _student(x)[0] == "indep" and x["model"] != r["model"]]
    D = []
    for batch in (e5e_dev, e5e_fresh):
        D += drops([r for r in batch if r.get("family") in H16_FAMILIES], loo)
    p = [d for *_, k, d in D if k == "distill"]
    n = [d for *_, k, d in D if k == "indep"]
    gap = min(p) - max(n) if p and n else None
    return {"r_star": (min(p) + max(n)) / 2 if gap is not None else None, "gap": gap,
            "dev": [[f, m, k, round(d, 4)] for f, m, k, d in D]}


def h16(e5e_dev, e5e_fresh, e5e_h16) -> dict:
    st = ("Behaviour calibrated against reference students (dark-knowledge drop below the lowest reference > r*, "
          "r* from dev) identifies fresh distilled students: TPR >= 90% at FPR <= 5%.")
    rule = h16_rule(e5e_dev, e5e_fresh)
    base = {"id": "H16", "statement": st, "follows": "H9, H12, H14, H14b"}
    if rule["gap"] is None or rule["gap"] <= 0:
        return {**base, "verdict": "not supported", "numbers": rule, "evidence": "no positive dev gap"}
    rows = [r for r in e5e_h16 if r.get("family") in H16_FAMILIES]
    if not rows:
        return {**base, "verdict": "pending", "numbers": rule,
                "evidence": f"dev rule: drop > {rule['r_star']:.3f} (dev gap {rule['gap']:.3f}); "
                            "fresh students wait for GPU quota"}

    def refs(r, fr):
        return [x for x in fr if _student(x)[0] == "indep" and 18 <= _student(x)[1] <= 20]
    sus = [x for x in drops(rows, refs) if 13 <= int(x[1].rsplit("_s", 1)[1]) <= 17]
    tp = sum(k == "distill" and d > rule["r_star"] for _, _, k, d in sus)
    P = sum(k == "distill" for _, _, k, _ in sus)
    fp = sum(k == "indep" and d > rule["r_star"] for _, _, k, d in sus)
    N = sum(k == "indep" for _, _, k, _ in sus)
    done = {f for f, *_ in sus} >= set(H16_FAMILIES)
    ok = (P > 0 and N > 0 and tp / P >= 0.9 and fp / N <= 0.05) if done else None
    return {**base, "verdict": verdict(ok), "numbers": {**rule, "fresh": [[f, m, k, round(d, 4)] for f, m, k, d in sus],
            "tpr": [tp, P], "fpr": [fp, N]}, "evidence": f"drop > {rule['r_star']:.3f}: TPR {tp}/{P}, FPR {fp}/{N}"}


def h16b(e5e_h16b) -> dict:
    st = ("With students trained long enough to learn the task (text encoder, vision transformer), the "
          "reference-calibrated rule from a new dev batch identifies fresh distilled students: TPR >= 90% at FPR <= 5%.")
    base = {"id": "H16b", "statement": st, "follows": "H14b"}
    if not e5e_h16b:
        return {**base, "verdict": "pending", "evidence": "dev and fresh students wait for GPU quota"}
    return {**base, "verdict": "pending", "evidence": "judged when both batches are recorded"}


def _verdict_plain(x: dict | None, T, rule) -> str:
    if not x or x.get("error") or (x.get("compared") or 0) < MIN_COMPARED or num(x.get("median_cosine")) is None:
        return "not comparable"
    return "related" if x["median_cosine"] > T["plain"][rule] else "unrelated"


def h17(h15r, e2f_before, e2s, T, rule) -> list[dict]:
    sa = ("On fresh GGUF pairs where names align >= 8 tensors, structural alignment (names ignored) gives the same "
          "related / unrelated verdict for >= 98%.")
    sb = "Structural alignment makes >= 50% of E2's declared GGUF pairs with no name match comparable."
    pairs = []
    for m in h15r:
        for a in [d.get("alignments") for d in m.get("declared", [])] + list(m.get("pool_alignments", [])):
            if not a or not a.get("names"):
                continue
            vn = _verdict_plain(a["names"], T, rule)
            if vn != "not comparable":
                pairs.append((vn, _verdict_plain(a.get("struct"), T, rule)))
    agree = sum(a == b for a, b in pairs)
    neg = sum(a == "unrelated" for a, _ in pairs)
    judged = len(pairs) >= 20 and neg >= 5
    if len(h15r) < 360:
        va, ea = "pending", "fresh sample running"
    else:
        va = verdict(agree / len(pairs) >= 0.98) if judged else "not judged"
        ea = f"{agree}/{len(pairs)} agree ({neg} unrelated by names)"
    out = [{"id": "H17a", "statement": sa, "follows": "unverifiable declared bases", "verdict": va,
            "numbers": {"agree": [agree, len(pairs)], "unrelated_by_names": neg}, "evidence": ea}]
    target = [(m["id"], d["base"]) for m in e2f_before if not m.get("reason") for d in m.get("declared", [])
              if (m.get("format") == "gguf" or d.get("format") == "gguf") and not d.get("error")
              and (d.get("matched") or 0) == 0]
    got = {(r["model"], r["other"]): r for r in e2s if r.get("role") == "declared" and "lost" not in r}
    done = [got[t] for t in target if t in got]
    comp = sum((r.get("compared") or 0) >= MIN_COMPARED and not r.get("error") for r in done)
    complete = bool(target) and len(done) == len(target)
    out.append({"id": "H17b", "statement": sb, "follows": "unverifiable declared bases",
                "verdict": verdict(comp / len(target) >= 0.5) if complete else "pending",
                "numbers": {"comparable": [comp, len(target)], "scored": len(done)},
                "evidence": f"{comp}/{len(target)} comparable" if complete else
                f"E2s running ({len(target)} pairs with no name match)"})
    return out


def h18(h16r: dict, h16br: dict) -> dict:
    st = ("Behaviour (H16's rule) confirms >= 90% of the declared Hub pairs weights cannot compare that it can run "
          "(text models <= 14B sharing one tokenizer).")
    base = {"id": "H18", "statement": st, "follows": "behaviour above 2B parameters"}
    vs = (h16r["verdict"], h16br["verdict"])
    if "supported" not in vs:
        if "pending" in vs:
            return {**base, "verdict": "pending", "evidence": "runs only if H16 or H16b is supported"}
        return {**base, "verdict": "not run", "evidence": "not run: neither H16 nor H16b is supported"}
    return {**base, "verdict": "pending", "evidence": "runs after H16 / H16b"}

# --- main -------------------------------------------------------------------------------------------

# --- H14 / H14b: behaviour by shared mistakes (amendments 2026-11-17 (2) and (3)) ---------------

H14_STATS = {"error_agreement": 1, "dark_js": -1}   # sign: +1 higher = distilled, -1 lower = distilled
E5_FAMILIES = ["text_decoder", "text_encoder", "vision_cnn", "vision_transformer"]


def _admit(r, stat, sign, t) -> bool:
    return sign * r[stat] > sign * t


def h14_rule(dev: list[dict]) -> dict:
    """H14's pooled rule from E5e dev alone: the statistic with the larger gap (smallest distilled
    minus largest independent, signed); threshold at the gap's midpoint, or if no gap is positive the
    threshold admitting the most distilled models and no independent one."""
    P = [r for r in dev if r["label"] == "positive"]
    N = [r for r in dev if r["label"] == "negative"]
    gaps = {}
    for stat, sign in H14_STATS.items():
        p, n = [sign * r[stat] for r in P], [sign * r[stat] for r in N]
        gaps[stat] = {"gap": min(p) - max(n), "mid": sign * (min(p) + max(n)) / 2, "max_neg": sign * max(n),
                      "tpr_at_fpr0": sum(x > max(n) for x in p) / len(p)}
    stat = max(gaps, key=lambda s: gaps[s]["gap"])
    t = gaps[stat]["mid"] if gaps[stat]["gap"] > 0 else gaps[stat]["max_neg"]
    return {"statistic": stat, "sign": H14_STATS[stat], "threshold": t, "dev": gaps}


def _score(rows, stat, sign, t) -> dict:
    P = [r for r in rows if r["label"] == "positive"]
    N = [r for r in rows if r["label"] == "negative"]
    return {"tpr": [sum(_admit(r, stat, sign, t) for r in P), len(P)],
            "fpr": [sum(_admit(r, stat, sign, t) for r in N), len(N)]}


def _passes(sc) -> bool:
    (tp, np_), (fp, nn) = sc["tpr"], sc["fpr"]
    return np_ > 0 and nn > 0 and tp / np_ >= 0.9 and fp / nn <= 0.05


def h14(dev: list[dict], fresh: list[dict]) -> list[dict]:
    fams = {r["family"] for r in fresh}
    rule = h14_rule(dev)
    stat, sign, t = rule["statistic"], rule["sign"], rule["threshold"]
    per = {f: _score([r for r in fresh if r["family"] == f], stat, sign, t) for f in sorted(fams)}
    sc = _score(fresh, stat, sign, t)
    done = set(E5_FAMILIES) <= fams
    rel = "above" if sign > 0 else "below"
    out = [{"id": "H14", "statement": "On fresh students, one rule chosen on dev (pooled over families) identifies "
                                       "distilled students by shared mistakes (TPR >= 90% at FPR <= 5%).",
            "verdict": verdict(_passes(sc) if done else None),
            "numbers": {**rule, "fresh": sc, "per_family": per},
            "evidence": f"{stat} {rel} {t:.4f} (dev): TPR {sc['tpr'][0]}/{sc['tpr'][1]}, FPR {sc['fpr'][0]}/"
                        f"{sc['fpr'][1]}" + ("" if done else f"; waiting for {', '.join(sorted(set(E5_FAMILIES) - fams))}")}]
    rules = json.loads((ROOT / "results" / "h14b_rules.json").read_text(encoding="utf-8"))
    ruled = sorted(f for f, r in rules.items() if r["threshold"] is not None)
    per_b, ev = {}, []
    for f in ruled:
        r = rules[f]
        rs = [x for x in fresh if x["family"] == f]
        if rs:
            per_b[f] = _score(rs, r["statistic"], 1 if r["direction"] == "above" else -1, r["threshold"])
            ev.append(f"{f}: {r['statistic']} {r['direction']} {r['threshold']:.4f}, TPR {per_b[f]['tpr'][0]}/"
                      f"{per_b[f]['tpr'][1]}, FPR {per_b[f]['fpr'][0]}/{per_b[f]['fpr'][1]}")
    both = {"tpr": [sum(v["tpr"][0] for v in per_b.values()), sum(v["tpr"][1] for v in per_b.values())],
            "fpr": [sum(v["fpr"][0] for v in per_b.values()), sum(v["fpr"][1] for v in per_b.values())]}
    done_b = set(ruled) <= set(per_b)
    missing = sorted(set(ruled) - set(per_b))
    undecided = sorted(set(rules) - set(ruled))
    out.append({"id": "H14b", "statement": "On the families with a dev rule (text decoder, vision CNN), the per-family "
                                           "rule identifies fresh distilled students (TPR >= 90% at FPR <= 5%).",
                "verdict": verdict(_passes(both) if done_b else None),
                "numbers": {"rules": rules, "per_family": per_b, "pooled": both, "not_decidable": undecided},
                "evidence": "; ".join(ev) + (f"; waiting for {', '.join(missing)}" if missing else "")
                            + f"; no dev rule: {', '.join(undecided)}"})
    return out


def main() -> None:
    rows = load()
    e1 = latest_runs(rows, "E1", lambda r: r.get("family"))
    e1b = latest_runs(rows, "E1b", lambda r: r.get("family"))
    # E5 runs dev and fresh as separate runs: keep the latest of each (family, split)
    e5 = latest_runs(rows, "E5", lambda r: ((r.get("meta") or {}).get("family", r.get("family")),
                                            (r.get("meta") or {}).get("split", r.get("split"))))
    e6 = latest_runs(rows, "E6", lambda r: 0)
    e7 = latest_runs(rows, "E7", lambda r: 0)
    e7k = [r for r in latest_runs(rows, "E7k", lambda r: 0) if not r.get("error")]
    h7a = latest_runs(rows, "H7a", lambda r: r.get("family"))
    e2 = latest_runs(rows, "E2", lambda r: 0)
    e2w = latest_runs(rows, "E2W", lambda r: 0)
    e2b = latest_runs(rows, "E2B", lambda r: 0)
    e2r = latest_runs(rows, "E2r", lambda r: 0) + latest_runs(rows, "E2p", lambda r: 0)  # E2p: legacy .bin
    e2g = latest_runs(rows, "E2g", lambda r: (r["model"], r["other"], r["role"]))  # resumed runs: each pair's latest
    h13r = latest_runs(rows, "H13", lambda r: 0)
    h15r = latest_runs(rows, "H15", lambda r: 0)
    e2s = latest_runs(rows, "E2s", lambda r: (r["model"], r["other"], r["role"]))
    def by_family_split(exp):
        return latest_runs(rows, exp, lambda r: ((r.get("meta") or {}).get("family", r.get("family")),
                                                (r.get("meta") or {}).get("split", r.get("split"))))
    e5o, e5n = by_family_split("E5o"), by_family_split("E5n")
    for r in e5 + e5o + e5n:         # E5 keeps family/split in the run's metadata
        meta = r.get("meta") or {}
        r.setdefault("family", meta.get("family"))
        r.setdefault("split", meta.get("split"))
    e5e_dev, e5e_fresh = ([r for r in by_family_split(x) if "error_agreement" in r and r.get("label")]
                          for x in ("E5e-dev", "E5e-fresh"))
    e5e_h16, e5e_h16b = ([r for r in by_family_split(x) if "dark_js" in r]
                         for x in ("E5e-h16", "E5e-h16b"))
    for r in e5e_dev + e5e_fresh + e5e_h16 + e5e_h16b:
        r.setdefault("family", (r.get("meta") or {}).get("family"))
    T = thresholds(e1)
    out = {"thresholds": T, "families_in_e1": sorted({r["family"] for r in e1}), "blind_families": sorted(BLIND),
           "rules": {}}
    for rule in ("t", "midpoint"):
        A = audit(e2, T, rule) if e2 else None
        hs = [h1(e1, T, rule), *(h2h3(A, len(e2) == 600) if A else []), h4(e1, T, rule), h5(e1, e5, T, rule),
              h6(e1b, e2w, T, rule), h7(h7a, e2, T, rule), h8(e1b, e7k, e2, T, rule), h9(e5), h10(e6, T, rule),
              h11(e7, T, rule), h12(e5o), *h13(h13r, T, rule), *h14(e5e_dev, e5e_fresh)]
        h16r, h16br = h16(e5e_dev, e5e_fresh, e5e_h16), h16b(e5e_h16b)
        hs += [h15(e2, h15r, T, rule), h16r, h16br, *h17(h15r, with_rescores(e2, e2r, e2g), e2s, T, rule),
               h18(h16r, h16br)]
        if e2r:
            e2f = with_rescores(e2, e2r, e2g, e2s)
            Af = audit(e2f, T, rule)
            hf = {h["id"]: h for h in [*h2h3(Af, len(e2f) == 600), h7(h7a, e2f, T, rule)]}
            out.setdefault("reader_fixed", {})[rule] = {
                "rescored": len(e2r), "lost": sum("lost" in r for r in e2r),
                "still_errors": sum(bool(r.get("error")) for r in e2r), "audit": Af,
                "gguf_rescored": len(e2g), "gguf_comparable": sum((r.get("compared") or 0) >= MIN_COMPARED
                                                                  for r in e2g),
                "unverifiable": unverifiable(e2f, T, rule),
                "verdicts": {k: {"verdict": v["verdict"], "evidence": v["evidence"],
                                 "changes": v["verdict"] != next(h for h in hs if h["id"] == k)["verdict"]}
                             for k, v in hf.items()}}
        out["rules"][rule] = {"hypotheses": hs, "audit": A,
                              "behaviour": behaviour_audit(e2b, next(h for h in hs if h["id"] == "H9"))}
    # the CLI's thresholds are exactly the published rule's
    (ROOT / "pedigree" / "thresholds.json").write_text(json.dumps(
        {k: round(v["t"], 6) for k, v in T.items() if k in ("plain", "invariant")} |
        {"rule": "max(E1 dev negatives) + 5 SD, PLAN.md amendment 2026-11-07",
         "families": out["families_in_e1"]}, indent=1), encoding="utf-8")
    (ROOT / "results" / "verdicts.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    md = ["# Results", "", "Generated by `scripts/analyse.py` from `results/ledger.jsonl`. Do not edit by hand.", "",
          "## Thresholds (E1 dev negatives only; PLAN.md amendment 2026-11-07)", "",
          "| Score | Threshold (max + 5 SD) | Midpoint (sensitivity) | Dev negatives | Highest | Lowest dev fine-tune |",
          "|---|--:|--:|--:|--:|--:|"]
    for k, v in T.items():
        md.append(f"| {k} | {v['t']:.4f} | {v['midpoint']:.4f} | {v['n_negatives']} | {v['max_negative']:.4f} "
                  f"| {v['min_trained_positive']:.4f} |")
    md += ["", f"Families compared blind (after the rule was written): {', '.join(sorted(BLIND))}.", "",
           "## Hypotheses", "", "| | Hypothesis | Verdict | Evidence | Under the midpoint rule |", "|---|---|---|---|---|"]
    mid = {h["id"]: h for h in out["rules"]["midpoint"]["hypotheses"]}
    for h in sorted(out["rules"]["t"]["hypotheses"], key=lambda h: (int(h["id"][1:].rstrip("ab")), h["id"])):
        md.append(f"| {h['id']} | {h['statement']} | **{h['verdict']}** | {h['evidence']} | {mid[h['id']]['verdict']} |")
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "RESULTS.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    for h in sorted(out["rules"]["t"]["hypotheses"], key=lambda h: (int(h["id"][1:].rstrip("ab")), h["id"])):
        print(f"{h['id']:>4} {h['verdict']:<14} {h['evidence']}")
    print({k: round(v["t"], 4) for k, v in T.items()})


if __name__ == "__main__":
    main()
