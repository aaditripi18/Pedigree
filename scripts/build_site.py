"""Build Pedigree's site from results/verdicts.json (written by scripts/analyse.py).
No number on the site is typed by hand: every figure, count and verdict is read from
verdicts.json (and, for the Hub-drift note, results/e2_rereads.json and results/e2_sample.json).

    python scripts/analyse.py && python scripts/build_site.py

Writes site/index.html (findings, figures, hypotheses, caveats), site/check.html (the browser
checker; site/checker.js with the published thresholds) and site/assets/{site.css,site.js}.
The design follows Stowaway's and leakcheck's sites: the same dark surfaces and type, with
Pedigree's own accent (violet = derivative / related, orange = independent / unrelated; the pair
passes the dataviz palette validator on the panel colour #151924).
"""
from __future__ import annotations

import html
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site"
REPO_URL = "https://github.com/aaditripi18/Pedigree"
e = html.escape

FAMILY = {"text_decoder": "Text decoder", "text_encoder": "Text encoder", "vision_transformer": "Vision transformer",
          "vision_cnn": "Vision CNN", "speech": "Speech", "diffusion_unet": "Diffusion U-Net"}
KIND = {"ft_short": "short fine-tune", "ft_long": "long fine-tune", "lora_r8": "LoRA r8 merge",
        "lora_r64": "LoRA r64 merge", "scratch": "trained from scratch", "independent": "independent release"}
TIER = {"top_1k": "Top 1k", "1k_10k": "1k-10k", "10k_100k": "10k-100k"}
# which later hypothesis follows up which not-supported one (the verdicts themselves come from verdicts.json)
FOLLOWED_BY = {"H2": "a Hub finding", "H3": "a Hub finding", "H7": "H13", "H8": "H15", "H9": "H16", "H12": "H16",
               "H14": "H16", "H14b": "H16, H16b"}


def wilson(k: int, n: int, z: float = 1.96):
    if not n:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def pc(x: float, d: int = 0) -> str:
    return f"{x * 100:.{d}f}%"


def hsort(hs):
    return sorted(hs, key=lambda h: (int(h["id"][1:].rstrip("abcd")), h["id"]))


# --- shared page parts --------------------------------------------------------------------------

FONTS = ('<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" '
         'href="https://fonts.gstatic.com" crossorigin><link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
         'family=Hanken+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap">')
ICON = ("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'%3E%3Crect width='16' "
        "height='16' rx='3' fill='%230f131c'/%3E%3Ccircle cx='5' cy='5' r='2.2' fill='%239085e9'/%3E%3Ccircle cx='11' "
        "cy='11' r='2.2' fill='%239085e9'/%3E%3Cpath d='M5 5 L11 11' stroke='%239085e9' stroke-width='1.4'/%3E%3C/svg%3E")
NAV = [("index.html#audit", "Audit"), ("index.html#instrument", "Instrument"), ("index.html#hypotheses", "Hypotheses"),
       ("index.html#caveats", "Caveats"), ("index.html#reproduce", "Reproduce")]


def page(title: str, desc: str, body: str, extra: str = "") -> str:
    nav = "".join(f'<a href="{h}">{e(t)}</a>' for h, t in NAV)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{e(title)}</title>
<meta name="description" content="{e(desc)}">
<link rel="icon" href="{ICON}">
{FONTS}
<link rel="stylesheet" href="assets/site.css">
</head>
<body>
<a class="skip" href="#main">Skip to content</a>
<header class="top">
  <div class="wrap top-in">
    <a class="logo" href="index.html"><span class="dot"></span><span>PEDIGREE</span></a>
    <nav class="nav" aria-label="Main">{nav}<a href="{REPO_URL}" rel="noopener">GitHub &#8599;</a></nav>
    <a class="btn btn-primary btn-sm" href="check.html">Check a model</a>
  </div>
</header>
<main id="main">
<div class="wrap">
{body}
</div>
</main>
<footer class="foot">
  <div class="wrap foot-in">
    <div>
      <div class="logo"><span class="dot dot-muted"></span><span>PEDIGREE</span></div>
      <p>Every number on this site is generated from <code>results/ledger.jsonl</code> by
      <code>scripts/analyse.py</code>. Code under the <a href="{REPO_URL}/blob/main/LICENSE">MIT licence</a>.</p>
    </div>
    <nav class="foot-nav" aria-label="Footer"><a href="index.html">Findings</a><a href="check.html">Checker</a>
      <a href="{REPO_URL}/blob/main/PLAN.md">PLAN.md</a><a href="{REPO_URL}/blob/main/docs/RESULTS.md">RESULTS.md</a>
      <a href="{REPO_URL}">GitHub</a></nav>
  </div>
</footer>
<script src="assets/site.js"></script>
{extra}
</body>
</html>
"""


def sec_head(num: str, label: str, title: str, anchor: str) -> str:
    return (f'<div class="sec-head" id="{anchor}"><div class="sec-label"><span class="sec-num">{num}</span>'
            f'<span class="sq"></span><span>{e(label)}</span></div><h2>{title}</h2></div>')


def pill(v: str) -> str:
    cls = {"supported": "ok", "not supported": "no", "pending": "wait", "not run": "wait", "not judged": "wait"}.get(v, "wait")
    return f'<span class="pill pill-{cls}">{e(v.upper())}</span>'


def tip(*lines: str) -> str:
    return e("|".join(lines), quote=True)


def legend(items) -> str:
    """items: (css class of the swatch, label)"""
    return '<div class="legend">' + "".join(f'<span class="lg"><i class="sw {c}"></i>{e(t)}</span>' for c, t in items) \
        + "</div>"


def figure(title: str, sub: str, inner: str, note: str = "", table: str = "") -> str:
    t = f'<details class="dt"><summary>Data table</summary>{table}</details>' if table else ""
    n = f'<p class="fig-note">{note}</p>' if note else ""
    return (f'<figure class="fig"><figcaption class="fig-cap"><span class="fig-title">{title}</span>'
            f'<span class="fig-sub">{sub}</span></figcaption>{inner}{n}{t}</figure>')


def table(head, rows, num_from: int = 1) -> str:
    th = "".join(f'<th class="{"r" if i >= num_from else ""}">{h}</th>' for i, h in enumerate(head))
    tr = "".join("<tr>" + "".join(f'<td class="{"r" if i >= num_from else ""}">{c}</td>' for i, c in enumerate(r))
                 + "</tr>" for r in rows)
    return f'<div class="tbl-wrap"><table class="tbl"><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>'


# --- figures ------------------------------------------------------------------------------------

def fig_strip(h1: dict, t: float) -> str:
    """E1: every fresh pair's median cosine, derivatives vs independents, on a broken axis."""
    pts = h1["numbers"]["points"]
    der, ind = pts["derivative"], pts["independent"]
    lo_l = min(min(x[2] for x in ind), 0) - 0.006
    hi_l = max(t * 1.45, max(x[2] for x in ind) + 0.01)
    lo_r = math.floor(min(x[2] for x in der) * 10000) / 10000
    hi_r = 1.00006

    def xl(v):
        return (v - lo_l) / (hi_l - lo_l) * 100

    def xr(v):
        return (v - lo_r) / (hi_r - lo_r) * 100

    fams = [f for f in FAMILY if any(x[0] == f for x in der + ind)]
    rows = []
    for f in fams:
        left, right = [], []
        fi = [x for x in ind if x[0] == f]
        for i, (_, kind, v) in enumerate(fi):
            y = 50 if len(fi) == 1 else 18 + 64 * i / (len(fi) - 1)
            left.append(f'<span class="pt pt-ind" style="left:{xl(v):.2f}%;top:{y:.1f}%" tabindex="0" '
                        f'data-tip="{tip(FAMILY[f] + " · independent", KIND.get(kind, kind), f"median cosine {v:.5f}", "below threshold: unrelated")}"></span>')
        fd = [x for x in der if x[0] == f]
        for i, (_, kind, v) in enumerate(fd):
            y = 18 + 64 * i / max(len(fd) - 1, 1)
            right.append(f'<span class="pt pt-der" style="left:{xr(v):.2f}%;top:{y:.1f}%" tabindex="0" '
                         f'data-tip="{tip(FAMILY[f] + " · derivative", KIND.get(kind, kind), f"median cosine {v:.5f}", "above threshold: related")}"></span>')
        rows.append(f'<div class="strip-row"><span class="strip-fam">{e(FAMILY[f])}</span>'
                    f'<div class="strip-p strip-l"><i class="thr" style="left:{xl(t):.2f}%"></i>{"".join(left)}</div>'
                    f'<span class="strip-brk" aria-hidden="true">//</span>'
                    f'<div class="strip-p strip-r">{"".join(right)}</div></div>')
    lt = [x for x in (0.0, 0.02, 0.04) if lo_l <= x <= hi_l]
    rt = [lo_r, (lo_r + 1) / 2, 1.0]
    ticks_l = "".join(f'<span style="left:{xl(v):.2f}%">{v:g}</span>' for v in lt)
    ticks_r = "".join(f'<span style="left:{xr(v):.2f}%">{v:.4f}</span>' for v in rt)
    inner = (f'<div class="strip">'
             f'<div class="strip-row strip-hd"><span></span><span class="ax-t">unrelated range '
             f'<b class="thr-lab">threshold {t:.4f}</b></span><span></span><span class="ax-t">related range (zoomed)</span></div>'
             + "".join(rows) +
             f'<div class="strip-row strip-ax"><span></span><div class="ticks">{ticks_l}</div><span></span>'
             f'<div class="ticks">{ticks_r}</div></div>'
             f'<div class="strip-row strip-axt"><span></span><span>median cosine</span><span></span>'
             f'<span>median cosine</span></div></div>')
    leg = legend([("sw-der", "derivative: fine-tune or LoRA merge of the base"),
                  ("sw-ind", "independent: same architecture, own training"), ("sw-thr", "threshold (dev only)")])
    tab = table(["Family", "Kind", "Median cosine"],
                [[e(FAMILY[f]), e(KIND.get(k, k)), f"{v:.5f}"] for f, k, v in der + ind], num_from=2)
    tp, n_p = h1["numbers"]["tpr"][:2]
    fp, n_n = h1["numbers"]["fpr"][:2]
    return figure("Fresh ground truth: every pair the threshold never saw",
                  f"{tp}/{n_p} derivatives above it, {fp}/{n_n} independents · axis broken between the ranges",
                  leg + inner, table=tab)


def fig_curve(h11: dict, t: float) -> str:
    """H11: a final model against its own earlier checkpoints, and against other runs'."""
    own, other = h11["numbers"]["curve"], h11["numbers"].get("other_runs", {})
    steps = list(own)
    n = len(steps)

    def x(i):
        return 3 + 94 * i / (n - 1)

    def y(v):
        return 100 - max(0.0, min(1.0, v)) * 100

    def band(d, cls):
        top = " ".join(f"{x(i):.2f},{y(d[s][1]):.2f}" for i, s in enumerate(steps) if s in d)
        bot = " ".join(f"{x(i):.2f},{y(d[s][0]):.2f}" for i, s in reversed(list(enumerate(steps))) if s in d)
        hi = " ".join(f"{x(i):.2f},{y(d[s][1]):.2f}" for i, s in enumerate(steps) if s in d)
        lo = " ".join(f"{x(i):.2f},{y(d[s][0]):.2f}" for i, s in enumerate(steps) if s in d)
        return (f'<polygon class="{cls}-fill" points="{top} {bot}"/>'
                f'<polyline class="{cls}-line" points="{hi}" vector-effect="non-scaling-stroke"/>'
                f'<polyline class="{cls}-line" points="{lo}" vector-effect="non-scaling-stroke"/>')

    def lab(s):
        s = int(s)
        return f"{s // 1000}k" if s >= 1000 else str(s)
    svg = (f'<svg class="curve-svg" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">'
           + "".join(f'<line class="grid" x1="0" x2="100" y1="{y(v)}" y2="{y(v)}" vector-effect="non-scaling-stroke"/>'
                     for v in (0.25, 0.5, 0.75, 1.0))
           + band(other, "oth") + band(own, "own")
           + f'<line class="thr-line" x1="0" x2="100" y1="{y(t):.2f}" y2="{y(t):.2f}" vector-effect="non-scaling-stroke"/>'
           + "</svg>")
    hov = "".join(
        f'<span class="col-hit" style="left:{x(i):.2f}%" tabindex="0" data-tip="'
        + tip(f"checkpoint at step {int(s):,}", f"own run: {own[s][0]:.3f}-{own[s][1]:.3f}",
              (f"other runs: {other[s][0]:.3f} to {other[s][1]:.3f}" if s in other else "other runs: -"),
              f"threshold {t:.4f}") + '"><i class="mk" style="top:' + f'{y(own[s][1]):.2f}%"></i></span>'
        for i, s in enumerate(steps))
    yt = "".join(f'<span style="top:{y(v):.2f}%">{v:g}</span>' for v in (0, 0.25, 0.5, 0.75, 1.0))
    xt = "".join(f'<span style="left:{x(i):.2f}%">{lab(s)}</span>' for i, s in enumerate(steps))
    runs = h11["numbers"].get("runs", 0)
    inner = (legend([("sw-der", f"final model vs its own checkpoint (range over {runs} runs)"),
                     ("sw-ind", f"vs the same step of the other {runs - 1} runs"), ("sw-thr", f"threshold {t:.4f}")])
             + f'<div class="curve"><div class="curve-y">{yt}</div><div class="curve-plot">{svg}{hov}'
               f'<span class="dlab own-lab" style="left:{x(n - 3):.1f}%;top:{y(own[steps[-3]][0]) + 6:.1f}%">own run</span>'
               f'<span class="dlab oth-lab" style="left:{x(n - 3):.1f}%;top:{y(0) - 12:.1f}%">other runs</span>'
               f'</div><div class="curve-x">{xt}</div><div class="curve-xt">checkpoint step (the final model is step 400k)</div></div>')
    tab = table(["Checkpoint step", "Own run (min)", "Own run (max)", "Other runs (min)", "Other runs (max)"],
                [[f"{int(s):,}", f"{own[s][0]:.4f}", f"{own[s][1]:.4f}",
                  f"{other[s][0]:.4f}" if s in other else "-", f"{other[s][1]:.4f}" if s in other else "-"]
                 for s in steps])
    return figure("How far back lineage is visible",
                  f"Stanford CRFM's {runs} GPT-2 runs: the final weights against checkpoints of the same run and of the others",
                  inner, table=tab)


def hbars(rows, scale: float, ref: tuple[float, str] | None = None, cls: str = "b-v", fmt=None) -> str:
    """Horizontal bars, one series. rows: (label, value, display, tooltip lines, (lo, hi) interval or None)."""
    out = []
    rx = ref[0] / scale * 100 if ref else None
    if ref:
        out.append(f'<div class="hb-row hb-refrow"><span></span><div class="hb-track">'
                   f'<span class="ref-lab" style="left:{rx:.2f}%">{e(ref[1])}</span></div><span></span></div>')
    for row in rows:
        lab, v, disp, tl, ci = row[:5]
        rc = row[5] if len(row) > 5 else cls
        w = v / scale * 100
        whisk = (f'<i class="ci" style="left:{ci[0] / scale * 100:.2f}%;width:{(ci[1] - ci[0]) / scale * 100:.2f}%"></i>'
                 if ci else "")
        rl = f'<i class="ref" style="left:{rx:.2f}%"></i>' if ref else ""
        out.append(f'<div class="hb-row"><span class="hb-lab">{e(lab)}</span><div class="hb-track">'
                   f'<span class="hb {rc}" style="width:{w:.2f}%" tabindex="0" data-tip="{tip(*tl)}"></span>{whisk}{rl}'
                   f'</div><span class="hb-val">{disp}</span></div>')
    return f'<div class="hbars">{"".join(out)}</div>'


def fig_h2(h2: dict) -> str:
    n = h2["numbers"]
    rows = []
    for tkey, (k, m) in n["per_tier"].items():
        ci = wilson(k, m)
        rows.append((TIER[tkey], k / m, f"{pc(k / m, 1)} <em>{k}/{m}</em>",
                     (f"{TIER[tkey]} downloads", f"{k} of {m} undeclared", f"95% interval {pc(ci[0], 1)}-{pc(ci[1], 1)}"), ci))
    ci = n["wilson"]
    rows.append(("All 600", n["share"], f"{pc(n['share'], 1)} <em>{n['k']}/{n['n']}</em>",
                 ("pooled sample", f"{n['k']} of {n['n']} undeclared", f"95% interval {pc(ci[0], 1)}-{pc(ci[1], 1)}"),
                 tuple(ci)))
    return hbars(rows, 0.5, ref=(0.30, "pre-registered bar 30%"))


def fig_audit(A: dict, FA: dict | None) -> str:
    LABEL = [("declared, related", "Declares a base the weights confirm"),
             ("declared, unrelated", "Declares a base the weights contradict"),
             ("declared, not comparable", "Declares a base weights cannot compare"),
             ("no declaration, parent found", "Declares none; an earlier relative found"),
             ("no declaration, no parent found", "Declares none; no relative found"),
             ("not readable", "Weights not readable"), ("adapter", "An adapter only")]

    def c(a, k):
        return sum(a["categories"].get(k, {}).values())
    scale = max(max(c(A, k), c(FA, k) if FA else 0) for k, _ in LABEL) * 1.12
    out = []
    for k, lab in LABEL:
        a, b = c(A, k), (c(FA, k) if FA else None)
        ta = tip(lab, "as audited: %d of %d" % (a, A["n_models"]))
        bars = (f'<div class="gb-line"><span class="hb b-g" style="width:{a / scale * 100:.2f}%" tabindex="0" '
                f'data-tip="{ta}"></span><span class="gb-v">{a}</span></div>')
        if b is not None:
            tb = tip(lab, "after reader fixes: %d of %d" % (b, FA["n_models"]))
            bars += (f'<div class="gb-line"><span class="hb b-v" style="width:{b / scale * 100:.2f}%" tabindex="0" '
                     f'data-tip="{tb}"></span><span class="gb-v">{b}</span></div>')
        out.append(f'<div class="gb-row"><span class="gb-lab">{e(lab)}</span><div class="gb-bars">{bars}</div></div>')
    leg = legend([("sw-g", "as audited (E2)"), ("sw-v", "after the reader fixes and re-reads (E2r, E2g, E2p, E2f, E2s)")] if FA else
                 [("sw-g", "as audited (E2)")])
    tab = table(["Category", "As audited", "After reader fixes"],
                [[e(lab), str(c(A, k)), str(c(FA, k)) if FA else "-"] for k, lab in LABEL])
    return figure("The 600-model audit, one category per model",
                  "Uniform random sample, 200 from each download tier of the Hub's top 100,000",
                  leg + f'<div class="gbars">{"".join(out)}</div>', table=tab)


def fig_pool(h13: dict) -> str:
    n = h13["numbers"]
    (k1, m1, c1), (k2, m2, c2) = n["v1"], n["v2"]
    top = hbars([("E2's pool", k1 / m1, f"{pc(k1 / m1)} <em>{k1}/{m1}</em>",
                  ("E2's candidate pool", f"holds the parent of {k1} of {m1}", f"95% interval {pc(c1[0])}-{pc(c1[1])}"),
                  tuple(c1), "b-g"),
                 ("Pool v2", k2 / m2, f"{pc(k2 / m2)} <em>{k2}/{m2}</em>",
                  ("pool v2 (H13)", f"holds the parent of {k2} of {m2}", f"95% interval {pc(c2[0])}-{pc(c2[1])}"),
                  tuple(c2))], 1.0, ref=(0.90, "bar 90%"))
    tiers = hbars([(TIER[t], k / m, f"{pc(k / m)} <em>{k}/{m}</em>", (f"pool v2, {TIER[t]} tier", f"{k} of {m}"),
                    wilson(k, m)) for t, (k, m) in n["v2_per_tier"].items()], 1.0, ref=(0.90, "bar 90%"))
    inner = (legend([("sw-g", "E2's candidate pool"), ("sw-v", "pool v2 (designed from E2's misses)")]) + top
             + f'<div class="sub-h">Pool v2 by download tier</div>' + tiers)
    return figure("Is the true parent among the candidates?",
                  f"{m1} fresh Hub models whose declared base the weights confirm · {n['v2_only']} found only by v2, "
                  f"{n['v1_only']} only by E2's pool",
                  inner)


def fig_behaviour(H: dict) -> str:
    h14, h14b, h16 = H["H14"], H["H14b"], H.get("H16")
    rules = h14b["numbers"]["rules"]
    per14 = h14["numbers"]["per_family"]
    perb = h14b["numbers"]["per_family"]
    rows = []
    for f in ("text_decoder", "vision_cnn", "text_encoder", "vision_transformer"):
        a = per14.get(f, {})
        r = rules.get(f, {})
        b = perb.get(f)
        rule = (f"{e(r['statistic'].replace('_', ' '))} {r['direction']} {r['threshold']:.3f}"
                if r.get("threshold") is not None else "<span class='muted'>no dev rule (gap " + f"{r.get('gap', 0):+.3f})</span>")
        rows.append([e(FAMILY[f]),
                     f"{a['tpr'][0]}/{a['tpr'][1]}" if a else "-", f"{a['fpr'][0]}/{a['fpr'][1]}" if a else "-",
                     rule, f"{b['tpr'][0]}/{b['tpr'][1]}" if b else "-", f"{b['fpr'][0]}/{b['fpr'][1]}" if b else "-"])
    t = table(["Family", "H14 TPR", "H14 FPR", "H14b rule (dev)", "H14b TPR", "H14b FPR"], rows)
    note = ""
    if h16 and (h16.get("numbers") or {}).get("r_star") is not None:
        nn = h16["numbers"]
        note = (f"H16's rule, fixed from dev before any fresh student exists: a student is called distilled when its "
                f"dark-knowledge divergence sits more than {nn['r_star'] * 100:.1f}% below the lowest of its reference "
                f"students (dev gap {nn['gap'] * 100:.1f} points). Verdict: {e(h16['verdict'])}.")
    return figure("Behaviour by shared mistakes, per family",
                  "Fresh students (5 distilled, 5 independent per family), rules chosen on dev students only",
                  t, note=note)


# --- index --------------------------------------------------------------------------------------

def index(V: dict) -> str:
    rules = V["rules"]["t"]
    hs = hsort(rules["hypotheses"])
    H = {h["id"]: h for h in hs}
    mid = {h["id"]: h["verdict"] for h in V["rules"]["midpoint"]["hypotheses"]}
    T = V["thresholds"]
    t = T["plain"]["t"]
    A = rules["audit"]
    RF = (V.get("reader_fixed") or {}).get("t") or {}
    FA = RF.get("audit")
    h1n, h2n, h3n = H["H1"]["numbers"], H["H2"]["numbers"], H["H3"]["numbers"]
    snap = json.loads((ROOT / "results" / "e2_sample.json").read_text(encoding="utf-8")).get("snapshot", "")
    snap_s = str(snap)[:10]
    n_hyp = len(hs)
    n_sup = sum(h["verdict"] == "supported" for h in hs)
    n_pend = sum(h["verdict"] in ("pending", "not run") for h in hs)

    hero = f"""
<section class="hero">
  <div class="hero-text">
    <div class="eyebrow"><span class="dot"></span><span>Hub lineage audit · snapshot {e(snap_s)}</span></div>
    <h1>Do models come from the model they say they do?</h1>
    <p class="lede">A model card's <code>base_model</code> is how licence checks, procurement reviews and the Hub itself
    learn what a model is built on. Nobody checks it. Pedigree checks it from the weights, on a frozen random sample of
    {A['n_models']} Hub models, with a verifier measured on ground truth first.</p>
    <div class="btns">
      <a class="btn btn-primary" href="check.html">Check a model</a>
      <a class="btn" href="#audit">Read the audit</a>
      <a class="btn btn-ghost" href="#reproduce"><span class="acc">&gt;_</span> Reproduce it</a>
    </div>
  </div>
  <figure class="hero-fig">
    <figcaption class="hero-cap"><span>Weights show an earlier public parent · no base declared</span>
    <span class="acc">H2</span></figcaption>
    <div class="hero-num"><span class="big">{pc(h2n['share'], 1)}</span>
    <span class="hero-numt">{h2n['k']} of {h2n['n']} models whose weights show a public parent do not declare it
    (95% interval {pc(h2n['wilson'][0], 1)}-{pc(h2n['wilson'][1], 1)}).</span></div>
    {fig_h2(H['H2'])}
    <p class="hero-foot">Commoner among the most downloaded models than in the long tail. The pre-registered bar was
    30%, so H2 is <b>not supported</b>: a measurement of the Hub, reported as it came out.</p>
  </figure>
</section>
<section class="kpis" aria-label="Key numbers">
  <div class="kpi"><span class="dot"></span><span><b>{A['n_models']}</b> Hub models audited, every one accounted for</span></div>
  <div class="kpi"><span class="dot"></span><span><b>{h1n['tpr'][0]}/{h1n['tpr'][1]}</b> fresh derivatives verified ·
  <b>{h1n['fpr'][0]}/{h1n['fpr'][1]}</b> independents flagged</span></div>
  <div class="kpi"><span class="dot dot-o"></span><span><b>{h3n['k']} of {h3n['n']}</b> comparable declared bases
  contradicted by the weights</span></div>
</section>"""

    problem = f"""
<section class="sec">
{sec_head("01", "The problem", "A declared base is a claim nobody checks", "problem")}
<div class="cols2">
  <div class="prose">
    <p>A fine-tune of a model with a restrictive licence can be re-uploaded under a permissive one with no parent
    named; a card can name a base the model does not descend from. The Hub shows the claim as fact.</p>
    <p>Weight-based lineage methods already exist and are strong (AWM, ICLR 2026; HuRef, NeurIPS 2024; MoTHer,
    ICLR 2025). Pedigree does not claim a new detector. It uses that idea as an instrument, measures the instrument on
    ground truth first, and then measures the Hub, with every hypothesis and decision rule committed in
    <a href="{REPO_URL}/blob/main/PLAN.md">PLAN.md</a> before its data existed.</p>
  </div>
  <ol class="steps">
    <li><span class="step-n">1</span><div><b>Read, don't download.</b> File headers and the first values of up to 24
    tensors, by HTTP range request: a 70B model costs megabytes.</div></li>
    <li><span class="step-n">2</span><div><b>Align.</b> Pair tensors by name and shape (GGUF, MLX, PyTorch and
    safetensors names mapped to one form).</div></li>
    <li><span class="step-n">3</span><div><b>Decide.</b> Median cosine over aligned tensors against a threshold fixed
    on development negatives only: {t:.4f}.</div></li>
  </ol>
</div>
</section>"""

    h4, h5, h6 = H["H4"]["numbers"], H["H5"]["numbers"], H["H6"]["numbers"]
    instrument = f"""
<section class="sec">
{sec_head("02", "Ground truth first", "The instrument, measured before it measures the Hub", "instrument")}
<p class="lead">Thresholds come from development pairs only: the highest unrelated score plus five standard deviations,
wide because the audit makes thousands of comparisons. Then each was judged once on pairs it never saw, across six
families (two compared blind, after the rule was written).</p>
{fig_strip(H['H1'], t)}
<div class="tiles">
  <div class="tile"><span class="tile-l">Permutation attack (H4)</span><span class="tile-n">{h4['hidden'][0]}/{h4['hidden'][1]}
  &rarr; {h4['restored'][0]}/{h4['restored'][1]}</span><p>A function-preserving permutation of the residual stream hides
  lineage from the plain score; weight matching before comparing restores it.</p>{pill(H['H4']['verdict'])}</div>
  <div class="tile"><span class="tile-l">Quantised and distilled (H5)</span><span class="tile-n">{h5['q4'][0]}/{h5['q4'][1]}
  · {h5['students_related'][0]}/{h5['students_related'][1]}</span><p>4-bit quantised derivatives still verify; students
  distilled from a random start share no weights, so weights cannot see them.</p>{pill(H['H5']['verdict'])}</div>
  <div class="tile"><span class="tile-l">Samples vs whole files (H6)</span><span class="tile-n">{h6['e1'][0]}/{h6['e1'][1]}
  · {h6['e2'][0]}/{h6['e2'][1]}</span><p>The sampled verdict equals the verdict over every aligned tensor read in full,
  on fresh ground-truth pairs and on Hub pairs.</p>{pill(H['H6']['verdict'])}</div>
</div>
{fig_curve(H['H11'], t)}
<p class="fig-after">Lineage stays visible across the whole run: the final model is related to its own checkpoints from
step 10,000 on in all {H['H11']['numbers'].get('runs')} runs, and to no checkpoint of another run (H11, {e(H['H11']['verdict'])}). Shared
initialisation, not shared data, is what the score detects (H10: {e(H['H10']['evidence'])}; {e(H['H10']['verdict'])}).</p>
</section>"""

    rf_line = ""
    if RF:
        ch = [k for k, x in RF["verdicts"].items() if x["changes"]]
        rf_line = (f"After four reader fixes (MLX bit packing, GGUF names in multimodal checkpoints, pre-2020 PyTorch "
                   f"files, GGUF tensor types decoded from their public sources) the same 600 models were scored again: "
                   + ("no verdict changes" if not ch else f"{', '.join(ch)} change") + " ("
                   + "; ".join(f"{k} {e(x['evidence'])}" for k, x in RF["verdicts"].items()) + ").")
    audit = f"""
<section class="sec">
{sec_head("03", "The Hub audit (E2)", "What 600 random Hub models declare, and what their weights show", "audit")}
{fig_audit(A, FA)}
<div class="cols2 findings">
  <div class="card"><div class="card-top"><span class="tag">H2</span>{pill(H['H2']['verdict'])}</div>
  <h3>Undeclared lineage is common, but under the bar</h3>
  <p>{h2n['k']} of {h2n['n']} models whose weights show an earlier public relative declare no base
  ({pc(h2n['share'], 1)}; weighted by tier size {pc(h2n['tier_weighted'], 1)}). The bar was 30%. A finding about the
  Hub, not a fault of the instrument.</p></div>
  <div class="card"><div class="card-top"><span class="tag">H3</span>{pill(H['H3']['verdict'])}</div>
  <h3>When a card names a base, it is almost always right</h3>
  <p>{h3n['k']} of {h3n['n']} comparable declared bases are contradicted ({pc(h3n['share'], 1)}, interval
  {pc(h3n['wilson'][0], 1)}-{pc(h3n['wilson'][1], 1)}); in each the weights show no lineage at all. The bar was 5%:
  the Hub's declarations are more honest than the hypothesis guessed.</p></div>
</div>
<p class="fig-after">{rf_line}</p>
</section>"""

    h13 = H.get("H13a")
    pool = f"""
<section class="sec">
{sec_head("04", "Finding a parent nobody declared", "The candidate pool, rebuilt from its own misses", "pool")}
<p class="lead">To find an undeclared parent, each model is compared with a pool of candidates. E2's pool held the
declared parent for only {H['H7']['numbers']['b'][0]} of {H['H7']['numbers']['b'][1]} Hub models (H7, not supported).
Every miss had a recorded cause; pool v2 answers each without reading the model's own declaration, and was judged once
on {h13['numbers']['n']} fresh models drawn before any was read.</p>
{fig_pool(h13)}
<p class="fig-after">H13a {e(h13['verdict'])}; H13b {e(H['H13b']['verdict'])} (exact McNemar p =
{h13['numbers']['mcnemar_p']:.1e}), at a mean of {h13['numbers']['mean_pool_size']:.0f} candidates per model.</p>
</section>"""

    beh = rules.get("behaviour") or {}
    na = beh.get("not_applicable", {})
    behaviour = f"""
<section class="sec">
{sec_head("05", "The second instrument", "Behaviour, where weights cannot be compared", "behaviour")}
<p class="lead">A distilled student shares no weights with its teacher, so behaviour was pre-registered for those cases.
Four hypotheses tested it on fresh students; none passed (H9, H12, H14, H14b). The data shows why: within each training
batch the vision-CNN students do separate, but the whole fresh batch sits below the threshold chosen on the dev batch,
so an absolute threshold does not carry from one batch to the next. H16 compares each student with reference students
trained the same way instead; H16b trains text-encoder and vision-transformer students long enough to learn the task.</p>
{fig_behaviour(H)}
<p class="fig-after">On the Hub, behaviour applied to none of the {beh.get('pairs', 0)} declared pairs weights could not
compare ({"; ".join(f"{v} with {e(k.replace('model above', 'a model above'))}" for k, v in na.items())}). It runs on larger models (H18) only if H16 or H16b passes
first: behaviour on a 7B model proves nothing until the behaviour check itself has passed on ground truth.</p>
</section>"""

    rows = []
    for h in hs:
        fol = FOLLOWED_BY.get(h["id"]) if h["verdict"] == "not supported" else None
        meta = []
        if fol:
            meta.append(f"followed up by {e(fol)}" if fol.startswith("H") else e(fol))
        if h.get("follows"):
            meta.append(f"follows {e(h['follows'])}" if h["follows"].startswith("H") else f"follows: the {e(h['follows'])}")
        if mid.get(h["id"]) and mid[h["id"]] != h["verdict"]:
            meta.append(f"midpoint rule: {e(mid[h['id']])}")
        m = f'<span class="h-meta">{" · ".join(meta)}</span>' if meta else ""
        rows.append(f'<div class="h-row"><div class="h-main"><span class="tag">{e(h["id"])}</span><div>'
                    f'<p class="h-st">{e(h["statement"])}</p><p class="h-ev">{e(h["evidence"])}</p>{m}</div></div>'
                    f'<div class="h-v">{pill(h["verdict"])}</div></div>')
    hyps = f"""
<section class="sec">
{sec_head("06", "Pre-registered", f"{n_hyp} hypotheses, decided by rules fixed before the runs", "hypotheses")}
<p class="lead">{n_sup} supported, {n_hyp - n_sup - n_pend} not supported, {n_pend} running or waiting. A verdict is
never reworded; a failed one is followed by a new hypothesis, designed from the failure's cause and judged on fresh
data.</p>
<div class="h-list">{"".join(rows)}</div>
</section>"""

    # every caveat, measured
    U = RF.get("unverifiable") or {}
    rr = json.loads((ROOT / "results" / "e2_rereads.json").read_text(encoding="utf-8"))
    d8 = H["H8"]["numbers"]
    h15 = H.get("H15", {})
    dev15 = (h15.get("numbers") or {}).get("dev") or {}
    f15 = (h15.get("numbers") or {}).get("fresh")
    n17 = H["H17a"].get("numbers") or {}
    flips = [k for k, v in mid.items() if v != {h["id"]: h["verdict"] for h in hs}.get(k)]
    big = na.get("model above 2B parameters", 0)
    jw = [w for w in d8["d1_wrong"] if w[2] and abs(w[3]) <= 30]
    joint, joint_days = len(jw), max((abs(w[3]) for w in jw), default=0)
    cnn = H["H14b"]["numbers"]["per_family"]["vision_cnn"]
    STEPS = int(re.search(r"^STEPS = (?:int\(os\.environ\.get\(\"PEDIGREE_STEPS\", \")?(\d+)", (ROOT / "experiments" / "distill.py").read_text(encoding="utf-8"),
                          re.M).group(1))
    cav = [
        ("Declared bases weights cannot compare",
         f"{U.get('models', 0)} models ({U.get('pairs', 0)} pairs) after the reader fixes: "
         + "; ".join(f"{v} {e(k)}" for k, v in (U.get("reasons") or {}).items()) + ".",
         f"H17 aligns GGUF tensors by block and unique shape when names fail. Against name alignment on fresh pairs "
         f"(H17a, {e(H['H17a']['verdict'])}) it agreed on {n17.get('agree', [0, 0])[0]}/{n17.get('agree', [0, 0])[1]}, "
         f"gave the opposite verdict on {n17.get('opposite', 0)} and aligned nothing on {n17.get('abstained', 0)}; "
         f"H17b: {e(H['H17b']['verdict'])}. GGUF types newer than the gguf package are decoded only from their public "
         f"sources; gated bases are read through a byte-identical public copy."),
        ("The candidate pool",
         f"E2's pool missed the parent of {H['H7']['numbers']['b'][1] - H['H7']['numbers']['b'][0]} of "
         f"{H['H7']['numbers']['b'][1]} Hub models; every miss has a recorded cause (results/h7_misses.json).",
         f"Pool v2 holds {h13['numbers']['v2'][0]} of {h13['numbers']['v2'][1]} fresh parents (H13a {e(h13['verdict'])})."),
        ("Direction: which model is the parent",
         f"Hub dates are right on {d8['d1'][0]}/{d8['d1'][1]} pairs; {joint} of the {len(d8['d1_wrong'])} misses are "
         f"one organisation's joint releases, where the derived repo was created at most {joint_days:g} days before "
         f"its base. Kurtosis is right on "
         f"{d8['d2'][0]}/{d8['d2'][1]} decided pairs and leaves {d8['d2'][2] - d8['d2'][1]}/{d8['d2'][2]} undecided "
         f"(short fine-tunes and LoRA merges barely move it). H8: {e(H['H8']['verdict'])}.",
         (f"D3 (dates across owners, names within one release, kurtosis last): right on "
          f"{dev15.get('d3', [0, 0])[0]}/{dev15.get('d3', [0, 0])[1]} decided dev pairs; on the fresh sample (H15, "
          f"{e(h15.get('verdict', 'pending'))}) {f15['d3'][0]}/{f15['d3'][1]} decided, no better than dates alone "
          f"({f15['d1'][0]}/{f15['d1'][1]}); kurtosis is right on only {f15['d2'][0]}/{f15['d2'][1]} fresh Hub pairs, "
          f"the reverse of its assumed direction. The audit keeps dates." if f15 else
          f"D3 (dates across owners, names within one release, kurtosis last) is right on "
          f"{dev15.get('d3', [0, 0])[0]}/{dev15.get('d3', [0, 0])[1]} decided dev pairs; H15 judges it on a fresh "
          f"sample ({e(h15.get('verdict', 'pending'))}).")),
        ("Behaviour as evidence of lineage",
         f"Absolute thresholds do not transfer between training batches (vision CNN: fresh TPR {cnn['tpr'][0]}/"
         f"{cnn['tpr'][1]} at FPR {cnn['fpr'][0]}/{cnn['fpr'][1]}); text-encoder and vision-transformer students never "
         f"learned the task in {STEPS:,} steps.",
         f"H16: reference-calibrated rule, fixed on dev ({e(H['H16']['verdict'])}); H16b: students trained to a "
         f"competence floor ({e(H['H16b']['verdict'])})."),
        ("Behaviour above 2B parameters",
         f"{big} of the declared pairs weights could not compare have a model above 2B parameters; behaviour was not "
         "run on them because no behaviour rule has passed on ground truth.",
         f"H18 runs on text models up to 14B only if H16 or H16b is supported ({e(H['H18']['verdict'])}); most of these "
         "pairs are GGUF files that H17 may make comparable by weights instead."),
        ("The threshold rule",
         f"Under the pre-registered sensitivity rule (midpoint between the classes) {len(flips)} verdicts change: "
         f"{', '.join(flips)}.",
         "The published rule is max + 5 SD; both columns are in docs/RESULTS.md."),
        ("Hub drift during the audit",
         f"Pass 2 re-read {rr['reread']} models independently: {rr['identical']} identical; the others changed in "
         f"their commit, declaration or pool.",
         "Each model keeps its first reading (results/e2_rereads.json)."),
        ("Sampled tensors vs whole files",
         f"Whole-file and sampled verdicts agree on {h6['e1'][0]}/{h6['e1'][1]} fresh pairs and "
         f"{h6['e2'][0]}/{h6['e2'][1]} Hub pairs; largest score difference {h6['max_abs_difference']:.3f}.",
         "The threshold sits far from both classes, so that difference never moves a verdict."),
    ]
    cards = "".join(f'<div class="cav"><h3>{e(a)}</h3><p>{b}</p><p class="remedy"><span>&rarr;</span> {c}</p></div>'
                    for a, b, c in cav)
    caveats = f"""
<section class="sec">
{sec_head("07", "Scope", "Every caveat, measured", "caveats")}
<p class="lead">Each weakness found along the way is either fixed and re-tested on fresh data, or measured, with what
is being done about it.</p>
<div class="cavs">{cards}</div>
</section>"""

    repro = f"""
<section class="sec sec-last">
{sec_head("08", "Reproduce", "Every number, from the ledger", "reproduce")}
<div class="term">
  <div class="term-bar"><span class="tdots"><i></i><i></i><i></i></span><span class="term-t">from a clone of the repo</span>
  <button type="button" class="copy" data-copy="#repro">Copy</button></div>
<pre id="repro"><span class="c"># install, run the tests</span>
<span class="p">$ </span>pip install -e ".[dev,experiments]" &amp;&amp; pytest
<span class="c"># check one model against its declared base_model, or a base you name</span>
<span class="p">$ </span>pedigree verify hf:org/model
<span class="p">$ </span>pedigree verify hf:org/model --base hf:org/base
<span class="c"># every verdict from results/ledger.jsonl, then this site</span>
<span class="p">$ </span>python scripts/analyse.py
<span class="p">$ </span>python scripts/build_site.py</pre>
</div>
<p class="fine">The heavy runs (ground-truth fine-tunes on GPU, the 600-model audit on CPU) ran on Kaggle with
<code>scripts/kaggle_run.py</code>, which uploads only committed code and stamps every result with its commit. CI reruns
the analysis and the site build and fails on any difference.</p>
</section>"""
    return page("Pedigree — do models come from where they say?",
                "Pedigree checks a model's declared base_model against its weights, and audits a frozen random sample "
                "of 600 Hugging Face models for missing, unverifiable and contradicted lineage.",
                hero + problem + instrument + audit + pool + behaviour + hyps + caveats + repro)


# --- checker ------------------------------------------------------------------------------------

def check(T: dict) -> str:
    js = (SITE / "checker.js").read_text(encoding="utf-8")
    n_t = int(re.search(r"N_TENSORS = (\d+)", js).group(1))
    n_s = int(re.search(r"SAMPLE = (\d+)", js).group(1))
    body = f"""
<section class="hero hero-check">
  <div class="hero-text">
    <div class="eyebrow"><span class="dot"></span><span>Browser checker · nothing uploaded</span></div>
    <h1>Check a model's declared base</h1>
    <p class="lede">Compares a model's weights with its declared <code>base_model</code>, or a base you name, in this
    page. It reads file headers and the first {n_s:,} values of up to {n_t} matched tensors from huggingface.co; nothing
    is sent anywhere else.</p>
  </div>
</section>
<div class="check-grid">
  <section class="panel">
    <form id="f" class="form">
      <label>Model<input id="model" placeholder="org/model" autocomplete="off" spellcheck="false" required></label>
      <label>Base <span class="muted">(empty: the declared base_model)</span>
      <input id="base" placeholder="org/base" autocomplete="off" spellcheck="false"></label>
      <div class="btns"><button class="btn btn-primary" id="go">Compare</button>
      <button type="button" class="btn" id="ex" data-model="microsoft/DialoGPT-small"
      data-base="openai-community/gpt2">Try an example</button></div>
    </form>
    <p id="log" class="log" aria-live="polite"></p>
    <div id="out" class="out" aria-live="polite"></div>
  </section>
  <aside class="panel side">
    <h3>How to read the verdict</h3>
    <ul class="kv">
      <li><span class="pill pill-ok">RELATED</span> median cosine above {T['plain']} (the published threshold)</li>
      <li><span class="pill pill-un">UNRELATED</span> at or below it, with at least 8 tensors compared</li>
      <li><span class="pill pill-wait">NOT COMPARABLE</span> fewer than 8 shared tensors: never read as unrelated</li>
    </ul>
    <p class="fine">Safetensors models only. For PyTorch, GGUF and MLX files, gated models and the
    permutation-invariant check, use the CLI: <code>pedigree verify hf:org/model</code>.</p>
    <p class="fine"><a href="index.html#instrument">How the threshold was chosen and tested</a></p>
  </aside>
</div>"""
    script = f"""<script>globalThis.THRESHOLDS = {json.dumps({k: v for k, v in T.items() if k in ("plain", "invariant")})};</script>
<script type="module">
import {{ openModel, compare, setWaitLog }} from "./checker.js";
const $ = id => document.getElementById(id), log = t => $("log").textContent = t;
setWaitLog(log);
$("ex").addEventListener("click", () => {{ $("model").value = $("ex").dataset.model; $("base").value = $("ex").dataset.base; $("f").requestSubmit(); }});
function card(base, r) {{
  const cls = r.verdict === "related" ? "ok" : r.verdict === "unrelated" ? "un" : "wait";
  const d = document.createElement("div"); d.className = "res";
  const top = document.createElement("div"); top.className = "res-top";
  const b = document.createElement("span"); b.className = "res-base"; b.textContent = "vs " + base;
  const p = document.createElement("span"); p.className = "pill pill-" + cls; p.textContent = r.verdict.toUpperCase() + (r.partial ? " (PARTIAL)" : "");
  top.append(b, p);
  const n = document.createElement("div"); n.className = "res-num"; n.textContent = r.plain === null ? "n/a" : r.plain.toFixed(4);
  const s = document.createElement("p"); s.className = "res-sub";
  s.textContent = `median cosine over ${{r.compared}} tensors (${{r.matched}} matched by name and shape) · threshold ${{THRESHOLDS.plain}}`;
  const g = document.createElement("div"); g.className = "gauge";
  const pos = v => Math.max(0, Math.min(100, (v + 0.1) / 1.1 * 100));
  const mk = document.createElement("i"); mk.className = "g-mk g-" + cls; if (r.plain !== null) mk.style.left = pos(r.plain) + "%"; else mk.hidden = true;
  const th = document.createElement("i"); th.className = "g-th"; th.style.left = pos(THRESHOLDS.plain) + "%";
  const l0 = document.createElement("span"); l0.className = "g-l"; l0.textContent = "-0.1";
  const l1 = document.createElement("span"); l1.className = "g-r"; l1.textContent = "1.0";
  g.append(th, mk, l0, l1);
  d.append(top, n, s, g); return d;
}}
$("f").addEventListener("submit", async ev => {{
  ev.preventDefault(); $("out").textContent = ""; $("go").disabled = true;
  try {{
    const m = await openModel($("model").value.trim(), log);
    const bases = $("base").value.trim() ? [$("base").value.trim()] : m.declared;
    if (!bases.length) {{ log(""); $("out").textContent = "This model card declares no base_model. Name a base to compare with."; return; }}
    for (const b of bases) $("out").append(card(b, await compare(m, await openModel(b, log), log)));
    log("done");
  }} catch (err) {{ log(""); $("out").textContent = "Could not compare: " + err.message; }}
  finally {{ $("go").disabled = false; }}
  const q = new URLSearchParams({{ model: $("model").value.trim(), base: $("base").value.trim() }});
  if (!$("base").value.trim()) q.delete("base");
  history.replaceState(null, "", "?" + q);
}});
// a shareable link: check.html?model=org/model&base=org/base runs the comparison on load
const params = new URLSearchParams(location.search);
if (params.get("model")) {{ $("model").value = params.get("model"); $("base").value = params.get("base") || ""; $("f").requestSubmit(); }}
</script>"""
    return page("Pedigree checker — verify a declared base model",
                "Compare a Hugging Face model's weights with its declared base_model in your browser.", body, script)


CSS = r"""
:root{--surface:#0f131c;--lowest:#0a0e17;--strip:#121622;--panel:#151924;--panel-hover:#1b202e;--line:#272d3f;--hair:#23293b;
--fg:#dfe2ef;--fg2:#bdc8d1;--muted:#87929a;--outline-v:#3e484f;--hi:#262a34;--sec:#bfc6dc;
--acc:#9085e9;--orange:#d95926;--mono:"JetBrains Mono",ui-monospace,SFMono-Regular,Menlo,monospace;
--sans:"Hanken Grotesk",ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;color-scheme:dark}
*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:72px}
body{margin:0;background:var(--surface);color:var(--fg);font:15px/1.6 var(--sans);-webkit-font-smoothing:antialiased}
a{color:var(--fg);text-decoration:underline;text-decoration-color:var(--outline-v);text-underline-offset:3px}
a:hover{text-decoration-color:var(--acc)}
code{font-family:var(--mono);font-size:.86em;color:var(--fg);background:var(--lowest);border:1px solid var(--hair);padding:1px 5px;border-radius:2px}
.skip{position:absolute;left:-999px}.skip:focus{left:8px;top:8px;z-index:99;background:var(--panel);padding:8px}
.wrap{max-width:1100px;margin:0 auto;padding:0 16px}@media(min-width:768px){.wrap{padding:0 32px}}
.top{position:fixed;top:0;left:0;right:0;z-index:50;background:rgba(15,19,28,.9);backdrop-filter:blur(10px);border-bottom:1px solid var(--outline-v)}
.top-in{height:56px;display:flex;align-items:center;justify-content:space-between;gap:16px}
.logo{display:flex;align-items:center;gap:8px;font:600 12px/1 var(--mono);letter-spacing:.12em;text-decoration:none;color:var(--fg)}
.dot{width:7px;height:7px;border-radius:50%;background:var(--acc);display:inline-block;flex:none}.dot-muted{background:var(--muted)}.dot-o{background:var(--orange)}
.nav{display:none;gap:22px}.nav a{font:12px var(--mono);color:var(--fg2);text-decoration:none}.nav a:hover{color:var(--fg)}
@media(min-width:900px){.nav{display:flex}}
.btn{display:inline-flex;align-items:center;justify-content:center;gap:6px;height:38px;padding:0 16px;border:1px solid #353c52;
background:transparent;color:var(--fg);font:500 12.5px var(--mono);text-decoration:none;border-radius:2px;cursor:pointer;transition:background .15s}
.btn:hover{background:var(--panel-hover)}.btn-primary{background:var(--acc);border-color:var(--acc);color:#0f131c;font-weight:600}
.btn-primary:hover{background:#a39af0}.btn-ghost{border-color:transparent;color:var(--fg2)}.btn-sm{height:32px;padding:0 12px}
.btn:disabled{opacity:.5;cursor:wait}.acc{color:var(--acc)}.muted{color:var(--muted)}
main{padding-top:56px}
.hero{display:grid;grid-template-columns:1fr;gap:32px;padding:48px 0;border-bottom:1px solid var(--hair);align-items:start}
@media(min-width:1024px){.hero{grid-template-columns:7fr 5fr;padding:64px 0 56px}}
.hero-text{display:flex;flex-direction:column;gap:20px;max-width:600px}
.eyebrow{display:inline-flex;align-items:center;gap:8px;font:500 11px var(--mono);letter-spacing:.08em;text-transform:uppercase;color:var(--acc)}
h1{font:600 34px/1.14 var(--sans);letter-spacing:-.02em;margin:0}@media(min-width:768px){h1{font-size:42px}}
.lede{font-size:17px;line-height:1.65;color:var(--fg2);margin:0}.btns{display:flex;flex-wrap:wrap;gap:8px}
.hero-fig{margin:0;background:var(--panel);border:1px solid var(--line);border-radius:4px;padding:16px;display:flex;flex-direction:column;gap:12px}
.hero-cap{display:flex;justify-content:space-between;gap:8px;font:500 11px var(--mono);letter-spacing:.06em;text-transform:uppercase;
color:var(--muted);padding-bottom:10px;border-bottom:1px solid var(--hair)}
.hero-num{display:flex;flex-direction:column;gap:4px}.big{font:700 52px/1 var(--mono);letter-spacing:-.03em;color:#fff}
.hero-numt{font-size:13.5px;color:var(--fg2)}.hero-foot{font-size:13px;color:var(--fg2);margin:0;padding-top:10px;border-top:1px solid var(--hair)}
.hero-foot b{color:var(--fg)}
.kpis{display:grid;grid-template-columns:1fr;background:var(--strip);border-bottom:1px solid var(--hair)}
@media(min-width:768px){.kpis{grid-template-columns:repeat(3,1fr)}.kpi+.kpi{border-left:1px solid var(--hair)}}
.kpi{display:flex;align-items:center;gap:10px;padding:14px 20px;font:13px var(--mono);color:var(--fg2)}.kpi b{color:var(--fg);font-weight:600}
.kpi+.kpi{border-top:1px solid var(--hair)}@media(min-width:768px){.kpi+.kpi{border-top:0}}
.sec{padding:56px 0;border-bottom:1px solid var(--hair);display:flex;flex-direction:column;gap:24px}.sec-last{border-bottom:0}
.sec-head{scroll-margin-top:72px}.sec-label{display:flex;align-items:center;gap:8px;font:500 11px var(--mono);letter-spacing:.08em;
text-transform:uppercase;color:var(--fg2)}.sec-num{color:var(--acc);font-weight:700}.sq{width:5px;height:5px;background:var(--outline-v)}
h2{font:600 26px/1.25 var(--sans);letter-spacing:-.015em;margin:6px 0 0}h3{font:600 16px/1.4 var(--sans);margin:0}
.lead{font-size:16px;color:var(--fg2);max-width:780px;margin:0}.prose p{color:var(--fg2);margin:0 0 12px;font-size:15.5px}
.cols2{display:grid;grid-template-columns:1fr;gap:16px}@media(min-width:900px){.cols2{grid-template-columns:1fr 1fr;gap:24px}}
.steps{list-style:none;margin:0;padding:0;display:flex;flex-direction:column;gap:10px}
.steps li{display:flex;gap:12px;background:var(--panel);border:1px solid var(--line);border-radius:4px;padding:14px;font-size:14px;color:var(--fg2)}
.steps b{color:var(--fg)}.step-n{font:700 12px var(--mono);color:var(--acc);background:rgba(144,133,233,.12);width:24px;height:24px;
display:flex;align-items:center;justify-content:center;border-radius:2px;flex:none}
.fig{margin:0;background:var(--panel);border:1px solid var(--line);border-radius:4px;padding:16px;display:flex;flex-direction:column;gap:14px}
@media(min-width:768px){.fig{padding:20px 24px}}
.fig-cap{display:flex;flex-direction:column;gap:3px;padding-bottom:12px;border-bottom:1px solid var(--hair)}
.fig-title{font:600 12.5px var(--mono);letter-spacing:.04em;text-transform:uppercase;color:var(--fg)}.fig-sub{font-size:13px;color:var(--muted)}
.fig-note{font-size:13px;color:var(--fg2);margin:0}.fig-after{font-size:14px;color:var(--fg2);margin:0;max-width:860px}
.legend{display:flex;flex-wrap:wrap;gap:6px 18px;font:12px var(--mono);color:var(--fg2)}.lg{display:inline-flex;align-items:center;gap:7px}
.sw{display:inline-block;width:10px;height:10px;flex:none}.sw-der{background:var(--acc);transform:rotate(45deg) scale(.85)}
.sw-ind{background:var(--orange);border-radius:50%}.sw-v{background:var(--acc)}.sw-g{background:var(--muted)}.sw-thr{width:14px;height:0;border-top:2px dashed var(--fg2)}
/* strip plot */
.strip{display:flex;flex-direction:column;gap:2px}
.strip-row{display:grid;grid-template-columns:1fr 18px 1fr;grid-template-areas:"fam fam fam" "l b r";column-gap:6px;align-items:center}
.strip-row>.strip-fam{grid-area:fam;font:12px var(--mono);color:var(--fg2);padding-top:6px}
.strip-row>:nth-child(2){grid-area:l}.strip-row>:nth-child(3){grid-area:b}.strip-row>:nth-child(4){grid-area:r}
.strip-hd>span:first-child,.strip-ax>span:first-child,.strip-axt>span:first-child{display:none}
@media(min-width:720px){.strip-row{grid-template-columns:150px 1fr 18px 1fr;grid-template-areas:"fam l b r"}
.strip-row>.strip-fam{padding-top:0}.strip-hd>span:first-child,.strip-ax>span:first-child,.strip-axt>span:first-child{display:block}}
.strip-p{position:relative;height:46px;background:var(--lowest);border:1px solid var(--hair)}
.strip-brk{text-align:center;font:12px var(--mono);color:var(--muted)}
.ax-t{font:11px var(--mono);color:var(--muted);text-transform:uppercase;letter-spacing:.05em;display:flex;justify-content:space-between;gap:6px}
.thr-lab{font-weight:500;color:var(--fg2);text-transform:none;letter-spacing:0}
.thr{position:absolute;top:-2px;bottom:-2px;border-left:2px dashed var(--fg2);opacity:.8}
.pt{position:absolute;width:9px;height:9px;margin:-4.5px 0 0 -4.5px;box-shadow:0 0 0 2px var(--lowest);cursor:pointer}
.pt::after{content:"";position:absolute;inset:-6px}.pt:focus-visible{outline:2px solid #fff;outline-offset:2px}
.pt-der{background:var(--acc);transform:rotate(45deg)}.pt-ind{background:var(--orange);border-radius:50%}
.ticks{position:relative;height:16px;font:10.5px var(--mono);color:var(--muted)}.ticks span{position:absolute;transform:translateX(-50%);white-space:nowrap}
.ticks span:last-child{transform:translateX(-90%)}.strip-axt{font:10.5px var(--mono);color:var(--muted)}.strip-axt>span:nth-child(2),.strip-axt>span:nth-child(4){text-align:center}
/* range-band curve */
.curve{display:grid;grid-template-columns:36px 1fr;grid-template-rows:240px 18px 18px}
.curve-y{position:relative;font:10.5px var(--mono);color:var(--muted)}.curve-y span{position:absolute;right:6px;transform:translateY(-50%)}
.curve-plot{position:relative;border-left:1px solid var(--line);border-bottom:1px solid var(--line)}
.curve-svg{position:absolute;inset:0;width:100%;height:100%}.grid{stroke:var(--hair);stroke-width:1}
.own-fill{fill:var(--acc);fill-opacity:.22}.own-line{fill:none;stroke:var(--acc);stroke-width:2}
.oth-fill{fill:var(--orange);fill-opacity:.25}.oth-line{fill:none;stroke:var(--orange);stroke-width:2}
.thr-line{stroke:var(--fg2);stroke-width:1.5;stroke-dasharray:5 4}
.col-hit{position:absolute;top:0;bottom:0;width:36px;margin-left:-18px;cursor:pointer}
.col-hit:hover,.col-hit:focus-visible{background:rgba(144,133,233,.08);outline:none}
.mk{position:absolute;left:50%;width:9px;height:9px;margin:-4.5px 0 0 -4.5px;background:var(--acc);transform:rotate(45deg);box-shadow:0 0 0 2px var(--panel)}
.dlab{position:absolute;font:11px var(--mono);color:var(--fg2);transform:translateX(-50%);white-space:nowrap;pointer-events:none}
.curve-x{grid-column:2;position:relative;font:10.5px var(--mono);color:var(--muted)}.curve-x span{position:absolute;top:3px;transform:translateX(-50%)}
.curve-xt{grid-column:2;text-align:center;font:10.5px var(--mono);color:var(--muted)}
/* horizontal bars */
.hbars{position:relative;display:flex;flex-direction:column;gap:10px}
.hb-row{display:grid;grid-template-columns:64px 1fr 92px;gap:10px;align-items:center}
@media(min-width:600px){.hb-row{grid-template-columns:96px 1fr 110px}}
.hb-lab{font:12px var(--mono);color:var(--fg2)}.hb-track{position:relative;height:20px}
.hb{display:block;height:100%;border-radius:0 4px 4px 0;cursor:pointer;min-width:2px}.b-v{background:var(--acc)}.b-g{background:var(--muted)}
.hb:focus-visible{outline:2px solid #fff;outline-offset:2px}
.ci{position:absolute;top:50%;height:0;border-top:2px solid var(--fg);opacity:.8;pointer-events:none}
.ci::before,.ci::after{content:"";position:absolute;top:-6px;height:10px;border-left:2px solid var(--fg)}.ci::before{left:0}.ci::after{right:0}
.hb-val{font:600 12px var(--mono);color:var(--fg);white-space:nowrap;text-align:right}
.hb-val em{font-style:normal;font-weight:400;color:var(--muted);margin-left:4px}
.hb-refrow .hb-track{height:14px}.ref-lab{position:absolute;bottom:0;transform:translateX(-50%);font:10.5px var(--mono);color:var(--fg2);white-space:nowrap}
.ref{position:absolute;top:-5px;bottom:-5px;border-left:2px dashed var(--fg2);pointer-events:none}
.sub-h{font:500 11px var(--mono);text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin-top:4px}
/* grouped bars */
.gbars{display:flex;flex-direction:column;gap:14px}.gb-row{display:grid;grid-template-columns:1fr;gap:6px}
@media(min-width:720px){.gb-row{grid-template-columns:300px 1fr;gap:16px;align-items:center}}
.gb-lab{font-size:13.5px;color:var(--fg)}.gb-bars{display:flex;flex-direction:column;gap:3px}
.gb-line{display:flex;align-items:center;gap:8px;height:14px}.gb-line .hb{height:14px}.gb-v{font:600 12px var(--mono);color:var(--fg)}
/* tables */
.tbl-wrap{overflow-x:auto}.tbl{width:100%;border-collapse:collapse;font-size:13px;min-width:520px}
.tbl th{font:500 10.5px var(--mono);text-transform:uppercase;letter-spacing:.05em;color:#8c93a8;text-align:left;padding:9px 12px;
background:var(--panel-hover);border-bottom:1px solid var(--hair)}.tbl td{padding:9px 12px;border-bottom:1px solid var(--hair);color:var(--fg)}
.tbl .r{text-align:right;font-family:var(--mono);font-size:12.5px}.dt summary{cursor:pointer;font:12px var(--mono);color:var(--muted)}
.dt[open] summary{margin-bottom:10px}
.tiles{display:grid;grid-template-columns:1fr;gap:12px}@media(min-width:900px){.tiles{grid-template-columns:repeat(3,1fr)}}
.tile{background:var(--panel);border:1px solid var(--line);border-radius:4px;padding:16px;display:flex;flex-direction:column;gap:6px;align-items:flex-start}
.tile-l{font:500 11px var(--mono);text-transform:uppercase;letter-spacing:.06em;color:var(--muted)}
.tile-n{font:700 28px/1.1 var(--mono);color:#fff;letter-spacing:-.02em}.tile p{margin:0;font-size:13.5px;color:var(--fg2);flex:1}
.findings .card,.card{background:var(--panel);border:1px solid var(--line);border-radius:4px;padding:18px;display:flex;flex-direction:column;gap:8px}
.card p{margin:0;font-size:14px;color:var(--fg2)}.card-top{display:flex;justify-content:space-between;align-items:center}
.tag{font:700 12px var(--mono);color:var(--fg);background:var(--hi);padding:2px 0;min-width:46px;text-align:center;border-radius:2px;flex:none;align-self:flex-start}
.pill{display:inline-flex;align-items:center;padding:2px 9px;font:600 11px var(--mono);letter-spacing:.04em;border:1px solid;border-radius:2px;white-space:nowrap}
.pill-ok{background:rgba(144,133,233,.15);color:#b3abf3;border-color:rgba(144,133,233,.4)}
.pill-no{background:var(--hi);color:var(--sec);border-color:var(--outline-v)}
.pill-un{background:rgba(217,89,38,.14);color:#f08a5e;border-color:rgba(217,89,38,.4)}
.pill-wait{background:transparent;color:var(--muted);border-color:var(--outline-v);border-style:dashed}
.h-list{border:1px solid var(--line);border-radius:4px;background:var(--panel)}
.h-row{display:flex;flex-direction:column;gap:10px;padding:14px 16px;border-bottom:1px solid var(--hair)}.h-row:last-child{border-bottom:0}
.h-row:hover{background:var(--panel-hover)}@media(min-width:768px){.h-row{flex-direction:row;justify-content:space-between;align-items:center;gap:24px}}
.h-main{display:flex;gap:12px;max-width:820px}.h-st{margin:0;font-size:14.5px;color:var(--fg);font-weight:500}
.h-ev{margin:2px 0 0;font:12px/1.5 var(--mono);color:var(--fg2)}.h-meta{display:block;margin-top:4px;font:11px var(--mono);color:var(--acc)}
.h-v{flex:none}
.cavs{display:grid;grid-template-columns:1fr;gap:12px}@media(min-width:900px){.cavs{grid-template-columns:1fr 1fr}}
.cav{background:var(--panel);border:1px solid var(--line);border-radius:4px;padding:18px;display:flex;flex-direction:column;gap:8px}
.cav p{margin:0;font-size:14px;color:var(--fg2)}.cav .remedy{color:var(--fg);font-size:13.5px}.remedy span{color:var(--acc);font-family:var(--mono)}
.term{background:var(--panel);border:1px solid var(--line);border-radius:4px;overflow:hidden}
.term-bar{display:flex;align-items:center;gap:10px;padding:8px 12px;background:var(--strip);border-bottom:1px solid var(--hair)}
.tdots{display:flex;gap:6px}.tdots i{width:10px;height:10px;border-radius:50%;background:var(--line)}
.term-t{font:11px var(--mono);color:var(--muted);flex:1}.copy{font:12px var(--mono);color:var(--fg2);background:none;border:1px solid var(--line);padding:3px 10px;cursor:pointer;border-radius:2px}
.term pre{margin:0;padding:18px;font:13px/1.75 var(--mono);color:var(--fg);overflow-x:auto}.term .c{color:#8c93a8}.term .p{color:var(--acc);user-select:none}
.fine{font-size:13px;color:var(--muted);margin:0;max-width:860px}
.foot{background:var(--lowest);border-top:1px solid var(--outline-v);padding:32px 0;margin-top:0}
.foot-in{display:flex;flex-direction:column;gap:18px}@media(min-width:768px){.foot-in{flex-direction:row;justify-content:space-between;align-items:center}}
.foot p{font-size:13px;color:var(--fg2);margin:8px 0 0;max-width:520px}.foot-nav{display:flex;flex-wrap:wrap;gap:6px 18px}
.foot-nav a{font:12px var(--mono);color:var(--fg2);text-decoration:none;padding:4px 0}.foot-nav a:hover{color:var(--acc)}
.tip{position:fixed;z-index:80;display:none;pointer-events:none;background:#1f2433;border:1px solid var(--line);border-radius:4px;
padding:8px 10px;font:12px/1.5 var(--mono);color:var(--fg2);max-width:280px;box-shadow:0 8px 24px rgba(0,0,0,.45)}.tip .tip-h{color:var(--fg);font-weight:600}
/* checker */
.hero-check{padding-bottom:32px}.check-grid{display:grid;grid-template-columns:1fr;gap:16px;padding:32px 0 64px}
@media(min-width:900px){.check-grid{grid-template-columns:7fr 4fr}}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:4px;padding:20px;display:flex;flex-direction:column;gap:14px}
.form{display:flex;flex-direction:column;gap:14px}.form label{display:flex;flex-direction:column;gap:6px;font:500 12px var(--mono);color:var(--fg2)}
.form input{font:14px var(--mono);padding:10px 12px;background:var(--lowest);border:1px solid var(--line);color:var(--fg);border-radius:2px}
.form input:focus{outline:none;border-color:var(--acc)}.log{font:12px var(--mono);color:var(--muted);min-height:1.5em;margin:0}
.out{display:flex;flex-direction:column;gap:12px;font-size:14px;color:var(--fg2)}
.res{border:1px solid var(--line);background:var(--lowest);border-radius:4px;padding:16px;display:flex;flex-direction:column;gap:6px}
.res-top{display:flex;justify-content:space-between;gap:8px;align-items:center;flex-wrap:wrap}.res-base{font:13px var(--mono);color:var(--fg);word-break:break-all}
.res-num{font:700 40px/1.1 var(--mono);color:#fff;letter-spacing:-.02em}.res-sub{margin:0;font-size:12.5px;color:var(--muted)}
.gauge{position:relative;height:28px;margin-top:6px;border-top:1px solid var(--line)}
.g-th{position:absolute;top:-6px;height:14px;border-left:2px dashed var(--fg2)}.g-mk{position:absolute;top:-5px;width:10px;height:10px;margin-left:-5px;transform:rotate(45deg)}
.g-ok{background:var(--acc)}.g-un{background:var(--orange);border-radius:50%;transform:none}.g-wait{background:var(--muted)}
.g-l,.g-r{position:absolute;top:10px;font:10.5px var(--mono);color:var(--muted)}.g-l{left:0}.g-r{right:0}
.side h3{font-size:15px}.kv{list-style:none;margin:0;padding:0;display:flex;flex-direction:column;gap:10px;font-size:13px;color:var(--fg2)}
.kv li{display:flex;flex-direction:column;gap:4px;align-items:flex-start}
"""

JS = r"""
(() => {
  const tip = document.createElement("div"); tip.className = "tip"; tip.setAttribute("role", "tooltip"); document.body.append(tip);
  let cur = null;
  function show(el) {
    cur = el; tip.textContent = "";
    el.dataset.tip.split("|").forEach((line, i) => { const d = document.createElement("div"); d.textContent = line; if (!i) d.className = "tip-h"; tip.append(d); });
    tip.style.display = "block";
    const r = el.getBoundingClientRect(), w = tip.offsetWidth, h = tip.offsetHeight;
    let x = r.left + r.width / 2 - w / 2, y = r.top - h - 10;
    if (y < 64) y = r.bottom + 10;
    x = Math.max(8, Math.min(window.innerWidth - w - 8, x));
    tip.style.left = x + "px"; tip.style.top = y + "px";
  }
  function hide() { cur = null; tip.style.display = "none"; }
  document.querySelectorAll("[data-tip]").forEach(el => {
    el.addEventListener("mouseenter", () => show(el)); el.addEventListener("mouseleave", hide);
    el.addEventListener("focus", () => show(el)); el.addEventListener("blur", hide);
    el.addEventListener("click", ev => { ev.stopPropagation(); cur === el ? hide() : show(el); });
  });
  document.addEventListener("click", hide); window.addEventListener("scroll", hide, { passive: true });
  document.querySelectorAll("[data-copy]").forEach(b => b.addEventListener("click", async () => {
    const src = document.querySelector(b.dataset.copy);
    const text = src.innerText.split("\n").filter(l => !l.startsWith("#")).map(l => l.replace(/^\$ /, "")).join("\n");
    try { await navigator.clipboard.writeText(text); b.textContent = "Copied"; setTimeout(() => b.textContent = "Copy", 1500); } catch (e) {}
  }));
})();
"""


def main() -> None:
    V = json.loads((ROOT / "results" / "verdicts.json").read_text(encoding="utf-8"))
    T = json.loads((ROOT / "pedigree" / "thresholds.json").read_text(encoding="utf-8"))
    (SITE / "assets").mkdir(parents=True, exist_ok=True)
    (SITE / "assets" / "site.css").write_text(CSS.strip() + "\n", encoding="utf-8")
    (SITE / "assets" / "site.js").write_text(JS.strip() + "\n", encoding="utf-8")
    (SITE / "index.html").write_text(index(V), encoding="utf-8")
    (SITE / "check.html").write_text(check(T), encoding="utf-8")
    print("wrote site/index.html, site/check.html, site/assets/")


if __name__ == "__main__":
    main()
