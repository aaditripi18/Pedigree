"""Build Pedigree's site from results/verdicts.json (written by scripts/analyse.py).
No number on the site is typed by hand.

    python scripts/analyse.py && python scripts/build_site.py

Writes site/index.html (findings, hypotheses, method) and site/check.html (the browser
checker, site/checker.js, with the published thresholds).
"""
from __future__ import annotations

import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site"
REPO_URL = "https://github.com/aaditripi18/Pedigree"
e = html.escape

CSS = """
:root{--bg:#fbfaf7;--fg:#1d1d1b;--muted:#5d5b55;--line:#e3e0d8;--card:#fff;--accent:#1f5f8b;--ok:#2d6a3e;--no:#9a3b2e;--mono:ui-monospace,SFMono-Regular,Menlo,monospace}
@media (prefers-color-scheme:dark){:root{--bg:#141413;--fg:#ecebe6;--muted:#a3a19a;--line:#2e2d2a;--card:#1c1c1a;--accent:#7fb4dc;--ok:#7cc48f;--no:#e28c7c}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.6 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:920px;margin:0 auto;padding:40px 16px 80px}h1{font-size:2rem;line-height:1.2;margin:0 0 12px}
h2{font-size:1.25rem;margin:48px 0 12px;padding-top:16px;border-top:1px solid var(--line)}p,li{color:var(--fg)}
.lede{font-size:1.1rem;color:var(--muted)}nav{display:flex;gap:16px;margin-bottom:32px;font-size:.95rem}
a{color:var(--accent)}table{width:100%;border-collapse:collapse;font-size:.92rem;display:block;overflow-x:auto}
th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line);vertical-align:top}th{color:var(--muted);font-weight:600}
.num{font-variant-numeric:tabular-nums;text-align:right}.ok{color:var(--ok);font-weight:600}.no{color:var(--no);font-weight:600}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px;margin:20px 0}
.card{background:var(--card);border:1px solid var(--line);padding:16px}.card b{display:block;font-size:1.6rem;font-variant-numeric:tabular-nums}
code,pre{font-family:var(--mono);font-size:.88rem}pre{background:var(--card);border:1px solid var(--line);padding:12px;overflow-x:auto}
input{font:inherit;padding:8px;border:1px solid var(--line);background:var(--card);color:var(--fg);width:100%}
button{font:inherit;padding:8px 16px;border:1px solid var(--accent);background:var(--accent);color:#fff;cursor:pointer}
#log{color:var(--muted);font-size:.9rem;min-height:1.5em}.verdict{font-size:1.4rem;font-weight:700}
"""


def page(title: str, body: str, extra_head: str = "") -> str:
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{e(title)}</title>
<style>{CSS}</style>{extra_head}</head><body><main>
<nav><a href="index.html">Findings</a><a href="check.html">Check a model</a><a href="{REPO_URL}">Code</a></nav>
{body}</main></body></html>"""


def verdict_cell(v: str) -> str:
    cls = "ok" if v == "supported" else "no" if v == "not supported" else ""
    return f'<span class="{cls}">{e(v)}</span>'


def index(V: dict) -> str:
    rules = V["rules"]["t"]
    hs = sorted(rules["hypotheses"], key=lambda h: (int(h["id"][1:].rstrip("ab")), h["id"]))
    mid = {h["id"]: h["verdict"] for h in V["rules"]["midpoint"]["hypotheses"]}
    rows = "".join(f"<tr><td>{h['id']}</td><td>{e(h['statement'])}</td><td>{verdict_cell(h['verdict'])}</td>"
                   f"<td>{e(h['evidence'])}</td><td>{e(mid[h['id']])}</td></tr>" for h in hs)
    T = V["thresholds"]
    trows = "".join(f"<tr><td>{k}</td><td class='num'>{v['t']:.4f}</td><td class='num'>{v['max_negative']:.4f}</td>"
                    f"<td class='num'>{v['n_negatives']}</td></tr>" for k, v in T.items())
    A = rules.get("audit") or {}
    cards = ""
    if A:
        h2, h3 = A["h2"], A["h3"]
        cards = f"""<div class="cards">
<div class="card"><b>{h2['share'] * 100:.0f}%</b>of Hub models whose weights show a public parent do not declare it
({h2['k']} of {h2['n']}; 95% interval {h2['wilson'][0] * 100:.0f}–{h2['wilson'][1] * 100:.0f}%)</div>
<div class="card"><b>{h3['share'] * 100:.0f}%</b>of declared bases are contradicted by the weights
({h3['k']} of {h3['n']}; 95% interval {h3['wilson'][0] * 100:.0f}–{h3['wilson'][1] * 100:.0f}%)</div>
<div class="card"><b>{A['n_models']}</b>models audited, drawn at random from three download tiers of the Hub's
top 100,000; every one is accounted for</div></div>"""
    LABEL = [("declared, related", "Declares a base, and the weights confirm it"),
             ("declared, unrelated", "Declares a base the weights contradict"),
             ("declared, not comparable", "Declares a base that cannot be compared by weights"),
             ("no declaration, parent found", "Declares no base; an earlier related model was found"),
             ("no declaration, no parent found", "Declares no base; no related earlier model found"),
             ("not readable", "Weights not readable"), ("adapter", "An adapter only")]
    crows = "".join(f"<tr><td>{e(lbl)}</td><td class='num'>{sum(A['categories'].get(k, {}).values())}</td></tr>"
                    for k, lbl in LABEL) if A else ""
    B = rules.get("behaviour") or {}
    beh = ""
    if B:
        na = B["not_applicable"]
        beh = (f"<p>Weights cannot be compared for {B['pairs']} declared pairs. The behaviour instrument, pre-registered "
               f"for them, applied to {len(B['verified_by_behaviour']) + len(B['below_rule'])}: "
               + "; ".join(f"{n} {e(k)}" for k, n in sorted(na.items(), key=lambda x: -x[1]))
               + ". On ground truth it did not separate distilled students from independent ones (H9, H12), so it "
               "is reported as its own category and never moves H3.</p>")
    F = (V.get("reader_fixed") or {}).get("t")
    fixed = ""
    if F:
        ch = [k for k, x in F["verdicts"].items() if x["changes"]]
        fixed = (f"<p>{F['rescored']} comparisons in the audit ended in a file-reader error. After the readers were "
                 f"fixed they were scored again (E2r): all but {F['still_errors']} now score (those use GGUF tensor "
                 "types outside llama.cpp's published list), and "
                 + ("no verdict changes: " if not ch else f"the verdict of {', '.join(ch)} changes: ")
                 + "; ".join(f"{k} {e(x['evidence'])}" for k, x in F["verdicts"].items()) + ".</p>")
    h13 = {h["id"]: h for h in hs}.get("H13a")
    pool2 = ""
    if h13 and h13.get("numbers"):
        n = h13["numbers"]
        pool2 = (f"<p>Pool v2, pre-registered and judged on a fresh sample: it held the confirmed parent for "
                 f"{n['v2'][0]} of {n['v2'][1]} fresh models (E2's pool: {n['v1'][0]}), at a mean of "
                 f"{n['mean_pool_size']:.0f} candidates; H13a {e(h13['verdict'])}, H13b "
                 f"{e({h['id']: h for h in hs}['H13b']['verdict'])}.</p>")
    body = f"""<h1>Pedigree: do models come from where they say they do?</h1>
<p class="lede">A model card's <code>base_model</code> tells licence checks, procurement reviews and the Hub
what a model is built on. Nobody checks it. Pedigree checks it from the weights, on a frozen random sample of the
Hub, with a verifier measured on ground truth first.</p>
{cards}
<h2>The audit</h2>
<table><tr><th>Category</th><th>Models</th></tr>{crows}</table>
{beh}{fixed}
<h2>Hypotheses</h2>
<p>Every decision rule was committed in <a href="{REPO_URL}/blob/main/PLAN.md">PLAN.md</a> before the runs it
governs; verdicts are never reworded. The last column repeats each verdict under a second threshold rule.</p>
<table><tr><th></th><th>Hypothesis</th><th>Verdict</th><th>Evidence</th><th>Midpoint rule</th></tr>{rows}</table>
{"<h2>Finding a parent nobody declared</h2>" + pool2 if pool2 else ""}
<h2>Thresholds</h2>
<p>Fixed from development negatives only: the highest score of any unrelated pair plus five standard deviations,
wide because the audit makes thousands of comparisons.</p>
<table><tr><th>Score</th><th>Threshold</th><th>Highest dev negative</th><th>Dev negatives</th></tr>{trows}</table>
<h2>Check a model yourself</h2>
<p><a href="check.html">The browser checker</a> runs the same comparison in this page, reading only file headers
and small slices of tensors from the Hub. The CLI does the same for PyTorch, GGUF and MLX files too:</p>
<pre>pip install git+{REPO_URL}
pedigree verify hf:org/model --base hf:org/base</pre>"""
    return page("Pedigree", body)


def check(T: dict) -> str:
    body = """<h1>Check a model's declared base</h1>
<p class="lede">Compares a model's weights with its declared <code>base_model</code> (or a base you name) in this
page. It reads file headers and the first 8,192 values of up to 24 matched tensors from huggingface.co; nothing is
sent anywhere else.</p>
<form id="f"><p><label>Model<br><input id="model" placeholder="org/model" required></label></p>
<p><label>Base (leave empty to use the declared base_model)<br><input id="base" placeholder="org/base"></label></p>
<p><button>Compare</button></p></form>
<p id="log"></p><div id="out"></div>
<p>Safetensors models only; for PyTorch, GGUF and MLX files, gated models and the permutation-invariant check, use the
CLI.</p>"""
    script = f"""<script>globalThis.THRESHOLDS = {json.dumps({k: v for k, v in T.items() if k in ("plain", "invariant")})};</script>
<script type="module">
import {{ openModel, compare, setWaitLog }} from "./checker.js";
const $ = id => document.getElementById(id), log = t => $("log").textContent = t;
setWaitLog(log);
$("f").addEventListener("submit", async ev => {{
  ev.preventDefault(); $("out").textContent = "";
  try {{
    const m = await openModel($("model").value.trim(), log);
    const bases = $("base").value.trim() ? [$("base").value.trim()] : m.declared;
    if (!bases.length) {{ log(""); $("out").textContent = "This model card declares no base_model. Name a base to compare with."; return; }}
    for (const b of bases) {{
      const r = await compare(m, await openModel(b, log), log);
      const div = document.createElement("div");
      const v = document.createElement("p"); v.className = "verdict"; v.textContent = `vs ${{b}}: ${{r.verdict}}${{r.partial ? " (partial: few shared tensors)" : ""}}`;
      const d = document.createElement("p"); d.textContent = `median cosine ${{r.plain === null ? "n/a" : r.plain.toFixed(4)}} over ${{r.compared}} tensors (${{r.matched}} matched by name and shape); threshold ${{THRESHOLDS.plain}}`;
      div.append(v, d); $("out").append(div);
    }}
    log("done");
  }} catch (err) {{ log(""); $("out").textContent = "Could not compare: " + err.message; }}
}});
</script>"""
    return page("Pedigree checker", body, script)


def main() -> None:
    V = json.loads((ROOT / "results" / "verdicts.json").read_text(encoding="utf-8"))
    T = json.loads((ROOT / "pedigree" / "thresholds.json").read_text(encoding="utf-8"))
    SITE.mkdir(exist_ok=True)
    (SITE / "index.html").write_text(index(V), encoding="utf-8")
    (SITE / "check.html").write_text(check(T), encoding="utf-8")
    print("wrote site/index.html, site/check.html")


if __name__ == "__main__":
    main()
