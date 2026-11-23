"""H16c dev pilot (2026-11-23, design data before any H16c pre-registration): why H16b's text-encoder
students stalled at one output, and which training escapes it.

H16b's 22 students (RoBERTa-base teacher, fresh split) all predict the same token at every masked
position (accuracy 0.0548, agreement 0.0680, identical across seeds and kinds); their losses level off
near the unigram entropy. This pilot trains independent half-depth students on the DEV teacher
(bert-base-uncased) only, so the fresh family stays unread, under a few training settings, and scores
each on E5's wikitext-2 test probes exactly as E5e does (own masked-token accuracy, agreement with the
teacher).

    python -m experiments.h16c_pilot [--out PATH]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from experiments import distill, e5_behaviour as eb, finetune as ft

STEPS = 5000
SEED = 90
# name: (learning rate, warm-up, gradient clip, batch)
CONFIGS = {
    "h16b": (5e-4, 1000, 0.0, 8),          # H16b's setting, shortened: the stall should reproduce
    "clip": (5e-4, 1000, 1.0, 8),          # + gradient clipping
    "lr1e-4_clip": (1e-4, 1000, 1.0, 8),   # lower rate + clipping
    "lr2e-4_clip_b32": (2e-4, 1000, 1.0, 32),  # + 4x batch
}


def run(name: str, out: Path) -> dict:
    lr, warm, clip, batch = CONFIGS[name]
    distill.LR, distill.WARMUP, distill.CLIP, distill.STEPS, ft.BATCH = lr, warm, clip, STEPS, batch
    t0 = time.time()
    r = distill.train_student("text_encoder", "dev", "indep", SEED)
    student = r["student"].eval()
    teacher, _, _, _, _ = distill.teacher_and_student("text_encoder", "dev", SEED)
    teacher.to(ft.DEVICE).eval()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ft.bases()["text_encoder"]["dev"]["base"])
    lines, digest = eb.text_probes()
    eb.TRUTH = True
    res = eb.compare_behaviour(teacher, student, lambda m: eb._text_run(m, tok, lines, False))
    rec = {"config": name, "lr": lr, "warmup": warm, "clip": clip, "batch": batch, "steps": STEPS, "seed": SEED,
           "teacher": ft.bases()["text_encoder"]["dev"]["base"], "probes": digest,
           "accuracy": res.get("accuracy"), "teacher_accuracy": res.get("teacher_accuracy"),
           "agreement": res["agreement"], "losses": r["losses"], "seconds": round(time.time() - t0)}
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print(name, "accuracy", rec["accuracy"], "teacher", rec["teacher_accuracy"], "agreement", rec["agreement"],
          "loss", r["losses"][0], "->", r["losses"][-1], f"{rec['seconds']} s", flush=True)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/kaggle/working/h16c_pilot.jsonl")
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    out = Path(a.out)
    done = {json.loads(l)["config"] for l in out.read_text(encoding="utf-8").splitlines()} if out.exists() else set()
    for name in ([a.only] if a.only else CONFIGS):
        if name not in done:
            run(name, out)


if __name__ == "__main__":
    main()
