"""H16c dev pilot (2026-11-23, design data before any H16c pre-registration): why H16b's text-encoder
students stalled at one output, and which training escapes it.

H16b's 22 students (RoBERTa-base teacher, fresh split) all predict the same token at every masked
position (accuracy 0.0548, agreement 0.0680, identical across seeds and kinds); their losses level off
near the unigram entropy. This pilot trains independent half-depth students on the DEV teacher
(bert-base-uncased) only, so the fresh family stays unread, under a few training settings, and scores
each on E5's wikitext-2 test probes exactly as E5e does (own masked-token accuracy, agreement with the
teacher).

    python -m experiments.h16c_pilot [--family text_encoder|vision_transformer] [--out PATH]

The vision-transformer pilot (amendment 2026-11-23 (2)) asks the same question of H16b's other family,
whose independents learned but stayed under the floor (0.275-0.377 against 0.484): it trains
independent half-depth ViT students on the dev teacher only, and scores them on E5's CIFAR-10 probes.
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
CONFIGS = {   # text encoder (2026-11-23)
    "h16b": (5e-4, 1000, 0.0, 8),          # H16b's setting, shortened: the stall should reproduce
    "clip": (5e-4, 1000, 1.0, 8),          # + gradient clipping
    "lr1e-4_clip": (1e-4, 1000, 1.0, 8),   # lower rate + clipping
    "lr2e-4_clip_b32": (2e-4, 1000, 1.0, 32),  # + 4x batch
}
VIT_CONFIGS = {   # vision transformer (2026-11-23 (2)), same four questions
    "h16b": (5e-4, 1000, 0.0, 8),          # H16b's setting, shortened
    "clip_b32": (5e-4, 1000, 1.0, 32),     # + clipping and 4x batch
    "lr2e-4_clip_b32": (2e-4, 1000, 1.0, 32),  # the text encoder's best
    "lr1e-3_clip_b64": (1e-3, 1000, 1.0, 64),  # higher rate, 8x batch
}
FAMILY_CONFIGS = {"text_encoder": CONFIGS, "vision_transformer": VIT_CONFIGS}


def run(name: str, out: Path, family: str = "text_encoder") -> dict:
    lr, warm, clip, batch = FAMILY_CONFIGS[family][name]
    distill.LR, distill.WARMUP, distill.CLIP, distill.STEPS, ft.BATCH = lr, warm, clip, STEPS, batch
    t0 = time.time()
    r = distill.train_student(family, "dev", "indep", SEED)
    student = r["student"].to(ft.DEVICE).eval()
    teacher, _, _, _, _ = distill.teacher_and_student(family, "dev", SEED)
    teacher.to(ft.DEVICE).eval()
    eb.TRUTH = True
    if family == "text_encoder":
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(ft.bases()[family]["dev"]["base"])
        lines, digest = eb.text_probes()
        res = eb.compare_behaviour(teacher, student, lambda m: eb._text_run(m, tok, lines, False))
    else:   # E5e's CIFAR-10 probes for the vision transformer
        x, digest, y = eb.image_probes(224, (0.5,) * 3, (0.5,) * 3, labels=True)
        res = eb.compare_behaviour(teacher, student, lambda m: eb._image_run(m, x, y))
    rec = {"family": family, "config": name, "lr": lr, "warmup": warm, "clip": clip, "batch": batch, "steps": STEPS, "seed": SEED,
           "teacher": ft.bases()[family]["dev"]["base"], "probes": digest,
           "accuracy": res.get("accuracy"), "teacher_accuracy": res.get("teacher_accuracy"),
           "agreement": res["agreement"], "losses": r["losses"], "seconds": round(time.time() - t0)}
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print(name, "accuracy", rec["accuracy"], "teacher", rec["teacher_accuracy"], "agreement", rec["agreement"],
          "loss", r["losses"][0], "->", r["losses"][-1], f"{rec['seconds']} s", flush=True)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="text_encoder", choices=sorted(FAMILY_CONFIGS))
    ap.add_argument("--out", default="/kaggle/working/h16c_pilot.jsonl")
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    out = Path(a.out)
    done = {json.loads(l)["config"] for l in out.read_text(encoding="utf-8").splitlines()} if out.exists() else set()
    for name in ([a.only] if a.only else FAMILY_CONFIGS[a.family]):
        if name not in done:
            run(name, out, a.family)


if __name__ == "__main__":
    main()
