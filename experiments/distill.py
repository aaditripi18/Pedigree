"""E5 ground truth (PLAN.md, second instrument): students that share no weights with
their teacher.

For a family and split, with teacher T:

  distill_s{k}  half-depth student, random initialisation, trained only to imitate T's
                output distribution (KL divergence, temperature 2)      positive for H9
  indep_s{k}    the same student architecture, same data, same steps, trained on the
                task's own labels with no teacher                       negative for H9

k = 0, 1, 2 (three seeds). T is the base for language models (their outputs are the
pre-training task) and the family's ft_long derivative for image classifiers (the base
has no CIFAR-10 head). Students train on the train splits; E5's probes come from the
test splits only.

    python -m experiments.distill text_decoder dev
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch

from experiments import finetune as ft

# E5 used seeds 0-2; H12's new students are fresh seeds 3-7 (PLAN.md amendment 2026-11-08 (3))
SEEDS = tuple(int(x) for x in os.environ.get("PEDIGREE_SEEDS", "0,1,2").split(","))
STEPS = int(os.environ.get("PEDIGREE_STEPS", "2000"))
TEMPERATURE = 2.0
LR = float(os.environ.get("PEDIGREE_LR", "1e-4"))
# H16b and H24 (amendments 2026-11-17 (4), 2026-11-20): 20,000 steps, learning rate 5e-4, 1,000
# linear warm-up steps then cosine decay; 0 keeps E5's constant learning rate
WARMUP = int(os.environ.get("PEDIGREE_WARMUP", "0"))
# H16c dev pilot (2026-11-23): gradient-norm clipping; 0 keeps every earlier run's training unchanged
CLIP = float(os.environ.get("PEDIGREE_CLIP", "0"))
# a batch of students that is not a split of its own trains against that split's teacher
TEACHER_SPLIT = {"h16b": "fresh"}


def tsplit(split: str) -> str:
    return TEACHER_SPLIT.get(split, split)
# image teachers are ft_long derivatives, possibly in another (read-only) folder
TEACHERS = Path(os.environ.get("PEDIGREE_TEACHERS", ft.OUT))


def _half(config, *names):
    for n in names:
        if hasattr(config, n):
            setattr(config, n, max(1, getattr(config, n) // 2))
            return config
    raise ValueError(f"no depth field among {names}")


def teacher_and_student(family: str, split: str, seed: int):
    """(teacher, student factory, batch iterator, data info, causal flag or None)."""
    repo = ft.bases()[family][tsplit(split)]["base"]
    if family in ("text_encoder", "text_decoder"):
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer
        causal = family == "text_decoder"
        cls = AutoModelForCausalLM if causal else AutoModelForMaskedLM
        tok = AutoTokenizer.from_pretrained(repo)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        teacher = cls.from_pretrained(repo).float()

        def student():
            torch.manual_seed(seed)
            return cls.from_config(_half(AutoConfig.from_pretrained(repo), "num_hidden_layers", "n_layer")).float()
        batches, info = ft.text_batches(tok, causal, 100 + seed)
        return teacher, student, batches, info, causal
    if family in ("vision_transformer", "vision_cnn"):
        from safetensors.torch import load_file
        src = TEACHERS / family / tsplit(split) / "ft_long"
        if family == "vision_transformer":
            from transformers import AutoConfig, AutoModelForImageClassification
            cfg = AutoConfig.from_pretrained(repo, num_labels=10)
            teacher = AutoModelForImageClassification.from_config(cfg)

            def student():
                torch.manual_seed(seed)
                return AutoModelForImageClassification.from_config(
                    _half(AutoConfig.from_pretrained(repo, num_labels=10), "num_hidden_layers"))
            size, norm = 224, {}
        else:
            import timm
            teacher = timm.create_model("hf-hub:" + repo, pretrained=False, num_classes=10)
            small = {"timm/resnet50.a1_in1k": "resnet26", "timm/resnet18.a1_in1k": "resnet10t"}[repo]

            def student():
                torch.manual_seed(seed)
                return timm.create_model(small, pretrained=False, num_classes=10)
            size, norm = 160, {"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)}
        missing, unexpected = teacher.load_state_dict(load_file(str(src / "model.safetensors")), strict=False)
        if missing or unexpected:
            raise ValueError(f"teacher weights do not fit: {missing[:3]} {unexpected[:3]}")
        batches, info = ft.image_batches(size, 100 + seed, **norm)
        info = {**info, "teacher": f"local:{family}/{tsplit(split)}/ft_long"}
        return teacher.float(), student, batches, info, None
    raise ValueError(f"E5 students are not built for {family}")


def _logits(model, batch, causal):
    if causal is None:
        x, _ = batch
        out = model(x)
        return out.logits if hasattr(out, "logits") else out
    inputs = {k: v for k, v in batch.items() if k != "labels"}
    return model(**inputs).logits


def train_student(family: str, split: str, kind: str, seed: int) -> dict:
    teacher, make, batches, info, causal = teacher_and_student(family, split, seed)
    student = make().to(ft.DEVICE)
    teacher.to(ft.DEVICE).eval()
    opt = torch.optim.AdamW(student.parameters(), lr=LR)
    sched = None
    if WARMUP:
        import math
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda i: (i + 1) / WARMUP if i < WARMUP
            else 0.5 * (1 + math.cos(math.pi * (i - WARMUP) / max(1, STEPS - WARMUP))))
    student.train()
    losses = []
    for i in range(STEPS):
        batch = next(batches)
        if kind == "distill":
            with torch.no_grad():
                t = _logits(teacher, batch, causal)
            s = _logits(student, batch, causal)
            if causal is not None:   # only real tokens (and only masked ones for an MLM)
                keep = batch["labels"] != -100 if causal is False else batch["attention_mask"] == 1
                t, s = t[keep], s[keep]
            loss = torch.nn.functional.kl_div(torch.log_softmax(s / TEMPERATURE, -1),
                                              torch.softmax(t / TEMPERATURE, -1),
                                              reduction="batchmean") * TEMPERATURE ** 2
        else:
            if causal is None:
                x, y = batch
                loss = torch.nn.functional.cross_entropy(_logits(student, batch, None), y)
            else:
                loss = student(**batch).loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"loss is {float(loss)} at step {i}: nothing saved")
        opt.zero_grad()
        loss.backward()
        if CLIP:
            torch.nn.utils.clip_grad_norm_(student.parameters(), CLIP)
        opt.step()
        if sched is not None:
            sched.step()
        if i % 50 == 0 or i == STEPS - 1:
            losses.append(round(float(loss), 4))
    return {"student": student, "losses": losses, "info": info}


def main(family: str, split: str) -> None:
    for seed in SEEDS:
        for kind in ("distill", "indep"):
            out = ft.OUT / family / split / f"{kind}_s{seed}"
            if (out / "meta.json").exists():
                print(f"{family}/{split}/{kind}_s{seed}: exists", flush=True)
                continue
            t0 = time.time()
            r = train_student(family, split, kind, seed)
            ft.save(r["student"], out, {"family": family, "split": split, "kind": kind, "seed": seed,
                                        "teacher": r["info"].get("teacher", ft.bases()[family][tsplit(split)]["base"]),
                                        "steps": STEPS, "lr": LR, "warmup": WARMUP, "temperature": TEMPERATURE if kind == "distill" else None,
                                        "batch": ft.BATCH, "device": str(ft.DEVICE), "data": r["info"],
                                        "losses": r["losses"], "versions": ft._versions(), "code": ft._code(),
                                        "seconds": round(time.time() - t0)})
            print(f"{family}/{split}/{kind}_s{seed}: loss {r['losses'][0]} -> {r['losses'][-1]}, "
                  f"{time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
