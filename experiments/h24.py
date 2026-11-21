"""H24 (PLAN.md, amendment 2026-11-20): behaviour for speech and diffusion models, with students
trained as in H16b (20,000 steps, learning rate 5e-4, 1,000 warm-up steps, cosine decay).

  python -m experiments.h24 train speech 50,51,52        # distilled and independent students of these seeds
  python -m experiments.h24 score speech 50,51,52,...    # ledger E5e-h24: each student against the teacher

Speech: teacher facebook/wav2vec2-base-960h; students are half-depth wav2vec2 CTC models from a
random initialisation. Distilled students imitate the teacher's per-frame CTC distributions (KL,
temperature 2) on LibriSpeech train-clean-100 audio; independent students train with CTC on its
transcripts. Score: dark-knowledge divergence of per-frame distributions on the first 200
LibriSpeech test-clean utterances; task score 1 - character error rate (greedy CTC decoding).

Diffusion: teacher google/ddpm-cifar10-32; students are its UNet with half the layers per block,
from a random initialisation. Distilled students regress the teacher's noise prediction, independent
students the true noise, on CIFAR-10 train. Score: mean squared difference from the teacher's noise
prediction on the first 1,000 CIFAR-10 test images at 10 fixed timesteps, divided by the teacher's
own noise error (lower = closer); task score 1 - normalised noise error.
"""
from __future__ import annotations

import http.client
import io
import json
import math
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch

from experiments import finetune as ft
from experiments import ledger

STEPS = int(os.environ.get("PEDIGREE_STEPS", "20000"))
LR = float(os.environ.get("PEDIGREE_LR", "5e-4"))
WARMUP = int(os.environ.get("PEDIGREE_WARMUP", "1000"))
TEMPERATURE = 2.0
SPEECH_TEACHER = "facebook/wav2vec2-base-960h"
DIFF_TEACHER = "google/ddpm-cifar10-32"
LIBRI = "https://huggingface.co/datasets/openslr/librispeech_asr/resolve/2b9f39377850ffce6bf6358257ae9f84b2349497/clean/"
TRAIN_SHARDS = [f"train.100/{i:04d}.parquet" for i in range(int(os.environ.get("PEDIGREE_H24_SHARDS", "4")))]
TEST_SHARD = "test/0000.parquet"
MAX_SECONDS = 10.0
SPEECH_BATCH, DIFF_BATCH = 8, 32
SPEECH_PROBES = int(os.environ.get("PEDIGREE_H24_SPEECH_PROBES", "200"))
DIFF_PROBES = int(os.environ.get("PEDIGREE_H24_DIFF_PROBES", "1000"))
TIMESTEPS = [50, 150, 250, 350, 450, 550, 650, 750, 850, 950]
STUDENTS = Path(os.environ.get("PEDIGREE_STUDENTS", ft.OUT))


def _whole_parquet(p: Path) -> bool:
    """A parquet file begins and ends with the magic bytes PAR1."""
    with open(p, "rb") as fh:
        head = fh.read(4)
        fh.seek(-4, 2)
        return head == b"PAR1" and fh.read(4) == b"PAR1"


def _parquet(name: str) -> Path:
    p = ft.DATA / "librispeech" / name.replace("/", "_")
    if p.exists() and not _whole_parquet(p):
        p.unlink()                         # a short file from an earlier run: fetched again
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(8):
            try:
                with urllib.request.urlopen(urllib.request.Request(LIBRI + name, headers={"User-Agent": "pedigree"}),
                                            timeout=900) as r:
                    expected = int(r.headers.get("Content-Length") or 0)
                    tmp = p.with_suffix(".part")
                    with open(tmp, "wb") as fh:
                        while True:
                            chunk = r.read(1 << 22)
                            if not chunk:
                                break
                            fh.write(chunk)
                # a connection can close early without an error: keep only a whole file
                if (expected and tmp.stat().st_size != expected) or not _whole_parquet(tmp):
                    raise OSError(f"{name}: short download ({tmp.stat().st_size} of {expected} bytes)")
                tmp.replace(p)
                break
            except (OSError, http.client.HTTPException):
                if attempt == 7:
                    raise
                time.sleep(min(300, 15 * 2 ** attempt))
    return p


def _audio(row) -> np.ndarray:
    import soundfile as sf
    a, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
    if sr != 16000:
        raise ValueError(f"sampling rate {sr}")
    return a


def speech_utterances(names: list[str], limit: int | None = None):
    import pandas as pd
    rows = []
    for n in names:
        df = pd.read_parquet(_parquet(n), columns=["audio", "text", "id"])
        rows += df.to_dict("records")
        if limit and len(rows) >= limit:
            break
    return rows[:limit] if limit else rows


# --- models --------------------------------------------------------------------------------------

def teacher_and_student(family: str, seed: int):
    if family == "speech":
        from transformers import AutoConfig, Wav2Vec2ForCTC, Wav2Vec2Processor
        proc = Wav2Vec2Processor.from_pretrained(SPEECH_TEACHER)
        teacher = Wav2Vec2ForCTC.from_pretrained(SPEECH_TEACHER).float()
        cfg = AutoConfig.from_pretrained(SPEECH_TEACHER)
        cfg.num_hidden_layers = cfg.num_hidden_layers // 2

        def student():
            torch.manual_seed(seed)
            return Wav2Vec2ForCTC(cfg)
        return teacher, student, proc
    if family == "diffusion_unet":
        from diffusers import DDPMScheduler, UNet2DModel
        teacher = UNet2DModel.from_pretrained(DIFF_TEACHER).float()
        cfg = dict(teacher.config)
        cfg["layers_per_block"] = max(1, cfg["layers_per_block"] // 2)

        def student():
            torch.manual_seed(seed)
            return UNet2DModel.from_config(cfg)
        return teacher, student, DDPMScheduler.from_pretrained(DIFF_TEACHER)
    raise ValueError(family)


def speech_batches(proc, seed: int):
    rows = [r for r in speech_utterances(TRAIN_SHARDS) if True]
    rng = np.random.default_rng(seed)
    audio_ok = []
    for r in rows:
        a = _audio(r)
        if len(a) <= MAX_SECONDS * 16000:
            audio_ok.append((a, r["text"]))
    while True:
        idx = rng.choice(len(audio_ok), SPEECH_BATCH, replace=False)
        xs, ts = [audio_ok[i][0] for i in idx], [audio_ok[i][1] for i in idx]
        enc = proc(xs, sampling_rate=16000, padding=True, return_tensors="pt")
        lab = proc.tokenizer(ts, padding=True, return_tensors="pt")
        labels = lab.input_ids.masked_fill(lab.attention_mask == 0, -100)
        yield enc.input_values.to(ft.DEVICE), labels.to(ft.DEVICE)


def diffusion_batches(seed: int):
    import pandas as pd
    from PIL import Image
    path, _ = ft.task_file("cifar10")
    df = pd.read_parquet(path)
    imgs = torch.stack([torch.from_numpy(np.asarray(Image.open(io.BytesIO(b["bytes"])).convert("RGB")))
                        .permute(2, 0, 1).float() / 127.5 - 1 for b in df["img"]])
    g = torch.Generator().manual_seed(seed)
    while True:
        idx = torch.randint(0, len(imgs), (DIFF_BATCH,), generator=g)
        yield imgs[idx].to(ft.DEVICE)


def train(family: str, kind: str, seed: int):
    teacher, make, aux = teacher_and_student(family, seed)
    teacher.to(ft.DEVICE).eval()
    student = make().to(ft.DEVICE).train()
    opt = torch.optim.AdamW(student.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: (i + 1) / WARMUP if i < WARMUP else 0.5 * (1 + math.cos(math.pi * (i - WARMUP) / max(1, STEPS - WARMUP))))
    batches = speech_batches(aux, seed) if family == "speech" else diffusion_batches(seed)
    g = torch.Generator(device=ft.DEVICE).manual_seed(seed)
    losses = []
    for i in range(STEPS):
        if family == "speech":
            x, labels = next(batches)
            if kind == "distill":
                with torch.no_grad():
                    t = teacher(x).logits
                s = student(x).logits
                n = min(t.shape[1], s.shape[1])
                loss = torch.nn.functional.kl_div(torch.log_softmax(s[:, :n] / TEMPERATURE, -1).flatten(0, 1),
                                                  torch.softmax(t[:, :n] / TEMPERATURE, -1).flatten(0, 1),
                                                  reduction="batchmean") * TEMPERATURE ** 2
            else:
                loss = student(x, labels=labels).loss
        else:
            x0 = next(batches)
            noise = torch.randn(x0.shape, generator=g, device=ft.DEVICE)
            ts = torch.randint(0, aux.config.num_train_timesteps, (x0.shape[0],), generator=g, device=ft.DEVICE)
            xt = aux.add_noise(x0, noise, ts)
            s = student(xt, ts).sample
            if kind == "distill":
                with torch.no_grad():
                    target = teacher(xt, ts).sample
            else:
                target = noise
            loss = torch.nn.functional.mse_loss(s, target)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"loss is {float(loss)} at step {i}: nothing saved")
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        sched.step()
        if i % 100 == 0 or i == STEPS - 1:
            losses.append(round(float(loss), 4))
    return student, losses


def main_train(family: str, seeds: list[int]) -> None:
    for seed in seeds:
        for kind in ("distill", "indep"):
            out = ft.OUT / family / "h24" / f"{kind}_s{seed}"
            if (out / "meta.json").exists():
                print(f"{family}/h24/{kind}_s{seed}: exists", flush=True)
                continue
            t0 = time.time()
            student, losses = train(family, kind, seed)
            ft.save(student, out, {"family": family, "split": "h24", "kind": kind, "seed": seed,
                                   "teacher": SPEECH_TEACHER if family == "speech" else DIFF_TEACHER,
                                   "steps": STEPS, "lr": LR, "warmup": WARMUP, "temperature": TEMPERATURE,
                                   "batch": SPEECH_BATCH if family == "speech" else DIFF_BATCH,
                                   "device": str(ft.DEVICE), "losses": losses, "versions": ft._versions(),
                                   "code": ft._code(), "seconds": round(time.time() - t0)})
            print(f"{family}/h24/{kind}_s{seed}: loss {losses[0]} -> {losses[-1]}, {time.time() - t0:.0f} s", flush=True)


# --- scoring -------------------------------------------------------------------------------------

def _cer(ref: str, hyp: str) -> float:
    a, b = ref, hyp
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] / max(len(a), 1)


def score_speech(teacher, proc, student, probes) -> dict:
    from experiments.e5_behaviour import Stats
    st, cer_t, cer_s = Stats(), [], []
    for r in probes:
        x = proc(_audio(r), sampling_rate=16000, return_tensors="pt").input_values.to(ft.DEVICE)
        with torch.no_grad():
            lt, ls = teacher(x).logits[0], student(x).logits[0]
        n = min(len(lt), len(ls))
        st.add(lt[:n], ls[:n], torch.full((n,), -1))
        cer_t.append(_cer(r["text"], proc.batch_decode(lt.argmax(-1)[None])[0]))
        cer_s.append(_cer(r["text"], proc.batch_decode(ls.argmax(-1)[None])[0]))
    res = st.result()
    return {"score": res["dark_js"], "dark_js": res["dark_js"], "agreement": res["agreement"],
            "teacher_accuracy": 1 - float(np.mean(cer_t)), "accuracy": 1 - float(np.mean(cer_s)), "probes": len(probes)}


def score_diffusion(teacher, sched, student, x0) -> dict:
    g = torch.Generator().manual_seed(0)
    d2t = err_t = err_s = n = 0.0
    for t in TIMESTEPS:
        for i in range(0, len(x0), 100):
            xb = x0[i:i + 100]
            noise = torch.randn(xb.shape, generator=g)
            ts = torch.full((len(xb),), t, dtype=torch.long)
            xt = sched.add_noise(xb, noise, ts).to(ft.DEVICE)
            with torch.no_grad():
                pt, ps = teacher(xt, ts.to(ft.DEVICE)).sample.cpu(), student(xt, ts.to(ft.DEVICE)).sample.cpu()
            d2t += float(((ps - pt) ** 2).sum())
            err_t += float(((pt - noise) ** 2).sum())
            err_s += float(((ps - noise) ** 2).sum())
            n += noise.numel()
    return {"score": d2t / err_t, "teacher_mse": d2t / n, "teacher_accuracy": 1 - err_t / n,
            "accuracy": 1 - err_s / n, "probes": len(x0), "timesteps": TIMESTEPS}


def main_score(family: str, seeds: list[int]) -> None:
    from safetensors.torch import load_file
    teacher, make, aux = teacher_and_student(family, 0)
    teacher.to(ft.DEVICE).eval()
    if family == "speech":
        probes = speech_utterances([TEST_SHARD], SPEECH_PROBES)
        meta = {"probes": "librispeech_test_clean", "n": len(probes)}
    else:
        from experiments.e5_behaviour import image_probes
        x0, digest = image_probes(32, (0.5,) * 3, (0.5,) * 3)
        x0 = x0[:DIFF_PROBES]
        meta = {"probes": "cifar10_test", "sha256": digest, "n": len(x0)}
    run = ledger.Run("E5e-h24", family=family, split="h24", teacher=SPEECH_TEACHER if family == "speech" else DIFF_TEACHER,
                     **meta)
    for seed in seeds:
        for kind in ("distill", "indep"):
            name = f"{kind}_s{seed}"
            d = STUDENTS / family / "h24" / name
            if not (d / "meta.json").exists():
                run.record(model=name, kind=kind, error="student not built")
                print(f"  {name}: MISSING", flush=True)
                continue
            m = make()
            missing, unexpected = m.load_state_dict(load_file(str(d / "model.safetensors")), strict=False)
            if missing or unexpected:
                raise ValueError(f"{name}: weights do not fit ({missing[:2]} {unexpected[:2]})")
            m.to(ft.DEVICE).eval()
            r = score_speech(teacher, aux, m, probes) if family == "speech" else score_diffusion(teacher, aux, m, x0)
            run.record(model=name, kind=kind, label="positive" if kind == "distill" else "negative", **r)
            print(f"  {name}: score {r['score']:.4f}  task {r['accuracy']:.3f} (teacher {r['teacher_accuracy']:.3f})",
                  flush=True)


if __name__ == "__main__":
    action, family, seeds = sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3].split(",")]
    (main_train if action == "train" else main_score)(family, seeds)
