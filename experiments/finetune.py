"""E1 ground truth: derivatives made from each base with real training (PLAN.md).

For one base (a family's dev or fresh model in results/e1_bases.json), make:

  ft_short    full fine-tune, STEPS_SHORT optimiser steps on a public task
  ft_long     full fine-tune, 10x as many steps
  lora_r8     a rank-8 LoRA trained on the same task, merged into the weights
  lora_r64    rank 64, merged
  scratch     the same architecture from a fresh random initialisation, trained
              ft_long's steps (a negative: same shapes, unrelated weights)

Each is saved as safetensors under data/derivatives/<family>/<split>/<kind>/ with a
JSON beside it (base commit, task data SHA-256, steps, seed, loss curve). Tasks:
masked / causal language modelling on WikiText-2 (text), CIFAR-10 classification
(vision), ASR on the LibriSpeech dummy split (speech), noise prediction on CIFAR-10
(diffusion). The point is a genuine optimisation trajectory from the base, not task
accuracy.

    python -m experiments.finetune text_encoder dev
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("PEDIGREE_DERIVATIVES", ROOT / "data" / "derivatives"))
DATA = Path(os.environ.get("PEDIGREE_TASKS", ROOT / "data" / "tasks"))
STEPS_SHORT = 300
LONG_FACTOR = 10
BATCH = 8
LR = 2e-5
SEED = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TASK_DATA = {
    "wikitext2": "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/train-00000-of-00001.parquet",
    "cifar10": "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/train-00000-of-00001.parquet",
    "librispeech_dummy": "https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy/resolve/main/clean/validation-00000-of-00001.parquet",
}


def task_file(name: str) -> tuple[Path, str]:
    """Download once; return (path, sha256)."""
    DATA.mkdir(parents=True, exist_ok=True)
    p = DATA / f"{name}.parquet"
    for attempt in range(8):  # a lost connection retries; the file is written only when complete
        if p.exists():
            break
        try:
            with urllib.request.urlopen(urllib.request.Request(TASK_DATA[name], headers={"User-Agent": "pedigree"}),
                                        timeout=600) as r:
                body = r.read()
            tmp = p.with_suffix(".part")
            tmp.write_bytes(body)
            tmp.replace(p)
        except OSError:
            if attempt == 7:
                raise
            time.sleep(min(300, 15 * 2 ** attempt))
    return p, hashlib.sha256(p.read_bytes()).hexdigest()


def bases() -> dict:
    return json.loads((ROOT / "results" / "e1_bases.json").read_text(encoding="utf-8"))["families"]


# --- per-family model, batches and loss --------------------------------------------------


def text_batches(tok, causal: bool, seed: int):
    import pandas as pd
    path, digest = task_file("wikitext2")
    lines = [t for t in pd.read_parquet(path)["text"].tolist() if len(t.split()) > 20]
    rng = np.random.default_rng(seed)

    def gen():
        while True:
            idx = rng.choice(len(lines), BATCH, replace=False)
            enc = tok([lines[i] for i in idx], truncation=True, max_length=128, padding="max_length",
                      return_tensors="pt")
            labels = enc["input_ids"].clone()
            if causal:
                labels[enc["attention_mask"] == 0] = -100
            else:  # 15% masking, as BERT
                m = (torch.rand(labels.shape, generator=torch.Generator().manual_seed(int(rng.integers(1 << 30))))
                     < 0.15) & (enc["attention_mask"] == 1)
                labels[~m] = -100
                enc["input_ids"] = enc["input_ids"].masked_fill(m, tok.mask_token_id)
            yield {k: v.to(DEVICE) for k, v in {**enc, "labels": labels}.items()}
    return gen(), {"task": "wikitext2", "sha256": digest}


def image_batches(size: int, seed: int, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    import pandas as pd
    from PIL import Image
    path, digest = task_file("cifar10")
    df = pd.read_parquet(path)
    rng = np.random.default_rng(seed)
    m, s = torch.tensor(mean).view(3, 1, 1), torch.tensor(std).view(3, 1, 1)

    def load(i):
        img = Image.open(io.BytesIO(df["img"].iloc[i]["bytes"])).convert("RGB").resize((size, size))
        return (torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float() / 255 - m) / s

    def gen():
        while True:
            idx = rng.choice(len(df), BATCH, replace=False)
            yield (torch.stack([load(i) for i in idx]).to(DEVICE),
                   torch.tensor(df["label"].iloc[idx].to_numpy()).to(DEVICE))
    return gen(), {"task": "cifar10", "sha256": digest}


def build(family: str, repo: str, scratch: bool):
    """(model, step function returning a loss, data info)."""
    if family in ("text_encoder", "text_decoder"):
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer
        causal = family == "text_decoder"
        cls = AutoModelForCausalLM if causal else AutoModelForMaskedLM
        tok = AutoTokenizer.from_pretrained(repo)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = cls.from_config(AutoConfig.from_pretrained(repo)) if scratch else cls.from_pretrained(repo)
        batches, info = text_batches(tok, causal, SEED)
        return model, lambda: model(**next(batches)).loss, info
    if family == "vision_transformer":
        from transformers import AutoConfig, AutoModelForImageClassification
        if scratch:
            model = AutoModelForImageClassification.from_config(AutoConfig.from_pretrained(repo, num_labels=10))
        else:
            model = AutoModelForImageClassification.from_pretrained(repo, num_labels=10, ignore_mismatched_sizes=True)
        batches, info = image_batches(224, SEED)

        def step():
            x, y = next(batches)
            return model(pixel_values=x, labels=y).loss
        return model, step, info
    if family == "vision_cnn":
        import timm
        model = timm.create_model("hf-hub:" + repo, pretrained=not scratch, num_classes=10)
        batches, info = image_batches(160, SEED, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        def step():
            x, y = next(batches)
            return torch.nn.functional.cross_entropy(model(x), y)
        return model, step, info
    if family == "diffusion_unet":
        from diffusers import DDPMScheduler, UNet2DModel
        model = (UNet2DModel.from_config(UNet2DModel.load_config(repo)) if scratch
                 else UNet2DModel.from_pretrained(repo))
        sched = DDPMScheduler.from_pretrained(repo)
        size = model.config.sample_size
        # the 256-pixel UNet at batch 8 does not fit a 16 GB GPU: recompute activations in
        # the backward pass instead of storing them (same gradients, less memory)
        model.enable_gradient_checkpointing()
        batches, info = image_batches(size, SEED)
        g = torch.Generator().manual_seed(SEED)

        def step():
            x, _ = next(batches)
            noise = torch.randn(x.shape, generator=g)
            t = torch.randint(0, sched.config.num_train_timesteps, (x.shape[0],), generator=g)
            noise, t = noise.to(DEVICE), t.to(DEVICE)
            return torch.nn.functional.mse_loss(model(sched.add_noise(x, noise, t), t).sample, noise)
        return model, step, info
    if family == "speech":
        return build_speech(repo, scratch)
    raise ValueError(family)


def build_speech(repo: str, scratch: bool):
    import pandas as pd
    import soundfile as sf
    from transformers import AutoConfig, AutoProcessor
    path, digest = task_file("librispeech_dummy")
    df = pd.read_parquet(path)
    proc = AutoProcessor.from_pretrained(repo)
    audio = [sf.read(io.BytesIO(a["bytes"]))[0].astype(np.float32) for a in df["audio"]]
    text = df["text"].str.lower().tolist() if "whisper" in repo else df["text"].tolist()
    if "whisper" not in repo:
        # wav2vec2 trains on whole clips: the three longer than 15 s (of 73) exhausted a
        # 30 GB CPU session; Whisper always pads to 30 s, so its memory does not change
        keep = [i for i, a in enumerate(audio) if len(a) <= 15 * 16000]
        audio, text = [audio[i] for i in keep], [text[i] for i in keep]
    rng = np.random.default_rng(SEED)
    if "whisper" in repo:
        from transformers import WhisperForConditionalGeneration as M
    else:
        from transformers import Wav2Vec2ForCTC as M
    model = M(AutoConfig.from_pretrained(repo)) if scratch else M.from_pretrained(repo)
    if "whisper" not in repo:
        # a clip too short for its transcript makes CTC infinite; the standard fine-tuning
        # setting drops those terms instead of turning the whole loss into NaN
        model.config.ctc_zero_infinity = True

    def step():
        idx = rng.choice(len(audio), BATCH, replace=False)
        if "whisper" in repo:
            feats = proc.feature_extractor([audio[i] for i in idx], sampling_rate=16000, return_tensors="pt")
            labels = proc.tokenizer([text[i] for i in idx], padding=True, return_tensors="pt").input_ids
            return model(input_features=feats.input_features.to(DEVICE), labels=labels.to(DEVICE)).loss
        inp = proc([audio[i] for i in idx], sampling_rate=16000, return_tensors="pt", padding=True)
        labels = proc.tokenizer([text[i] for i in idx], padding=True, return_tensors="pt").input_ids
        labels[labels == proc.tokenizer.pad_token_id] = -100
        return model(input_values=inp.input_values.to(DEVICE), labels=labels.to(DEVICE)).loss
    return model, step, {"task": "librispeech_dummy", "sha256": digest}


# --- training and saving -------------------------------------------------------------------


def train(model, step, steps: int) -> list[float]:
    torch.manual_seed(SEED)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    model.train()
    losses = []
    for i in range(steps):
        loss = step()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"loss is {float(loss)} at step {i}: nothing saved")
        if i % 25 == 0 or i == steps - 1:
            losses.append(round(float(loss), 4))
    return losses


def lora(model, family: str, rank: int):
    from peft import LoraConfig, get_peft_model
    # attention query/value projections; transformers 5 renamed some (ViT: query -> q_proj)
    targets = {"text_encoder": ["query", "value", "q_proj", "v_proj"],
               "text_decoder": ["c_attn", "query_key_value"],
               "vision_transformer": ["query", "value", "q_proj", "v_proj"], "speech": ["q_proj", "v_proj"],
               "vision_cnn": ["conv2"], "diffusion_unet": ["to_q", "to_v"]}[family]
    names = {n.split(".")[-1] for n, _ in model.named_modules()}
    found = [t for t in targets if t in names]
    if not found:
        raise ValueError(f"no LoRA target among {targets} in {family}")
    cfg = LoraConfig(r=rank, lora_alpha=2 * rank, target_modules=found)
    return get_peft_model(model, cfg)


def save(model, out: Path, meta: dict) -> None:
    from safetensors.torch import save_file
    out.mkdir(parents=True, exist_ok=True)
    sd = {k: v.detach().cpu().contiguous().clone() for k, v in model.state_dict().items()}
    save_file(sd, str(out / "model.safetensors"))
    (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")


def _code() -> dict:
    from experiments.ledger import git_state
    return git_state()


def _versions() -> dict:
    import peft
    import transformers
    return {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__}


def main(family: str, split: str, only: list[str] | None = None) -> None:
    from pedigree import hub
    repo = bases()[family][split]["base"]
    commit = hub.resolve("hf:" + repo).commit
    kinds = {"ft_short": (False, STEPS_SHORT, None), "ft_long": (False, STEPS_SHORT * LONG_FACTOR, None),
             "lora_r8": (False, STEPS_SHORT * LONG_FACTOR, 8), "lora_r64": (False, STEPS_SHORT * LONG_FACTOR, 64),
             "scratch": (True, STEPS_SHORT * LONG_FACTOR, None)}
    for kind, (scratch, steps, rank) in kinds.items():
        if only and kind not in only:
            continue
        out = OUT / family / split / kind
        if (out / "meta.json").exists():
            print(f"{family}/{split}/{kind}: exists", flush=True)
            continue
        t0 = time.time()
        model, step, info = build(family, repo, scratch)
        # transformers 5 loads a checkpoint in its stored precision (Pythia is float16);
        # every derivative is trained in float32 so the optimiser is well defined
        model = model.float()
        if rank:
            model = lora(model, family, rank)
        model.to(DEVICE)
        losses = train(model, step, steps)
        if rank:
            model = model.merge_and_unload()
        save(model, out, {"family": family, "split": split, "kind": kind, "base": repo, "base_commit": commit,
                          "steps": steps, "batch": BATCH, "lr": LR, "seed": SEED, "lora_rank": rank,
                          "scratch": scratch, "device": str(DEVICE), "versions": _versions(), "code": _code(), "data": info, "losses": losses, "seconds": round(time.time() - t0)})
        print(f"{family}/{split}/{kind}: {steps} steps, loss {losses[0]} -> {losses[-1]}, "
              f"{time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:] or None)
