"""E5: lineage from behaviour where weights cannot be compared (PLAN.md, H9).

Probes (frozen, never trained on): the first PROBE_TEXTS lines of WikiText-2's *test*
split with more than 20 words, and the first PROBE_IMAGES images of CIFAR-10's *test*
split. For each family and split, the teacher is compared with:

  distill_s*          students distilled from it                    positive
  indep_s*            same student architecture, same data, labels  negative (same task)
  public derivatives  documented distillations (DistilBERT, DistilGPT-2)          positive
  E1 independents     public models sharing the teacher's tokenizer               negative

Measures over the probes: top-1 agreement and mean Jensen-Shannon divergence of the
output distributions (masked positions for a masked LM, every real position for a causal
LM, the 10 classes for images), and linear CKA of the final representation (mean-pooled
over tokens, or the pooled image feature). Rules and thresholds are chosen on dev only.

    python -m experiments.e5_behaviour text_decoder [dev|fresh]
"""
from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from experiments import distill, ledger
from experiments import finetune as ft
from pedigree.behaviour import linear_cka
from pedigree.verify import InMemory, compare

ROOT = Path(__file__).resolve().parent.parent
PROBE_TEXTS = 256
PROBE_IMAGES = 1000
MAX_LEN = 64
BATCH = 16
STUDENTS = Path(os.environ.get("PEDIGREE_STUDENTS", ft.OUT))
PROBE_DATA = {
    "wikitext2_test": "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/test-00000-of-00001.parquet",
    "cifar10_test": "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/test-00000-of-00001.parquet",
}
ft.TASK_DATA.update(PROBE_DATA)


def text_probes() -> tuple[list[str], str]:
    import pandas as pd
    path, digest = ft.task_file("wikitext2_test")
    lines = [t.strip() for t in pd.read_parquet(path)["text"].tolist() if len(t.split()) > 20]
    return lines[:PROBE_TEXTS], digest


def image_probes(size: int, mean, std) -> tuple[torch.Tensor, str]:
    import pandas as pd
    from PIL import Image
    path, digest = ft.task_file("cifar10_test")
    df = pd.read_parquet(path).iloc[:PROBE_IMAGES]
    m, s = torch.tensor(mean).view(3, 1, 1), torch.tensor(std).view(3, 1, 1)
    xs = [(torch.from_numpy(np.asarray(Image.open(io.BytesIO(b["bytes"])).convert("RGB").resize((size, size))))
           .permute(2, 0, 1).float() / 255 - m) / s for b in df["img"]]
    return torch.stack(xs), digest


class Stats:
    """Streaming agreement and JS divergence, plus pooled representations for CKA."""

    def __init__(self):
        self.n = 0
        self.agree = 0.0
        self.js = 0.0

    def add(self, p_logits: torch.Tensor, q_logits: torch.Tensor) -> None:
        p, q = torch.softmax(p_logits.double(), -1), torch.softmax(q_logits.double(), -1)
        m = (p + q) / 2
        kl = lambda a, b: (a * (torch.log2(a.clamp_min(1e-12)) - torch.log2(b.clamp_min(1e-12)))).sum(-1)  # noqa: E731
        self.js += float(((kl(p, m) + kl(q, m)) / 2).sum())
        self.agree += float((p.argmax(-1) == q.argmax(-1)).sum())
        self.n += p.shape[0]

    def result(self) -> dict:
        return {"positions": self.n, "agreement": self.agree / max(self.n, 1), "js": self.js / max(self.n, 1)}


def _text_run(model, tok, lines, causal: bool):
    """Per batch: (logits at compared positions, pooled final representation)."""
    g = torch.Generator().manual_seed(0)
    for i in range(0, len(lines), BATCH):
        enc = tok(lines[i:i + BATCH], truncation=True, max_length=MAX_LEN, padding="max_length", return_tensors="pt")
        mask = enc["attention_mask"] == 1
        if not causal:   # the same 15% of positions masked for every model compared
            pick = (torch.rand(enc["input_ids"].shape, generator=g) < 0.15) & mask
            enc["input_ids"] = enc["input_ids"].masked_fill(pick, tok.mask_token_id)
            mask = pick
        enc = {k: v.to(ft.DEVICE) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        h = out.hidden_states[-1]
        am = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
        yield out.logits[mask.to(ft.DEVICE)], ((h * am).sum(1) / am.sum(1)).cpu()


def _image_run(model, x):
    for i in range(0, len(x), BATCH * 4):
        xb = x[i:i + BATCH * 4].to(ft.DEVICE)
        with torch.no_grad():
            if hasattr(model, "forward_head"):     # timm
                f = model.forward_head(model.forward_features(xb), pre_logits=True)
                logits = model.get_classifier()(f)
            else:                                   # transformers ViT
                out = model(pixel_values=xb, output_hidden_states=True)
                f, logits = out.hidden_states[-1][:, 0], out.logits
        yield logits, f.cpu()


def compare_behaviour(teacher, other, runner) -> dict:
    st = Stats()
    rt, ro = [], []
    for (lt, ft_), (lo, fo) in zip(runner(teacher), runner(other)):
        if lt.shape != lo.shape:
            raise ValueError(f"output spaces differ: {tuple(lt.shape)} vs {tuple(lo.shape)}")
        st.add(lt, lo)
        rt.append(ft_)
        ro.append(fo)
    return {**st.result(), "cka": linear_cka(torch.cat(rt).numpy(), torch.cat(ro).numpy())}


def _load_student(family: str, split: str, name: str, make):
    from safetensors.torch import load_file
    d = STUDENTS / family / split / name
    if not (d / "meta.json").exists():
        return None
    m = make()
    missing, unexpected = m.load_state_dict(load_file(str(d / "model.safetensors")), strict=False)
    if missing or unexpected:
        raise ValueError(f"{name}: weights do not fit ({missing[:2]} {unexpected[:2]})")
    return m


def main(family: str, splits: list[str]) -> None:
    spec = json.loads((ROOT / "results" / "e1_bases.json").read_text(encoding="utf-8"))
    for split in splits:
        teacher, make, _, info, causal = distill.teacher_and_student(family, split, 0)
        teacher.to(ft.DEVICE).eval()
        repo = ft.bases()[family][split]["base"]
        if family in ("text_encoder", "text_decoder"):
            from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer
            tok = AutoTokenizer.from_pretrained(repo)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            lines, digest = text_probes()
            runner = lambda m: _text_run(m, tok, lines, causal)  # noqa: E731
            cls = AutoModelForCausalLM if causal else AutoModelForMaskedLM
            public = [(r, "positive", "public_derivative") for r in spec.get("public_derivatives", {}).get(repo, [])]
            public += [(r, "negative", "independent") for r in ft.bases()[family][split]["independent"]]
            probe = {"probes": "wikitext2_test", "sha256": digest, "n": len(lines)}
        else:
            size = 224 if family == "vision_transformer" else 160
            norm = ((0.5,) * 3, (0.5,) * 3) if family == "vision_transformer" else \
                ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            x, digest = image_probes(size, *norm)
            runner = lambda m: _image_run(m, x)  # noqa: E731
            public, cls = [], None
            probe = {"probes": "cifar10_test", "sha256": digest, "n": len(x)}
        run = ledger.Run("E5", family=family, split=split, teacher=info.get("teacher", repo), **probe)
        cands = [(f"{k}_s{s}", "positive" if k == "distill" else "negative", k)
                 for s in distill.SEEDS for k in ("distill", "indep")]
        for name, label, kind in cands + public:
            try:
                if kind in ("distill", "indep"):
                    m = _load_student(family, split, name, make)
                    if m is None:
                        run.record(model=name, kind=kind, label=label, error="student not built")
                        print(f"  {name}: MISSING", flush=True)
                        continue
                else:
                    # a public model only counts if it reads the same tokens as the teacher
                    from transformers import AutoTokenizer
                    if AutoTokenizer.from_pretrained(name).get_vocab() != tok.get_vocab():
                        run.record(model=name, kind=kind, label=label, error="different tokenizer: no shared output space")
                        print(f"  {name}: different tokenizer, skipped (recorded)", flush=True)
                        continue
                    m = cls.from_pretrained(name).float()
                r = compare_behaviour(teacher, m.to(ft.DEVICE).eval(), runner)
                # H5: the weight verifier on the same pair (students share no initialisation)
                w = compare(InMemory(m.state_dict()), InMemory(teacher.state_dict()))
                r.update(weight_median_cosine=w["median_cosine"], weight_compared=w["compared"],
                         weight_matched=w["matched"])
                run.record(model=name, kind=kind, label=label, **r)
                print(f"  {name:<45} agree {r['agreement']:.3f}  js {r['js']:.4f}  cka {r['cka']:.3f}", flush=True)
                del m
            except Exception as e:  # recorded, never dropped
                run.record(model=name, kind=kind, label=label, error=f"{type(e).__name__}: {e}")
                print(f"  {name}: ERROR {e}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:] or ["dev", "fresh"])
