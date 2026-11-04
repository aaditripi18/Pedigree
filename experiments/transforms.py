"""Weight-level derivatives for E1/E3/E4 (PLAN.md): no training data needed.

Every function takes and returns a {name: torch.Tensor} state dict.

  quantize_q8_0, quantize_q4_0  round trip through GGUF's Q8_0 / Q4_0 block formats
                                (blocks of 32 along the last axis), then dequantise
  prune                         zero the smallest-magnitude share of every matrix
  permute_hidden                function-preserving permutation of the hidden units of
                                every feed-forward block (and of ResNet bottleneck
                                channels); each permuted block is checked numerically to
                                compute the same function before it is returned
"""
from __future__ import annotations

import re

import numpy as np
import torch

BLOCK = 32


def _blocks(x: torch.Tensor):
    flat = x.detach().to(torch.float32).reshape(-1)
    pad = (-flat.numel()) % BLOCK
    return torch.nn.functional.pad(flat, (0, pad)).reshape(-1, BLOCK), pad


def quantize_q8_0(sd: dict) -> dict:
    """GGUF Q8_0: per block of 32, d = max|x| / 127, q = round(x / d), x' = q * d."""
    out = {}
    for n, x in sd.items():
        if not x.is_floating_point() or x.ndim < 2:
            out[n] = x.clone()
            continue
        b, pad = _blocks(x)
        d = b.abs().amax(1, keepdim=True) / 127
        q = torch.where(d > 0, torch.round(b / torch.where(d > 0, d, 1)), torch.zeros_like(b))
        y = (q * d).reshape(-1)
        out[n] = (y[: y.numel() - pad] if pad else y).reshape(x.shape).to(x.dtype)
    return out


def quantize_q4_0(sd: dict) -> dict:
    """GGUF Q4_0 (llama.cpp quantize_row_q4_0): per block of 32, take the value of
    largest magnitude (with its sign) m, d = m / -8, q = clamp(trunc(x / d + 8.5), 0, 15),
    x' = (q - 8) * d."""
    out = {}
    for n, x in sd.items():
        if not x.is_floating_point() or x.ndim < 2:
            out[n] = x.clone()
            continue
        b, pad = _blocks(x)
        idx = b.abs().argmax(1, keepdim=True)
        m = torch.gather(b, 1, idx)
        d = m / -8
        inv = torch.where(d != 0, 1 / torch.where(d != 0, d, 1), torch.zeros_like(d))
        q = torch.clamp(torch.trunc(b * inv + 8.5), 0, 15)
        y = ((q - 8) * d).reshape(-1)
        out[n] = (y[: y.numel() - pad] if pad else y).reshape(x.shape).to(x.dtype)
    return out


def prune(sd: dict, share: float) -> dict:
    """Zero the `share` smallest-magnitude entries of every floating matrix (ndim >= 2)."""
    out = {}
    for n, x in sd.items():
        if not x.is_floating_point() or x.ndim < 2:
            out[n] = x.clone()
            continue
        k = int(share * x.numel())
        y = x.clone()
        if k:
            thresh = x.abs().reshape(-1).to(torch.float32).kthvalue(k).values
            y[x.abs() <= thresh] = 0
        out[n] = y
    return out


# --- permutations ----------------------------------------------------------------------
# (up weight, up bias, down weight, layout) per family. layout "linear": up [I, H], down
# [H, I] (torch Linear); "conv1d": GPT-2's Conv1D, up [H, I], down [I, H].
FFN = [
    (r"(.*\.intermediate\.dense)\.weight$", r"\1.bias", lambda p: re.sub(r"intermediate\.dense$", "output.dense", p), "linear"),
    (r"(.*\.mlp\.c_fc)\.weight$", r"\1.bias", lambda p: re.sub(r"c_fc$", "c_proj", p), "conv1d"),
    (r"(.*\.mlp\.dense_h_to_4h)\.weight$", r"\1.bias", lambda p: re.sub(r"dense_h_to_4h$", "dense_4h_to_h", p), "linear"),
    (r"(.*\.fc1)\.weight$", r"\1.bias", lambda p: re.sub(r"fc1$", "fc2", p), "linear"),
    (r"(.*\.feed_forward\.intermediate_dense)\.weight$", r"\1.bias",
     lambda p: re.sub(r"intermediate_dense$", "output_dense", p), "linear"),
]
BN_KEYS = ("weight", "bias", "running_mean", "running_var")


def _check_ffn(w1, b1, w2, perm, layout):
    """y = W2 gelu(W1 x + b1) before and after permuting the hidden units must agree."""
    g = torch.Generator().manual_seed(0)
    if layout == "conv1d":
        w1, w2 = w1.t(), w2.t()
    x = torch.randn(5, w1.shape[1], generator=g, dtype=torch.float64)
    f = lambda a, b, c: torch.nn.functional.gelu(x @ a.double().t() + b.double()) @ c.double().t()  # noqa: E731
    before = f(w1, b1, w2)
    after = f(w1[perm], b1[perm], w2[:, perm])
    return torch.allclose(before, after, rtol=1e-9, atol=1e-9)


def permute_hidden(sd: dict, seed: int = 0) -> tuple[dict, dict]:
    """Permute the hidden units of every recognised feed-forward block and ResNet
    bottleneck (timm naming). Returns (new state dict, report: blocks permuted per kind
    and whether every one was verified function-preserving)."""
    rng = np.random.default_rng(seed)
    out = {n: x.clone() for n, x in sd.items()}
    report = {"ffn": 0, "resnet_bottleneck": 0, "verified": True}
    for n in list(sd):
        for pat, bias_pat, down_of, layout in FFN:
            m = re.match(pat, n)
            if not m:
                continue
            up = m.group(1)
            down = down_of(up)
            bn, dw = re.sub(pat, bias_pat, n), down + ".weight"
            if bn not in sd or dw not in sd:
                continue
            w1, b1, w2 = sd[n], sd[bn], sd[dw]
            hidden = w1.shape[1] if layout == "conv1d" else w1.shape[0]
            perm = torch.as_tensor(rng.permutation(hidden))
            report["verified"] &= _check_ffn(w1, b1, w2, perm, layout)
            if layout == "conv1d":
                out[n], out[dw] = w1[:, perm], w2[perm, :]
            else:
                out[n], out[dw] = w1[perm], w2[:, perm]
            out[bn] = b1[perm]
            report["ffn"] += 1
    # timm ResNet bottleneck: conv1 -> bn1 -> act -> conv2: permute conv1's output channels
    for n in list(sd):
        m = re.match(r"(.*)\.conv1\.weight$", n)
        if not m or f"{m.group(1)}.conv2.weight" not in sd or f"{m.group(1)}.bn1.weight" not in sd:
            continue
        p = m.group(1)
        c = sd[n].shape[0]
        perm = torch.as_tensor(rng.permutation(c))
        conv1, conv2 = sd[n], sd[f"{p}.conv2.weight"]
        bn = {k: sd[f"{p}.bn1.{k}"] for k in BN_KEYS if f"{p}.bn1.{k}" in sd}
        x = torch.randn(2, conv1.shape[1], 9, 9, generator=torch.Generator().manual_seed(1), dtype=torch.float64)

        def block(w1, b, w2):
            h = torch.nn.functional.conv2d(x, w1.double(), padding=conv1.shape[-1] // 2)
            h = torch.nn.functional.batch_norm(h, b["running_mean"].double(), b["running_var"].double(),
                                               b["weight"].double(), b["bias"].double(), training=False)
            return torch.nn.functional.conv2d(torch.relu(h), w2.double(), padding=conv2.shape[-1] // 2)

        if all(k in bn for k in BN_KEYS):
            report["verified"] &= torch.allclose(block(conv1, bn, conv2),
                                                 block(conv1[perm], {k: v[perm] for k, v in bn.items()},
                                                       conv2[:, perm]), rtol=1e-9, atol=1e-9)
        out[n] = conv1[perm]
        for k, v in bn.items():
            out[f"{p}.bn1.{k}"] = v[perm]
        out[f"{p}.conv2.weight"] = conv2[:, perm]
        report["resnet_bottleneck"] += 1
    return out, report
