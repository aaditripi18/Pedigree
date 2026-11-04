"""The residual stream of a transformer, and permuting it without changing the function.

Every layer of a pre- or post-LayerNorm transformer reads from or writes to a stream of
width d. A permutation P of those d coordinates, applied consistently, leaves the
function unchanged: embeddings and position embeddings write in P's order, LayerNorm
is permutation-equivariant (its gain and bias are permuted with it), every layer that
reads the stream gets its input axis permuted, every layer that writes it its output
axis. Attention's own head space is left alone: query/key/value projections are only
read-side, output projections only write-side.

residual_axes(model, d) names, for every parameter, which of its axes is a stream axis.
The attack (experiments/permute_full.py) permutes those axes; the defence
(pedigree/rebasin.py) undoes a recovered permutation on them. Needs torch.
"""
from __future__ import annotations

import re

import torch

QKV = re.compile(r"(^|\.)(query|key|value|q_proj|k_proj|v_proj|q_lin|k_lin|v_lin)$")
ATTN_OUT = re.compile(r"(attention\.output\.dense|attention\.dense|attn\.c_proj|self_attn\.out_proj|"
                      r"attention\.o_proj|attention\.out_proj|attn\.out_proj|out_lin|\.o_proj)$")


def _conv1d():
    try:
        from transformers.pytorch_utils import Conv1D
        return Conv1D
    except Exception:  # pragma: no cover
        return ()


def residual_axes(model: torch.nn.Module, d: int) -> dict[str, list[int]]:
    """{parameter name: axes that index the residual stream}. Tied parameters appear
    under their first name only (they are one tensor)."""
    conv1d = _conv1d()
    axes: dict[str, set[int]] = {}
    seen: dict[int, str] = {}

    def add(pname: str, p: torch.Tensor, ax: int):
        key = seen.setdefault(id(p), pname)
        axes.setdefault(key, set()).add(ax)

    for mname, mod in model.named_modules():
        pre = mname + "." if mname else ""
        if isinstance(mod, torch.nn.Embedding) and mod.embedding_dim == d:
            add(pre + "weight", mod.weight, 1)
        elif isinstance(mod, torch.nn.LayerNorm) and tuple(mod.normalized_shape) == (d,):
            if mod.weight is not None:
                add(pre + "weight", mod.weight, 0)
            if mod.bias is not None:
                add(pre + "bias", mod.bias, 0)
        elif isinstance(mod, torch.nn.Linear) or (conv1d and isinstance(mod, conv1d)):
            is_conv1d = not isinstance(mod, torch.nn.Linear)
            w = mod.weight
            fan_in, fan_out = (w.shape[0], w.shape[1]) if is_conv1d else (w.shape[1], w.shape[0])
            in_ax, out_ax = (0, 1) if is_conv1d else (1, 0)
            reads = fan_in == d and not ATTN_OUT.search(mname)
            writes = fan_out == d and not QKV.search(mname)
            if reads:
                add(pre + "weight", w, in_ax)
            if writes:
                add(pre + "weight", w, out_ax)
                if mod.bias is not None:
                    add(pre + "bias", mod.bias, 0)
        elif isinstance(mod, torch.nn.Conv2d) and mod.out_channels == d:   # ViT patch embedding
            add(pre + "weight", mod.weight, 0)
            if mod.bias is not None:
                add(pre + "bias", mod.bias, 0)
        for pname, p in mod.named_parameters(recurse=False):            # cls token, position table
            if pname in ("cls_token", "position_embeddings", "mask_token", "distillation_token") \
                    and p.shape[-1] == d:
                add(pre + pname, p, p.ndim - 1)
    return {k: sorted(v) for k, v in axes.items()}


def apply(state: dict, axes: dict[str, list[int]], perm) -> dict:
    """A new state dict with `perm` applied on every listed residual axis."""
    perm = torch.as_tensor(perm)
    out = dict(state)
    for name, ax in axes.items():
        if name not in out:
            continue
        x = out[name]
        for a in ax:
            x = x.index_select(a, perm.to(x.device))
        out[name] = x
    return out
