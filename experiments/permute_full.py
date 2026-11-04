"""E3 attack (b): permute the residual stream and every feed-forward block's hidden units,
then prove the model still computes the same function (PLAN.md, amendment 2026-11-05 (2)).

    model, report = attack(model, inputs, seed)

`inputs` are keyword arguments for one forward pass (random token ids or pixels). The
report gives the largest absolute difference of the outputs before and after, and
`verified` (allclose at float32 tolerance); a model that fails is reported, never used.
"""
from __future__ import annotations

import numpy as np
import torch

from experiments.transforms import permute_hidden
from pedigree.residual import residual_axes

ATOL = 1e-4
RTOL = 1e-4


def hidden_size(model) -> int:
    c = model.config
    return int(getattr(c, "hidden_size", None) or getattr(c, "n_embd"))


def _outputs(model, inputs) -> torch.Tensor:
    with torch.no_grad():
        out = model(**inputs)
    return (out.logits if hasattr(out, "logits") else out[0]).detach().double()


def permute_residual_(model, perm) -> dict[str, list[int]]:
    """In place, so tied parameters (an embedding and its output head) stay one tensor."""
    axes = residual_axes(model, hidden_size(model))
    perm = torch.as_tensor(perm)
    with torch.no_grad():
        for name, ax in axes.items():
            p = model.get_parameter(name)
            x = p.data
            for a in ax:
                x = x.index_select(a, perm.to(x.device))
            p.data = x.contiguous()
    return axes


def attack(model, inputs: dict, seed: int = 0) -> tuple[torch.nn.Module, dict]:
    model.eval()
    before = _outputs(model, inputs)
    d = hidden_size(model)
    perm = np.random.default_rng(seed).permutation(d)
    axes = permute_residual_(model, perm)
    sd, ffn = permute_hidden({k: v.detach().clone() for k, v in model.state_dict().items()}, seed=seed + 1)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    after = _outputs(model, inputs)
    diff = float((before - after).abs().max())
    report = {"residual_dim": d, "residual_tensors": len(axes), "ffn_blocks": ffn["ffn"],
              "ffn_verified": bool(ffn["verified"]), "max_abs_diff": diff,
              "verified": bool(torch.allclose(before, after, rtol=RTOL, atol=ATOL)) and bool(ffn["verified"])
              and not unexpected}
    return model, report


def random_inputs(model, family: str, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    c = model.config
    if family in ("text_encoder", "text_decoder"):
        return {"input_ids": torch.randint(0, c.vocab_size, (2, 16), generator=g)}
    if family == "vision_transformer":
        return {"pixel_values": torch.randn(2, c.num_channels, c.image_size, c.image_size, generator=g)}
    raise ValueError(f"attack (b) is not implemented for {family}")
