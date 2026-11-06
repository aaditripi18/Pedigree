"""Pair each tensor of a model with the tensor of a candidate relative it corresponds to.

Fine-tunes often wrap their base (`bert.x` saved as `x`, or `model.x`) or use old
LayerNorm names (`gamma`/`beta` for `weight`/`bias`). Adapted from Stowaway's
S5 alignment (MIT), which E8 showed must be used everywhere a pair is compared.
"""
from __future__ import annotations

import re

_LEGACY_NAMES = {"gamma": "weight", "beta": "bias"}
# transformers 5 renamed the layers of several architectures (ViT: encoder.layer.N ->
# layers.N, attention.attention.query -> attention.q_proj, intermediate.dense -> mlp.fc1,
# output.dense -> mlp.fc2). A model saved with one version and its base saved with the
# other are the same tensors under two names, so both sides are mapped to one form first.
_RENAMES = [
    (re.compile(r"(^|\.)encoder\.layer\.(\d+)\."), r"\1layers.\2."),
    (re.compile(r"\.attention\.(?:attention|self)\.query\."), ".attention.q_proj."),
    (re.compile(r"\.attention\.(?:attention|self)\.key\."), ".attention.k_proj."),
    (re.compile(r"\.attention\.(?:attention|self)\.value\."), ".attention.v_proj."),
    (re.compile(r"\.attention\.output\.dense\."), ".attention.o_proj."),
    (re.compile(r"\.attention\.output\.LayerNorm\."), ".attention_layernorm."),
    (re.compile(r"(layers\.\d+)\.intermediate\.dense\."), r"\1.mlp.fc1."),
    (re.compile(r"(layers\.\d+)\.output\.dense\."), r"\1.mlp.fc2."),
    (re.compile(r"(layers\.\d+)\.output\.LayerNorm\."), r"\1.output_layernorm."),
]
MAX_DROP = 2


def canonical(name: str) -> str:
    """One spelling for a tensor saved under any transformers version's names."""
    for pat, rep in _RENAMES:
        name = pat.sub(rep, name)
    return name


def _canon(name: str, drop: int) -> str:
    parts = canonical(name).split(".")[drop:]
    if parts and parts[-1] in _LEGACY_NAMES:
        parts[-1] = _LEGACY_NAMES[parts[-1]]
    return ".".join(parts)


def align(model: dict[str, tuple], base: dict[str, tuple]) -> tuple[dict[str, str], int, int]:
    """model / base map a tensor name to (dtype class, shape). Tries dropping 0..MAX_DROP
    leading name components on each side and keeps the choice that pairs the most
    tensors of identical shape (fewest drops on a tie). A name that is ambiguous after
    dropping is left out. Returns (model name -> base name, dropped here, dropped there).

    Unlike Stowaway, dtypes may differ (a bf16 fine-tune of an fp32 base is still a
    pair): callers pass a dtype *class* ("float") so values are compared after decoding."""
    best: tuple[dict[str, str], int, int] = ({}, 0, 0)
    for dm in range(MAX_DROP + 1):
        for db in range(MAX_DROP + 1):
            bkeys: dict[str, list] = {}
            for b in base:
                bkeys.setdefault(_canon(b, db), []).append(b)
            mkeys: dict[str, list] = {}
            for m in model:
                mkeys.setdefault(_canon(m, dm), []).append(m)
            pairs = {}
            for key, ms in mkeys.items():
                bs = bkeys.get(key)
                if key and len(ms) == 1 and bs and len(bs) == 1 and model[ms[0]] == base[bs[0]]:
                    pairs[ms[0]] = bs[0]
            if len(pairs) > len(best[0]):
                best = (pairs, dm, db)
    return best
