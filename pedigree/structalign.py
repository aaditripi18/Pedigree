"""Structural alignment of a GGUF file's tensors with a Hugging Face model's, for when
llama.cpp's name table does not know the architecture (PLAN.md, amendment 2026-11-17 (4), H17).

Both sides are grouped by block index (``blk.N.`` in GGUF; the first ``.layers.N.``,
``.layer.N.``, ``.h.N.``, ``.blocks.N.`` or ``.block.N.`` on the other side; tensors without an
index form one group). The two models must have the same number of blocks. Within a group a
tensor is paired only if its shape occurs exactly once in that group on both sides; ambiguous
shapes are left out, so a wrong pairing can only come from two different tensors that happen to
be the only ones of one shape in the same block on both sides.
"""
from __future__ import annotations

import re
from collections import defaultdict

_GGUF_BLOCK = re.compile(r"^blk\.(\d+)\.")
_HF_BLOCK = re.compile(r"(?:^|\.)(?:layers|layer|h|blocks|block)\.(\d+)\.")


def _block(name: str, gguf_side: bool) -> int | None:
    m = (_GGUF_BLOCK.match(name) if gguf_side else _HF_BLOCK.search(name))
    return int(m.group(1)) if m else None


def _groups(shapes: dict, gguf_side: bool) -> dict:
    """{block index or None: {shape: [names]}}"""
    g: dict = defaultdict(lambda: defaultdict(list))
    for name, (_kind, shape) in shapes.items():
        g[_block(name, gguf_side)][tuple(shape)].append(name)
    return g


def struct_names(gguf_shapes: dict, hf_shapes: dict) -> dict[str, str]:
    """{gguf tensor name: hf tensor name}, paired by block index and unique shape.
    Shapes are dicts {name: (kind, shape)} with the GGUF shape already in Hugging Face order."""
    g, h = _groups(gguf_shapes, True), _groups(hf_shapes, False)
    nb_g = len([k for k in g if k is not None])
    nb_h = len([k for k in h if k is not None])
    if nb_g != nb_h:
        return {}
    out = {}
    for key, gs in g.items():
        hs = h.get(key)
        if not hs:
            continue
        for shape, names in gs.items():
            other = hs.get(shape)
            if len(names) == 1 and other and len(other) == 1:
                out[names[0]] = other[0]
    return out
