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


# --- v2 (PLAN.md, amendment 2026-11-18, H20): stacks, so unequal block counts still align ----------

_GGUF_STACK = re.compile(r"^(?:(.+)\.)?blk\.(\d+)\.")
_HF_STACK = re.compile(r"(?:^|\.)(layers|layer|h|blocks|block)\.(\d+)\.")


def _stack(name: str, gguf_side: bool) -> tuple[str, int] | None:
    if gguf_side:
        m = _GGUF_STACK.match(name)
        return ((m.group(1) + ".blk") if m.group(1) else "blk", int(m.group(2))) if m else None
    m = _HF_STACK.search(name)
    return (name[:m.end(1)], int(m.group(2))) if m else None


def _stacks(shapes: dict, gguf_side: bool):
    """({stack: {block: {shape: [names]}}}, {shape: [names]} for tensors without a block index)"""
    st: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    free: dict = defaultdict(list)
    for name, (_kind, shape) in shapes.items():
        s = _stack(name, gguf_side)
        if s is None:
            free[tuple(shape)].append(name)
        else:
            st[s[0]][s[1]][tuple(shape)].append(name)
    return st, free


def _unique_pairs(gs: dict, hs: dict, out: dict) -> None:
    for shape, names in gs.items():
        other = hs.get(shape)
        if len(names) == 1 and other and len(other) == 1:
            out[names[0]] = other[0]


def struct_names_v2(gguf_shapes: dict, hf_shapes: dict) -> dict[str, str]:
    """{gguf tensor name: hf tensor name}. GGUF stacks, largest first, each paired with the unpaired
    stack of the other side with the same block count, else the nearest (ties: more tensors); block
    i with block i below the smaller count; unique shapes within each block, and among tensors with
    no block index."""
    gst, gfree = _stacks(gguf_shapes, True)
    hst, hfree = _stacks(hf_shapes, False)

    def size(st, k):
        return sum(len(v) for blk in st[k].values() for v in blk.values())
    out: dict = {}
    free_h = set(hst)
    for g in sorted(gst, key=lambda k: -size(gst, k)):
        if not free_h:
            break
        ng = len(gst[g])
        h = min(free_h, key=lambda k: (abs(len(hst[k]) - ng), -size(hst, k)))
        free_h.discard(h)
        for i in range(min(len(gst[g]), len(hst[h]))):     # block i with block i
            if i in gst[g] and i in hst[h]:
                _unique_pairs(gst[g][i], hst[h][i], out)
    _unique_pairs(gfree, hfree, out)
    return out
