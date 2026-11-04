"""E3 defence (ii): undo a residual-stream permutation, then compare (PLAN.md).

A model whose residual stream was permuted still carries its parent's embedding
matrix, column for column, only in another order. Matching the suspect's embedding
columns to the candidate parent's (Hungarian algorithm on their correlations, the
weight-matching idea of Git Re-Basin, Ainsworth et al. 2023) recovers the permutation;
undoing it on every residual axis (pedigree.residual) leaves only one-sided
permutations (feed-forward hidden units), which the Gram invariant ignores.

The same matching is run on every pair, related or not, so the similarity matching
manufactures between unrelated models is measured on the negatives.
"""
from __future__ import annotations

import numpy as np

from .align import align
from .verify import full

ANCHOR_BYTES = 32 * 1024 * 1024
ANCHOR_HINT = ("word_embeddings", "wte", "embed_in", "embed_tokens", "patch_embeddings")


def _std(x: np.ndarray) -> np.ndarray:
    x = x - x.mean(0, keepdims=True)
    return x / (np.linalg.norm(x, axis=0, keepdims=True) + 1e-12)


def match_columns(suspect: np.ndarray, parent: np.ndarray) -> np.ndarray:
    """r with suspect[:, r] ~ parent: r[k] is the suspect column matched to parent column k."""
    from scipy.optimize import linear_sum_assignment
    n = min(suspect.shape[0], parent.shape[0])
    c = _std(parent[:n]).T @ _std(suspect[:n])
    rows, cols = linear_sum_assignment(-c)
    r = np.empty(c.shape[0], dtype=np.int64)
    r[rows] = cols
    return r


def _as_stream_last(x: np.ndarray, axis: int) -> np.ndarray:
    return np.moveaxis(x, axis, -1).reshape(-1, x.shape[axis])


def _anchor(model, name: str, shape: tuple, axis: int) -> np.ndarray:
    if axis == len(shape) - 1:   # leading bytes are whole rows: read only those
        d = shape[-1]
        rows = max(1, min(int(np.prod(shape[:-1])), ANCHOR_BYTES // (4 * d)))
        return model.sample(name, max_bytes=rows * d * 4)[: rows * d].reshape(rows, d)
    return _as_stream_last(full(model, name), axis)


class Unpermuted:
    """A model seen through the inverse of a recovered residual permutation."""

    def __init__(self, model, axes: dict[str, list[int]], arch_shapes: dict, r: np.ndarray):
        self.repo, self.commit, self.model, self.r = model.repo, model.commit, model, r
        pairing, _, _ = align(model.float_shapes(), arch_shapes)
        self.axes = {n: axes[a] for n, a in pairing.items() if a in axes}
        self._cache: dict[str, np.ndarray] = {}

    def float_shapes(self):
        return self.model.float_shapes()

    def sample(self, name: str, max_bytes: int = 32 * 1024) -> np.ndarray:
        if name not in self.axes:
            return self.model.sample(name, max_bytes)
        if name not in self._cache:
            x = full(self.model, name)
            for a in self.axes[name]:
                if x.shape[a] == len(self.r):
                    x = np.take(x, self.r, axis=a)
            self._cache[name] = x.ravel()
        return self._cache[name][: max_bytes // 4]


def unpermute(suspect, parent, axes: dict[str, list[int]], arch_shapes: dict) -> tuple[Unpermuted, dict]:
    """Recover the residual permutation from the best anchor both models share."""
    s_pair, _, _ = align(suspect.float_shapes(), arch_shapes)
    p_pair, _, _ = align(parent.float_shapes(), arch_shapes)
    s_of = {a: n for n, a in s_pair.items()}
    p_of = {a: n for n, a in p_pair.items()}
    cands = [a for a, ax in axes.items() if len(ax) == 1 and a in s_of and a in p_of
             and len(arch_shapes[a][1]) >= 2]
    if not cands:
        raise ValueError("no shared tensor carries the residual stream alone")
    cands.sort(key=lambda a: (not any(h in a for h in ANCHOR_HINT), -int(np.prod(arch_shapes[a][1]))))
    a = cands[0]
    ax = axes[a][0]
    shape = arch_shapes[a][1]
    r = match_columns(_anchor(suspect, s_of[a], shape, ax), _anchor(parent, p_of[a], shape, ax))
    return Unpermuted(suspect, axes, arch_shapes, r), {"anchor": a, "identity_share": float(np.mean(r == np.arange(len(r))))}
