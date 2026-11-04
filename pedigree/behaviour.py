"""Second instrument (PLAN.md, E5): lineage from behaviour, where weights cannot be compared.

A distilled student or a resized model shares no tensor with its source, so the
weight verifier can only say "unverifiable". Behaviour can still be compared:

  linear_cka   centred kernel alignment (Kornblith et al. 2019) between two models'
               representations of the same probe inputs. Invariant to rotation and
               isotropic scaling of either representation, and defined for different
               widths, so it applies to a smaller student and its teacher.
  agreement    share of probe inputs where the two models' top-1 outputs agree, and
  js           mean Jensen-Shannon divergence of their output distributions, both only
               where the two models share an output space (same labels / vocabulary).

Behaviour is weaker evidence than weights: two independently trained models of the
same task can agree for honest reasons. E5 therefore calibrates against same-task
independent models, and the audit reports a behavioural verdict separately.
"""
from __future__ import annotations

import numpy as np


def _centre(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x - x.mean(0, keepdims=True)


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Linear CKA of two representation matrices with the same rows (probe inputs) and
    any number of columns: ||Y'X||_F^2 / (||X'X||_F ||Y'Y||_F), on centred features."""
    if x.shape[0] != y.shape[0]:
        raise ValueError("representations must cover the same probe inputs")
    x, y = _centre(x), _centre(y)
    num = np.linalg.norm(y.T @ x, "fro") ** 2
    den = np.linalg.norm(x.T @ x, "fro") * np.linalg.norm(y.T @ y, "fro")
    return float(num / den) if den > 0 else float("nan")


def agreement(p: np.ndarray, q: np.ndarray) -> float:
    """Share of rows where the two output distributions have the same argmax."""
    return float((np.asarray(p).argmax(-1) == np.asarray(q).argmax(-1)).mean())


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """Mean Jensen-Shannon divergence (base 2, in [0, 1]) between row distributions."""
    p = np.clip(np.asarray(p, dtype=np.float64), eps, None)
    q = np.clip(np.asarray(q, dtype=np.float64), eps, None)
    p /= p.sum(-1, keepdims=True)
    q /= q.sum(-1, keepdims=True)
    m = (p + q) / 2
    kl = lambda a, b: (a * np.log2(a / b)).sum(-1)  # noqa: E731
    return float(((kl(p, m) + kl(q, m)) / 2).mean())
