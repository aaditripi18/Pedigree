"""The invariant check survives a permutation that defeats the plain cosine, and stays
low for unrelated matrices."""
import numpy as np

from pedigree.verify import cosine, gram_cosine


def test_invariant_to_row_and_column_permutation():
    rng = np.random.default_rng(0)
    base = rng.standard_normal((512, 128)) @ np.diag(np.linspace(0.2, 2, 128))  # structured, as trained weights
    child = base + 0.1 * rng.standard_normal(base.shape)                          # a fine-tune
    rows = child[rng.permutation(512)]
    cols = child[:, rng.permutation(128)]
    assert cosine(base.ravel(), rows.ravel()) < 0.2                                # the plain check is fooled
    assert gram_cosine(base, rows) > 0.9 and gram_cosine(base, cols) > 0.9


def test_low_for_unrelated_matrices():
    rng = np.random.default_rng(1)
    a, b = rng.standard_normal((512, 128)), rng.standard_normal((512, 128))
    assert abs(gram_cosine(a, b)) < 0.1
