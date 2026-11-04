"""The behavioural measures behave as their definitions say."""
import numpy as np
import pytest

from pedigree.behaviour import agreement, js_divergence, linear_cka


def test_cka_is_invariant_to_rotation_scale_and_width():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((200, 32))
    q, _ = np.linalg.qr(rng.standard_normal((32, 32)))
    assert linear_cka(x, 3.0 * x @ q) == pytest.approx(1.0, abs=1e-9)
    # a wider representation that contains x (a "student" vs "teacher" of another width)
    y = np.hstack([x @ q, np.zeros((200, 16))])
    assert linear_cka(x, y) == pytest.approx(1.0, abs=1e-9)


def test_cka_is_low_for_unrelated_representations():
    rng = np.random.default_rng(1)
    x, y = rng.standard_normal((2000, 16)), rng.standard_normal((2000, 16))
    assert linear_cka(x, y) < 0.05


def test_cka_needs_the_same_probes():
    with pytest.raises(ValueError):
        linear_cka(np.zeros((3, 2)), np.zeros((4, 2)))


def test_js_and_agreement():
    p = np.array([[0.9, 0.1], [0.2, 0.8]])
    assert js_divergence(p, p) == pytest.approx(0.0, abs=1e-12)
    assert js_divergence(np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]])) == pytest.approx(1.0, abs=1e-6)
    assert agreement(p, p[::-1]) == 0.0 and agreement(p, p) == 1.0
