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


def test_error_agreement_and_dark_knowledge():
    """E5e: on the teacher's wrong answers, how often the other model gives the same wrong answer;
    and the divergence once the teacher's top class is removed. Without truth, E5's numbers only."""
    import torch
    from experiments.e5_behaviour import Stats
    t = torch.tensor([[5.0, 1.0, 0.0], [0.0, 5.0, 1.0], [1.0, 0.0, 5.0], [5.0, 0.0, 1.0]])
    copy = t.clone()                                     # same errors, same dark knowledge
    other = torch.tensor([[0.0, 5.0, 1.0], [0.0, 5.0, 1.0], [5.0, 0.0, 1.0], [0.0, 1.0, 5.0]])
    truth = torch.tensor([1, 1, 0, 0])                   # teacher wrong at positions 0 and 2
    a, b, c = Stats(), Stats(), Stats()
    a.add(t, copy, truth)
    b.add(t, other, truth)
    c.add(t, other)
    ra, rb, rc = a.result(), b.result(), c.result()
    assert ra["teacher_wrong"] == 2 and ra["error_agreement"] == 1.0 and abs(ra["dark_js"]) < 1e-12
    assert rb["error_agreement"] == 0.0 and rb["dark_js"] > 0.05
    assert set(rc) == {"positions", "agreement", "js"} and rc["agreement"] == rb["agreement"]
