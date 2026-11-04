"""Defence (ii) recovers a residual permutation from a fine-tuned, attacked model."""
import numpy as np
import pytest
import torch

pytest.importorskip("scipy")
transformers = pytest.importorskip("transformers")

from experiments.permute_full import attack, random_inputs  # noqa: E402
from pedigree.rebasin import match_columns, unpermute  # noqa: E402
from pedigree.residual import residual_axes  # noqa: E402
from pedigree.verify import compare, compare_invariant  # noqa: E402


class Mem:
    def __init__(self, sd, repo):
        self.repo, self.commit = repo, "mem"
        self.t = {k: v.detach().double().numpy() for k, v in sd.items() if v.is_floating_point()}

    def float_shapes(self):
        return {k: ("float", v.shape) for k, v in self.t.items()}

    def sample(self, name, max_bytes=32 * 1024):
        return self.t[name].ravel()[: max_bytes // 4]


def test_match_columns_recovers_a_permutation():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2000, 64))
    p = rng.permutation(64)
    noisy = x[:, p] + 0.05 * rng.standard_normal(x.shape)
    r = match_columns(noisy, x)
    assert np.array_equal(noisy[:, r].argmax(0), x.argmax(0)) or np.array_equal(p[r], np.arange(64))


def test_attacked_model_is_recovered():
    from transformers import GPT2Config, GPT2LMHeadModel
    torch.manual_seed(0)
    cfg = GPT2Config(vocab_size=2000, n_embd=64, n_layer=2, n_head=4)
    parent = GPT2LMHeadModel(cfg)
    child = GPT2LMHeadModel(cfg)
    child.load_state_dict({k: v + 0.004 * torch.randn_like(v) if v.is_floating_point() else v
                           for k, v in parent.state_dict().items()})       # a fine-tune: 20% of the init scale
    child, report = attack(child, random_inputs(child, "text_decoder"), seed=5)
    assert report["verified"]
    stranger = GPT2LMHeadModel(cfg)                                          # independent init
    axes = residual_axes(parent, 64)
    arch = {k: ("float", tuple(v.shape)) for k, v in parent.state_dict().items() if v.is_floating_point()}
    P, C, S = Mem(parent.state_dict(), "p"), Mem(child.state_dict(), "c"), Mem(stranger.state_dict(), "s")
    assert compare(C, P)["median_cosine"] < 0.5                              # the attack works
    fixed, info = unpermute(C, P, axes, arch)
    assert compare_invariant(fixed, P)["median_invariant"] > 0.9            # the defence restores it
    fixed_s, _ = unpermute(S, P, axes, arch)
    assert compare_invariant(fixed_s, P)["median_invariant"] < 0.3          # and invents nothing
