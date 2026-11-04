"""The E1/E3/E4 weight transforms do what they claim."""
import pytest

torch = pytest.importorskip("torch")

from experiments.transforms import permute_hidden, prune, quantize_q4_0, quantize_q8_0  # noqa: E402


def _w(*shape, seed=0):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))


def test_q8_0_error_is_within_half_a_step_per_block():
    x = _w(64, 96)
    y = quantize_q8_0({"w": x})["w"]
    blocks = x.reshape(-1, 32)
    step = blocks.abs().amax(1, keepdim=True) / 127
    assert ((y.reshape(-1, 32) - blocks).abs() <= step / 2 + 1e-6).all()


def test_q4_0_matches_llama_cpp_rule_on_a_hand_checked_block():
    x = torch.tensor([[-8.0] + [1.0] * 31])        # m = -8 -> d = 1
    y = quantize_q4_0({"w": x})["w"]
    assert y[0, 0] == -8.0 and (y[0, 1:] == 1.0).all()
    big = _w(32, 64)
    err = (quantize_q4_0({"w": big})["w"] - big).abs()
    assert err.max() <= big.abs().max() / 8 + 1e-6  # at most one step


def test_prune_zeroes_the_requested_share_and_keeps_vectors():
    x, b = _w(100, 100), _w(100)
    y = prune({"w": x, "b": b}, 0.9)
    assert abs((y["w"] == 0).float().mean().item() - 0.9) < 0.01
    assert torch.equal(y["b"], b)


@pytest.mark.parametrize("names,layout", [
    (("enc.layer.0.intermediate.dense", "enc.layer.0.output.dense"), "linear"),
    (("h.0.mlp.c_fc", "h.0.mlp.c_proj"), "conv1d"),
    (("layers.0.fc1", "layers.0.fc2"), "linear"),
])
def test_permutation_changes_weights_but_not_the_function(names, layout):
    up, down = names
    H, I = 8, 32
    if layout == "conv1d":
        sd = {f"{up}.weight": _w(H, I, seed=1), f"{up}.bias": _w(I, seed=2), f"{down}.weight": _w(I, H, seed=3)}
    else:
        sd = {f"{up}.weight": _w(I, H, seed=1), f"{up}.bias": _w(I, seed=2), f"{down}.weight": _w(H, I, seed=3)}
    out, rep = permute_hidden(sd, seed=0)
    assert rep["ffn"] == 1 and rep["verified"]
    assert not torch.equal(out[f"{up}.weight"], sd[f"{up}.weight"])
    assert sorted(out[f"{up}.bias"].tolist()) == sorted(sd[f"{up}.bias"].tolist())


def test_resnet_bottleneck_permutation_is_function_preserving():
    c_in, c = 6, 10
    sd = {"layer1.0.conv1.weight": _w(c, c_in, 1, 1, seed=4), "layer1.0.conv2.weight": _w(c, c, 3, 3, seed=5),
          "layer1.0.bn1.weight": _w(c, seed=6), "layer1.0.bn1.bias": _w(c, seed=7),
          "layer1.0.bn1.running_mean": _w(c, seed=8), "layer1.0.bn1.running_var": _w(c, seed=9).abs() + 0.5}
    out, rep = permute_hidden(sd, seed=0)
    assert rep["resnet_bottleneck"] == 1 and rep["verified"]
    assert not torch.equal(out["layer1.0.conv1.weight"], sd["layer1.0.conv1.weight"])
