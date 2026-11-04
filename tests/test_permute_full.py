"""Attack (b) preserves the function of every transformer family it covers, and changes
every residual tensor (small random configs, no downloads)."""
import pytest
import torch

from experiments.permute_full import attack, random_inputs

transformers = pytest.importorskip("transformers")


def _models():
    from transformers import (BertConfig, BertForMaskedLM, GPT2Config, GPT2LMHeadModel, GPTNeoXConfig,
                              GPTNeoXForCausalLM, RobertaConfig, RobertaForMaskedLM, ViTConfig,
                              ViTForImageClassification)
    small = dict(hidden_size=64, num_hidden_layers=2, num_attention_heads=4, intermediate_size=128)
    yield "text_encoder", BertForMaskedLM(BertConfig(vocab_size=100, **small))
    yield "text_encoder", RobertaForMaskedLM(RobertaConfig(vocab_size=100, **small))
    yield "text_decoder", GPT2LMHeadModel(GPT2Config(vocab_size=100, n_embd=64, n_layer=2, n_head=4))
    yield "text_decoder", GPTNeoXForCausalLM(GPTNeoXConfig(vocab_size=100, **small))
    yield "vision_transformer", ViTForImageClassification(ViTConfig(image_size=32, patch_size=8, num_labels=10,
                                                                    **small))


@pytest.mark.parametrize("family,model", list(_models()), ids=lambda x: type(x).__name__ if not isinstance(x, str) else x)
def test_attack_preserves_function_and_moves_weights(family, model):
    torch.manual_seed(0)
    original = {k: v.clone() for k, v in model.state_dict().items()}
    model, report = attack(model, random_inputs(model, family), seed=3)
    assert report["verified"], report
    assert report["ffn_blocks"] == 2 and report["residual_tensors"] > 10
    changed = [k for k, v in model.state_dict().items()
               if v.is_floating_point() and v.numel() >= 64 and not torch.equal(v, original[k])]
    big = [k for k, v in original.items() if v.is_floating_point() and v.numel() >= 64 * 64]
    assert set(big) <= set(changed), set(big) - set(changed)   # every large matrix moved
