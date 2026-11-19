"""Our GPT with OpenAI's GPT-2 (124M) weights must produce the same logits as Hugging Face's implementation."""
import pytest
import torch

pytest.importorskip("transformers")


def test_gpt2_logits_match_huggingface():
    from gpt2 import load_gpt2

    ours, hf = load_gpt2("gpt2")
    torch.manual_seed(0)
    idx = torch.randint(ours.cfg.vocab_size, (2, 64))
    with torch.no_grad():
        a, _ = ours(idx)
        b = hf(idx).logits
    assert (a - b).abs().max().item() < 1e-3
    # and with the KV cache, one token at a time
    with torch.no_grad():
        assert torch.equal(ours.generate(idx[:1, :8], 20, temperature=0, use_cache=True),
                           ours.generate(idx[:1, :8], 20, temperature=0, use_cache=False))
