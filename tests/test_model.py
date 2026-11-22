"""Sanity checks that catch most transformer bugs before you spend GPU time."""
import math
from dataclasses import replace

import pytest
import torch

from model import GPT, GPTConfig

GPT2_STYLE = GPTConfig(vocab_size=65, block_size=16, n_layer=2, n_head=2, n_embd=32)
LLAMA_STYLE = replace(GPT2_STYLE, bias=False, norm="rmsnorm", mlp="swiglu", pos="rope")
CONFIGS = {
    "gpt2": GPT2_STYLE,
    "gpt2-manual-attn": replace(GPT2_STYLE, flash=False),
    "llama": LLAMA_STYLE,
    "llama-manual-attn": replace(LLAMA_STYLE, flash=False),
}
all_configs = pytest.mark.parametrize("cfg", CONFIGS.values(), ids=CONFIGS.keys())


@all_configs
def test_initial_loss_is_ln_vocab(cfg):
    torch.manual_seed(0)
    model = GPT(cfg)
    idx = torch.randint(cfg.vocab_size, (8, cfg.block_size))
    _, loss = model(idx, torch.randint(cfg.vocab_size, (8, cfg.block_size)))
    assert abs(loss.item() - math.log(cfg.vocab_size)) < 0.1


@all_configs
def test_no_future_leak(cfg):
    torch.manual_seed(0)
    model = GPT(cfg).eval()
    idx = torch.randint(cfg.vocab_size, (2, cfg.block_size))
    perturbed = idx.clone()
    perturbed[:, -1] = (perturbed[:, -1] + 1) % cfg.vocab_size
    a, _ = model(idx)
    b, _ = model(perturbed)
    assert torch.allclose(a[:, :-1], b[:, :-1], atol=1e-6), "earlier positions saw a future token"
    assert not torch.allclose(a[:, -1], b[:, -1]), "last position ignored its own token"


@all_configs
def test_overfit_one_batch(cfg):
    torch.manual_seed(0)
    model = GPT(cfg)
    x = torch.randint(cfg.vocab_size, (4, cfg.block_size))
    y = torch.randint(cfg.vocab_size, (4, cfg.block_size))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(300):
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1, f"could not memorize one batch, loss {loss.item():.3f}"


@pytest.mark.parametrize("base", [GPT2_STYLE, LLAMA_STYLE], ids=["gpt2", "llama"])
def test_manual_attention_matches_sdpa(base):
    torch.manual_seed(0)
    manual = GPT(replace(base, flash=False)).eval()
    flash = GPT(replace(base, flash=True)).eval()
    flash.load_state_dict(manual.state_dict())
    idx = torch.randint(base.vocab_size, (3, base.block_size))
    assert torch.allclose(manual(idx)[0], flash(idx)[0], atol=1e-5)


@all_configs
def test_kv_cache_matches_full_forward(cfg):
    from model import KVCache

    torch.manual_seed(0)
    model = GPT(cfg).eval()
    idx = torch.randint(cfg.vocab_size, (2, cfg.block_size))
    full, _ = model(idx)
    cache = KVCache(cfg.n_layer, cfg.block_size)
    first, _ = model(idx[:, :5], cache=cache)  # prefill 5, then one token at a time
    steps = [first] + [model(idx[:, t : t + 1], cache=cache)[0] for t in range(5, cfg.block_size)]
    assert torch.allclose(torch.cat(steps, dim=1), full, atol=1e-5)


@all_configs
def test_cached_generation_equals_uncached(cfg):
    torch.manual_seed(0)
    model = GPT(cfg).eval()
    start = torch.zeros((1, 1), dtype=torch.long)
    n = 2 * cfg.block_size  # runs past block_size, exercising the cache rebuild
    a = model.generate(start, n, temperature=0, use_cache=True)
    b = model.generate(start, n, temperature=0, use_cache=False)
    assert a.shape == (1, n + 1) and torch.equal(a, b)


def test_top_p_keeps_only_the_nucleus():
    torch.manual_seed(0)
    model = GPT(GPT2_STYLE).eval()
    with torch.no_grad():  # make the head strongly prefer tokens 3 then 7, regardless of input
        model.lm_head.weight.zero_()
    model.lm_head.weight.data[3] += 1.0
    model.lm_head.weight.data[7] += 0.9
    model.ln_f.weight.data.zero_()
    model.ln_f.bias.data.fill_(5.0)  # every hidden state -> all 5s, so logit_3 = 5*32, logit_7 = 4.5*32
    out = model.generate(torch.zeros((1, 1), dtype=torch.long), 50, temperature=1.0, top_p=0.5)
    assert set(out[0, 1:].tolist()) == {3}


def test_left_padded_generation_matches_unpadded():
    import torch

    from model import GPT, GPTConfig

    torch.manual_seed(0)
    m = GPT(GPTConfig(vocab_size=50, block_size=32, n_layer=2, n_head=2, n_embd=32, bias=False, norm="rmsnorm",
                      mlp="swiglu", pos="rope")).eval()
    short, long = [5, 6, 7], [1, 2, 3, 4, 8, 9, 10]
    alone = m.generate(torch.tensor([short]), 6, temperature=0)[0, 3:].tolist()
    batch = torch.tensor([[0] * 4 + short, long])
    pad = torch.tensor([[True] * 4 + [False] * 3, [False] * 7])
    both = m.generate(batch, 6, temperature=0, pad=pad)
    assert both[0, 7:].tolist() == alone
    assert both[1, 7:].tolist() == m.generate(torch.tensor([long]), 6, temperature=0)[0, 7:].tolist()
