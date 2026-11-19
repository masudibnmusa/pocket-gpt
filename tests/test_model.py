"""Sanity checks that catch most transformer bugs before you spend GPU time."""
import math

import torch

from model import GPT, GPTConfig

CFG = GPTConfig(vocab_size=65, block_size=16, n_layer=2, n_head=2, n_embd=32)


def test_initial_loss_is_ln_vocab():
    torch.manual_seed(0)
    model = GPT(CFG)
    idx = torch.randint(CFG.vocab_size, (8, CFG.block_size))
    _, loss = model(idx, torch.randint(CFG.vocab_size, (8, CFG.block_size)))
    assert abs(loss.item() - math.log(CFG.vocab_size)) < 0.1


def test_no_future_leak():
    torch.manual_seed(0)
    model = GPT(CFG).eval()
    idx = torch.randint(CFG.vocab_size, (2, CFG.block_size))
    perturbed = idx.clone()
    perturbed[:, -1] = (perturbed[:, -1] + 1) % CFG.vocab_size
    a, _ = model(idx)
    b, _ = model(perturbed)
    assert torch.allclose(a[:, :-1], b[:, :-1], atol=1e-6), "earlier positions saw a future token"
    assert not torch.allclose(a[:, -1], b[:, -1]), "last position ignored its own token"


def test_overfit_one_batch():
    torch.manual_seed(0)
    model = GPT(CFG)
    x = torch.randint(CFG.vocab_size, (4, CFG.block_size))
    y = torch.randint(CFG.vocab_size, (4, CFG.block_size))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(300):
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1, f"could not memorize one batch, loss {loss.item():.3f}"


def test_generate_shape():
    model = GPT(CFG).eval()
    out = model.generate(torch.zeros((1, 1), dtype=torch.long), 20, top_k=5)
    assert out.shape == (1, 21)  # also exercises cropping past block_size=16
