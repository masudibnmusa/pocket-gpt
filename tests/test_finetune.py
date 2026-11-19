import random

import torch

from finetune import make_batch, make_request, parse
from model import GPT, GPTConfig


def test_parse_fields_before_and_after_story():
    fields, story = parse("Words: oak, quit\nStory: \nTom saw an oak.\n\"Hi!\" he said.\nSummary: Tom sees a tree.\n")
    assert fields == {"Words": "oak, quit", "Summary": "Tom sees a tree."}
    assert story == 'Tom saw an oak.\n"Hi!" he said.'


def test_loss_only_on_response_tokens():
    # ids = prompt [10, 11, 12] + response [20, 21, 99]; second example is shorter, so it gets padded
    x, y = make_batch([([10, 11, 12, 20, 21, 99], 3, {}), ([10, 11, 20, 99], 2, {})], "cpu")
    assert x[0].tolist() == [10, 11, 12, 20, 21]
    assert y[0].tolist() == [-100, -100, 20, 21, 99]  # position 2 (last prompt token) predicts the first story token
    assert y[1].tolist() == [-100, 20, 99, -100, -100]  # prompt and padding ignored


def test_chat_request_always_names_every_required_word():
    fields = {"Words": "oak, quit, gloomy", "Features": "Dialogue, BadEnding", "Summary": "A tree story."}
    for seed in range(20):
        req = make_request(fields, random.Random(seed))
        assert all(f'"{w}"' in req for w in ("oak", "quit", "gloomy"))
        assert "Include some dialogue." in req and "Give it a sad ending." in req


def test_grow_vocab_keeps_old_logits_and_tying():
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=50, block_size=8, n_layer=1, n_head=2, n_embd=16)).eval()
    idx = torch.randint(50, (2, 8))
    before, _ = model(idx)
    model.grow_vocab(2)
    after, _ = model(idx)
    assert after.shape[-1] == 52 and model.cfg.vocab_size == 52
    assert torch.allclose(after[..., :50], before, atol=1e-6)
    assert model.lm_head.weight is model.tok_emb.weight
    model(torch.tensor([[50, 51]]))  # new ids embed without error
