import math

import torch
import torch.nn.functional as F

from dpo import seq_logprob
from finetune import make_batch
from model import GPT, GPTConfig

CFG = GPTConfig(vocab_size=40, block_size=16, n_layer=1, n_head=2, n_embd=16)


def test_seq_logprob_counts_only_response_tokens():
    torch.manual_seed(0)
    model = GPT(CFG).eval()
    ids, plen = [1, 2, 3, 4, 5, 6], 3  # prompt 1 2 3, response 4 5 6
    x, y = make_batch([(ids, plen, None)], "cpu")
    with torch.no_grad():
        lp = seq_logprob(model, x, y)
        logits, _ = model(torch.tensor([ids[:-1]]))
    manual = sum(torch.log_softmax(logits[0, j], -1)[ids[j + 1]].item() for j in range(plen - 1, len(ids) - 1))
    assert math.isclose(lp.item(), manual, rel_tol=1e-5)


def test_dpo_loss_is_ln2_when_policy_equals_reference():
    torch.manual_seed(0)
    model = GPT(CFG).eval()
    rows = [([1, 2, 3, 4], 2, None), ([1, 2, 7, 8, 9], 2, None)]  # chosen, rejected
    x, y = make_batch(rows, "cpu")
    with torch.no_grad():
        pol = ref = seq_logprob(model, x, y)
    margin = 0.1 * ((pol[0] - ref[0]) - (pol[1] - ref[1]))
    assert math.isclose(-F.logsigmoid(margin).item(), math.log(2), rel_tol=1e-6)


def test_fluency_nll_matches_unbatched_loss_and_skips_short_replies():
    from dpo import fluency_nll

    torch.manual_seed(0)
    lm = GPT(CFG).eval()
    replies = [[1, 2, 3, 4, 5], [6, 7], [8], [9, 10, 50, 11]]  # 50 >= vocab: cut there (a role token)
    got = fluency_nll(lm, replies, "cpu", batch_size=3)
    with torch.no_grad():
        for r, g in zip([[1, 2, 3, 4, 5], [6, 7], [9, 10]], [got[0], got[1], got[3]]):
            x = torch.tensor([r])
            assert math.isclose(g, lm(x[:, :-1], x[:, 1:])[1].item(), rel_tol=1e-4)
    assert got[2] is None


def test_fluency_penalty_is_a_hinge_above_the_margin():
    from bpe import BPETokenizer
    from dpo import fluency_nll, make_reward

    torch.manual_seed(0)
    lm = GPT(CFG).eval()
    tok = BPETokenizer.train("ab" * 50, 260, ["<|endoftext|>"])
    replies = [[1, 2, 3, 4, 5], [6, 7, 8]]
    nll = fluency_nll(lm, replies, "cpu")
    m = (nll[0] + nll[1]) / 2  # between the two losses: exactly one reply is penalised
    got = make_reward(tok, "cpu", fluency_lm=lm, fluency_weight=10, fluency_margin=m)([([0], {}, r) for r in replies])
    assert sorted(got) == [-10 * (max(nll) - m), 0.0]


def test_token_kl_is_zero_for_same_model_and_matches_manual():
    from dpo import token_kl

    torch.manual_seed(0)
    p, q = torch.randn(2, 5, 7), torch.randn(2, 5, 7)
    y = torch.tensor([[-100, 1, 2, -100, 3], [-100, -100, 4, 5, 6]])
    assert token_kl(p, p, y).abs().item() < 1e-6
    lp, lq = torch.log_softmax(p, -1), torch.log_softmax(q, -1)
    manual = [(lp[i, t].exp() * (lp[i, t] - lq[i, t])).sum() for i in range(2) for t in range(5) if y[i, t] != -100]
    assert math.isclose(token_kl(p, q, y).item(), (sum(manual) / len(manual)).item(), rel_tol=1e-5)
