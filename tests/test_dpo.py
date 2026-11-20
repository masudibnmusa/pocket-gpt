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
