import random

import torch

from judge import Judge, features_of, judge_request, score
from model import GPTConfig


def test_score_ignores_padding_and_batch_mates():
    torch.manual_seed(0)
    judge = Judge(GPTConfig(vocab_size=30, block_size=32, n_layer=1, n_head=2, n_embd=16)).eval()
    a, b = [1, 2, 3, 4], [5, 6, 7, 8, 9, 10, 11]
    alone = score(judge, [a], "cpu")[0]
    together = score(judge, [a, b], "cpu")  # a is right-padded to b's length here
    assert abs(alone - together[0]) < 1e-5


def test_feature_request_mentions_only_features():
    fields = {"Features": "Dialogue, BadEnding", "Words": "oak", "Summary": "A tree."}
    req = judge_request(fields, "features", random.Random(0))
    assert "dialogue" in req and "sad ending" in req and "oak" not in req and "tree" not in req.lower()
    assert features_of({"Features": "Dialogue, Unknown"}) == ["Dialogue"]


def test_paste_request_inserts_summary_between_sentences_and_keeps_the_story():
    from judge import paste_request

    story = 'Tom saw a dog.\nThe dog said, "Hi!" Tom smiled. They played all day.'
    fields = {"Summary": "Tom meets a dog."}
    for k in range(20):
        bad = paste_request(story, fields, random.Random(k))
        assert bad.count("Tom meets a dog.") in (2, 3)
        assert bad.replace("Tom meets a dog. ", "") == story  # only whole phrases added, newlines kept
