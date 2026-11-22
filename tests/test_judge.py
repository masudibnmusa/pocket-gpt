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


def test_paste_request_swaps_sentences_for_the_summary_at_about_the_same_length():
    from finetune import parse
    from judge import paste_request

    story = ('Tom saw a dog.\nThe dog said, "Hi!" Tom smiled. They played all day. Then they ran home. '
             "Mom made soup. Tom told her about the dog. She laughed. They ate together.")
    fields = {"Summary": "Tom meets a dog and they play."}
    for k in range(30):
        bad = paste_request(story, fields, random.Random(k))
        assert bad != story and "Tom meets a dog and they play." in bad
        assert 0.8 < len(bad) / len(story) < 1.25  # no length cue for the judge


def test_conversation_pairs_share_history_and_fit():
    from bpe import BPETokenizer
    from finetune import ASSISTANT, USER
    from judge import MAX_LEN, conversation_pairs
    from multiturn import build_conversations

    story = "Once there was a little dog named Max. Max liked to run. He was happy."
    entries = [f"Words: run, happy\nSummary: A dog runs.\nStory: \n{story}\n",
               "Words: moon, boat\nSummary: A girl sails.\nStory: \nA girl named Sue sailed a boat to the moon.\n"] * 40
    tok = BPETokenizer.train("".join(entries) * 3, 300, ["<|endoftext|>", USER, ASSISTANT])
    pairs = conversation_pairs(tok, build_conversations(tok, entries, 1024, random.Random(0)), random.Random(0))
    assert pairs
    for good, bad, kind in pairs:
        assert kind == "conv" and good != bad and max(len(good), len(bad)) <= MAX_LEN
        assert good[0] == bad[0] == tok.special_ids[USER] and good[-1] == bad[-1] == tok.special_ids["<|endoftext|>"]
        a = len(good) - 1 - good[::-1].index(tok.special_ids[ASSISTANT])  # the judged reply follows the last <|assistant|>
        assert good[: a + 1] == bad[: a + 1]
