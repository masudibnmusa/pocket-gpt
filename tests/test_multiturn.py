import random

from bpe import BPETokenizer
from finetune import ASSISTANT, USER, make_batch
from multiturn import build_conversations, encode_turn, name_fact

EOT = "<|endoftext|>"
STORY = "Once there was a little dog named Max. Max liked to run. He was happy."
ENTRY = f"Words: run, happy\nFeatures: Dialogue\nSummary: A dog runs.\nStory: \n{STORY}\n"
TOK = BPETokenizer.train((STORY + " " + ENTRY) * 5, 300, [EOT, USER, ASSISTANT])


def test_name_fact():
    assert name_fact(STORY) == ("dog", "Max")
    assert name_fact("There was a girl. She was named Sue.") is None  # "was" is not a noun


def test_turn_mask_covers_assistant_and_eot_only():
    ids, mask = encode_turn(TOK, "hi", "hello")
    a = ids.index(TOK.special_ids[ASSISTANT])
    assert mask == [False] * (a + 1) + [True] * (len(ids) - a - 1)
    assert ids[-1] == TOK.special_ids[EOT]


def test_conversations_fit_and_learn_every_assistant_turn():
    convs = build_conversations(TOK, [ENTRY] * 30, 256, random.Random(0))
    assert convs and all(len(ids) <= 257 for ids, _, _ in convs)
    ids, mask, meta = max(convs, key=lambda c: c[0].count(TOK.special_ids[USER]))
    assert ids.count(TOK.special_ids[USER]) >= 2 and meta["fact"] == ("dog", "Max")
    # every learned span starts right after an <|assistant|> and ends with EOT; user turns are never learned
    starts = [j for j in range(1, len(ids)) if mask[j] and not mask[j - 1]]
    assert all(ids[j - 1] == TOK.special_ids[ASSISTANT] for j in starts)
    assert all(not m for t, m in zip(ids, mask) if t == TOK.special_ids[USER])
    # make_batch turns the mask into targets: learned tokens appear as targets, nothing else
    x, y = make_batch([(ids, mask, meta)], "cpu")
    assert sorted(y[0][y[0] != -100].tolist()) == sorted(t for t, m in zip(ids, mask) if m)
