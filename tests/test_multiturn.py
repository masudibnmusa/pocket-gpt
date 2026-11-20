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


def test_word_questions_refer_to_the_right_story():
    from multiturn import WORDS_FIRST_QS, WORDS_LAST_QS, WORDS_QS, follow_ups

    s1 = ({"Words": "oak, quit"}, "A girl named Sue saw an oak.")
    s2 = ({"Words": "moon, boat"}, "A girl named Ann sailed a boat.")
    qa = dict(follow_ups([s1, s2], random.Random(0)))
    latest, first = 'You asked me to use the words "moon" and "boat".', 'You asked me to use the words "oak" and "quit".'
    assert [qa[q] for q in qa if q in WORDS_QS] == [latest]  # unqualified means the latest story
    assert [qa[q] for q in qa if q in WORDS_LAST_QS] == [latest]
    assert [qa[q] for q in qa if q in WORDS_FIRST_QS] == [first]
    assert not any("name" in q for q in qa)  # "the girl" would be ambiguous: both stories have one


def test_small_talk_gets_fixed_replies_and_eval_phrasings_are_held_out():
    from multiturn import (EVAL_GREETINGS, EVAL_OFF_TOPIC, GREETING_REPLY, GREETINGS, OFF_TOPIC, OFF_TOPIC_REPLY,
                           small_talk)

    rng = random.Random(0)
    talk = [small_talk(rng) for _ in range(300)]
    assert {a for _, a in talk} == {GREETING_REPLY, OFF_TOPIC_REPLY}
    assert all((q in GREETINGS) == (a == GREETING_REPLY) for q, a in talk)
    assert not set(EVAL_GREETINGS) & set(GREETINGS) and not set(EVAL_OFF_TOPIC) & set(OFF_TOPIC)
    convs = build_conversations(TOK, [ENTRY] * 200, 256, random.Random(0))
    replies = [TOK.decode([t for t, m in zip(ids, mask) if m]) for ids, mask, _ in convs]
    assert any(GREETING_REPLY in r for r in replies) and any(OFF_TOPIC_REPLY in r for r in replies)



def test_casual_requests_never_get_word_questions():
    from multiturn import CASUAL, WORDS_QS, follow_ups, story_request

    fields = {"Words": "run, happy", "Summary": "A dog runs."}
    reqs = [story_request(fields, random.Random(k)) for k in range(100)]
    casual = [asked for req, asked in reqs if req in CASUAL]
    assert casual and len(casual) < len(reqs)
    for asked in casual:
        assert not any(q in WORDS_QS for q, _ in follow_ups([(asked, STORY)], random.Random(0)))
