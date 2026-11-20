"""Multi-turn chat fine-tuning: conversations where later turns can only be answered from earlier ones.

Turn 1 is a story request and its story (TinyStories-Instruct). Follow-ups (1-2 per conversation, random):
  - "What was the dog's name?"            -> from the story the assistant told   (when it says "a dog named Max")
  - "Can you sum it up in one sentence?"  -> the entry's summary
  - "Which words did I ask you to use?"   -> copied from the user's first message
  - "Tell me another story ..."           -> a second story, with the first still in context
Each assistant turn ends with <|endoftext|>; the loss covers every assistant turn and nothing else.

python multiturn.py --base runs/ts_xl_1024/ckpt.pt --out runs/ts_xl_chat
python chat.py --ckpt runs/ts_xl_chat/ckpt.pt
"""
import argparse
import os
import random
import re
from dataclasses import asdict

import torch

from data import DATA_DIR, EOT, download, load_tokenizer
from finetune import ASSISTANT, CHAT_TOKENS, URL, USER, make_request, parse, train_sft
from model import GPT

NAMED = re.compile(r"\b([a-z]+) (?:named|called) ([A-Z][a-z]+)\b")
NOT_NOUNS = {"was", "were", "and", "friends", "old", "boys", "children", "girls", "is", "are", "be", "also", "then"}
SUMMARY_QS = ["Can you sum it up in one sentence?", "What was the story about?", "Tell me what happened, briefly.",
              "Summarize the story for me."]
WORDS_QS = ["Which words did I ask you to use?", "What words did I want in the story?"]
WORDS_FIRST_QS = ["Which words did I ask for in the first story?", "What words did I want in the first story?"]
WORDS_LAST_QS = ["Which words did I ask for in the last story?", "What words did I want in the latest story?"]
ANOTHER = ["Now tell me another story.", "Thanks! One more, please.", "Can you tell me a different story?"]


def name_fact(story):
    """('dog', 'Max') from the first '<noun> named/called <Name>' in the story, else None."""
    m = NAMED.search(story)
    return (m.group(1), m.group(2)) if m and m.group(1) not in NOT_NOUNS else None


def words_of(fields):
    return [w.strip() for w in fields.get("Words", "").split(",") if w.strip()]


def quote_list(words):
    q = [f'"{w}"' for w in words]
    return q[0] if len(q) == 1 else f"{', '.join(q[:-1])} and {q[-1]}"


def words_answer(words):
    return f"You asked me to use the words {quote_list(words)}."


def follow_ups(stories, rng):
    """Candidate (question, answer) pairs about the conversation so far; stories = [(fields, story)] in order.
    Unqualified questions ("Which words did I ask you to use?", "sum it up") refer to the latest story;
    "first"/"last" questions name the story. Name questions only when the noun picks out a single story."""
    out = []
    facts = [name_fact(s) for _, s in stories]
    nouns = [f[0] for f in facts if f]
    for fact in facts:
        if fact and nouns.count(fact[0]) == 1:
            noun, name = fact
            out.append((f"What was the {noun}'s name?", f"The {noun}'s name was {name}."))
    fields = stories[-1][0]
    if "Summary" in fields:
        out.append((rng.choice(SUMMARY_QS), fields["Summary"]))
    if words_of(fields):
        out.append((rng.choice(WORDS_QS), words_answer(words_of(fields))))
    if len(stories) > 1:
        if words_of(stories[0][0]):
            out.append((rng.choice(WORDS_FIRST_QS), words_answer(words_of(stories[0][0]))))
        if words_of(fields):
            out.append((rng.choice(WORDS_LAST_QS), words_answer(words_of(fields))))
    return out


def encode_turn(tok, user_text, assistant_text):
    """-> (ids, mask): mask is True on the assistant's tokens and its closing EOT (what the model learns)."""
    u = [tok.special_ids[USER]] + tok.encode_ordinary(user_text) + [tok.special_ids[ASSISTANT]]
    a = tok.encode_ordinary(assistant_text) + [tok.special_ids[EOT]]
    return u + a, [False] * len(u) + [True] * len(a)


def build_conversations(tok, entries, block_size, rng):
    """[(ids, mask, meta)]; meta has turn 1's name fact (if any) for the eval. Conversations fit in block_size + 1.
    Each conversation: a story request, then 1-3 turns, each either a second story (once) or a question."""
    parsed = [(f, s) for f, s in map(parse, entries) if f and s]
    convs = []
    for i, (fields, story) in enumerate(parsed):
        stories = [(fields, story)]
        ids, mask = encode_turn(tok, make_request(fields, rng), story)
        for _ in range(rng.randint(1, 3)):
            if len(stories) == 1 and rng.random() < 0.35:
                f2, s2 = parsed[(i + 1) % len(parsed)]
                q, a, new = rng.choice(ANOTHER) + " " + make_request(f2, rng), s2, (f2, s2)
            else:
                options = follow_ups(stories, rng)
                if not options:
                    break
                (q, a), new = rng.choice(options), None
            t_ids, t_mask = encode_turn(tok, q, a)
            if len(ids) + len(t_ids) > block_size + 1:
                break
            ids, mask = ids + t_ids, mask + t_mask
            if new:
                stories.append(new)
        if len(ids) <= block_size + 1 and any(mask):
            convs.append((ids, mask, {"fields": fields, "story": story, "fact": name_fact(story)}))
    return convs


def load_entries(split, max_mb):
    raw = os.path.join(DATA_DIR, f"tinystories_instruct_{split}{f'_{max_mb}mb' if max_mb else ''}.txt")
    download(URL.format("train" if split == "train" else "valid"), raw, max_mb * 10**6 if max_mb else None)
    return open(raw, encoding="utf-8", errors="ignore").read().split(EOT)[1:-1]


@torch.no_grad()
def name_recall(model, tok, eval_items, device, shuffle_history=False):
    """Teacher-force turn 1 (request + dataset story), ask for a character's name, generate greedily.
    shuffle_history: put a *different* conversation's story in turn 1, to check the answer comes from context."""
    eot, hits = tok.special_ids[EOT], 0
    for k, (fields, story, (noun, name)) in enumerate(eval_items):
        hist_story = eval_items[(k + 1) % len(eval_items)][1] if shuffle_history else story
        ids, _ = encode_turn(tok, make_request(fields, random.Random(k)), hist_story)
        q = [tok.special_ids[USER]] + tok.encode_ordinary(f"What was the {noun}'s name?") + [tok.special_ids[ASSISTANT]]
        prompt = (ids + q)[-(model.cfg.block_size - 20) :]
        out = model.generate(torch.tensor([prompt], device=device), 20, temperature=0, eos_id=eot)
        hits += name in tok.decode(out[0, len(prompt) :].tolist()).split(EOT)[0]
    return hits / len(eval_items)


@torch.no_grad()
def words_recall(model, tok, story_pairs, device, which):
    """Teacher-force two story turns (both with required words), then ask which words were requested:
    which = "plain" (unqualified, means the latest), "first" or "last". Correct = every expected word quoted in the
    answer and none of the other request's words."""
    q_text = {"plain": WORDS_QS[0], "first": WORDS_FIRST_QS[0], "last": WORDS_LAST_QS[0]}[which]
    hits = 0
    for k, ((f1, s1), (f2, s2)) in enumerate(story_pairs):
        r = random.Random(k)
        t1, _ = encode_turn(tok, make_request(f1, r), s1)
        t2, _ = encode_turn(tok, ANOTHER[0] + " " + make_request(f2, r), s2)
        q = [tok.special_ids[USER]] + tok.encode_ordinary(q_text) + [tok.special_ids[ASSISTANT]]
        prompt = (t1 + t2 + q)[-(model.cfg.block_size - 40) :]
        out = model.generate(torch.tensor([prompt], device=device), 40, temperature=0, eos_id=tok.special_ids[EOT])
        ans = tok.decode(out[0, len(prompt) :].tolist()).split(EOT)[0].lower()
        want = words_of(f1 if which == "first" else f2)
        other = set(words_of(f2 if which == "first" else f1)) - set(want)
        hits += all(f'"{w.lower()}"' in ans for w in want) and not any(f'"{o.lower()}"' in ans for o in other)
    return hits / len(story_pairs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="runs/ts_xl_1024/ckpt.pt")
    p.add_argument("--out", default="runs/ts_xl_chat")
    p.add_argument("--train_mb", type=int, default=100)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch_size", type=int, default=8, help="16 x 1024 tokens does not fit in 8 GB")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--eval_interval", type=int, default=500)
    p.add_argument("--eval_n", type=int, default=200)
    p.add_argument("--eval_only", action="store_true")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(1337)
    base, ckpt = GPT.from_checkpoint(args.base, device)
    data_dir = ckpt["args"]["data"]
    tok = load_tokenizer(data_dir, CHAT_TOKENS)
    bs = base.cfg.block_size
    val = build_conversations(tok, load_entries("val", None), bs, random.Random(1))
    print(f"val: {len(val):,} conversations, {sum(v[2]['fact'] is not None for v in val):,} with a name fact")
    ckpt_path = os.path.join(args.out, "ckpt.pt")

    if not args.eval_only:
        train = build_conversations(tok, load_entries("train", args.train_mb), bs, random.Random(0))
        n_tok = sum(len(c[0]) for c in train)
        print(f"train: {len(train):,} conversations, {n_tok / len(train):.0f} tokens on average")
        model = base
        if model.cfg.vocab_size < tok.vocab_size:  # a pretrained base: add <|user|> / <|assistant|>
            model.grow_vocab(tok.vocab_size - model.cfg.vocab_size)
        train_sft(model, train, val, device, args.steps, args.batch_size, args.lr, args.warmup, args.eval_interval,
                  label="chat")
        os.makedirs(args.out, exist_ok=True)
        torch.save({"model": model.state_dict(), "config": asdict(model.cfg), "step": args.steps,
                    "args": {**vars(args), "data": data_dir, "extra_special_tokens": CHAT_TOKENS}}, ckpt_path)
        print(f"saved {ckpt_path}")

    model, _ = GPT.from_checkpoint(ckpt_path, device)
    items = [(v[2]["fields"], v[2]["story"], v[2]["fact"]) for v in val if v[2]["fact"]][: args.eval_n]
    real = name_recall(model, tok, items, device)
    swapped = name_recall(model, tok, items, device, shuffle_history=True)
    print(f"\nname recall ({len(items)} held-out conversations): {real:.0%} with the real story in context, "
          f"{swapped:.0%} with a different story swapped in")
    with_words = [(f, s) for f, s in map(parse, load_entries("val", None)) if s and len(words_of(f)) >= 2]
    story_pairs = list(zip(with_words[0::2], with_words[1::2]))[: args.eval_n]
    print(f"words recall ({len(story_pairs)} two-story conversations): " + ", ".join(
        f"{w} {words_recall(model, tok, story_pairs, device, w):.0%}" for w in ("plain", "first", "last")))


if __name__ == "__main__":
    main()
