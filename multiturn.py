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
# Small talk, off-topic questions and gibberish get fixed replies, so the model stops answering every message that
# isn't a story question with a summary. The EVAL_ lists are held out to check it generalises beyond these phrasings.
GREET_WORDS = ["hi", "hello", "hey", "hey there", "hi there", "hello there", "good morning", "good afternoon", "good day",
               "greetings", "hi friend", "hello friend", "morning", "heya", "hullo", "hi buddy", "hey friend",
               "hello bot", "hi storyteller", "hey you", "hello again", "hi again",
               # short interjections, so a short unknown word that sounds friendly isn't taken for random letters
               "sup", "ahoy", "aloha", "hola", "bonjour", "g'day", "heyo", "hi hi", "hello hello", "hey buddy",
               "what's up", "wassup", "hellooo", "hii", "heyyy"]
GREET_ENDS = ["", "!", ".", "!!", " :)", "~", ","]
GREET_MORE = ["How are you?", "How are you doing?", "What's up?", "Nice to meet you.", "Are you there?",
              "How's it going?", "Anyone there?", "How is your day?"]
GREETING_REPLY = "Hi! I can tell you a story. What should it be about?"
# Off-topic messages are assembled too, half of them commands: with questions only, the model learned "question ->
# off-topic, anything else -> greeting" and greeted "Translate cat into French." (the Translate pattern stays held out).
OFF_TOPIC_QS = ["What is {thing}?", "Do you know {thing}?", "Tell me about {thing}.", "Why does {event}?",
                "How do I {task}?", "Can you {task}?", "Could you please {task}?", "Who {did}?", "When did {event2}?",
                "Where is {place}?", "How far is {place}?", "Is it true that {fact}?"]
OFF_TOPIC_CMDS = ["{Task}.", "{Task}, please.", "Please {task}.", "I need you to {task}.", "Help me {task}.",
                  "Explain {thing}.", "Calculate {a} times {b}.", "Solve {a} + {b}.", "Spell {word} backwards.",
                  "Define the word {word}.", "Give me directions to {place}.", "List {things}.", "Book a flight to {place}.",
                  "Set an alarm for {a} am.", "Convert {a} miles to kilometers.", "Recommend a good movie.",
                  "Summarize today's news.", "Check my email.", "Turn on the lights.", "Write an email to my boss."]
OFF_TOPIC_FILL = {
    "thing": ["the capital of Spain", "gravity", "a black hole", "the stock market", "photosynthesis", "the speed of light",
              "an atom", "the internet", "the weather today", "the time", "your favorite color", "quantum physics",
              "the price of gold", "World War II", "how vaccines work", "the meaning of life"],
    "task": ["fix my bike", "cook pasta", "do my homework", "write some code", "pay my bills", "lose weight",
             "learn to drive", "fix my computer", "file my taxes", "change a tire", "make a website", "do my math"],
    "did": ["wrote Hamlet", "painted the Mona Lisa", "won the game yesterday", "is the president", "discovered gravity",
            "built the pyramids", "is the richest person"],
    "event": ["the sky look blue", "ice float", "the sun rise", "my phone get hot", "bread rise"],
    "event2": ["the war end", "people land on the moon", "the internet start", "dinosaurs live"],
    "place": ["Paris", "the nearest bank", "Mount Everest", "the moon", "Tokyo", "the train station", "my house"],
    "fact": ["fish can fly", "the earth is flat", "bananas are berries", "cats have nine lives"],
    "word": ["apple", "house", "friend", "computer", "river", "happy"],
    "things": ["the planets", "all the countries", "the colors of the rainbow", "the days of the week", "ten prime numbers"],
}
OFF_TOPIC_REPLY = "Sorry, I only know how to tell stories. What should the story be about?"
EVAL_GREETINGS = ["hiya", "yo", "Good evening!", "howdy", "hey hey"]
EVAL_OFF_TOPIC = ["What is the biggest planet?", "How many legs does a spider have?", "Translate cat into French.",
                  "What is the square root of 81?", "Who invented the phone?"]
EVAL_NEW_STORY = ["write short children story", "story please", "I want a story about a frog.", "tell me a story now",
                  "Give me a story about a happy cat."]


CASUAL = ["tell me a story", "write a story", "Story time!", "Can I have a story?", "write me a short story",
          "a story for kids please", "I want a bedtime story.", "Tell me a fun story.", "make up a story",
          "Can you tell me a story?", "one story please", "write a kids story"]


def story_request(fields, rng):
    """(request, fields the follow-ups may ask about). A quarter are short casual requests with no constraints, so
    any plain "story please" starts a story; those drop the words, since the user never asked for any."""
    if rng.random() < 0.25:
        return rng.choice(CASUAL), {k: v for k, v in fields.items() if k != "Words"}
    return make_request(fields, rng), fields


def gibberish(rng):
    return "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(3, 10)))


def greeting(rng):
    """A greeting assembled from parts (word, capitalisation, punctuation, a follow-on question), so the model learns
    the pattern rather than a list. Never one of the held-out EVAL_GREETINGS."""
    held_out = {e.lower().strip("!.") for e in EVAL_GREETINGS}
    while True:
        if rng.random() < 0.1:
            msg = rng.choice(GREET_MORE)
        else:
            w = rng.choice(GREET_WORDS)
            w = rng.choice([w, w, w.capitalize(), w.title(), w.upper()])
            msg = (w + rng.choice(GREET_ENDS) + (" " + rng.choice(GREET_MORE) if rng.random() < 0.3 else "")).strip()
        if msg.lower().strip("!.~:), ") not in held_out:
            return msg


def off_topic(rng):
    """An off-topic question or command assembled from OFF_TOPIC_QS / OFF_TOPIC_CMDS; never one of EVAL_OFF_TOPIC."""
    while True:
        t = rng.choice(OFF_TOPIC_QS if rng.random() < 0.5 else OFF_TOPIC_CMDS)
        fill = {k: rng.choice(v) for k, v in OFF_TOPIC_FILL.items()}
        msg = t.format(**fill, Task=fill["task"][0].upper() + fill["task"][1:], a=rng.randint(2, 12), b=rng.randint(2, 99))
        if rng.random() < 0.2:
            msg = msg.lower().rstrip(".?")
        if msg not in EVAL_OFF_TOPIC:
            return msg


def small_talk(rng):
    """(message, fixed reply): a greeting, an off-topic question or command, or random letters."""
    r = rng.random()
    if r < 0.35:
        return greeting(rng), GREETING_REPLY
    return (off_topic(rng) if r < 0.8 else gibberish(rng)), OFF_TOPIC_REPLY


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
    """[(ids, mask, meta)]; meta has the first story's name fact (if any) for the eval, and "turns": one
    (reply start, reply end, kind, fields) per assistant turn, kind "story" / "talk" / "question" (fields: what a story
    turn was asked for, else {}). Conversations fit in block_size + 1. Each conversation: sometimes a greeting first, a
    story request, then 1-3 turns, each a second story (once), small talk / an off-topic message (fixed reply), or a
    question about the stories so far."""
    parsed = [(f, s) for f, s in map(parse, entries) if f and s]
    convs = []
    for i, (fields, story) in enumerate(parsed):
        req, asked = story_request(fields, rng)
        stories = [(asked, story)]
        ids, mask, turns = [], [], []

        def add(q, a, kind, f):
            t_ids, t_mask = encode_turn(tok, q, a)
            if len(ids) + len(t_ids) > block_size + 1:
                return False
            turns.append((len(ids) + t_mask.index(True), len(ids) + len(t_ids), kind, f))
            ids.extend(t_ids)
            mask.extend(t_mask)
            return True

        if rng.random() < 0.3:
            add(*small_talk(rng), "talk", {})
        add(req, story, "story", asked)
        for _ in range(rng.randint(1, 3)):
            if rng.random() < 0.2:
                (q, a), kind, new = small_talk(rng), "talk", None
            elif len(stories) == 1 and rng.random() < 0.35:
                f2, s2 = parsed[(i + 1) % len(parsed)]
                req, asked2 = story_request(f2, rng)  # half with no "another", so any story request starts a story
                q, a, kind, new = (rng.choice(ANOTHER) + " " + req if rng.random() < 0.5 else req), s2, "story", (asked2, s2)
            else:
                options = follow_ups(stories, rng)
                if not options:
                    break
                (q, a), kind, new = rng.choice(options), "question", None
            if not add(q, a, kind, new[0] if new else {}):
                break
            if new:
                stories.append(new)
        if any(mask):
            convs.append((ids, mask, {"fields": fields, "story": story, "fact": name_fact(story), "turns": turns}))
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


@torch.no_grad()
def small_talk_eval(model, tok, parsed, device, n=100):
    """Held-out phrasings only. After a story turn (teacher-forced), send a greeting / off-topic question / gibberish /
    story request; also a bare greeting with no history. Greedy replies; returns the fraction handled correctly:
    the fixed reply for small talk, and for a story request a new story (40+ tokens, not the previous summary)."""
    eot = tok.special_ids[EOT]
    rng = random.Random(5)

    def answer(history, msg, max_new=120):
        q = [tok.special_ids[USER]] + tok.encode_ordinary(msg) + [tok.special_ids[ASSISTANT]]
        prompt = (history + q)[-(model.cfg.block_size - max_new) :]
        out = model.generate(torch.tensor([prompt], device=device), max_new, temperature=0, eos_id=eot)
        reply = out[0, len(prompt) :].tolist()
        return reply[: reply.index(eot)] if eot in reply else reply

    res = {"greeting": 0, "off_topic": 0, "gibberish": 0, "new_story": 0, "greeting_first": 0}
    for k, (fields, story) in enumerate(parsed[:n]):
        hist, _ = encode_turn(tok, make_request(fields, random.Random(k)), story)
        res["greeting"] += tok.decode(answer(hist, rng.choice(EVAL_GREETINGS))) == GREETING_REPLY
        res["off_topic"] += tok.decode(answer(hist, rng.choice(EVAL_OFF_TOPIC))) == OFF_TOPIC_REPLY
        res["gibberish"] += tok.decode(answer(hist, gibberish(random.Random(10_000 + k)))) == OFF_TOPIC_REPLY
        new = answer(hist, rng.choice(EVAL_NEW_STORY))
        res["new_story"] += len(new) >= 40 and tok.decode(new) != fields.get("Summary")
        res["greeting_first"] += tok.decode(answer([], rng.choice(EVAL_GREETINGS))) == GREETING_REPLY
    return {k: v / n for k, v in res.items()}


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
    talk = small_talk_eval(model, tok, with_words, device)
    print("held-out small talk handled: " + ", ".join(f"{k} {v:.0%}" for k, v in talk.items()))


if __name__ == "__main__":
    main()
