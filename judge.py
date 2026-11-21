"""Learned judge (reward model): scores how well a story fits a request, features and summary included.

Architecture: the chat model's transformer + a linear head reading the hidden state at the closing <|endoftext|>.
Training: pairwise (Bradley-Terry) loss, -log sigmoid(r(request, matching story) - r(request, mismatched story)).
The matching story is the one written for the request; the mismatched one depends on the request type:
  - full     (summary + words + features): a random other story
  - features (e.g. "Give it a sad ending. Include some dialogue."): a story lacking a requested feature
  - summary  ("It should be about this: ..."): a random other story
  - copy     (full request with a summary): the matching story with the request's phrases pasted in 2-3 times, the
             reward hack seen in iterative DPO, so "the summary's words appear" stops meaning "this story matches"
Two judges are trained on disjoint halves of the data: A gives the DPO reward, B only evaluates, so the policy
cannot score well on the evaluation by exploiting one judge's quirks.

python judge.py --sft runs/ts_xl_chat2/ckpt.pt --half 0 --out runs/judge_a
python judge.py --sft runs/ts_xl_chat2/ckpt.pt --half 1 --out runs/judge_b
"""
import argparse
import os
import random
import re
import time
from dataclasses import asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import EOT, load_tokenizer
from finetune import ASSISTANT, CHAT_TOKENS, FEATURE_TEXT, OPENERS, USER, make_request, parse
from model import GPT, GPTConfig
from multiturn import load_entries

# Judged sequences are request (~100 tokens) + reply (DPO samples <= 320), so longer training pairs add nothing
# but memory (32 x 1024 tokens does not fit next to the desktop on an 8 GB GPU).
MAX_LEN = 600


def features_of(fields):
    return [f.strip() for f in fields.get("Features", "").split(",") if f.strip() in FEATURE_TEXT]


def judge_request(fields, kind, rng):
    if kind == "full":
        return make_request(fields, rng)
    if kind == "features":
        return " ".join([rng.choice(OPENERS)] + [FEATURE_TEXT[f] for f in features_of(fields)])
    return f"{rng.choice(OPENERS)} It should be about this: {fields['Summary']}"


def paste_request(story, fields, rng):
    """The story with the summary / required sentence pasted in at 2-3 random sentence ends (formatting kept)."""
    phrases = [fields["Summary"]] + ([fields["Random sentence"]] if "Random sentence" in fields else [])
    ends = [m.end() for m in re.finditer(r'[.!?]"? ', story)] or [len(story)]
    for pos in sorted(rng.sample(ends, min(len(ends), rng.randint(2, 3))), reverse=True):
        story = story[:pos] + rng.choice(phrases) + " " + story[pos:]
    return story


class Judge(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gpt = GPT(cfg)
        self.head = nn.Linear(cfg.n_embd, 1)

    def forward(self, idx, last):
        """idx (B, T) right-padded; last (B,) index of each row's final token. Returns (B,) scores."""
        h = self.gpt.hidden(idx)
        return self.head(h[torch.arange(idx.size(0), device=idx.device), last]).squeeze(-1)

    @classmethod
    def load(cls, path, device):
        ckpt = torch.load(path, map_location=device)
        judge = cls(GPTConfig(**ckpt["config"])).to(device)
        judge.load_state_dict(ckpt["model"])
        return judge.eval()


def encode_pair(tok, request, story):
    """<|user|>request<|assistant|>story<|endoftext|> (user text can't produce special tokens)."""
    return ([tok.special_ids[USER]] + tok.encode_ordinary(request) + [tok.special_ids[ASSISTANT]]
            + tok.encode_ordinary(story) + [tok.special_ids[EOT]])


@torch.no_grad()
def score(judge, seqs, device, batch_size=32):
    """Scores for a list of token-id sequences (each ending in EOT)."""
    out = []
    autocast = torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda")
    for i in range(0, len(seqs), batch_size):
        chunk = [s[-judge.gpt.cfg.block_size :] for s in seqs[i : i + batch_size]]
        x = torch.zeros(len(chunk), max(map(len, chunk)), dtype=torch.long)
        for j, s in enumerate(chunk):
            x[j, : len(s)] = torch.tensor(s)
        with autocast:
            out += judge(x.to(device), torch.tensor([len(s) - 1 for s in chunk], device=device)).float().tolist()
    return out


def make_pairs(tok, parsed, rng, block_size):
    """[(matching seq, mismatched seq, kind)] from parsed (fields, story) entries."""
    pairs = []
    for fields, story in parsed:
        kinds = ["full"] + (["features"] if features_of(fields) else []) + (["summary", "copy"] if "Summary" in fields else [])
        kind = rng.choice(kinds)
        if kind == "features":
            want = set(features_of(fields))
            for _ in range(30):  # a story missing at least one requested feature
                f2, s2 = rng.choice(parsed)
                if not want <= set(features_of(f2)):
                    break
            else:
                continue
        elif kind == "copy":
            s2 = paste_request(story, fields, rng)
        else:
            f2, s2 = rng.choice(parsed)
            if s2 == story:
                continue
        req = judge_request(fields, "full" if kind == "copy" else kind, rng)
        good, bad = encode_pair(tok, req, story), encode_pair(tok, req, s2)
        if max(len(good), len(bad)) <= min(block_size, MAX_LEN):
            pairs.append((good, bad, kind))
    return pairs


def accuracy(judge, pairs, device):
    """Per request type: fraction of pairs where the matching story scores higher."""
    s_good = score(judge, [g for g, _, _ in pairs], device)
    s_bad = score(judge, [b for _, b, _ in pairs], device)
    res = {}
    for kind in ("full", "features", "summary", "copy"):
        idx = [i for i, p in enumerate(pairs) if p[2] == kind]
        res[kind] = sum(s_good[i] > s_bad[i] for i in idx) / max(1, len(idx))
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sft", default="runs/ts_xl_chat2/ckpt.pt", help="chat model to initialise the judge from")
    p.add_argument("--half", type=int, choices=[0, 1], default=0, help="which half of the train entries to use")
    p.add_argument("--out", default="runs/judge_a")
    p.add_argument("--steps", type=int, default=2500)
    p.add_argument("--batch_size", type=int, default=16, help="pairs per step")
    p.add_argument("--lr", type=float, default=3e-5)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(1337 + args.half)
    sft, ckpt = GPT.from_checkpoint(args.sft, device)
    tok = load_tokenizer(ckpt["args"]["data"], CHAT_TOKENS)
    judge = Judge(sft.cfg).to(device)
    judge.gpt.load_state_dict(sft.state_dict())
    del sft

    parsed = [(f, s) for f, s in map(parse, load_entries("train", 100)) if f and s][args.half :: 2]
    bs = judge.gpt.cfg.block_size
    train = make_pairs(tok, parsed, random.Random(args.half), bs)
    val_parsed = [(f, s) for f, s in map(parse, load_entries("val", None)) if f and s][:3000]
    val = make_pairs(tok, val_parsed, random.Random(99), bs)  # same val pairs for both judges
    print(f"{len(train):,} train pairs, {len(val):,} val pairs", flush=True)
    print(f"step     0 | val accuracy {accuracy(judge, val, device)}", flush=True)

    opt = torch.optim.AdamW(judge.parameters(), lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 100))
    autocast = torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda")
    rng, t0 = random.Random(args.half), time.time()
    judge.train()
    for step in range(1, args.steps + 1):
        batch = rng.sample(train, args.batch_size)
        seqs = [g for g, _, _ in batch] + [b for _, b, _ in batch]
        x = torch.zeros(len(seqs), max(map(len, seqs)), dtype=torch.long)
        for j, s in enumerate(seqs):
            x[j, : len(s)] = torch.tensor(s)
        last = torch.tensor([len(s) - 1 for s in seqs], device=device)
        with autocast:
            r = judge(x.to(device), last).float()
        n = len(batch)
        loss = -F.logsigmoid(r[:n] - r[n:]).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(judge.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 500 == 0 or step == args.steps:
            judge.eval()
            acc = {k: round(v, 3) for k, v in accuracy(judge, val, device).items()}
            judge.train()
            print(f"step {step:5d} | loss {loss.item():.4f} | val accuracy {acc} | {time.time() - t0:.0f}s", flush=True)
    os.makedirs(args.out, exist_ok=True)
    torch.save({"model": judge.state_dict(), "config": asdict(judge.gpt.cfg),
                "args": {**vars(args), "data": ckpt["args"]["data"]}}, os.path.join(args.out, "ckpt.pt"))
    print(f"saved {args.out}/ckpt.pt")


if __name__ == "__main__":
    main()
