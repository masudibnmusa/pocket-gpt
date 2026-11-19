"""Instruction fine-tuning (SFT) on TinyStories-Instruct: teach a pretrained model to write a story that follows
given constraints (required words, features like Dialogue/BadEnding, a summary, a sentence to include).

python finetune.py --base runs/ts_llama/ckpt.pt --out runs/ts_llama_sft
python finetune.py --base runs/ts_llama/ckpt.pt --out runs/ts_llama_sft --eval_only   # just compare base vs SFT

Prompt format (plain text, so the pretraining tokenizer needs no new tokens); loss is on the story only:
    Words: quit, oak, gloomy
    Features: Dialogue
    Story:
    <story><|endoftext|>
"""
import argparse
import math
import os
import random
import re
import time

import torch

from data import DATA_DIR, EOT, download, load_tokenizer
from model import GPT

URL = "https://huggingface.co/datasets/roneneldan/TinyStoriesInstruct/resolve/main/TinyStories-Instruct-{}.txt"
FIELDS = ("Words", "Features", "Summary", "Random sentence")
FIELD_LINE = re.compile(r"^(Words|Features|Summary|Random sentence): (.*)$")


def parse(entry):
    """One instruct entry -> ({field: value}, story). Fields may appear before or after the story."""
    fields, story, in_story = {}, [], False
    for line in entry.strip().split("\n"):
        m = FIELD_LINE.match(line)
        if line.startswith("Story:"):
            in_story = True
        elif m:
            fields[m.group(1)] = m.group(2).strip()
            in_story = False
        elif in_story:
            story.append(line)
    return fields, "\n".join(story).strip()


def make_prompt(fields):
    return "".join(f"{k}: {fields[k]}\n" for k in FIELDS if k in fields) + "Story:\n"


def load_examples(tok, split, max_mb, block_size):
    """Tokenized (ids, prompt_len) pairs that fit in block_size + 1 (inputs and shifted targets), plus raw fields."""
    raw = os.path.join(DATA_DIR, f"tinystories_instruct_{split}{f'_{max_mb}mb' if max_mb else ''}.txt")
    download(URL.format("train" if split == "train" else "valid"), raw, max_mb * 10**6 if max_mb else None)
    text = open(raw, encoding="utf-8", errors="ignore").read()
    entries = text.split(EOT)[1:-1]  # first/last may be cut mid-entry by the byte-range download
    eot = tok.special_ids[EOT]
    out, total = [], 0
    for e in entries:
        fields, story = parse(e)
        if not story or not fields:
            continue
        total += 1
        prompt = tok.encode(make_prompt(fields))
        ids = prompt + tok.encode(story) + [eot]
        if len(ids) <= block_size + 1:
            out.append((ids, len(prompt), fields))
    print(f"{split}: {len(out):,} of {total:,} examples fit in {block_size} tokens")
    return out


def make_batch(examples, device):
    """Right-pad; targets are -100 (ignored by cross_entropy) on the prompt and padding."""
    T = max(len(ids) for ids, _, _ in examples) - 1
    x = torch.zeros(len(examples), T, dtype=torch.long)
    y = torch.full((len(examples), T), -100, dtype=torch.long)
    for i, (ids, plen, _) in enumerate(examples):
        t = torch.tensor(ids)
        x[i, : len(ids) - 1] = t[:-1]
        y[i, plen - 1 : len(ids) - 1] = t[plen:]  # predict story tokens (and EOT) only
    return x.to(device), y.to(device)


def word_follow_rate(model, tok, examples, device, n=100, seed=0):
    """Generate a story per prompt that has a Words field; fraction of required words that appear in it."""
    rng = random.Random(seed)
    pool = [e for e in examples if "Words" in e[2]]
    picked, hit, total, samples = rng.sample(pool, min(n, len(pool))), 0, 0, []
    torch.manual_seed(seed)
    for ids, plen, fields in picked:
        prompt = ids[:plen]
        out = model.generate(torch.tensor([prompt], device=device), model.cfg.block_size - plen,
                             temperature=0.7, top_p=0.9, eos_id=tok.special_ids[EOT])
        story = tok.decode(out[0, plen:].tolist()).split(EOT)[0]
        words = [w.strip().lower() for w in fields["Words"].split(",") if w.strip()]
        hit += sum(bool(re.search(r"\b" + re.escape(w), story.lower())) for w in words)  # prefix: "quit" ~ "quitting"
        total += len(words)
        samples.append((make_prompt(fields), story))
    return hit / total, samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="runs/ts_llama/ckpt.pt")
    p.add_argument("--out", default="runs/ts_llama_sft")
    p.add_argument("--train_mb", type=int, default=100, help="MB of the 2.7GB instruct train file to use")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--eval_interval", type=int, default=500)
    p.add_argument("--eval_n", type=int, default=100, help="prompts for the word-following eval")
    p.add_argument("--eval_only", action="store_true")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(1337)
    base, ckpt = GPT.from_checkpoint(args.base, device)
    data_dir = ckpt["args"]["data"]
    tok = load_tokenizer(data_dir)
    assert EOT in tok.special_ids, "needs a tokenizer with <|endoftext|> (the TinyStories BPE)"
    bs = base.cfg.block_size
    val = load_examples(tok, "val", None, bs)

    ckpt_path = os.path.join(args.out, "ckpt.pt")
    if not args.eval_only:
        train = load_examples(tok, "train", args.train_mb, bs)
        model, _ = GPT.from_checkpoint(args.base, device)
        model.train()
        decay = [p for p in model.parameters() if p.dim() >= 2]
        no_decay = [p for p in model.parameters() if p.dim() < 2]
        opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
                                lr=args.lr)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup) * 0.5 *
                                                  (1 + math.cos(math.pi * min(1.0, s / args.steps))))
        autocast = torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda")
        rng = random.Random(0)

        @torch.no_grad()
        def val_loss():
            model.eval()
            losses = []
            for i in range(0, 50 * args.batch_size, args.batch_size):
                with autocast:
                    losses.append(model(*make_batch(val[i : i + args.batch_size], device))[1].item())
            model.train()
            return sum(losses) / len(losses)

        print(f"step     0 | story val loss {val_loss():.4f}")
        t0 = time.time()
        for step in range(1, args.steps + 1):
            x, y = make_batch(rng.sample(train, args.batch_size), device)
            with autocast:
                _, loss = model(x, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            if step % args.eval_interval == 0 or step == args.steps:
                print(f"step {step:5d} | train {loss.item():.4f} | story val loss {val_loss():.4f} | "
                      f"{time.time() - t0:.0f}s")
        os.makedirs(args.out, exist_ok=True)
        torch.save({"model": model.state_dict(), "config": ckpt["config"], "step": args.steps,
                    "args": {**vars(args), "data": data_dir}}, ckpt_path)
        print(f"saved {ckpt_path}")

    sft, _ = GPT.from_checkpoint(ckpt_path, device)
    for name, m in [("base", base), ("sft", sft)]:
        rate, samples = word_follow_rate(m, tok, val, device, n=args.eval_n)
        print(f"\n{name}: required words used {rate:.0%}")
        prompt, story = samples[0]
        print(f"--- example prompt ---\n{prompt}--- {name} story ---\n{story[:600]}")


if __name__ == "__main__":
    main()
