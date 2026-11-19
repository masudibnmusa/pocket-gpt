"""Train a character-level GPT on Tiny Shakespeare.  python train.py --help"""
import argparse
import math
import os
import time
from dataclasses import asdict, fields

import torch

from data import get_batch, load_split, load_tokenizer
from model import GPT, GPTConfig

p = argparse.ArgumentParser()
for f in fields(GPTConfig):
    if f.name != "vocab_size":
        p.add_argument(f"--{f.name}", type=f.type, default=f.default)
p.add_argument("--batch_size", type=int, default=64)
p.add_argument("--steps", type=int, default=5000)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--eval_interval", type=int, default=250)
p.add_argument("--eval_iters", type=int, default=100)
p.add_argument("--out", default="runs/char_shakespeare")
p.add_argument("--seed", type=int, default=1337)
args = p.parse_args()

torch.manual_seed(args.seed)
device = "cuda" if torch.cuda.is_available() else "cpu"
tok = load_tokenizer()
data = {s: load_split(s) for s in ("train", "val")}

cfg = GPTConfig(vocab_size=tok.vocab_size, **{f.name: getattr(args, f.name) for f in fields(GPTConfig) if f.name != "vocab_size"})
model = GPT(cfg).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
print(f"{model.num_params() / 1e6:.2f}M params on {device}; expected initial loss ln({cfg.vocab_size}) = {math.log(cfg.vocab_size):.3f}")


@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split, d in data.items():
        losses = [model(*get_batch(d, args.batch_size, cfg.block_size, device))[1].item() for _ in range(args.eval_iters)]
        out[split] = sum(losses) / len(losses)
    model.train()
    return out


t0 = time.time()
for step in range(args.steps + 1):
    if step % args.eval_interval == 0 or step == args.steps:
        tok_s = args.eval_interval * args.batch_size * cfg.block_size / (time.time() - t0) if step else 0
        losses = estimate_loss()
        print(f"step {step:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} | {tok_s:,.0f} tok/s")
        t0 = time.time()
    if step == args.steps:
        break
    x, y = get_batch(data["train"], args.batch_size, cfg.block_size, device)
    _, loss = model(x, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()

os.makedirs(args.out, exist_ok=True)
torch.save({"model": model.state_dict(), "config": asdict(cfg)}, os.path.join(args.out, "ckpt.pt"))
print(f"saved {args.out}/ckpt.pt\n---")
model.eval()
print(tok.decode(model.generate(torch.zeros((1, 1), dtype=torch.long, device=device), 300)[0].tolist()))
