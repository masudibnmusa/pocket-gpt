"""Train a character-level GPT on Tiny Shakespeare.  python train.py --help"""
import argparse
import math
import os
import time
from dataclasses import asdict, fields

import torch

from data import get_batch, load_meta, load_split, load_tokenizer
from model import GPT, GPTConfig

p = argparse.ArgumentParser()
for f in fields(GPTConfig):
    if f.name != "vocab_size":
        p.add_argument(f"--{f.name}", type=f.type, default=f.default)
p.add_argument("--batch_size", type=int, default=64)
p.add_argument("--steps", type=int, default=5000)
p.add_argument("--lr", type=float, default=1e-3, help="peak learning rate, reached after warmup")
p.add_argument("--min_lr", type=float, default=1e-4, help="learning rate at the end of cosine decay")
p.add_argument("--warmup", type=int, default=100)
p.add_argument("--grad_clip", type=float, default=1.0, help="max global grad norm; 'inf' disables clipping")
p.add_argument("--weight_decay", type=float, default=0.1, help="applied to matmul weights and embeddings only")
p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16", help="autocast dtype for forward passes (CUDA only)")
p.add_argument("--eval_interval", type=int, default=250)
p.add_argument("--eval_iters", type=int, default=100)
p.add_argument("--data", default="data/char", help="dir made by data.py (train.bin, val.bin, meta.json)")
p.add_argument("--out", default="runs/char_shakespeare")
p.add_argument("--seed", type=int, default=1337)
p.add_argument("--resume", action="store_true", help="continue from <out>/ckpt.pt (model flags are taken from the checkpoint)")
args = p.parse_args()

torch.manual_seed(args.seed)
device = "cuda" if torch.cuda.is_available() else "cpu"
tok = load_tokenizer(args.data)
meta = load_meta(args.data)
data = {s: load_split(args.data, s) for s in ("train", "val")}
# Loss per token isn't comparable across tokenizers; bits per byte is.
val_bits_per_byte = lambda loss: loss / math.log(2) * meta["val_tokens"] / meta["val_bytes"]

ckpt_path = os.path.join(args.out, "ckpt.pt")
ckpt = None
if args.resume:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = GPTConfig(**ckpt["config"])
else:
    if os.path.exists(ckpt_path):
        raise SystemExit(f"{ckpt_path} already exists; pass --resume to continue it or --out to start a new run")
    cfg = GPTConfig(vocab_size=tok.vocab_size, **{f.name: getattr(args, f.name) for f in fields(GPTConfig) if f.name != "vocab_size"})
model = GPT(cfg).to(device)
# Decay 2D+ tensors (Linear weights, embeddings); never biases or LayerNorm gains/shifts.
decay = [p for p in model.parameters() if p.dim() >= 2]
no_decay = [p for p in model.parameters() if p.dim() < 2]
opt = torch.optim.AdamW(
    [{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}], lr=args.lr
)
# bf16 has fp32's exponent range, so unlike fp16 it needs no GradScaler.
autocast = torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda" and args.dtype == "bf16")
print(f"{model.num_params() / 1e6:.2f}M params on {device} ({args.dtype if device == 'cuda' else 'fp32'}); "
      f"decayed {sum(p.numel() for p in decay):,} / not decayed {sum(p.numel() for p in no_decay):,}; "
      f"expected initial loss ln({cfg.vocab_size}) = {math.log(cfg.vocab_size):.3f}")


@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split, d in data.items():
        losses = []
        for _ in range(args.eval_iters):
            with autocast:
                losses.append(model(*get_batch(d, args.batch_size, cfg.block_size, device))[1].item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out


def get_lr(step, max_lr, min_lr, warmup, total):
    """Linear warmup to max_lr, then cosine decay to min_lr at step `total`."""
    if step < warmup:
        return max_lr * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


assert get_lr(0, 1.0, 0.1, 10, 100) == 0.1 and get_lr(10, 1.0, 0.1, 10, 100) == 1.0
assert abs(get_lr(100, 1.0, 0.1, 10, 100) - 0.1) < 1e-12 and abs(get_lr(55, 1.0, 0.1, 10, 100) - 0.55) < 1e-12

def rng_state():
    return {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state() if device == "cuda" else None}


def save(step, rng):
    os.makedirs(args.out, exist_ok=True)
    state = {"model": model.state_dict(), "optimizer": opt.state_dict(), "config": asdict(cfg),
             "step": step, "rng": rng, "args": vars(args)}
    torch.save(state, ckpt_path + ".tmp")
    os.replace(ckpt_path + ".tmp", ckpt_path)  # atomic: a crash mid-save never corrupts the last good checkpoint


start_step = 0
if ckpt is not None:
    model.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["optimizer"])
    torch.set_rng_state(ckpt["rng"]["cpu"].cpu())
    if device == "cuda" and ckpt["rng"]["cuda"] is not None:
        torch.cuda.set_rng_state(ckpt["rng"]["cuda"].cpu())
    start_step = ckpt["step"]
    print(f"resumed from {ckpt_path} at step {start_step}")
    del ckpt

t0 = time.time()
grad_norm = torch.tensor(float("nan"))
for step in range(start_step, args.steps + 1):
    lr = get_lr(step, args.lr, args.min_lr, args.warmup, args.steps)
    if step % args.eval_interval == 0 or step == args.steps:
        tok_s = args.eval_interval * args.batch_size * cfg.block_size / (time.time() - t0) if step > start_step else 0
        rng = rng_state()  # taken before eval, so a resumed run replays this exact eval and the batches after it
        losses = estimate_loss()
        print(f"step {step:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} | "
              f"val bpb {val_bits_per_byte(losses['val']):.3f} | "
              f"lr {lr:.2e} | grad norm {grad_norm.item():.2f} | {tok_s:,.0f} tok/s")
        save(step, rng)
        t0 = time.time()
    if step == args.steps:
        break
    for g in opt.param_groups:
        g["lr"] = lr
    x, y = get_batch(data["train"], args.batch_size, cfg.block_size, device)
    with autocast:
        _, loss = model(x, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)  # returns the pre-clip norm
    opt.step()

print(f"saved {ckpt_path}\n---")
model.eval()
start = torch.tensor([tok.encode("\n")], dtype=torch.long, device=device)
print(tok.decode(model.generate(start, 300)[0].tolist()))
