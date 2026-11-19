"""Generate text from a trained checkpoint.  python sample.py --prompt "ROMEO:" """
import argparse
import sys
import time

import torch

from data import load_tokenizer
from model import GPT

p = argparse.ArgumentParser()
p.add_argument("--ckpt", default="runs/char_shakespeare/ckpt.pt")
p.add_argument("--prompt", default="\n")
p.add_argument("--tokens", type=int, default=500)
p.add_argument("--temperature", type=float, default=0.8)
p.add_argument("--top_k", type=int, default=None)
p.add_argument("--top_p", type=float, default=None, help="nucleus sampling, e.g. 0.9")
p.add_argument("--no_cache", action="store_true", help="recompute the full context every token (slow)")
p.add_argument("--seed", type=int, default=None)
args = p.parse_args()

if args.seed is not None:
    torch.manual_seed(args.seed)
device = "cuda" if torch.cuda.is_available() else "cpu"
model, ckpt = GPT.from_checkpoint(args.ckpt, device)

tok = load_tokenizer(ckpt.get("args", {}).get("data", "data/char"))  # the tokenizer the model was trained with
idx = torch.tensor([tok.encode(args.prompt)], dtype=torch.long, device=device)
t0 = time.time()
out = model.generate(idx, args.tokens, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
                     use_cache=not args.no_cache)
dt = time.time() - t0
print(tok.decode(out[0].tolist()))
print(f"\n[{args.tokens} tokens in {dt:.2f}s, {args.tokens / dt:.0f} tok/s, KV cache {'off' if args.no_cache else 'on'}]",
      file=sys.stderr)
