"""Talk to a chat-tuned model (made with finetune.py --chat). Single-turn: each message gets a fresh story.

python chat.py --ckpt runs/ts_llama_chat/ckpt.pt
python chat.py --ckpt runs/ts_llama_chat/ckpt.pt --prompt 'Tell me a story about a brave cat. Use the word "moon".'
"""
import argparse

import torch

from data import EOT, load_tokenizer
from finetune import ASSISTANT, USER
from model import GPT


def reply(model, tok, message, temperature=0.7, top_p=0.9):
    bs = model.cfg.block_size
    # encode_ordinary: whatever the user types, it can never become a role or end-of-text token
    ids = [tok.special_ids[USER]] + tok.encode_ordinary(message)[: bs // 2] + [tok.special_ids[ASSISTANT]]
    idx = torch.tensor([ids], device=next(model.parameters()).device)
    out = model.generate(idx, bs - len(ids), temperature=temperature, top_p=top_p, eos_id=tok.special_ids[EOT])
    return tok.decode(out[0, len(ids):].tolist()).split(EOT)[0].strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="runs/ts_llama_chat/ckpt.pt")
    p.add_argument("--prompt", default=None, help="answer one message and exit")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt = GPT.from_checkpoint(args.ckpt, device)
    extra = ckpt["args"].get("extra_special_tokens", [])
    assert USER in extra, f"{args.ckpt} is not chat-tuned; run finetune.py --chat first"
    tok = load_tokenizer(ckpt["args"]["data"], extra)

    if args.prompt is not None:
        print(reply(model, tok, args.prompt, args.temperature, args.top_p))
        return
    print("Ask for a story (Ctrl+D or 'quit' to exit).")
    while True:
        try:
            msg = input("\nyou> ").strip()
        except EOFError:
            break
        if msg.lower() in ("quit", "exit"):
            break
        if msg:
            print("\nmodel> " + reply(model, tok, msg, args.temperature, args.top_p))


if __name__ == "__main__":
    main()
