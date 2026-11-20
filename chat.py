"""Talk to a chat-tuned model (finetune.py --chat or multiturn.py). The conversation history stays in context until
it no longer fits; then the oldest turns are dropped. Type /reset to start over.

python chat.py --ckpt runs/ts_xl_chat/ckpt.pt
python chat.py --ckpt runs/ts_xl_chat/ckpt.pt --prompt 'Tell me a story about a brave cat. Use the word "moon".'
"""
import argparse

import torch

from data import EOT, load_tokenizer
from finetune import ASSISTANT, USER
from model import GPT


def reply(model, tok, message, history=(), temperature=0.7, top_p=0.9, max_reply=300):
    """history: list of previous turns, each a list of token ids (user + assistant + EOT).
    Returns (reply text, updated history)."""
    bs = model.cfg.block_size
    # encode_ordinary: whatever the user types, it can never become a role or end-of-text token
    turn = [tok.special_ids[USER]] + tok.encode_ordinary(message)[: bs // 4] + [tok.special_ids[ASSISTANT]]
    history = list(history)
    budget = bs - min(max_reply, bs // 2)  # leave room for the answer
    while history and sum(map(len, history)) + len(turn) > budget:
        history.pop(0)  # forget the oldest turn
    prompt = [t for h in history for t in h] + turn
    idx = torch.tensor([prompt], device=next(model.parameters()).device)
    out = model.generate(idx, bs - len(prompt), temperature=temperature, top_p=top_p, eos_id=tok.special_ids[EOT])
    answer = out[0, len(prompt):].tolist()
    eot = tok.special_ids[EOT]
    answer = answer[: answer.index(eot)] if eot in answer else answer
    return tok.decode(answer).strip(), history + [turn + answer + [eot]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="runs/ts_xl_chat/ckpt.pt")
    p.add_argument("--prompt", nargs="+", default=None, help="one or more messages, answered in order, then exit")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt = GPT.from_checkpoint(args.ckpt, device)
    extra = ckpt["args"].get("extra_special_tokens", [])
    assert USER in extra, f"{args.ckpt} is not chat-tuned; run finetune.py --chat or multiturn.py first"
    tok = load_tokenizer(ckpt["args"]["data"], extra)

    history = []
    if args.prompt is not None:
        for msg in args.prompt:
            answer, history = reply(model, tok, msg, history, args.temperature, args.top_p)
            print(f"you> {msg}\nmodel> {answer}\n")
        return
    print("Chat with the model (/reset to clear the conversation, Ctrl+D or 'quit' to exit).")
    while True:
        try:
            msg = input("\nyou> ").strip()
        except EOFError:
            break
        if msg.lower() in ("quit", "exit"):
            break
        if msg == "/reset":
            history = []
            print("(conversation cleared)")
        elif msg:
            answer, history = reply(model, tok, msg, history, args.temperature, args.top_p)
            print("\nmodel> " + answer)


if __name__ == "__main__":
    main()
