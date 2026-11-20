"""Preference tuning with DPO (Direct Preference Optimization, Rafailov et al. 2023) on a chat model.

1. Sample: for each story request with required words, sample K stories from the chat model (the policy).
2. Score with a rule-based reward: fraction of required words used (unfinished stories are dropped).
   chosen = best sample, rejected = worst; requests whose samples all score the same are skipped.
3. Train: loss = -log sigmoid(beta * [(log pi(c) - log ref(c)) - (log pi(r) - log ref(r))]), where ref is a frozen copy
   of the starting model. No reward model and no RL loop: the preference pairs are the whole signal.
4. Evaluate on held-out requests: word use, reply length, and fluency (per-token loss of the replies under the
   pretrained base model, which catches "just list the words" reward hacking).

python dpo.py --sft runs/ts_xl_chat/ckpt.pt --out runs/ts_xl_dpo
"""
import argparse
import os
import random
import time
from collections import defaultdict
from dataclasses import asdict

import torch
import torch.nn.functional as F

from data import EOT, load_tokenizer
from finetune import CHAT_TOKENS, encode_prompt, load_examples, make_batch, parse, word_follow_rate, words_used
from model import GPT
from multiturn import load_entries


def seq_logprob(model, x, y):
    """Sum of log p(target) over the learned (non -100) positions of each row."""
    logits, _ = model(x)
    lp = torch.log_softmax(logits.float(), dim=-1).gather(-1, y.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return (lp * (y != -100)).sum(-1)


@torch.no_grad()
def sample_pairs(model, tok, requests, k, device, batch_seqs=96, max_new=320):
    """[(prompt ids, chosen ids, rejected ids, chosen score, rejected score)]. Prompts of equal token length are
    batched together, so generate() needs no padding."""
    eot = tok.special_ids[EOT]
    buckets = defaultdict(list)
    for prompt, fields in requests:
        buckets[len(prompt)].append((prompt, fields))
    pairs, done, t0 = [], 0, time.time()
    autocast = torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda")
    for group in buckets.values():
        per = max(1, batch_seqs // k)
        for i in range(0, len(group), per):
            chunk = group[i : i + per]
            idx = torch.tensor([p for p, _ in chunk for _ in range(k)], device=device)
            with autocast:
                # prompt + reply (incl. EOT) must fit the context, or training on the pair is impossible
                out = model.generate(idx, min(max_new, model.cfg.block_size - len(chunk[0][0])), temperature=1.0,
                                     top_p=0.95, eos_id=eot)
            for j, (prompt, fields) in enumerate(chunk):
                scored = []
                for row in out[j * k : (j + 1) * k, len(prompt) :].tolist():
                    if eot not in row:
                        continue  # unfinished: never a valid chosen or rejected reply
                    resp = row[: row.index(eot)]
                    hit, n = words_used(tok.decode(resp), fields)
                    scored.append((hit / n, resp))
                if len(scored) >= 2:
                    scored.sort(key=lambda s: s[0])
                    (lo, rej), (hi, cho) = scored[0], scored[-1]
                    if hi > lo:
                        pairs.append((prompt, cho, rej, hi, lo))
            if (done + len(chunk)) // 100 > done // 100:
                print(f"  sampled {done + len(chunk)}/{len(requests)} requests, {len(pairs)} pairs, "
                      f"{time.time() - t0:.0f}s", flush=True)
            done += len(chunk)
    return pairs


def dpo_train(policy, ref, pairs, eot, device, steps, batch_size, lr, beta):
    opt = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 20))
    autocast = torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda")
    rng = random.Random(0)
    policy.train()
    for step in range(1, steps + 1):
        batch = rng.sample(pairs, batch_size)
        rows = [(p + c + [eot], len(p), None) for p, c, *_ in batch] + [(p + r + [eot], len(p), None) for p, _, r, *_ in batch]
        x, y = make_batch(rows, device)
        with autocast:
            pol = seq_logprob(policy, x, y)
            with torch.no_grad():
                rf = seq_logprob(ref, x, y)
        n = len(batch)
        margin = beta * ((pol[:n] - rf[:n]) - (pol[n:] - rf[n:]))  # implicit reward of chosen minus rejected
        loss = -F.logsigmoid(margin).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 25 == 0 or step == steps:
            print(f"step {step:4d} | dpo loss {loss.item():.4f} | reward acc {(margin > 0).float().mean().item():.2f} | "
                  f"margin {margin.mean().item():.3f}", flush=True)
    return policy.eval()


@torch.no_grad()
def evaluate(model, judge, tok, val, device, n):
    """Word use, mean reply length (tokens), and fluency = judge's per-token loss on the replies."""
    torch.manual_seed(0)
    rate, samples = word_follow_rate(model, tok, val, device, n=n)
    lens, nll = [], []
    for _, story in samples:
        ids = tok.encode_ordinary(story)[: judge.cfg.block_size]
        lens.append(len(ids))
        if len(ids) > 1:
            x = torch.tensor([ids], device=device)
            nll.append(judge(x[:, :-1], x[:, 1:])[1].item())
    return rate, sum(lens) / len(lens), sum(nll) / len(nll), samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sft", default="runs/ts_xl_chat/ckpt.pt")
    p.add_argument("--judge", default="runs/ts_xl_1024/ckpt.pt", help="pretrained base, for the fluency check")
    p.add_argument("--out", default="runs/ts_xl_dpo")
    p.add_argument("--n_requests", type=int, default=1500)
    p.add_argument("--k", type=int, default=6, help="samples per request")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=8, help="pairs per step (x2 sequences, + reference)")
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--eval_n", type=int, default=200)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(1337)
    policy, ckpt = GPT.from_checkpoint(args.sft, device)
    ref, _ = GPT.from_checkpoint(args.sft, device)
    judge, _ = GPT.from_checkpoint(args.judge, device)
    for m in (ref, judge):
        m.requires_grad_(False)
    data_dir = ckpt["args"]["data"]
    tok = load_tokenizer(data_dir, CHAT_TOKENS)
    eot = tok.special_ids[EOT]

    # Requests from the instruct *train* split (the eval uses the held-out val split)
    rng = random.Random(0)
    entries = [parse(e) for e in load_entries("train", 100)]
    requests = [(encode_prompt(tok, f, True, rng), f) for f, s in entries if s and len(f.get("Words", "").split(",")) >= 3]
    requests = random.Random(1).sample(requests, args.n_requests)
    pairs_path = os.path.join(args.out, "pairs.pt")
    if os.path.exists(pairs_path):
        pairs = torch.load(pairs_path)
    else:
        print(f"sampling {args.k} replies for each of {len(requests)} requests")
        pairs = sample_pairs(policy, tok, requests, args.k, device)
        os.makedirs(args.out, exist_ok=True)
        torch.save(pairs, pairs_path)
    print(f"{len(pairs)} preference pairs; chosen uses {sum(p[3] for p in pairs) / len(pairs):.0%} of words, "
          f"rejected {sum(p[4] for p in pairs) / len(pairs):.0%}")

    val = load_examples(tok, "val", None, policy.cfg.block_size, chat=True)
    before = evaluate(policy, judge, tok, val, device, args.eval_n)
    dpo_train(policy, ref, pairs, eot, device, args.steps, args.batch_size, args.lr, args.beta)
    torch.save({"model": policy.state_dict(), "config": asdict(policy.cfg), "step": args.steps,
                "args": {**vars(args), "data": data_dir, "extra_special_tokens": CHAT_TOKENS}},
               os.path.join(args.out, "ckpt.pt"))
    after = evaluate(policy, judge, tok, val, device, args.eval_n)

    print(f"\n{'':6} {'words used':>10} {'reply tokens':>12} {'fluency (judge loss)':>21}")
    for name, (rate, length, nll, _) in [("sft", before), ("dpo", after)]:
        print(f"{name:6} {rate:10.0%} {length:12.0f} {nll:21.3f}")
    for name, res in [("sft", before), ("dpo", after)]:
        prompt, story = res[3][0]
        print(f"\n--- {name} ---\n{prompt}\n{story[:700]}")


if __name__ == "__main__":
    main()
