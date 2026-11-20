"""Preference tuning with DPO (Direct Preference Optimization, Rafailov et al. 2023), optionally over several rounds.

Each round:
1. Sample K replies per story request from the current policy (fresh requests every round: on-policy data).
2. Reward each finished reply:
     --reward words : fraction of required words used (rule-based)
     --reward judge : learned judge A's score (fit to features + summary + words) + word_weight * word fraction
   chosen = best reply, rejected = worst; requests whose replies all score the same are skipped.
3. Train: loss = -log sigmoid(beta * [(log pi(c) - log ref(c)) - (log pi(r) - log ref(r))]), ref = the policy at the
   start of the round (round 1: the SFT model).
4. Evaluate on held-out requests: required words, dialogue when asked for, judge B (trained on disjoint data, never
   used as reward) score and win rate vs the SFT model, fluency under the pretrained LM, reply length.

python dpo.py --sft runs/ts_xl_chat2/ckpt.pt --reward judge --judge_rm runs/judge_a/ckpt.pt \
              --eval_judge runs/judge_b/ckpt.pt --rounds 3 --out runs/ts_xl_dpo_judge
python dpo.py --sft runs/ts_xl_chat/ckpt.pt --reward words --rounds 1 --out runs/ts_xl_dpo     # words-only, 1 round
"""
import argparse
import copy
import os
import random
import time
from collections import defaultdict
from dataclasses import asdict

import torch
import torch.nn.functional as F

from data import EOT, load_tokenizer
from finetune import CHAT_TOKENS, encode_prompt, make_batch, parse, words_used
from judge import Judge, score
from model import GPT
from multiturn import load_entries


def seq_logprob(model, x, y):
    """Sum of log p(target) over the learned (non -100) positions of each row."""
    logits, _ = model(x)
    lp = torch.log_softmax(logits.float(), dim=-1).gather(-1, y.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return (lp * (y != -100)).sum(-1)


@torch.no_grad()
def generate_replies(model, tok, prompts, k, device, temperature, top_p, batch_seqs=96, max_new=320,
                     keep_unfinished=False):
    """k replies per prompt -> list (per prompt) of k reply id lists. A reply that never reached EOT is None
    (training pairs need finished replies) unless keep_unfinished. Prompts of equal token length are batched
    together, so generate() needs no padding."""
    eot = tok.special_ids[EOT]
    buckets = defaultdict(list)
    for i, p in enumerate(prompts):
        buckets[len(p)].append(i)
    out = [None] * len(prompts)
    autocast = torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda")
    per = max(1, batch_seqs // k)
    for plen, members in buckets.items():
        for c in range(0, len(members), per):
            chunk = members[c : c + per]
            idx = torch.tensor([prompts[i] for i in chunk for _ in range(k)], device=device)
            with autocast:
                # prompt + reply (incl. EOT) must fit the context, or training on the pair is impossible
                gen = model.generate(idx, min(max_new, model.cfg.block_size - plen), temperature=temperature,
                                     top_p=top_p, eos_id=eot)
            rows = gen[:, plen:].tolist()
            for j, i in enumerate(chunk):
                out[i] = [r[: r.index(eot)] if eot in r else (r if keep_unfinished else None)
                          for r in rows[j * k : (j + 1) * k]]
    return out


def word_fraction(tok, reply, fields):
    hit, n = words_used(tok.decode(reply), fields)
    return hit / n if n else 0.0


def make_reward(tok, device, judge=None, word_weight=2.0):
    """reward(list of (prompt, fields, reply)) -> list of floats."""
    def reward(items):
        words = [word_fraction(tok, r, f) for _, f, r in items]
        if judge is None:
            return words
        js = score(judge, [p + r + [tok.special_ids[EOT]] for p, _, r in items], device)
        return [j + word_weight * w for j, w in zip(js, words)]
    return reward


def build_pairs(requests, replies, reward):
    """(prompt, chosen, rejected, chosen reward, rejected reward) per request whose finished replies differ."""
    flat = [(i, r) for i, rs in enumerate(replies) for r in rs if r is not None]
    rewards = reward([(requests[i][0], requests[i][1], r) for i, r in flat])
    by_req = defaultdict(list)
    for (i, r), s in zip(flat, rewards):
        by_req[i].append((s, r))
    pairs = []
    for i, scored in by_req.items():
        if len(scored) >= 2:
            scored.sort(key=lambda t: t[0])
            (lo, rej), (hi, cho) = scored[0], scored[-1]
            if hi > lo:
                pairs.append((requests[i][0], cho, rej, hi, lo))
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
        if step % 50 == 0 or step == steps:
            print(f"  step {step:4d} | dpo loss {loss.item():.4f} | reward acc {(margin > 0).float().mean().item():.2f} | "
                  f"margin {margin.mean().item():.3f}", flush=True)
    return policy.eval()


@torch.no_grad()
def evaluate(model, tok, items, device, fluency_lm, eval_judge=None):
    """items: [(prompt ids, fields)]. One reply each (temperature 0.7, top-p 0.9, fixed seed). Returns (metrics,
    per-item judge-B scores, replies)."""
    torch.manual_seed(0)
    # no length cap here: an evaluated reply runs until EOT or the context is full, and counts either way
    replies = [rs[0] for rs in generate_replies(model, tok, [p for p, _ in items], 1, device, 0.7, 0.9,
                                                max_new=model.cfg.block_size, keep_unfinished=True)]
    texts = [tok.decode(r) for r in replies]
    worded = [(t, f) for t, (_, f) in zip(texts, items) if f.get("Words")]
    dialogue = [t for t, (_, f) in zip(texts, items) if "Dialogue" in f.get("Features", "")]
    m = {"words": sum(words_used(t, f)[0] for t, f in worded) / sum(words_used(t, f)[1] for t, f in worded),
         "dialogue": sum('"' in t for t in dialogue) / max(1, len(dialogue)),
         "length": sum(map(len, replies)) / len(replies)}
    nll = []
    for r in replies:
        ids = r[: fluency_lm.cfg.block_size]
        if len(ids) > 1:
            x = torch.tensor([ids], device=device)
            nll.append(fluency_lm(x[:, :-1], x[:, 1:])[1].item())
    m["fluency"] = sum(nll) / len(nll)
    jb = None
    if eval_judge is not None:
        jb = score(eval_judge, [p + r + [tok.special_ids[EOT]] for (p, _), r in zip(items, replies)], device)
        m["judge_b"] = sum(jb) / len(jb)
    return m, jb, texts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sft", default="runs/ts_xl_chat2/ckpt.pt")
    p.add_argument("--fluency_lm", default="runs/ts_xl_1024/ckpt.pt", help="pretrained base, for the fluency check")
    p.add_argument("--reward", choices=["words", "judge"], default="judge")
    p.add_argument("--judge_rm", default="runs/judge_a/ckpt.pt", help="learned judge used as reward (--reward judge)")
    p.add_argument("--eval_judge", default="runs/judge_b/ckpt.pt", help="held-out judge, evaluation only ('' = none)")
    p.add_argument("--word_weight", type=float, default=2.0, help="reward = judge score + word_weight * word fraction")
    p.add_argument("--out", default="runs/ts_xl_dpo_judge")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--n_requests", type=int, default=1500, help="fresh requests per round")
    p.add_argument("--k", type=int, default=6, help="samples per request")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=8, help="pairs per step (x2 sequences, + reference)")
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--eval_n", type=int, default=300)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(1337)
    policy, ckpt = GPT.from_checkpoint(args.sft, device)
    fluency_lm, _ = GPT.from_checkpoint(args.fluency_lm, device)
    fluency_lm.requires_grad_(False)
    data_dir = ckpt["args"]["data"]
    tok = load_tokenizer(data_dir, CHAT_TOKENS)
    eot = tok.special_ids[EOT]
    reward_judge = Judge.load(args.judge_rm, device) if args.reward == "judge" else None
    eval_judge = Judge.load(args.eval_judge, device) if args.eval_judge else None
    reward = make_reward(tok, device, reward_judge, args.word_weight)

    # Requests: instruct *train* split, a fresh disjoint slice each round; eval: held-out val split.
    rng = random.Random(0)
    pool = [(encode_prompt(tok, f, True, rng), f) for f, s in map(parse, load_entries("train", 100)) if f and s and f.get("Words")]
    random.Random(1).shuffle(pool)
    vrng = random.Random(2)
    items = [(encode_prompt(tok, f, True, vrng), f) for f, s in map(parse, load_entries("val", None)) if f and s][: args.eval_n]
    os.makedirs(args.out, exist_ok=True)

    sft_m, sft_jb, sft_texts = evaluate(policy, tok, items, device, fluency_lm, eval_judge)
    table = [("sft", sft_m, None)]
    for rnd in range(1, args.rounds + 1):
        requests = pool[(rnd - 1) * args.n_requests : rnd * args.n_requests]
        pairs_path = os.path.join(args.out, f"pairs_round{rnd}.pt")
        if os.path.exists(pairs_path):
            pairs = torch.load(pairs_path)
        else:
            t0 = time.time()
            replies = generate_replies(policy, tok, [p for p, _ in requests], args.k, device, 1.0, 0.95)
            pairs = build_pairs(requests, replies, reward)
            torch.save(pairs, pairs_path)
            print(f"round {rnd}: sampled {args.k} x {len(requests)} replies in {time.time() - t0:.0f}s", flush=True)
        print(f"round {rnd}: {len(pairs)} pairs; mean reward chosen {sum(p[3] for p in pairs) / len(pairs):.2f}, "
              f"rejected {sum(p[4] for p in pairs) / len(pairs):.2f}", flush=True)
        ref = copy.deepcopy(policy).requires_grad_(False)  # each round is anchored to where it started
        dpo_train(policy, ref, pairs, eot, device, args.steps, args.batch_size, args.lr, args.beta)
        del ref
        torch.save({"model": policy.state_dict(), "config": asdict(policy.cfg), "step": rnd * args.steps,
                    "args": {**vars(args), "data": data_dir, "extra_special_tokens": CHAT_TOKENS}},
                   os.path.join(args.out, "ckpt.pt"))
        m, jb, texts = evaluate(policy, tok, items, device, fluency_lm, eval_judge)
        win = sum((a > b) + 0.5 * (a == b) for a, b in zip(jb, sft_jb)) / len(jb) if jb else None  # ties: half
        table.append((f"round {rnd}", m, win))
        print(f"round {rnd}: {m} | judge-B win rate vs sft {win}", flush=True)

    print(f"\n{'model':9} {'words':>6} {'dialogue':>9} {'judge B':>8} {'win vs sft':>10} {'fluency':>8} {'length':>7}")
    for name, m, win in table:
        print(f"{name:9} {m['words']:6.0%} {m['dialogue']:9.0%} {m.get('judge_b', float('nan')):8.2f} "
              f"{'-' if win is None else f'{win:.0%}':>10} {m['fluency']:8.3f} {m['length']:7.0f}")
    print(f"\n--- request ---\n{tok.decode(items[0][0])}\n--- sft ---\n{sft_texts[0][:600]}\n--- final ---\n{texts[0][:600]}")


if __name__ == "__main__":
    main()
