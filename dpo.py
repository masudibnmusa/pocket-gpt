"""Preference tuning with DPO (Direct Preference Optimization, Rafailov et al. 2023), optionally over several rounds.

Each round:
1. Sample K replies per story request from the current policy (fresh requests every round: on-policy data).
2. Reward each finished reply:
     --reward words : fraction of required words used (rule-based)
     --reward judge : learned judge A's score (fit to features + summary + words) + word_weight * word fraction
   minus fluency_weight * max(0, the reply's per-token loss under the pretrained LM - fluency_margin) (a fluency penalty:
   reward-hacked replies that pack the request's phrases into the story read worse to the base model; the margin
   leaves ordinary replies unpenalised, since a plain linear penalty also favours short replies).
   chosen = best reply, rejected = worst; requests whose replies all score the same are skipped.
3. Train: loss = -log sigmoid(beta * [(log pi(c) - log ref(c)) - (log pi(r) - log ref(r))]), ref = the policy at the
   start of the round (round 1: the SFT model) or, with --ref sft, the SFT model in every round, + kl_coef * the per-token KL(pi || SFT) over the pair's reply tokens
   (full next-token distributions), which anchors every round to the SFT model.
   --conv_frac of the requests are a random turn of a multi-turn conversation (story, small talk or a question
   about earlier stories) instead of a single story request; the judge sees the whole history.
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
from multiturn import build_conversations, load_entries


def logprob_of(logits, y):
    """Sum of log p(target) over the learned (non -100) positions of each row."""
    lp = torch.log_softmax(logits.float(), dim=-1).gather(-1, y.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return (lp * (y != -100)).sum(-1)


def seq_logprob(model, x, y):
    return logprob_of(model(x)[0], y)


def token_kl(logits, ref_logits, y):
    """Mean over learned positions of KL(softmax(logits) || softmax(ref_logits))."""
    lp, lq = torch.log_softmax(logits.float(), -1), torch.log_softmax(ref_logits.float(), -1)
    m = y != -100
    return ((lp.exp() * (lp - lq)).sum(-1) * m).sum() / m.sum()


@torch.no_grad()
def generate_replies(model, tok, prompts, k, device, temperature, top_p, batch_seqs=96, max_new=320,
                     keep_unfinished=False):
    """k replies per prompt -> list (per prompt) of k reply id lists. A reply that never reached EOT is None
    (training pairs need finished replies) unless keep_unfinished. Prompts are sorted by length and left-padded to the
    longest in their batch; generate() masks the padding."""
    eot = tok.special_ids[EOT]
    out = [None] * len(prompts)
    autocast = torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda")
    per = max(1, batch_seqs // k)
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    for c in range(0, len(order), per):
        chunk = order[c : c + per]
        plen = max(len(prompts[i]) for i in chunk)
        rows = [[eot] * (plen - len(prompts[i])) + prompts[i] for i in chunk for _ in range(k)]
        pad = [[True] * (plen - len(prompts[i])) + [False] * len(prompts[i]) for i in chunk for _ in range(k)]
        idx = torch.tensor(rows, device=device)
        with autocast:
            # prompt + reply (incl. EOT) must fit the context, or training on the pair is impossible
            gen = model.generate(idx, min(max_new, model.cfg.block_size - plen), temperature=temperature,
                                 top_p=top_p, eos_id=eot, pad=torch.tensor(pad, device=device))
        rows = gen[:, plen:].tolist()
        for j, i in enumerate(chunk):
            out[i] = [r[: r.index(eot)] if eot in r else (r if keep_unfinished else None)
                      for r in rows[j * k : (j + 1) * k]]
    return out


def word_fraction(tok, reply, fields):
    hit, n = words_used(tok.decode(reply), fields)
    return hit / n if n else 0.0


@torch.no_grad()
def fluency_nll(lm, replies, device, batch_size=16):
    """Per-token loss of each reply on its own (no prompt) under the pretrained LM; None for replies under 2 tokens."""
    out = []
    autocast = torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda")
    for i in range(0, len(replies), batch_size):
        chunk = []
        for r in replies[i : i + batch_size]:
            r = r[: lm.cfg.block_size]
            bad = [j for j, t in enumerate(r) if t >= lm.cfg.vocab_size]  # a role token the base LM never saw
            chunk.append(r[: bad[0]] if bad else r)
        x = torch.zeros(len(chunk), max(2, max(map(len, chunk))), dtype=torch.long)
        y = torch.full_like(x, -100)
        for j, r in enumerate(chunk):
            if len(r) > 1:
                x[j, : len(r)], y[j, : len(r) - 1] = torch.tensor(r), torch.tensor(r[1:])
        y = y.to(device)
        with autocast:
            logits, _ = lm(x.to(device))
        nll = F.cross_entropy(logits.float().transpose(1, 2), y, ignore_index=-100, reduction="none").sum(-1)
        n = (y != -100).sum(-1)
        out += [(a / b) if b else None for a, b in zip(nll.tolist(), n.tolist())]
    return out


def make_reward(tok, device, judge=None, word_weight=2.0, fluency_lm=None, fluency_weight=0.0, fluency_margin=0.0):
    """reward(list of (prompt, fields, reply)) -> list of floats."""
    def reward(items):
        out = [word_fraction(tok, r, f) for _, f, r in items]
        if judge is not None:
            js = score(judge, [p + r + [tok.special_ids[EOT]] for p, _, r in items], device)
            out = [j + word_weight * w for j, w in zip(js, out)]
        if fluency_lm is not None and fluency_weight:
            nll = fluency_nll(fluency_lm, [r for _, _, r in items], device)
            out = [o - fluency_weight * max(0.0, (n or 0.0) - fluency_margin) for o, n in zip(out, nll)]
        return out
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


def dpo_train(policy, ref, pairs, eot, device, steps, batch_size, lr, beta, kl_model=None, kl_coef=0.0):
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
            logits = policy(x)[0]
            pol = logprob_of(logits, y)
            with torch.no_grad():
                rf = seq_logprob(ref, x, y)
                kl_logits = kl_model(x)[0] if kl_coef else None
        n = len(batch)
        margin = beta * ((pol[:n] - rf[:n]) - (pol[n:] - rf[n:]))  # implicit reward of chosen minus rejected
        loss = -F.logsigmoid(margin).mean()
        kl = token_kl(logits, kl_logits, y) if kl_coef else torch.zeros(())
        loss = loss + kl_coef * kl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 50 == 0 or step == steps:
            print(f"  step {step:4d} | dpo loss {loss.item():.4f} | reward acc {(margin > 0).float().mean().item():.2f} | "
                  f"margin {margin.mean().item():.3f} | kl to sft {kl.item():.4f}", flush=True)
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
    nll = [n for n in fluency_nll(fluency_lm, replies, device) if n is not None]
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
    p.add_argument("--fluency_weight", type=float, default=0.0,
                   help="minus this x the reply's per-token loss under --fluency_lm (10 ~ the judge's spread)")
    p.add_argument("--fluency_margin", type=float, default=0.0,
                   help="penalise only loss above this (hinge); 0 = linear. ~75th percentile of the SFT's samples")
    p.add_argument("--out", default="runs/ts_xl_dpo_judge")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--n_requests", type=int, default=1500, help="fresh requests per round")
    p.add_argument("--k", type=int, default=6, help="samples per request")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=8, help="pairs per step (x2 sequences, + reference)")
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--eval_n", type=int, default=300)
    p.add_argument("--kl_coef", type=float, default=0.0, help="weight of the per-token KL(policy || SFT) in the loss")
    p.add_argument("--ref", choices=["round", "sft"], default="round",
                   help="DPO reference: the policy at the start of each round, or the SFT model in every round")
    p.add_argument("--conv_frac", type=float, default=0.0, help="share of requests that are a turn of a conversation")
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
    reward = make_reward(tok, device, reward_judge, args.word_weight, fluency_lm, args.fluency_weight,
                         args.fluency_margin)
    sft_model = copy.deepcopy(policy).requires_grad_(False) if args.kl_coef or args.ref == "sft" else None

    # Requests: instruct *train* split, a fresh disjoint slice each round; eval: held-out val split.
    rng = random.Random(0)
    train_entries = load_entries("train", 100)
    pool = [(encode_prompt(tok, f, True, rng), f) for f, s in map(parse, train_entries) if f and s and f.get("Words")]
    random.Random(1).shuffle(pool)
    if args.conv_frac:  # replace that share of the pool with conversation turns: history up to a reply, its fields
        n_conv = int(args.conv_frac * args.rounds * args.n_requests)
        crng = random.Random(3)
        convs = build_conversations(tok, train_entries[-4 * n_conv :], policy.cfg.block_size, crng)
        turns = []
        for ids, _, meta in convs:
            a, _, kind, f = crng.choice(meta["turns"])
            if a <= policy.cfg.block_size - 200:  # room for a reply
                turns.append((ids[:a], f if kind == "story" else {}))
        pool = pool[: args.rounds * args.n_requests]
        for j, t in zip(sorted(crng.sample(range(len(pool)), n_conv)), turns):
            pool[j] = t
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
        # each round anchored to where it started, or every round to the SFT model
        ref = sft_model if args.ref == "sft" else copy.deepcopy(policy).requires_grad_(False)
        dpo_train(policy, ref, pairs, eot, device, args.steps, args.batch_size, args.lr, args.beta, sft_model,
                  args.kl_coef)
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
