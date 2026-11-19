"""Look inside a trained model: attention maps, induction heads, and token-embedding geometry.

python analysis.py --ckpt runs/ts_llama/ckpt.pt     # writes assets/analysis/{attention,induction,embeddings}.png
"""
import argparse
import os
import re

import numpy as np
import torch
import torch.nn.functional as F

from data import load_tokenizer
from model import GPT

OUT = "assets/analysis"


def record_attention(model, idx):
    """Run idx through the hand-written attention path and return weights, (n_layer, n_head, T, T) per batch row."""
    attns = [b.attn for b in model.blocks]
    for a in attns:
        a.flash, a.keep_attn = False, True
    with torch.no_grad():
        logits, _ = model(idx)
    weights = torch.stack([a.last_attn for a in attns], dim=1)  # (B, n_layer, n_head, T, T)
    for a in attns:
        a.flash, a.keep_attn, a.last_attn = model.cfg.flash, False, None
    return logits, weights


def induction_scores(model, vocab_lo, vocab_hi, seq_len=50, batch=16, seed=0):
    """Feed [random tokens, same random tokens]. An induction head, at a token in the repeat, attends to the
    position right after that token's previous occurrence (offset seq_len - 1 back), so it can copy what came next.
    Score = mean attention on exactly that position. Also returns loss on each half: copying makes half 2 easy."""
    g = torch.Generator().manual_seed(seed)
    rand = torch.randint(vocab_lo, vocab_hi, (batch, seq_len), generator=g)
    idx = torch.cat([rand, rand], dim=1).to(next(model.parameters()).device)
    logits, w = record_attention(model, idx)
    q = torch.arange(seq_len, 2 * seq_len, device=idx.device)  # query positions in the repeat
    scores = w[:, :, :, q, q - seq_len + 1].mean(dim=(0, 3))  # (n_layer, n_head)
    prev = w[:, :, :, q, q - 1].mean(dim=(0, 3))  # previous-token heads: the other half of an induction circuit
    loss = F.cross_entropy(logits[:, :-1].transpose(1, 2), idx[:, 1:], reduction="none")
    return scores, prev, loss[:, : seq_len - 1].mean().item(), loss[:, seq_len:].mean().item()


def single_token_ids(tok, words):
    ids = {}
    for w in words:
        e = tok.encode(w)
        if len(e) == 1:
            ids[w] = e[0]
    return ids


def plot_induction_training(logs, plt):
    """Induction score, previous-token score and repeat gain over training, from train.py --induction logs."""
    pat = re.compile(r"^step +(\d+) .*\| induct ([\d.]+) \| prev ([\d.]+) \| repeat gain (-?[\d.]+)")
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for log in logs:
        rows = np.array([[float(x) for x in m.groups()] for m in map(pat.match, open(log)) if m])
        label = os.path.basename(log).removesuffix(".log")
        for ax, col in zip(axes, (1, 2, 3)):
            ax.plot(rows[:, 0], rows[:, col], label=label)
    for ax, title in zip(axes, ("best induction score", "best previous-token score", "repeat gain (loss drop, nats)")):
        ax.set(title=title, xlabel="step")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/induction_training.png", dpi=130)
    print(f"wrote {OUT}/induction_training.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="runs/ts_llama/ckpt.pt")
    p.add_argument("--prompt", default="Once upon a time, there was a girl named Lily. Lily had a red ball. "
                                       "One day, Lily lost her red ball. Lily looked for her red")
    p.add_argument("--induction_logs", nargs="+", help="only plot induction scores over training from these logs")
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(OUT, exist_ok=True)
    if args.induction_logs:
        return plot_induction_training(args.induction_logs, plt)
    model, ckpt = GPT.from_checkpoint(args.ckpt, "cpu")
    tok = load_tokenizer(ckpt["args"]["data"])
    cfg = model.cfg
    print(f"{args.ckpt}: {cfg.n_layer} layers x {cfg.n_head} heads, {model.num_params() / 1e6:.1f}M params")

    # 1) Induction heads ------------------------------------------------------------------------------------
    scores, prev, loss_first, loss_repeat = induction_scores(model, 256, tok.vocab_size - 1)
    print(f"random tokens: loss {loss_first:.2f} on first pass, {loss_repeat:.2f} on the repeat "
          f"(in-context copying)")
    heads = [(l, h) for l in range(cfg.n_layer) for h in range(cfg.n_head)]
    top = sorted(((scores[l, h].item(), l, h) for l, h in heads), reverse=True)
    print("top induction heads (layer.head: score): " + ", ".join(f"{l}.{h}: {s:.2f}" for s, l, h in top[:5]))
    top_prev = sorted(((prev[l, h].item(), l, h) for l, h in heads), reverse=True)
    print("top previous-token heads:                " + ", ".join(f"{l}.{h}: {s:.2f}" for s, l, h in top_prev[:5]))
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(scores.numpy(), cmap="magma", vmin=0, vmax=1)
    ax.set(xlabel="head", ylabel="layer", title="Induction score per head", xticks=range(cfg.n_head),
           yticks=range(cfg.n_layer))
    for l in range(cfg.n_layer):
        for h in range(cfg.n_head):
            s = scores[l, h].item()
            ax.text(h, l, f"{s:.2f}", ha="center", va="center", fontsize=8, color="black" if s > 0.5 else "white")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(f"{OUT}/induction.png", dpi=130)

    # 2) Attention maps on a real prompt: every head, plus the strongest induction head labelled ------------
    ids = tok.encode(args.prompt)
    labels = [tok.decode([i]).replace("\n", "\\n") for i in ids]
    _, w = record_attention(model, torch.tensor([ids]))
    w = w[0]  # (n_layer, n_head, T, T)
    fig, axes = plt.subplots(cfg.n_layer, cfg.n_head, figsize=(1.6 * cfg.n_head, 1.6 * cfg.n_layer))
    for l in range(cfg.n_layer):
        for h in range(cfg.n_head):
            ax = axes[l][h]
            ax.imshow(w[l, h].numpy() ** 0.5, cmap="viridis", vmin=0, vmax=1)  # sqrt so weak links stay visible
            ax.set_xticks([]), ax.set_yticks([])
            if h == 0:
                ax.set_ylabel(f"L{l}", fontsize=9)
            if l == 0:
                ax.set_title(f"H{h}", fontsize=9)
    fig.suptitle("Attention per layer (rows) and head (cols); query = row, key = column", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{OUT}/attention_all.png", dpi=110)

    _, bl, bh = top[0]
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(w[bl, bh].numpy(), cmap="viridis", vmin=0)
    ax.set_xticks(range(len(ids)), labels, rotation=90, fontsize=8)
    ax.set_yticks(range(len(ids)), labels, fontsize=8)
    ax.set(title=f"Layer {bl}, head {bh} (top induction head)", xlabel="attends to (key)", ylabel="query token")
    fig.tight_layout()
    fig.savefig(f"{OUT}/attention_head.png", dpi=120)

    # 3) Token embeddings: nearest neighbours and a 2-D PCA of a few semantic groups --------------------------
    E = model.tok_emb.weight.detach()
    En = F.normalize(E, dim=1)
    for w_ in [" happy", " dog", " red", " ran", " mom", " three", " big", " tree"]:
        (i,) = tok.encode(w_)
        sims = En @ En[i]
        sims[i] = -1
        nn_ids = sims.topk(6).indices.tolist()
        print(f"{w_!r:>9} -> " + ", ".join(repr(tok.decode([j])) for j in nn_ids))

    groups = {
        "colors": [" red", " blue", " green", " yellow", " pink", " purple", " orange", " brown", " white", " black"],
        "animals": [" dog", " cat", " bird", " fish", " bunny", " bear", " frog", " duck", " lion", " cow"],
        "numbers": [" one", " two", " three", " four", " five", " six", " seven", " ten"],
        "names": [" Lily", " Tim", " Sue", " Ben", " Tom", " Sam", " Anna", " Max", " Mia", " Jack"],
        "feelings": [" happy", " sad", " angry", " scared", " excited", " proud", " tired", " surprised"],
        "verbs": [" ran", " jumped", " ate", " played", " saw", " said", " walked", " looked"],
    }
    group_ids = {g: single_token_ids(tok, ws) for g, ws in groups.items()}
    rows = [(g, w_, i) for g, d in group_ids.items() for w_, i in d.items()]
    X = E[[i for _, _, i in rows]].numpy()
    X = X - X.mean(0)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    P = X @ Vt[:2].T
    fig, ax = plt.subplots(figsize=(8, 6.5))
    colors = dict(zip(groups, plt.cm.tab10.colors))
    for (g, w_, _), (x, y) in zip(rows, P):
        ax.scatter(x, y, color=colors[g], s=25)
        ax.annotate(w_.strip(), (x, y), fontsize=8, xytext=(3, 3), textcoords="offset points")
    for g in groups:
        ax.scatter([], [], color=colors[g], label=g)
    ax.legend(fontsize=8)
    ax.set(title="Token embeddings, PCA of selected words", xlabel="PC1", ylabel="PC2")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/embeddings.png", dpi=130)
    print(f"wrote {OUT}/induction.png, attention_all.png, attention_head.png, embeddings.png")


if __name__ == "__main__":
    main()
