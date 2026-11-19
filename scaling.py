"""Scaling sweep: train Llama-style models of increasing size on the same token budget, then fit
L(N) = c + a * N^-alpha to final val loss vs non-embedding parameters N (Kaplan et al. 2020).

python scaling.py          # runs missing/unfinished sizes (resumable), then fits and plots
python scaling.py --plot   # fit + plot only, from existing logs
"""
import argparse
import os
import re
import subprocess
import sys

import numpy as np

from model import GPT, GPTConfig

# name: (n_layer, n_head, n_embd)
SIZES = {"xs": (2, 2, 64), "s": (4, 4, 128), "m": (4, 4, 256), "l": (6, 6, 384), "xl": (8, 8, 512)}
DATA = "data/tinystories_bpe4096"
VOCAB, BLOCK = 4096, 256
COMMON = ["--data", DATA, "--block_size", str(BLOCK), "--batch_size", "64", "--steps", "3000", "--warmup", "200",
          "--eval_interval", "250", "--eval_iters", "50", "--bias", "false", "--norm", "rmsnorm", "--mlp", "swiglu",
          "--pos", "rope", "--compile"]
STEP_LINE = re.compile(r"^step +(\d+) \| train ([\d.]+) \| val ([\d.]+) \| val bpb ([\d.]+)")


def non_embedding_params(n_layer, n_head, n_embd):
    cfg = GPTConfig(vocab_size=VOCAB, block_size=BLOCK, n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                    bias=False, norm="rmsnorm", mlp="swiglu", pos="rope")
    return GPT(cfg).num_params() - VOCAB * n_embd  # tied embedding/head excluded, as in Kaplan et al.


def run(name, shape, out=None, extra=()):
    """Train one size; `extra` flags come last, so they override COMMON (argparse keeps the last value)."""
    out = out or f"runs/scale_{name}"
    log = out + ".log"
    if os.path.exists(log) and "saved " in open(log).read():
        return
    n_layer, n_head, n_embd = shape
    cmd = [sys.executable, "-u", "train.py", *COMMON, "--n_layer", str(n_layer), "--n_head", str(n_head),
           "--n_embd", str(n_embd), "--out", out, *extra]
    if os.path.exists(os.path.join(out, "ckpt.pt")):
        cmd.append("--resume")  # an earlier sweep was interrupted mid-run
    print(f"training {name} {shape} -> {log}", flush=True)
    with open(log, "a") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)


def read_curve(name, log=None):
    rows = [STEP_LINE.match(l) for l in open(log or f"runs/scale_{name}.log")]
    return np.array([[float(x) for x in m.groups()] for m in rows if m])  # step, train, val, val_bpb


def fit_power_law(N, L):
    """Grid-search the irreducible loss c; for each, a and alpha come from a linear fit in log space."""
    best = None
    for c in np.linspace(0, L.min() * 0.999, 2000):
        slope, intercept = np.polyfit(np.log(N), np.log(L - c), 1)
        pred = c + np.exp(intercept) * N**slope
        err = ((pred - L) ** 2).sum()
        if best is None or err < best[0]:
            best = (err, c, np.exp(intercept), -slope)
    return best[1:]  # c, a, alpha


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--plot", action="store_true", help="skip training")
    args = p.parse_args()
    if not args.plot:
        for name, shape in SIZES.items():
            run(name, shape)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [n for n in SIZES if os.path.exists(f"runs/scale_{n}.log")]
    N = np.array([non_embedding_params(*SIZES[n]) for n in names], dtype=float)
    curves = {n: read_curve(n) for n in names}
    L = np.array([curves[n][-1, 2] for n in names])
    c, a, alpha = fit_power_law(N, L)

    tokens_per_step = 64 * BLOCK
    print(f"{'size':4} {'shape':12} {'non-emb N':>10} {'val loss':>9} {'val bpb':>8}")
    for n, nn_, l in zip(names, N, L):
        print(f"{n:4} {str(SIZES[n]):12} {nn_ / 1e6:9.2f}M {l:9.4f} {curves[n][-1, 3]:8.3f}")
    print(f"fit: L(N) = {c:.3f} + {a:.3g} * N^-{alpha:.3f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(names)))
    for n, nn_, col in zip(names, N, colors):
        cv = curves[n]
        ax1.plot(cv[1:, 0] * tokens_per_step / 1e6, cv[1:, 2], color=col, label=f"{nn_ / 1e6:.2f}M")
    ax1.set(xlabel="tokens seen (M)", ylabel="val loss", title="Val loss during training", yscale="log")
    ax1.legend(title="non-emb params")
    ax1.grid(alpha=0.3)

    grid = np.logspace(np.log10(N.min() / 2), np.log10(N.max() * 4), 100)
    ax2.plot(grid, c + a * grid**-alpha, "--", color="gray", label=f"{c:.2f} + {a:.3g}·N^-{alpha:.2f}")
    ax2.scatter(N, L, c=colors, zorder=3, s=50)
    ax2.set(xscale="log", xlabel="non-embedding parameters N", ylabel="final val loss",
            title=f"Loss vs model size ({3000 * tokens_per_step / 1e6:.0f}M tokens each)")
    ax2.legend()
    ax2.grid(alpha=0.3, which="both")
    fig.tight_layout()
    os.makedirs("assets/analysis", exist_ok=True)
    fig.savefig("assets/analysis/scaling.png", dpi=130)
    print("wrote assets/analysis/scaling.png")


if __name__ == "__main__":
    main()
