"""Compute-optimal sweep (Chinchilla "IsoFLOP" method, Hoffmann et al. 2022): for each compute budget C, train every
model size N for exactly D = C / (6N) tokens (cosine schedule matched to that length), find the N with lowest loss,
then fit how the optimal N and D grow with C.

python isoflop.py          # runs missing/unfinished runs (resumable), then fits and plots
python isoflop.py --plot   # fit + plot only
"""
import argparse
import os

import numpy as np

from scaling import SIZES, VOCAB, BLOCK, non_embedding_params, read_curve, run

BUDGETS = [3e14, 1e15, 3e15]  # FLOPs
DATA = "data/tinystories_bpe4096_2300mb"  # ~550M train tokens: enough for every run in <= 2.5 epochs
TOKENS_PER_STEP = 64 * BLOCK
MIN_STEPS = 200  # below this a run is mostly warmup; skip it


def total_params(shape):
    return non_embedding_params(*shape) + VOCAB * shape[2]  # embedding counted once (tied head)


def plan():
    """[(budget, name, shape, steps, out)] for every run in the sweep."""
    runs = []
    for C in BUDGETS:
        for name, shape in SIZES.items():
            steps = round(C / (6 * total_params(shape)) / TOKENS_PER_STEP)
            if steps >= MIN_STEPS:
                runs.append((C, name, shape, steps, f"runs/iso_{C:.0e}_{name}"))
    return runs


def fit_optimum(N, L):
    """Parabola in log10(N); returns (N at the minimum, coefficients)."""
    coef = np.polyfit(np.log10(N), L, 2)
    return 10 ** (-coef[1] / (2 * coef[0])), coef


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--plot", action="store_true", help="skip training")
    args = p.parse_args()
    runs = plan()
    if not args.plot:
        for C, name, shape, steps, out in runs:
            run(name, shape, out=out, extra=["--data", DATA, "--steps", str(steps), "--warmup",
                                              str(min(200, steps // 10)), "--eval_interval", str(steps)])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = plt.cm.plasma(np.linspace(0.1, 0.8, len(BUDGETS)))
    opt = []
    print(f"{'budget':>7} {'size':4} {'params':>8} {'tokens':>8} {'tok/param':>9} {'val loss':>8}")
    for C, col in zip(BUDGETS, colors):
        rows = [(total_params(shape), steps * TOKENS_PER_STEP, read_curve(name, out + ".log")[-1, 2])
                for c, name, shape, steps, out in runs if c == C and os.path.exists(out + ".log")]
        N, D, L = map(np.array, zip(*rows))
        for n, d, l, (_, name, *_r) in zip(N, D, L, [r for r in runs if r[0] == C]):
            print(f"{C:7.0e} {name:4} {n / 1e6:7.2f}M {d / 1e6:7.0f}M {d / n:9.0f} {l:8.4f}")
        n_opt, coef = fit_optimum(N, L)
        opt.append((C, n_opt, C / (6 * n_opt)))
        grid = np.logspace(np.log10(N.min()) - 0.2, np.log10(N.max()) + 0.2, 100)
        ax1.plot(grid, np.polyval(coef, np.log10(grid)), "--", color=col, alpha=0.6)
        ax1.scatter(N, L, color=col, label=f"C = {C:.0e} FLOPs", zorder=3)
        ax1.axvline(n_opt, color=col, alpha=0.3)
    ax1.set(xscale="log", xlabel="parameters N", ylabel="final val loss", title="IsoFLOP curves")
    ax1.legend()
    ax1.grid(alpha=0.3, which="both")

    C, Nopt, Dopt = map(np.array, zip(*opt))
    a = np.polyfit(np.log(C), np.log(Nopt), 1)[0]
    b = np.polyfit(np.log(C), np.log(Dopt), 1)[0]
    print("\ncompute-optimal: " + ", ".join(f"C={c:.0e}: N={n / 1e6:.1f}M, D={d / 1e6:.0f}M ({d / n:.0f} tok/param)"
                                            for c, n, d in opt))
    print(f"N_opt ~ C^{a:.2f}, D_opt ~ C^{b:.2f}  (Chinchilla: ~0.5 / ~0.5, ~20 tokens per parameter)")
    ax2.plot(C, Nopt, "o-", label=f"N_opt ∝ C^{a:.2f}")
    ax2.plot(C, Dopt, "s-", label=f"D_opt ∝ C^{b:.2f}")
    ax2.set(xscale="log", yscale="log", xlabel="compute C (FLOPs)", ylabel="optimal N (params) / D (tokens)",
            title="Compute-optimal model size and data")
    ax2.legend()
    ax2.grid(alpha=0.3, which="both")
    fig.tight_layout()
    os.makedirs("assets/analysis", exist_ok=True)
    fig.savefig("assets/analysis/isoflop.png", dpi=130)
    print("wrote assets/analysis/isoflop.png")


if __name__ == "__main__":
    main()
