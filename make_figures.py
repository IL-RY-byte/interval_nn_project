"""Builds figures from results/*.csv.   python make_figures.py"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
})


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def fig_alpha(res):
    """Effect of the depth profile of gamma, with two ways of keeping the
    'average corridor' equal. Corridor-only variant."""
    series = [
        ("alpha_sweep.csv", "Same mean per layer", BLUE, "o"),
        ("alpha_sweep_params.csv", "Same mean per weight", ORANGE, "s"),
    ]
    panels = [("clean", "Clean accuracy"),
              ("noise0.3", "Accuracy under input noise, sigma = 0.3")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, (metric, title) in zip(axes, panels):
        for fn, label, color, marker in series:
            rows = [r for r in read(os.path.join(res, fn))
                    if r["variant"] == "corridor"]
            alphas = sorted({float(r["alpha"]) for r in rows})
            mean, sd = [], []
            for a in alphas:
                v = np.array([float(r[metric]) for r in rows
                              if abs(float(r["alpha"]) - a) < 1e-9])
                mean.append(v.mean())
                sd.append(v.std(ddof=1))
            ax.errorbar(alphas, mean, yerr=sd, color=color, marker=marker,
                        ms=6, lw=2, capsize=3, elinewidth=1,
                        markeredgecolor=SURFACE, markeredgewidth=1.5,
                        label=label)
        ax.set_title(title, loc="left", fontsize=10.5, color=INK)
        ax.set_xlabel("alpha  (< 0: wide corridor near input, > 0: near output)")
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.set_xticks(alphas)
    axes[0].set_ylabel("Test accuracy (mean, whiskers +/- 1 sd, 5 seeds)")
    axes[0].legend(frameon=False, loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(res, "fig_alpha_sweep.png"), dpi=170)
    plt.close(fig)


def fig_configs(res):
    rows = read(os.path.join(res, "summary.csv"))
    metrics = [("clean", "Clean"), ("noise0.3", "Input noise 0.3"),
               ("wnoise0.5", "Weight noise 50%"),
               ("cert0.05", "Certified, corridor 5%")]
    names = [r["config"] for r in rows]
    y = np.arange(len(names))[::-1]
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 5.0), sharey=True)
    for ax, (m, title) in zip(axes, metrics):
        mean = np.array([float(r[m]) for r in rows])
        sd = np.array([float(r[m + "_std"]) for r in rows])
        ax.errorbar(mean, y, xerr=sd, fmt="o", color=BLUE, ms=6, lw=0,
                    elinewidth=1.4, capsize=3, markeredgecolor=SURFACE,
                    markeredgewidth=1.2)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        ax.set_xlim(0, 1.02)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.set_yticks(y)
        ax.set_yticklabels(names)
        ax.set_xlabel("accuracy")
    fig.tight_layout()
    fig.savefig(os.path.join(res, "fig_configs.png"), dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    fig_alpha("results")
    fig_configs("results")
    print("ok")
