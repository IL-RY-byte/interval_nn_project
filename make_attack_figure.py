"""Certificate vs empirical attack: is the guarantee just a weak bound?
Reads results_v3/attack/raw.csv (digits) and results_v3/attack_mnist/raw.csv
(MNIST).   python make_attack_figure.py
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, GREY = "#2a78d6", "#8a8984"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
})

RHOS = (0.02, 0.05, 0.1)
PANELS = [("results_v3/attack/raw.csv", "Цифри 8x8 (3 seed, 200 прикладів)"),
          ("results_v3/attack_mnist/raw.csv", "MNIST (3 seed, 200 прикладів)")]


def stat(rows, cfg, col):
    v = np.array([float(r[col]) for r in rows if r["config"] == cfg])
    return v.mean()


fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True)
for ax, (path, title) in zip(axes, PANELS):
    rows = list(csv.DictReader(open(path)))
    for cfg, color, label in (("backprop", GREY, "Backprop"),
                              ("interval const 0.1", BLUE,
                               "Інтервальне, ρ = 0,1")):
        cert = [stat(rows, cfg, f"cert{r}") for r in RHOS]
        attack = [stat(rows, cfg, f"attack{r}") for r in RHOS]
        ax.plot(RHOS, cert, color=color, lw=2.2, marker="o", ms=6,
                markeredgecolor=SURFACE, markeredgewidth=1.3,
                label=f"{label}, сертифікат")
        ax.plot(RHOS, attack, color=color, lw=2.2, ls="--", marker="s",
                ms=6, markeredgecolor=SURFACE, markeredgewidth=1.3,
                label=f"{label}, атака")
    ax.set_title(title, loc="left", fontsize=10.5, color=INK)
    ax.set_xlabel("Коридор ρ (атака / сертифікат)")
    ax.set_xticks(RHOS)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_ylim(-0.03, 1.02)
axes[0].set_ylabel("Точність")
axes[1].legend(frameon=False, fontsize=8, loc="center left",
              bbox_to_anchor=(1.02, 0.5))
fig.suptitle("Сертифікат майже збігається з атакою; Backprop руйнується",
             x=0.02, ha="left", fontsize=11.5, color=INK)
fig.tight_layout(rect=[0, 0, 1, 0.94])
os.makedirs("results_v3/attack_fig", exist_ok=True)
out = "results_v3/attack_fig/fig_attack.png"
fig.savefig(out, dpi=170, bbox_inches="tight")
print("saved", out)
