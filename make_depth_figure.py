"""Does the certificate loosen with depth? Reads results_v3/depth/summary.csv.
   python make_depth_figure.py
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#8a8984"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
})

rows = list(csv.DictReader(open("results_v3/depth/summary.csv")))
depths = [int(r["depth"]) for r in rows]
clean = [float(r["clean"]) for r in rows]
cert05 = [float(r["cert0.05"]) for r in rows]
att05 = [float(r["attack0.05"]) for r in rows]
cert10 = [float(r["cert0.1"]) for r in rows]
att10 = [float(r["attack0.1"]) for r in rows]

fig, ax = plt.subplots(figsize=(6.6, 4.6))
ax.plot(depths, clean, color=INK2, lw=1.8, ls=":", marker="d", ms=6,
        markeredgecolor=SURFACE, markeredgewidth=1.2, label="Чиста точність")
ax.plot(depths, cert05, color=BLUE, lw=2.2, marker="o", ms=6,
        markeredgecolor=SURFACE, markeredgewidth=1.3, label="Сертифікат, ρ=0,05")
ax.plot(depths, att05, color=BLUE, lw=2.2, ls="--", marker="s", ms=6,
        markeredgecolor=SURFACE, markeredgewidth=1.3, label="Атака, ρ=0,05")
ax.plot(depths, cert10, color=ORANGE, lw=2.2, marker="o", ms=6,
        markeredgecolor=SURFACE, markeredgewidth=1.3, label="Сертифікат, ρ=0,1")
ax.plot(depths, att10, color=ORANGE, lw=2.2, ls="--", marker="s", ms=6,
        markeredgecolor=SURFACE, markeredgewidth=1.3, label="Атака, ρ=0,1")
ax.set_xticks(depths)
ax.set_xlabel("Глибина мережі (кількість шарів ваг)")
ax.set_ylabel("Точність")
ax.set_ylim(-0.03, 1.02)
ax.grid(color=GRID, lw=0.8)
ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=8, loc="lower left")
ax.set_title("Сертифікат слабшає з глибиною; за 8 шарами навчання ламається",
             loc="left", fontsize=10.5, color=INK)
fig.tight_layout()
os.makedirs("results_v3/depth_fig", exist_ok=True)
out = "results_v3/depth_fig/fig_depth.png"
fig.savefig(out, dpi=170, bbox_inches="tight")
print("saved", out)
