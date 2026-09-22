"""Does the depth curriculum fix the depth-10 collapse from run_depth.py?
Reads results_v3/depth/summary.csv (baseline) and
results_v3/depth_fix/final/summary.csv (with IntervalStrategy's
depth_curriculum, curriculum_span=0.05, otherwise identical settings).

    python make_depth_fix_figure.py
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, GREY = "#2a78d6", "#8a8984"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
})

base = list(csv.DictReader(open("results_v3/depth/summary.csv")))
fix = list(csv.DictReader(open("results_v3/depth_fix/final/summary.csv")))
depths = [int(r["depth"]) for r in base]

fig, ax = plt.subplots(figsize=(6.6, 4.6))
ax.plot(depths, [float(r["cert0.1"]) for r in base], color=GREY, lw=2.2,
        marker="o", ms=6, markeredgecolor=SURFACE, markeredgewidth=1.3,
        label="Базовий розклад, сертифікат ρ=0,1")
ax.plot(depths, [float(r["attack0.1"]) for r in base], color=GREY, lw=2.2,
        ls="--", marker="s", ms=6, markeredgecolor=SURFACE, markeredgewidth=1.3,
        label="Базовий розклад, атака ρ=0,1")
ax.plot(depths, [float(r["cert0.1"]) for r in fix], color=BLUE, lw=2.2,
        marker="o", ms=6, markeredgecolor=SURFACE, markeredgewidth=1.3,
        label="З curriculum, сертифікат ρ=0,1")
ax.plot(depths, [float(r["attack0.1"]) for r in fix], color=BLUE, lw=2.2,
        ls="--", marker="s", ms=6, markeredgecolor=SURFACE, markeredgewidth=1.3,
        label="З curriculum, атака ρ=0,1")
ax.set_xticks(depths)
ax.set_xlabel("Глибина мережі (кількість шарів ваг)")
ax.set_ylabel("Точність")
ax.set_ylim(-0.03, 1.02)
ax.grid(color=GRID, lw=0.8)
ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=8, loc="lower left")
ax.set_title("Розклад curriculum по глибині усуває колапс на 8-10 шарах",
             loc="left", fontsize=10.5, color=INK)
fig.tight_layout()
os.makedirs("results_v3/depth_fix_fig", exist_ok=True)
out = "results_v3/depth_fix_fig/fig_depth_fix.png"
fig.savefig(out, dpi=170, bbox_inches="tight")
print("saved", out)
