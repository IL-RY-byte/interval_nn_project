"""Figures for results_v3/frontier/raw.csv.  python make_frontier_figure.py"""
import csv
import os
import sys

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

src = sys.argv[1] if len(sys.argv) > 1 else "results_v3/frontier"
rows = list(csv.DictReader(open(os.path.join(src, "raw.csv"))))
configs = list(dict.fromkeys(r["config"] for r in rows))
cert_cols = [c for c in rows[0] if c.startswith("cert")]
rho_eval = [float(c[4:]) for c in cert_cols]


def stat(cfg, col):
    v = np.array([float(r[col]) for r in rows if r["config"] == cfg])
    return v.mean(), v.std(ddof=1)


fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.6))

# ---- left: clean vs certified at corridor 0.05 --------------------------
for fam, color, marker in (("interval const", BLUE, "o"),
                           ("interval inc", ORANGE, "s")):
    cfgs = [c for c in configs if c.startswith(fam + " ")]
    cfgs.sort(key=lambda c: float(c.rsplit(" ", 1)[1]))
    xs = [stat(c, "clean")[0] for c in cfgs]
    ys = [stat(c, "cert0.05")[0] for c in cfgs]
    ax.plot(xs, ys, color=color, lw=2, marker=marker, ms=7,
            markeredgecolor=SURFACE, markeredgewidth=1.5, label=fam)
    dy = 8 if fam == "interval const" else -14
    for c, x, y in zip(cfgs, xs, ys):
        if x >= 0.92:
            ax.annotate(c.rsplit(" ", 1)[1], (x, y),
                        textcoords="offset points", xytext=(0, dy),
                        ha="center", fontsize=8, color=color)
for c, m in (("backprop", "D"), ("weight-noise 0.15", "^")):
    ax.plot(stat(c, "clean")[0], stat(c, "cert0.05")[0], m, color=GREY,
            ms=8, markeredgecolor=SURFACE, markeredgewidth=1.2, label=c)
ax.set_xlim(0.92, 0.99)
ax.set_ylim(-0.03, 1.02)
ax.set_xlabel("Clean test accuracy (rho >= 0.2 falls left of this view)")
ax.set_ylabel("Certified accuracy, corridor 5%")
ax.set_title("Trade-off as training corridor rho grows (labels = rho)",
             loc="left", fontsize=10.5, color=INK)
ax.grid(color=GRID, lw=0.8)
ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=8.5, loc="center left")

# ---- right: certified accuracy vs evaluation corridor --------------------
styles = [("interval const 0.05", BLUE, "-", "trained rho 0.05"),
          ("interval const 0.15", BLUE, "--", "trained rho 0.15"),
          ("interval const 0.3", BLUE, ":", "trained rho 0.3"),
          ("backprop", GREY, "-", "backprop"),
          ("weight-noise 0.15", GREY, "--", "weight-noise")]
for c, color, ls, label in styles:
    ys = [stat(c, col)[0] for col in cert_cols]
    bx.plot(rho_eval, ys, color=color, ls=ls, lw=2, label=label)
bx.set_xlabel("Evaluation corridor rho (relative weight tolerance)")
bx.set_ylabel("Certified accuracy")
bx.set_title("How far the guarantee reaches", loc="left", fontsize=10.5,
             color=INK)
bx.grid(color=GRID, lw=0.8)
bx.set_axisbelow(True)
bx.legend(frameon=False, fontsize=8.5, loc="upper right")

fig.tight_layout()
out = os.path.join(src, "fig_frontier.png")
fig.savefig(out, dpi=170)
print("saved", out)
