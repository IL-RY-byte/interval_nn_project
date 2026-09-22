"""Paired tests over seeds with Holm correction. The comparison list is fixed
here, before looking at p-values.   python analyze.py"""
import csv

import numpy as np
from scipy import stats


def load(path):
    out = {}
    for r in csv.DictReader(open(path)):
        out.setdefault(r["config"], {})[int(r["seed"])] = r
    return out


def paired(d, a, b, m):
    seeds = sorted(set(d[a]) & set(d[b]))
    x = np.array([float(d[a][s][m]) for s in seeds])
    y = np.array([float(d[b][s][m]) for s in seeds])
    diff = x - y
    return diff.mean(), stats.ttest_rel(x, y).pvalue


def holm(ps):
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(ps) - rank) * ps[i])
        adj[i] = min(1.0, run)
    return adj


tests = []
P = load("results_v2/params/raw.csv")
C = load("results_v2/controls/raw.csv")
for m in ("clean", "noise0.3", "wnoise0.5", "cert0.05", "q3", "p0.9"):
    tests.append(("profile: interval inc - const (equal mean per weight)",
                  P, "interval inc", "interval const", m))
for m in ("clean", "wnoise0.5", "q3", "p0.9"):
    tests.append(("interval const - backprop+L1 0.003", C, "interval const",
                  "backprop+L1 0.003", m))
for m in ("clean", "wnoise0.5", "q3"):
    tests.append(("interval const - weight-noise const", C, "interval const",
                  "weight-noise const", m))

res = [paired(d, a, b, m) for _, d, a, b, m in tests]
adj = holm(np.array([p for _, p in res]))
print(f"{len(tests)} comparisons, Holm-adjusted p")
for (name, d, a, b, m), (diff, p), pa in zip(tests, res, adj):
    print(f"{name:55s} {m:10s} diff {diff:+.3f}  p {p:.4f}  adj {pa:.4f}"
          f"{'  *' if pa < 0.05 else ''}")
