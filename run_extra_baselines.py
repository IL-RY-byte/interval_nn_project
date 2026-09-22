"""Compare PNI and Bayes by Backprop against the existing reference configs
(backprop, weight-noise, interval-const) on the digits dataset.

    python run_extra_baselines.py --seeds 5 --epochs 80

Writes results_v3/extra_baselines/raw.csv and summary.csv, printing a
table with the same columns used elsewhere in the thesis and article:
clean accuracy, certified accuracy at rho=0.05/0.1 where available, and
empirical attack accuracy at the same rhos. BBB has no interval bound
propagation path, so its certificate is left blank.

PNI trains a plain interval_nn.model.MLP, just like backprop and
interval-const, so every metric in evaluate.py applies to it unmodified.
Bayes by Backprop uses its own BBBModel (interval_nn/bbb.py) with mu and
rho instead of a single W, so it gets a separate evaluation path below
that skips the two metrics tied to forward_bounds and a plain .dW
(certified_accuracy and empirical_robust_accuracy), and computes the rest
directly against the posterior mean weights.
"""
import argparse
import csv
import os
import time

import numpy as np

from interval_nn.data import batches, load
from interval_nn.diversity import ConstantDiversity
from interval_nn.evaluate import (accuracy, accuracy_noisy_input,
                                  accuracy_noisy_weights, accuracy_pruned,
                                  accuracy_quantized, certified_accuracy,
                                  empirical_robust_accuracy)
from interval_nn.model import ModelFactory
from interval_nn.strategies import (BackpropStrategy, IntervalStrategy,
                                    PNIStrategy, WeightNoiseStrategy)
from interval_nn.bbb import train_bbb

SIZES = {"digits": [64, 64, 48, 32, 10], "mnist": [784, 128, 64, 32, 10]}
RHO_CERT = (0.05, 0.1)
DELTAS = (0.1, 0.3, 0.5)


def train_plain(strategy, data, seed, epochs, batch, sizes):
    rng = np.random.default_rng(seed + 1000)
    model = ModelFactory.build(sizes, ConstantDiversity(), seed=seed,
                               normalize="params")
    for ep in range(epochs):
        for xb, yb in batches(data.x_train, data.y_train, batch, rng):
            strategy.train_step(model, xb, yb, ep / epochs, rng)
    return model


def evaluate_common(model, data, seed):
    rng = np.random.default_rng(seed + 2000)
    x, y = data.x_test, data.y_test
    row = {"clean": accuracy(model, x, y)}
    for d in DELTAS:
        row[f"wnoise{d}"] = float(np.mean(
            [accuracy_noisy_weights(model, x, y, d, rng) for _ in range(10)]))
    for b in (4, 3, 2):
        row[f"q{b}"] = accuracy_quantized(model, x, y, b)
    for f in (0.5, 0.8, 0.9):
        row[f"p{f}"] = accuracy_pruned(model, x, y, f)
    return row


def evaluate_certified(model, data, seed, n_attack):
    rng = np.random.default_rng(seed + 3000)
    x, y = data.x_test, data.y_test
    xa, ya = x[:n_attack], y[:n_attack]
    row = {}
    for r in RHO_CERT:
        row[f"cert{r}"] = certified_accuracy(model, x, y, r)
        row[f"attack{r}"] = empirical_robust_accuracy(model, xa, ya, r, seed=seed)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(SIZES), default="digits")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--rho", type=float, default=0.1)
    ap.add_argument("--n-attack", type=int, default=150)
    ap.add_argument("--out", default="results_v3/extra_baselines")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sizes = SIZES[a.dataset]

    configs = [
        ("backprop", lambda: ("plain", BackpropStrategy())),
        ("weight-noise-0.1", lambda: ("plain", WeightNoiseStrategy(a.rho))),
        ("interval-const-0.1", lambda: ("plain", IntervalStrategy(a.rho))),
        ("pni", lambda: ("plain", PNIStrategy())),
        ("bbb", lambda: ("bbb", None)),
    ]

    rows = []
    t0 = time.time()
    for seed in range(a.seeds):
        data = load(a.dataset, seed=seed)
        for name, make in configs:
            kind, strategy = make()
            if kind == "bbb":
                model = train_bbb(sizes, data, seed, a.epochs, a.batch)
                row = {"config": name, "seed": seed,
                      **evaluate_common(model, data, seed)}
                for r in RHO_CERT:
                    row[f"cert{r}"] = float("nan")
                    row[f"attack{r}"] = float("nan")
            else:
                model = train_plain(strategy, data, seed, a.epochs, a.batch, sizes)
                row = {"config": name, "seed": seed,
                      **evaluate_common(model, data, seed),
                      **evaluate_certified(model, data, seed, a.n_attack)}
            rows.append(row)
        print(f"seed {seed} done ({time.time() - t0:.0f}s)")

    keys = list(rows[0].keys())
    with open(os.path.join(a.out, "raw.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)

    metrics = keys[2:]
    names = list(dict.fromkeys(r["config"] for r in rows))
    summary = []
    for n in names:
        sel = [r for r in rows if r["config"] == n]
        s = {"config": n}
        for m in metrics:
            v = np.array([r[m] for r in sel], dtype=float)
            v = v[~np.isnan(v)]
            if len(v) == 0:
                s[m], s[m + "_std"] = float("nan"), float("nan")
            else:
                s[m] = round(float(v.mean()), 4)
                s[m + "_std"] = round(float(v.std(ddof=1)) if len(v) > 1 else 0.0, 4)
        summary.append(s)
    with open(os.path.join(a.out, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    cols = ["clean", "cert0.05", "attack0.05", "cert0.1", "attack0.1", "p0.9"]
    print("\n" + f"{'config':20s}" + "".join(f"{c:>11s}" for c in cols))
    for s in summary:
        vals = "".join(f"{s[c]:11.3f}" if not np.isnan(s[c]) else f"{'--':>11s}"
                       for c in cols)
        print(f"{s['config']:20s}{vals}")


if __name__ == "__main__":
    main()
