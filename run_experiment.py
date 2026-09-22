"""Benchmark: backprop baselines vs interval-corridor training with different
depth profiles of the diversity coefficient gamma.

    python run_experiment.py --seeds 5 --epochs 80
Writes results/raw.csv (one row per config x seed) and results/summary.csv.
"""
import argparse
import csv
import os
import time

import numpy as np

from interval_nn.data import batches, load
from interval_nn.diversity import ConstantDiversity, LinearDiversity
from interval_nn.evaluate import (accuracy, accuracy_noisy_input,
                                  accuracy_noisy_weights, accuracy_pruned,
                                  accuracy_quantized, certified_accuracy)
from interval_nn.model import ModelFactory
from interval_nn.monitor import HistoryRecorder, TrainingMonitor
from interval_nn.strategies import (BackpropStrategy, IntervalStrategy,
                                    NoiseAugStrategy, WeightNoiseStrategy)

SIZES = {"digits": [64, 64, 48, 32, 10], "mnist": [784, 128, 64, 32, 10]}
SIGMAS = (0.1, 0.2, 0.3)
DELTAS = (0.1, 0.3, 0.5)
RHO_CERT = (0.05, 0.1)
QBITS = (6, 4, 3, 2)
PRUNE = (0.5, 0.8, 0.9)


def configs(rho, eps, alpha):
    inc, dec = LinearDiversity(alpha), LinearDiversity(-alpha)
    const = ConstantDiversity()
    return [
        ("backprop", lambda: BackpropStrategy(), const),
        ("backprop+noise", lambda: NoiseAugStrategy(0.2), const),
        ("interval const", lambda: IntervalStrategy(rho), const),
        ("interval inc", lambda: IntervalStrategy(rho), inc),
        ("interval dec", lambda: IntervalStrategy(rho), dec),
        ("weight-noise const", lambda: WeightNoiseStrategy(rho), const),
        ("weight-noise inc", lambda: WeightNoiseStrategy(rho), inc),
        ("interval+box const", lambda: IntervalStrategy(rho, eps), const),
        ("interval+box inc", lambda: IntervalStrategy(rho, eps), inc),
        ("interval+box dec", lambda: IntervalStrategy(rho, eps), dec),
    ]


def train(strategy, diversity, data, seed, epochs, batch, normalize="layers",
          sizes=None):
    rng = np.random.default_rng(seed + 1000)
    model = ModelFactory.build(sizes or SIZES["digits"], diversity, seed=seed,
                               normalize=normalize)
    monitor, hist = TrainingMonitor(), HistoryRecorder()
    monitor.attach(hist)
    for ep in range(epochs):
        losses = []
        for xb, yb in batches(data.x_train, data.y_train, batch, rng):
            out = strategy.train_step(model, xb, yb, ep / epochs, rng)
            losses.append(out["loss"])
        monitor.notify("epoch_end", epoch=ep, loss=float(np.mean(losses)))
    return model


def evaluate(model, data, seed):
    rng = np.random.default_rng(seed + 2000)
    x, y = data.x_test, data.y_test
    row = {"clean": accuracy(model, x, y)}
    for s in SIGMAS:
        row[f"noise{s}"] = float(np.mean(
            [accuracy_noisy_input(model, x, y, s, rng) for _ in range(10)]))
    for d in DELTAS:
        row[f"wnoise{d}"] = float(np.mean(
            [accuracy_noisy_weights(model, x, y, d, rng) for _ in range(10)]))
    for r in RHO_CERT:
        row[f"cert{r}"] = certified_accuracy(model, x, y, r)
    for b in QBITS:
        row[f"q{b}"] = accuracy_quantized(model, x, y, b)
    for f in PRUNE:
        row[f"p{f}"] = accuracy_pruned(model, x, y, f)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(SIZES), default="digits")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--rho", type=float, default=0.15)
    ap.add_argument("--eps", type=float, default=0.03)
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--normalize", choices=["layers", "params"],
                    default="layers")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    rows = []
    t0 = time.time()
    for seed in range(a.seeds):
        data = load(a.dataset, seed=seed)
        for name, make_strategy, diversity in configs(a.rho, a.eps, a.alpha):
            model = train(make_strategy(), diversity, data, seed,
                          a.epochs, a.batch, a.normalize, SIZES[a.dataset])
            row = {"config": name, "seed": seed, **evaluate(model, data, seed)}
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
            v = np.array([r[m] for r in sel])
            s[m] = round(float(v.mean()), 4)
            s[m + "_std"] = round(float(v.std(ddof=1)) if len(v) > 1 else 0.0, 4)
        summary.append(s)
    with open(os.path.join(a.out, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    cols = ["clean", "noise0.3", "wnoise0.5", "cert0.05", "q4", "q3", "q2",
            "p0.8", "p0.9"]
    print("\n" + f"{'config':22s}" + "".join(f"{c:>9s}" for c in cols))
    for s in summary:
        print(f"{s['config']:22s}" + "".join(f"{s[c]:9.3f}" for c in cols))


if __name__ == "__main__":
    main()
