"""Does the IBP bound loosen with depth, as the theory predicts?

Every other experiment in this project uses a 4-weight-layer MLP. Here the
same interval-corridor training (rho_train=0.1, constant gamma profile) is
repeated on deeper networks (6, 8, 10 weight layers, same width pattern),
to see whether the gap between the certificate and the empirical attack
widens as IBP's interval bounds compound through more layers.

The default ramp and kappa are relaxed from ramp=0.4, kappa=0.5 to
ramp=0.85, kappa=0.9, with epochs raised to 160. This was needed: with
the shallow-network defaults, every depth from 6 layers up collapsed to
chance accuracy during training, since a deep network's compounding
bounds make the worst case loss far more destructive early on. The
relaxed schedule is the gentlest one that still trains the 4 and 6 layer
networks to the same accuracy as the rest of the project, and it is
applied identically to every depth so the comparison is fair. Depth 10
still fails to train past chance accuracy even under this schedule. See
README.md for the full write-up.

    python run_depth.py --seeds 5 --epochs 160

Writes results_v3/depth/raw.csv and summary.csv.
"""
import argparse
import csv
import os
import time

import numpy as np

from interval_nn.data import batches, load
from interval_nn.diversity import ConstantDiversity
from interval_nn.evaluate import (accuracy, certified_accuracy,
                                  empirical_robust_accuracy)
from interval_nn.model import ModelFactory
from interval_nn.strategies import IntervalStrategy

RHO_CERT = (0.05, 0.1)


def sizes_for_depth(depth, n_in=64, n_out=10):
    """depth = number of weight layers. Keeps the same width pattern as the
    rest of the project (constant 64 hidden width, tapering to 48/32 right
    before the output) so depth is the only thing that changes."""
    assert depth >= 3
    return [n_in] + [64] * (depth - 3) + [48, 32, n_out]


def train_interval(sizes, data, seed, epochs, batch, rho, ramp, kappa):
    rng = np.random.default_rng(seed + 1000)
    model = ModelFactory.build(sizes, ConstantDiversity(), seed=seed,
                               normalize="params")
    strat = IntervalStrategy(rho, ramp=ramp, kappa=kappa)
    for ep in range(epochs):
        for xb, yb in batches(data.x_train, data.y_train, batch, rng):
            strat.train_step(model, xb, yb, ep / epochs, rng)
    return model


def evaluate(model, data, seed, n_attack):
    x, y = data.x_test, data.y_test
    xa, ya = x[:n_attack], y[:n_attack]
    row = {"clean": accuracy(model, x, y)}
    for r in RHO_CERT:
        row[f"cert{r}"] = certified_accuracy(model, x, y, r)
        row[f"attack{r}"] = empirical_robust_accuracy(model, xa, ya, r, seed=seed)
        row[f"gap{r}"] = row[f"attack{r}"] - row[f"cert{r}"]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="digits")
    ap.add_argument("--depths", type=int, nargs="+", default=[4, 6, 8, 10])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=160)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--rho", type=float, default=0.1)
    ap.add_argument("--ramp", type=float, default=0.85)
    ap.add_argument("--kappa", type=float, default=0.9)
    ap.add_argument("--n-attack", type=int, default=150)
    ap.add_argument("--out", default="results_v3/depth")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    rows = []
    t0 = time.time()
    for seed in range(a.seeds):
        data = load(a.dataset, seed=seed)
        for depth in a.depths:
            sizes = sizes_for_depth(depth)
            model = train_interval(sizes, data, seed, a.epochs, a.batch, a.rho,
                                   a.ramp, a.kappa)
            row = {"depth": depth, "n_weights": sum(
                sizes[i] * sizes[i + 1] for i in range(len(sizes) - 1)),
                  "seed": seed, **evaluate(model, data, seed, a.n_attack)}
            rows.append(row)
        print(f"seed {seed} done ({time.time() - t0:.0f}s)")

    keys = list(rows[0].keys())
    with open(os.path.join(a.out, "raw.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)

    metrics = [k for k in keys if k not in ("depth", "n_weights", "seed")]
    summary = []
    for depth in a.depths:
        sel = [r for r in rows if r["depth"] == depth]
        s = {"depth": depth, "n_weights": sel[0]["n_weights"]}
        for m in metrics:
            v = np.array([r[m] for r in sel])
            s[m] = round(float(v.mean()), 4)
            s[m + "_std"] = round(float(v.std(ddof=1)) if len(v) > 1 else 0.0, 4)
        summary.append(s)
    with open(os.path.join(a.out, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    cols = ["depth", "clean", "cert0.05", "attack0.05", "gap0.05",
            "cert0.1", "attack0.1", "gap0.1"]
    print("\n" + "".join(f"{c:>11s}" for c in cols))
    for s in summary:
        print("".join(f"{s[c]:11.3f}" if c != "depth" else f"{s[c]:11d}"
                      for c in cols))


if __name__ == "__main__":
    main()
