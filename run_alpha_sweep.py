"""Ablation: how the depth profile of gamma (alpha) affects robustness.
alpha < 0: wide corridor near the input; alpha > 0: wide near the output.

    python run_alpha_sweep.py --seeds 5
Writes results/alpha_sweep.csv
"""
import argparse
import csv
import os

import numpy as np

from interval_nn.data import load_digits
from interval_nn.diversity import LinearDiversity
from interval_nn.strategies import IntervalStrategy
from run_experiment import evaluate, train

ALPHAS = (-0.9, -0.6, -0.3, 0.0, 0.3, 0.6, 0.9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--rho", type=float, default=0.15)
    ap.add_argument("--eps", type=float, default=0.03)
    ap.add_argument("--normalize", choices=["layers", "params"],
                    default="layers")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    rows = []
    for variant, eps in (("corridor", 0.0), ("corridor+box", a.eps)):
        for alpha in ALPHAS:
            for seed in range(a.seeds):
                data = load_digits(seed=seed)
                model = train(IntervalStrategy(a.rho, eps),
                              LinearDiversity(alpha), data, seed, a.epochs, 64,
                              a.normalize)
                rows.append({"variant": variant, "alpha": alpha, "seed": seed,
                             **evaluate(model, data, seed)})
            sel = [r for r in rows
                   if r["variant"] == variant and r["alpha"] == alpha]
            print(f"{variant:13s} alpha {alpha:+.1f}  clean "
                  f"{np.mean([r['clean'] for r in sel]):.3f}  noise0.3 "
                  f"{np.mean([r['noise0.3'] for r in sel]):.3f}")

    name = "alpha_sweep.csv" if a.normalize == "layers" \
        else "alpha_sweep_params.csv"
    with open(os.path.join(a.out, name), "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
