"""Clean accuracy vs certified accuracy as the training corridor rho grows.

Protocol fixed in advance: rho grid, alpha = 0.6 for the 'inc' profile with
mean gamma per weight normalised to 1 (no tuning of gamma), 10 seeds.

    python run_frontier.py --dataset digits --seeds 10
"""
import argparse
import csv
import os

import numpy as np

from interval_nn.data import load
from interval_nn.diversity import ConstantDiversity, LinearDiversity
from interval_nn.evaluate import accuracy, certified_accuracy
from interval_nn.strategies import (BackpropStrategy, IntervalStrategy,
                                    WeightNoiseStrategy)
from run_experiment import SIZES, train

RHO_TRAIN = (0.02, 0.05, 0.1, 0.15, 0.2, 0.3)
RHO_EVAL = (0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(SIZES), default="digits")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out", default="results_v3/frontier")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    const, inc = ConstantDiversity(), LinearDiversity(0.6)
    cfgs = [("backprop", None, lambda: BackpropStrategy(), const),
            ("weight-noise 0.15", 0.15, lambda: WeightNoiseStrategy(0.15),
             const)]
    for r in RHO_TRAIN:
        cfgs.append((f"interval const {r}", r,
                     (lambda r=r: IntervalStrategy(r)), const))
        cfgs.append((f"interval inc {r}", r,
                     (lambda r=r: IntervalStrategy(r)), inc))

    rows = []
    for seed in range(a.seeds):
        data = load(a.dataset, seed)
        for name, rho, mk, div in cfgs:
            m = train(mk(), div, data, seed, a.epochs, a.batch, "params",
                      SIZES[a.dataset])
            row = {"config": name, "family": name.rsplit(" ", 1)[0]
                   if rho is not None and name.startswith("interval")
                   else name, "rho_train": "" if rho is None else rho,
                   "seed": seed,
                   "clean": accuracy(m, data.x_test, data.y_test)}
            for r in RHO_EVAL:
                row[f"cert{r}"] = certified_accuracy(
                    m, data.x_test, data.y_test, r)
            rows.append(row)
        print(f"seed {seed} done", flush=True)

    with open(os.path.join(a.out, "raw.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    cols = ["clean"] + [f"cert{r}" for r in RHO_EVAL]
    print(f"{'config':22s}" + "".join(f"{c:>10s}" for c in cols))
    for n in dict.fromkeys(r["config"] for r in rows):
        sel = [r for r in rows if r["config"] == n]
        print(f"{n:22s}" + "".join(
            f"{np.mean([r[c] for r in sel]):10.3f}" for c in cols))


if __name__ == "__main__":
    main()
