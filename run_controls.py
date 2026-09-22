"""Control experiment: is the pruning robustness of corridor training just an
implicit L1 penalty?   python run_controls.py --seeds 10
Compares backprop, backprop + L1 (grid of lambdas, fixed in advance) and the
corridor / weight-noise references on the same metrics."""
import argparse
import csv
import os

import numpy as np

from interval_nn.data import load
from interval_nn.diversity import ConstantDiversity
from interval_nn.strategies import (BackpropStrategy, IntervalStrategy,
                                    L1Strategy, WeightNoiseStrategy)
from run_experiment import SIZES, evaluate, train

LAMBDAS = (1e-4, 3e-4, 1e-3, 3e-3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--rho", type=float, default=0.15)
    ap.add_argument("--out", default="results_v2/controls")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    const = ConstantDiversity()
    cfgs = [("backprop", lambda: BackpropStrategy())]
    cfgs += [(f"backprop+L1 {l:g}", (lambda l=l: L1Strategy(l)))
             for l in LAMBDAS]
    cfgs += [("weight-noise const", lambda: WeightNoiseStrategy(a.rho)),
             ("interval const", lambda: IntervalStrategy(a.rho))]
    rows = []
    for seed in range(a.seeds):
        data = load("digits", seed)
        for name, mk in cfgs:
            m = train(mk(), const, data, seed, a.epochs, 64, "layers",
                      SIZES["digits"])
            rows.append({"config": name, "seed": seed,
                         **evaluate(m, data, seed)})
    keys = list(rows[0].keys())
    with open(os.path.join(a.out, "raw.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)
    cols = ["clean", "wnoise0.5", "cert0.05", "q4", "q3", "p0.5", "p0.8",
            "p0.9"]
    print(f"{'config':22s}" + "".join(f"{c:>9s}" for c in cols))
    for n in dict.fromkeys(r["config"] for r in rows):
        sel = [r for r in rows if r["config"] == n]
        print(f"{n:22s}" + "".join(
            f"{np.mean([r[c] for r in sel]):9.3f}" for c in cols))


if __name__ == "__main__":
    main()
