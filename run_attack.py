"""Is the certificate meaningful? Compare it with an empirical per-sample
weight attack: certified <= true robust accuracy <= attacked accuracy.
    python run_attack.py --seeds 3
"""
import argparse
import csv
import os

import numpy as np

from interval_nn.data import load
from interval_nn.diversity import ConstantDiversity
from interval_nn.evaluate import (accuracy, certified_accuracy,
                                  empirical_robust_accuracy)
from interval_nn.strategies import (BackpropStrategy, IntervalStrategy,
                                    WeightNoiseStrategy)
from run_experiment import SIZES, train

RHOS = (0.02, 0.05, 0.1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(SIZES), default="digits")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--out", default="results_v3/attack")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    const = ConstantDiversity()
    cfgs = [("backprop", lambda: BackpropStrategy()),
            ("weight-noise 0.15", lambda: WeightNoiseStrategy(0.15)),
            ("interval const 0.05", lambda: IntervalStrategy(0.05)),
            ("interval const 0.1", lambda: IntervalStrategy(0.1)),
            ("interval const 0.15", lambda: IntervalStrategy(0.15))]
    rows = []
    for seed in range(a.seeds):
        data = load(a.dataset, seed)
        x, y = data.x_test[:a.n_test], data.y_test[:a.n_test]
        for name, mk in cfgs:
            m = train(mk(), const, data, seed, a.epochs, a.batch, "params",
                      SIZES[a.dataset])
            row = {"config": name, "seed": seed, "clean": accuracy(m, x, y)}
            for r in RHOS:
                row[f"cert{r}"] = certified_accuracy(m, x, y, r)
                row[f"attack{r}"] = empirical_robust_accuracy(m, x, y, r)
            rows.append(row)
        print(f"seed {seed} done", flush=True)
    with open(os.path.join(a.out, "raw.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    cols = ["clean"] + [f"{k}{r}" for r in RHOS for k in ("cert", "attack")]
    print(f"{'config':22s}" + "".join(f"{c:>12s}" for c in cols))
    for n in dict.fromkeys(r["config"] for r in rows):
        sel = [r for r in rows if r["config"] == n]
        print(f"{n:22s}" + "".join(
            f"{np.mean([r[c] for r in sel]):12.3f}" for c in cols))


if __name__ == "__main__":
    main()
