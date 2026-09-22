"""Follow-up to run_depth.py: depth 10 never trained past chance accuracy,
even after relaxing the ramp/kappa schedule (see README.md). This script
tries three independent fixes, alone and combined, at the depths that
struggled (8 and 10):

  --arch resnet       constant-width MLP with identity skip connections
                       on every hidden layer (ModelFactory.build_resnet),
                       so the corridor only compounds through each layer's
                       branch, not through the whole depth of the network.
  --beta-reg X         penalises the interval radius of every hidden
                       layer directly (IntervalStrategy.beta_reg), pushing
                       the optimiser toward weights that keep the box
                       narrow.
  --depth-curriculum   phases the corridor in output-layer-first instead
                       of ramping every layer in lockstep
                       (IntervalStrategy.depth_curriculum).

    python run_depth_fix.py --seeds 3 --epochs 160 --depths 8 10 \\
        --arch resnet --beta-reg 0.01 --depth-curriculum

Writes results_v3/depth_fix/<tag>/{raw,summary}.csv, tag built from the
active options so multiple configurations can be compared side by side.
"""
import argparse
import csv
import os
import time

import numpy as np

from interval_nn.data import batches, load
from interval_nn.diversity import ConstantDiversity, LinearDiversity
from interval_nn.evaluate import (accuracy, certified_accuracy,
                                  empirical_robust_accuracy)
from interval_nn.model import ModelFactory
from interval_nn.strategies import IntervalStrategy

RHO_CERT = (0.05, 0.1)


def sizes_for_depth(depth, n_in=64, n_out=10):
    assert depth >= 3
    return [n_in] + [64] * (depth - 3) + [48, 32, n_out]


def build_model(arch, depth, seed, gamma_alpha):
    diversity = ConstantDiversity() if gamma_alpha == 0.0 else LinearDiversity(gamma_alpha)
    if arch == "plain":
        sizes = sizes_for_depth(depth)
        return ModelFactory.build(sizes, diversity, seed=seed, normalize="params")
    if arch == "resnet":
        return ModelFactory.build_resnet(n_in=64, width=64, depth=depth, n_out=10,
                                         diversity=diversity, seed=seed,
                                         normalize="params")
    raise ValueError(arch)


def train(a, depth, data, seed):
    rng = np.random.default_rng(seed + 1000)
    model = build_model(a.arch, depth, seed, a.gamma_alpha)
    strat = IntervalStrategy(a.rho, ramp=a.ramp, kappa=a.kappa,
                             beta_reg=a.beta_reg,
                             depth_curriculum=a.depth_curriculum,
                             curriculum_span=a.curriculum_span)
    for ep in range(a.epochs):
        for xb, yb in batches(data.x_train, data.y_train, a.batch, rng):
            strat.train_step(model, xb, yb, ep / a.epochs, rng)
    return model


def evaluate(model, data, seed, n_attack):
    x, y = data.x_test, data.y_test
    xa, ya = x[:n_attack], y[:n_attack]
    row = {"clean": accuracy(model, x, y)}
    for r in RHO_CERT:
        row[f"cert{r}"] = certified_accuracy(model, x, y, r)
        row[f"attack{r}"] = empirical_robust_accuracy(model, xa, ya, r, seed=seed)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="digits")
    ap.add_argument("--depths", type=int, nargs="+", default=[8, 10])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=160)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--rho", type=float, default=0.1)
    ap.add_argument("--ramp", type=float, default=0.85)
    ap.add_argument("--kappa", type=float, default=0.9)
    ap.add_argument("--n-attack", type=int, default=150)
    ap.add_argument("--arch", choices=["plain", "resnet"], default="plain")
    ap.add_argument("--beta-reg", type=float, default=0.0)
    ap.add_argument("--depth-curriculum", action="store_true")
    ap.add_argument("--curriculum-span", type=float, default=0.3)
    ap.add_argument("--gamma-alpha", type=float, default=0.0)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", default="results_v3/depth_fix")
    a = ap.parse_args()

    tag = a.tag or "_".join(filter(None, [
        a.arch,
        f"beta{a.beta_reg}" if a.beta_reg > 0 else "",
        "curriculum" if a.depth_curriculum else "",
        f"gamma{a.gamma_alpha}" if a.gamma_alpha != 0.0 else "",
    ])) or "baseline"
    out_dir = os.path.join(a.out, tag)
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    t0 = time.time()
    for seed in range(a.seeds):
        data = load(a.dataset, seed=seed)
        for depth in a.depths:
            model = train(a, depth, data, seed)
            row = {"depth": depth, "seed": seed,
                  **evaluate(model, data, seed, a.n_attack)}
            rows.append(row)
            print(f"  [{tag}] depth={depth} seed={seed} clean={row['clean']:.3f} "
                 f"cert0.1={row['cert0.1']:.3f} attack0.1={row['attack0.1']:.3f} "
                 f"({time.time()-t0:.0f}s)")

    keys = list(rows[0].keys())
    with open(os.path.join(out_dir, "raw.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)

    metrics = [k for k in keys if k not in ("depth", "seed")]
    summary = []
    for depth in a.depths:
        sel = [r for r in rows if r["depth"] == depth]
        s = {"depth": depth}
        for m in metrics:
            v = np.array([r[m] for r in sel])
            s[m] = round(float(v.mean()), 4)
            s[m + "_std"] = round(float(v.std(ddof=1)) if len(v) > 1 else 0.0, 4)
        summary.append(s)
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    print(f"\n[{tag}]" + "".join(f"{c:>11s}" for c in
                                 ["depth", "clean", "cert0.1", "attack0.1"]))
    for s in summary:
        print("".join(f"{s[c]:11.3f}" if c != "depth" else f"{s[c]:11d}"
                      for c in ["depth", "clean", "cert0.1", "attack0.1"]))


if __name__ == "__main__":
    main()
