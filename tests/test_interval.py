"""Run from the project root:  python -m unittest discover -s tests -v"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from interval_nn.diversity import LinearDiversity  # noqa: E402
from interval_nn.model import ModelFactory  # noqa: E402
from interval_nn.strategies import (IntervalStrategy,  # noqa: E402
                                    WeightNoiseStrategy, softmax_ce)


def make_model(seed=0):
    return ModelFactory.build([6, 8, 8, 3], LinearDiversity(0.6), seed=seed)


class TestSoundness(unittest.TestCase):
    """Bounds must contain the output for EVERY weight set inside the
    corridor and EVERY input inside the box."""

    def test_bounds_contain_all_sampled_networks(self):
        rng = np.random.default_rng(1)
        model = make_model()
        x = rng.uniform(0, 1, (5, 6))
        eps, rho = 0.05, 0.2
        zm, zr = model.forward_bounds(x, eps, rho)
        lo, hi = zm - zr, zm + zr
        centre = [l.W.copy() for l in model.layers]
        for _ in range(3000):
            for l, c in zip(model.layers, centre):
                s = rho * l.gamma
                l.W[:] = c * (1 + s * rng.uniform(-1, 1, c.shape))
            out = model.forward(x + eps * rng.uniform(-1, 1, x.shape))
            self.assertTrue(np.all(out >= lo - 1e-9))
            self.assertTrue(np.all(out <= hi + 1e-9))

    def test_zero_corridor_equals_point_pass(self):
        rng = np.random.default_rng(2)
        model = make_model()
        x = rng.uniform(0, 1, (4, 6))
        zm, zr = model.forward_bounds(x, 0.0, 0.0)
        np.testing.assert_allclose(zm, model.forward(x), atol=1e-12)
        np.testing.assert_allclose(zr, 0.0, atol=1e-12)


class TestGradients(unittest.TestCase):
    """Hand-written gradients vs central finite differences."""

    def test_interval_loss_gradients(self):
        rng = np.random.default_rng(3)
        model = make_model(seed=4)
        x = rng.uniform(0, 1, (7, 6))
        y = rng.integers(0, 3, 7)
        strat = IntervalStrategy(rho=0.15, eps=0.03, kappa=0.4)

        def loss():
            c, w = strat.compute_grads(model, x, y, 0.15, 0.03)
            return 0.4 * c + 0.6 * w

        loss()
        analytic = [g.copy() for _, g in model.params()]
        h = 1e-6
        worst = 0.0
        for (p, _), g in zip(model.params(), analytic):
            flat, gflat = p.reshape(-1), g.reshape(-1)
            for i in rng.choice(flat.size, size=min(8, flat.size),
                                replace=False):
                old = flat[i]
                flat[i] = old + h
                up = loss()
                flat[i] = old - h
                dn = loss()
                flat[i] = old
                num = (up - dn) / (2 * h)
                rel = abs(num - gflat[i]) / max(1e-8, abs(num) + abs(gflat[i]))
                worst = max(worst, rel)
        self.assertLess(worst, 1e-4, f"max relative error {worst:.2e}")


class TestWeightNoise(unittest.TestCase):
    def test_centres_restored_after_step(self):
        rng = np.random.default_rng(5)
        model = make_model()
        before = [l.W.copy() for l in model.layers]
        x = rng.uniform(0, 1, (10, 6))
        y = rng.integers(0, 3, 10)
        WeightNoiseStrategy(rho=0.3, lr=0.0).train_step(model, x, y, 1.0, rng)
        for l, w in zip(model.layers, before):
            np.testing.assert_array_equal(l.W, w)


if __name__ == "__main__":
    unittest.main()
