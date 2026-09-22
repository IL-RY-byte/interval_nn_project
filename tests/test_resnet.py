"""Finite-difference checks for the depth-collapse fixes: the residual
(skip-connection) architecture (ResDense, ModelFactory.build_resnet), the
interval-width regularisation term (IntervalStrategy.beta_reg), and a
soundness/sanity check for the depth curriculum. Same style as
tests/test_interval.py: hand-derived gradients vs central finite
differences, tolerance 1e-4 relative error.

Run from the project root:  python -m unittest discover -s tests -v
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from interval_nn.diversity import ConstantDiversity  # noqa: E402
from interval_nn.evaluate import certified_accuracy  # noqa: E402
from interval_nn.model import ModelFactory  # noqa: E402
from interval_nn.strategies import IntervalStrategy, softmax_ce  # noqa: E402


def make_resnet(seed=0):
    return ModelFactory.build_resnet(n_in=5, width=6, depth=5, n_out=3,
                                     diversity=ConstantDiversity(), seed=seed)


class TestResDenseSoundness(unittest.TestCase):
    """Same soundness property as plain Dense: the interval bound must
    contain the output for every weight set inside the corridor."""

    def test_bounds_contain_all_sampled_networks(self):
        rng = np.random.default_rng(1)
        model = make_resnet()
        x = rng.uniform(0, 1, (5, 5))
        eps, rho = 0.05, 0.2
        zm, zr = model.forward_bounds(x, eps, rho)
        lo, hi = zm - zr, zm + zr
        centre = [l.W.copy() for l in model.layers]
        for _ in range(2000):
            for l, c in zip(model.layers, centre):
                s = rho * l.gamma
                l.W[:] = c * (1 + s * rng.uniform(-1, 1, c.shape))
            out = model.forward(x + eps * rng.uniform(-1, 1, x.shape))
            self.assertTrue(np.all(out >= lo - 1e-9))
            self.assertTrue(np.all(out <= hi + 1e-9))

    def test_zero_corridor_equals_point_pass(self):
        rng = np.random.default_rng(2)
        model = make_resnet()
        x = rng.uniform(0, 1, (4, 5))
        zm, zr = model.forward_bounds(x, 0.0, 0.0)
        np.testing.assert_allclose(zm, model.forward(x), atol=1e-12)
        np.testing.assert_allclose(zr, 0.0, atol=1e-12)

    def test_gamma_is_settable(self):
        """evaluate.certified_accuracy temporarily sets every layer's gamma
        to 1.0 (same corridor width everywhere) and restores it afterwards;
        ResDense must support that too, not just plain Dense."""
        rng = np.random.default_rng(10)
        model = make_resnet(seed=11)
        x = rng.uniform(0, 1, (5, 5))
        y = rng.integers(0, 3, 5)
        before = [l.gamma for l in model.layers]
        certified_accuracy(model, x, y, 0.1)
        after = [l.gamma for l in model.layers]
        self.assertEqual(before, after)


class TestResDenseGradients(unittest.TestCase):
    def test_point_pass_gradient(self):
        """Plain backprop through the residual model (kappa=1, no worst
        case) vs finite differences of the clean cross-entropy."""
        rng = np.random.default_rng(3)
        model = make_resnet(seed=4)
        x = rng.uniform(0, 1, (7, 5))
        y = rng.integers(0, 3, 7)

        def loss():
            return softmax_ce(model.forward(x), y)[0]

        model.zero_grad()
        _, d = softmax_ce(model.forward(x), y)
        model.backward(d)
        analytic = [g.copy() for _, g in model.params()]

        h = 1e-6
        worst = 0.0
        for (p, _), g in zip(model.params(), analytic):
            flat, gflat = p.reshape(-1), g.reshape(-1)
            for i in rng.choice(flat.size, size=min(6, flat.size), replace=False):
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

    def test_interval_loss_gradient(self):
        rng = np.random.default_rng(5)
        model = make_resnet(seed=6)
        x = rng.uniform(0, 1, (6, 5))
        y = rng.integers(0, 3, 6)
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
            for i in rng.choice(flat.size, size=min(6, flat.size), replace=False):
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


class TestIntervalRegularisation(unittest.TestCase):
    """beta_reg adds beta * sum_l (1/N_l) * ||radius_l||_1 to the loss via
    extra_de in model.backward_bounds. Check the resulting dW/db against
    finite differences of the full (worst-case-loss + regularisation) loss,
    on a plain (non-residual) model, where the effect is easiest to see."""

    def test_regularisation_gradient(self):
        rng = np.random.default_rng(7)
        model = ModelFactory.build([5, 6, 6, 3], ConstantDiversity(), seed=8)
        x = rng.uniform(0, 1, (6, 5))
        y = rng.integers(0, 3, 6)
        strat = IntervalStrategy(rho=0.2, eps=0.0, kappa=0.3, beta_reg=0.5)

        def loss():
            c, w = strat.compute_grads(model, x, y, 0.2, 0.0)
            return 0.3 * c + 0.7 * w + strat.last_reg_loss

        loss()
        analytic = [g.copy() for _, g in model.params()]
        h = 1e-6
        worst = 0.0
        for (p, _), g in zip(model.params(), analytic):
            flat, gflat = p.reshape(-1), g.reshape(-1)
            for i in rng.choice(flat.size, size=min(8, flat.size), replace=False):
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

    def test_beta_zero_matches_plain_interval(self):
        """beta_reg=0 must reproduce the un-regularised strategy exactly
        (extra_de stays None, nothing changes)."""
        rng = np.random.default_rng(9)
        model_a = ModelFactory.build([5, 6, 3], ConstantDiversity(), seed=1)
        model_b = ModelFactory.build([5, 6, 3], ConstantDiversity(), seed=1)
        x = rng.uniform(0, 1, (5, 5))
        y = rng.integers(0, 3, 5)
        IntervalStrategy(rho=0.1, beta_reg=0.0).compute_grads(model_a, x, y, 0.1, 0.0)
        IntervalStrategy(rho=0.1).compute_grads(model_b, x, y, 0.1, 0.0)
        for la, lb in zip(model_a.layers, model_b.layers):
            np.testing.assert_allclose(la.dW, lb.dW, atol=1e-12)


class TestDepthCurriculum(unittest.TestCase):
    """layer_scales: without the curriculum every layer shares one scale;
    with it, the output-side layer ramps first and the input-side layer
    ramps last."""

    def test_uniform_without_curriculum(self):
        strat = IntervalStrategy(ramp=0.4, depth_curriculum=False)
        scales = strat.layer_scales(progress=0.2, n_layers=5)
        self.assertTrue(np.allclose(scales, scales[0]))
        self.assertAlmostEqual(scales[0], 0.5)

    def test_output_layer_leads_input_layer(self):
        strat = IntervalStrategy(ramp=0.3, depth_curriculum=True,
                                 curriculum_span=0.3)
        scales = strat.layer_scales(progress=0.15, n_layers=5)
        self.assertEqual(len(scales), 5)
        # last layer (output side) starts at progress 0, so by 0.15 it is
        # already partway up; first layer (input side) starts at 0.3, so
        # at progress 0.15 it hasn't started at all yet.
        self.assertGreater(scales[-1], scales[0])
        self.assertEqual(scales[0], 0.0)
        self.assertGreater(scales[-1], 0.0)

    def test_all_layers_reach_full_scale_eventually(self):
        strat = IntervalStrategy(ramp=0.3, depth_curriculum=True,
                                 curriculum_span=0.3)
        scales = strat.layer_scales(progress=1.0, n_layers=5)
        np.testing.assert_allclose(scales, 1.0)


if __name__ == "__main__":
    unittest.main()
