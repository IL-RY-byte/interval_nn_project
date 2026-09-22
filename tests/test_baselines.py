"""Finite-difference gradient checks for the two new baselines, PNI and
Bayes by Backprop, in the same style as tests/test_interval.py: hand
derived analytic gradients vs central finite differences, tolerance 1e-4
relative error. These only exercise strategies.PNIStrategy and bbb.py,
not interval_nn.layers or model.

Run from the project root:  python -m unittest discover -s tests -v
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from interval_nn.diversity import LinearDiversity  # noqa: E402
from interval_nn.model import ModelFactory  # noqa: E402
from interval_nn.strategies import softmax_ce  # noqa: E402
from interval_nn.bbb import BBBDense, BBBModel, sigmoid, softplus  # noqa: E402


def make_model(seed=0):
    return ModelFactory.build([6, 8, 8, 3], LinearDiversity(0.6), seed=seed)


class TestPNIGradient(unittest.TestCase):
    """PNI's log_alpha gradient: W_l = centre_l + alpha_l * eps_l,
    alpha_l = exp(log_alpha_l), eps_l fixed (frozen noise sample) so the
    loss is an ordinary differentiable function of log_alpha alone."""

    def test_log_alpha_gradient(self):
        rng = np.random.default_rng(10)
        model = make_model(seed=11)
        x = rng.uniform(0, 1, (9, 6))
        y = rng.integers(0, 3, 9)

        centres = [l.W.copy() for l in model.layers]
        eps_list = [rng.standard_normal(l.W.shape) for l in model.layers]
        log_alpha = rng.uniform(-3.5, -1.5, len(model.layers))

        def set_noisy(log_a):
            alphas = np.exp(log_a)
            for l, c, eps, a in zip(model.layers, centres, eps_list, alphas):
                l.W = c + a * eps

        def restore():
            for l, c in zip(model.layers, centres):
                l.W = c

        def loss(log_a):
            set_noisy(log_a)
            out = model.forward(x)
            restore()
            return softmax_ce(out, y)[0]

        # analytic: forward+backward at log_alpha, then chain rule to alpha
        # and to log_alpha, exactly as PNIStrategy.train_step does.
        set_noisy(log_alpha)
        model.zero_grad()
        loss_val, d = softmax_ce(model.forward(x), y)
        model.backward(d)
        restore()
        alphas = np.exp(log_alpha)
        analytic = np.zeros_like(log_alpha)
        for k, (l, eps) in enumerate(zip(model.layers, eps_list)):
            dalpha = np.sum(l.dW * eps)
            analytic[k] = dalpha * alphas[k]

        h = 1e-5
        worst = 0.0
        for k in range(len(log_alpha)):
            la = log_alpha.copy()
            la[k] += h
            up = loss(la)
            la[k] -= 2 * h
            dn = loss(la)
            num = (up - dn) / (2 * h)
            rel = abs(num - analytic[k]) / max(1e-8, abs(num) + abs(analytic[k]))
            worst = max(worst, rel)
        self.assertLess(worst, 1e-4, f"max relative error {worst:.2e}")


class TestBBBGradient(unittest.TestCase):
    """Bayes by Backprop's dL/dmu and dL/drho for L = CE(mu + softplus(rho)*eps)
    + beta*KL, reparameterisation noise eps frozen so L is an ordinary
    function of (mu, rho)."""

    def _elbo(self, layers, x, y, eps_list, beta, sigma_prior):
        for l, eps in zip(layers, eps_list):
            l.W = l.mu + softplus(l.rho) * eps
        model = BBBModel(layers)
        out = model.forward(x, sample=False)
        ce = softmax_ce(out, y)[0]
        kl = sum(l.kl(sigma_prior) for l in layers)
        return ce + beta * kl

    def test_mu_and_rho_gradients(self):
        rng = np.random.default_rng(20)
        sizes = [5, 6, 3]
        layers = [BBBDense(sizes[i], sizes[i + 1], rho_init=-2.0, rng=rng)
                  for i in range(len(sizes) - 1)]
        model = BBBModel(layers)
        x = rng.uniform(0, 1, (8, 5))
        y = rng.integers(0, 3, 8)
        eps_list = [rng.standard_normal(l.mu.shape) for l in layers]
        sigma_prior, beta = 0.5, 0.1

        # analytic: forward with frozen eps, backward CE, add KL grads.
        for l, eps in zip(layers, eps_list):
            l.W = l.mu + softplus(l.rho) * eps
            l._eps, l._sigma = eps, softplus(l.rho)
        model.zero_grad()
        out = model.forward(x, sample=False)
        _, d = softmax_ce(out, y)
        model.backward(d)
        for l in layers:
            g_mu, g_rho = l.kl_grad(sigma_prior)
            l.dmu += beta * g_mu
            l.drho += beta * g_rho

        h = 1e-5
        worst = 0.0
        checked = 0
        for l in layers:
            for arr, grad in ((l.mu, l.dmu), (l.rho, l.drho)):
                flat, gflat = arr.reshape(-1), grad.reshape(-1)
                for i in rng.choice(flat.size, size=min(5, flat.size), replace=False):
                    old = flat[i]
                    flat[i] = old + h
                    up = self._elbo(layers, x, y, eps_list, beta, sigma_prior)
                    flat[i] = old - h
                    dn = self._elbo(layers, x, y, eps_list, beta, sigma_prior)
                    flat[i] = old
                    num = (up - dn) / (2 * h)
                    rel = abs(num - gflat[i]) / max(1e-8, abs(num) + abs(gflat[i]))
                    worst = max(worst, rel)
                    checked += 1
        self.assertGreater(checked, 0)
        self.assertLess(worst, 1e-4, f"max relative error {worst:.2e}")


if __name__ == "__main__":
    unittest.main()
