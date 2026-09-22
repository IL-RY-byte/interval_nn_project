"""Strategy pattern: interchangeable training algorithms + a tiny Adam."""
from abc import ABC, abstractmethod

import numpy as np


def softmax_ce(logits, y):
    """Mean cross-entropy and its gradient w.r.t. logits.

    loss_i = -log softmax(logits_i)[y_i] = -z_i[y_i] + log(sum(exp(z_i))),
    with z = logits - max(logits) (the standard log-sum-exp shift). This is
    computed directly from z instead of going through a clamped probability
    (a previous version added 1e-12 before the log to avoid log(0), which
    silently desynced the reported loss value from its gradient once a
    worst-case probability got small enough to be comparable to 1e-12,
    which happens routinely for a deep or residual network with wide
    corridors). The log-sum-exp form needs no epsilon and stays exact and
    finite for any input, so loss and gradient always agree."""
    z = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(z)
    denom = p.sum(axis=1, keepdims=True)
    p = p / denom
    n = len(y)
    loss = (-z[np.arange(n), y] + np.log(denom[:, 0])).mean()
    d = p
    d[np.arange(n), y] -= 1.0
    return loss, d / n


class Adam:
    def __init__(self, lr=3e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.t, self.m, self.v = 0, {}, {}

    def step(self, model):
        self.t += 1
        for k, (p, g) in enumerate(model.params()):
            m = self.m.setdefault(k, np.zeros_like(p))
            v = self.v.setdefault(k, np.zeros_like(p))
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * g * g
            mh = m / (1 - self.b1 ** self.t)
            vh = v / (1 - self.b2 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + self.eps)


class TrainingStrategy(ABC):
    name = "base"

    def __init__(self, lr=3e-3):
        self.opt = Adam(lr)

    @abstractmethod
    def train_step(self, model, x, y, progress, rng):
        """One optimisation step. progress in [0, 1] is the share of training
        already done (used for ramping). Returns a dict of scalar metrics."""


class BackpropStrategy(TrainingStrategy):
    name = "backprop"

    def train_step(self, model, x, y, progress, rng):
        model.zero_grad()
        loss, d = softmax_ce(model.forward(x), y)
        model.backward(d)
        self.opt.step(model)
        return {"loss": loss}


class NoiseAugStrategy(BackpropStrategy):
    """Baseline: ordinary backprop on inputs corrupted with Gaussian noise."""
    name = "backprop+noise"

    def __init__(self, sigma=0.2, lr=3e-3):
        super().__init__(lr)
        self.sigma = sigma

    def train_step(self, model, x, y, progress, rng):
        xn = x + self.sigma * rng.standard_normal(x.shape)
        return super().train_step(model, xn, y, progress, rng)


class IntervalStrategy(TrainingStrategy):
    """Worst-case training over the weight corridor and an input box.

    loss = kappa * CE(clean) + (1 - kappa) * CE(worst-case logits)
    Worst-case logits: lower bound for the true class, upper bound for the
    others. The corridor width rho is ramped from 0 during the first
    `ramp` share of training (standard IBP trick, keeps early training stable).

    Two optional extras, both aimed at the depth-collapse problem (IBP
    bounds blow up with depth, see run_depth.py): beta_reg penalises the
    interval radius of every hidden layer directly, so the optimiser is
    pushed towards weights that keep the box narrow instead of letting it
    balloon through the ReLU chain; depth_curriculum staggers the ramp so
    the corridor is switched on for the output-side layers first and the
    input-side layers last, instead of all layers ramping in lockstep.
    """
    name = "interval"

    def __init__(self, rho=0.1, eps=0.0, kappa=0.5, ramp=0.4, lr=3e-3,
                beta_reg=0.0, depth_curriculum=False, curriculum_span=0.3):
        super().__init__(lr)
        self.rho, self.eps, self.kappa, self.ramp = rho, eps, kappa, ramp
        self.beta_reg = beta_reg
        self.depth_curriculum = depth_curriculum
        self.curriculum_span = curriculum_span
        self.last_reg_loss = 0.0

    def layer_scales(self, progress, n_layers):
        """One ramp-scale per layer, in [0, 1]. Without depth_curriculum
        every layer shares the same scale (the original behaviour). With
        it, layer l's own ramp starts at progress = start_l and reaches 1
        after self.ramp further progress; start_l = 0 for the last
        (output-side) layer and grows to curriculum_span for the first
        (input-side) one, so the corridor switches on output-first."""
        if not self.depth_curriculum or n_layers <= 1:
            scale = min(1.0, progress / self.ramp) if self.ramp > 0 else 1.0
            return np.full(n_layers, scale)
        pos = np.linspace(1.0, 0.0, n_layers)  # layer 0 -> 1 (latest start)
        starts = pos * self.curriculum_span
        if self.ramp > 0:
            local = (progress - starts) / self.ramp
        else:
            local = np.where(progress >= starts, 1.0, 0.0)
        return np.clip(local, 0.0, 1.0)

    def compute_grads(self, model, x, y, rho, eps):
        """Fills model gradients, returns (clean loss, worst-case loss).
        rho may be a scalar or a per-layer array (see layer_scales)."""
        model.zero_grad()

        loss_c, d = softmax_ce(model.forward(x), y)
        model.backward(self.kappa * d)

        zm, zr = model.forward_bounds(x, eps, rho)
        sign = np.ones_like(zm)
        sign[np.arange(len(y)), y] = -1.0
        loss_w, d = softmax_ce(zm + zr * sign, y)

        extra_de = None
        self.last_reg_loss = 0.0
        if self.beta_reg > 0 and model._radii:
            B = x.shape[0]
            extra_de = []
            for e_l in model._radii:
                n_l = e_l.shape[1]
                extra_de.append(np.full_like(e_l, self.beta_reg / (n_l * B)))
                self.last_reg_loss += self.beta_reg * float(e_l.sum()) / (n_l * B)

        model.backward_bounds((1 - self.kappa) * d,
                              (1 - self.kappa) * d * sign,
                              extra_de=extra_de)
        return loss_c, loss_w

    def train_step(self, model, x, y, progress, rng):
        n_layers = len(model.layers)
        scale = self.layer_scales(progress, n_layers)
        eps_scale = min(1.0, progress / self.ramp) if self.ramp > 0 else 1.0
        loss_c, loss_w = self.compute_grads(
            model, x, y, self.rho * scale, self.eps * eps_scale)
        self.opt.step(model)
        return {"loss": loss_c, "loss_wc": loss_w, "loss_reg": self.last_reg_loss}


class WeightNoiseStrategy(TrainingStrategy):
    """Baseline: the stochastic counterpart of corridor training. Instead of
    the worst case, weights are sampled uniformly inside the same corridor
    W = C * (1 + s_l * U(-1, 1)), s_l = rho * gamma_l. Gradients computed at
    the sampled weights are applied to the clean centres (straight-through)."""
    name = "weight-noise"

    def __init__(self, rho=0.1, kappa=0.5, ramp=0.4, lr=3e-3):
        super().__init__(lr)
        self.rho, self.kappa, self.ramp = rho, kappa, ramp

    def train_step(self, model, x, y, progress, rng):
        scale = min(1.0, progress / self.ramp) if self.ramp > 0 else 1.0
        model.zero_grad()

        loss_c, d = softmax_ce(model.forward(x), y)
        model.backward(self.kappa * d)

        centres = [l.W.copy() for l in model.layers]
        for l in model.layers:
            s = self.rho * scale * l.gamma
            l.W *= 1.0 + s * rng.uniform(-1.0, 1.0, l.W.shape)
        loss_n, d = softmax_ce(model.forward(x), y)
        model.backward((1 - self.kappa) * d)
        for l, c in zip(model.layers, centres):
            l.W[:] = c

        self.opt.step(model)
        return {"loss": loss_c, "loss_noisy": loss_n}


class PNIStrategy(TrainingStrategy):
    """Baseline: Parametric Noise Injection (He, Rakin, Fan, CVPR 2019).

    Each layer l gets its own noise scale alpha_l >= 0 (parametrised as
    log_alpha to keep it positive), added to the weights:
    W' = W + alpha_l * eps, with eps ~ N(0, 1). Unlike WeightNoiseStrategy,
    where rho is fixed, alpha_l is trained by gradient descent. The
    gradient wrt alpha_l is the gradient wrt the noisy weights dotted with
    the noise sample used to produce them, as in the original paper."""
    name = "pni"

    def __init__(self, alpha_init=0.05, kappa=0.5, lr=3e-3, alpha_lr=3e-2):
        super().__init__(lr)
        self.kappa = kappa
        self.alpha_lr = alpha_lr
        self.alpha_init = alpha_init
        self.log_alpha = None

    def alphas(self):
        return None if self.log_alpha is None else np.exp(self.log_alpha)

    def train_step(self, model, x, y, progress, rng):
        if self.log_alpha is None:
            self.log_alpha = np.full(len(model.layers), np.log(self.alpha_init))

        model.zero_grad()
        loss_c, d = softmax_ce(model.forward(x), y)
        model.backward(self.kappa * d)
        dW_before = [l.dW.copy() for l in model.layers]

        centres = [l.W.copy() for l in model.layers]
        alphas = np.exp(self.log_alpha)
        eps_list = [rng.standard_normal(l.W.shape) for l in model.layers]
        for l, eps, a in zip(model.layers, eps_list, alphas):
            l.W = l.W + a * eps
        loss_n, d = softmax_ce(model.forward(x), y)
        model.backward((1 - self.kappa) * d)
        for l, c in zip(model.layers, centres):
            l.W = c

        # dL/dalpha_l = sum(dL/dW'_l * eps_l). Subtract the pre-noise
        # snapshot to isolate the noisy pass, then chain rule through
        # log_alpha_l = log(alpha_l).
        galpha = np.zeros_like(self.log_alpha)
        for k, (l, eps) in enumerate(zip(model.layers, eps_list)):
            d_noisy_only = l.dW - dW_before[k]
            galpha[k] = np.sum(d_noisy_only * eps)
        self.log_alpha = self.log_alpha - self.alpha_lr * (alphas * galpha)
        self.log_alpha = np.clip(self.log_alpha, np.log(1e-4), np.log(1.0))

        self.opt.step(model)
        return {"loss": loss_c, "loss_noisy": loss_n}


class L1Strategy(BackpropStrategy):
    """Control: backprop + L1 penalty lam * sum|W| (weights only). Worst
    case corridor training contains a term proportional to |W|, so it may
    act as an implicit L1. This baseline tests that explanation."""
    name = "backprop+L1"

    def __init__(self, lam=1e-3, lr=3e-3):
        super().__init__(lr)
        self.lam = lam

    def train_step(self, model, x, y, progress, rng):
        model.zero_grad()
        loss, d = softmax_ce(model.forward(x), y)
        model.backward(d)
        for l in model.layers:
            l.dW += self.lam * np.sign(l.W)
        self.opt.step(model)
        return {"loss": loss}
