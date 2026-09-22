"""Metrics shared by every training strategy."""
import numpy as np

from .data import add_gaussian_noise


def accuracy(model, x, y):
    return float((model.predict(x) == y).mean())


def accuracy_noisy_input(model, x, y, sigma, rng):
    return accuracy(model, add_gaussian_noise(x, sigma, rng), y)


def accuracy_noisy_weights(model, x, y, delta, rng):
    """Every weight is multiplied by (1 + delta * U(-1, 1))."""
    saved = [l.W.copy() for l in model.layers]
    for l in model.layers:
        l.W *= 1.0 + delta * rng.uniform(-1, 1, l.W.shape)
    acc = accuracy(model, x, y)
    for l, w in zip(model.layers, saved):
        l.W[:] = w
    return acc


def certified_accuracy(model, x, y, rho, eps=0.0):
    """Share of samples whose class cannot change for ANY weights in the
    corridor s = rho (same width in every layer, so all models are judged by
    the same yardstick) and any input in [x - eps, x + eps]."""
    saved = [l.gamma for l in model.layers]
    for l in model.layers:
        l.gamma = 1.0
    zm, zr = model.forward_bounds(x, eps, rho)
    for l, g in zip(model.layers, saved):
        l.gamma = g
    sign = np.ones_like(zm)
    sign[np.arange(len(y)), y] = -1.0
    return float(((zm + zr * sign).argmax(axis=1) == y).mean())


def quantize(W, bits):
    """Symmetric per-tensor uniform quantisation to `bits` bits."""
    levels = 2 ** (bits - 1) - 1
    scale = np.abs(W).max() / levels
    return np.round(W / scale) * scale if scale > 0 else W


def accuracy_quantized(model, x, y, bits):
    saved = [l.W.copy() for l in model.layers]
    for l in model.layers:
        l.W[:] = quantize(l.W, bits)
    acc = accuracy(model, x, y)
    for l, w in zip(model.layers, saved):
        l.W[:] = w
    return acc


def accuracy_pruned(model, x, y, frac):
    """Zero the smallest-|W| share `frac` of weights in every layer."""
    saved = [l.W.copy() for l in model.layers]
    for l in model.layers:
        k = int(frac * l.W.size)
        if k > 0:
            thr = np.partition(np.abs(l.W).ravel(), k - 1)[k - 1]
            l.W[np.abs(l.W) <= thr] = 0.0
    acc = accuracy(model, x, y)
    for l, w in zip(model.layers, saved):
        l.W[:] = w
    return acc


def empirical_robust_accuracy(model, x, y, rho, steps=20, step=0.3, seed=0):
    """Per-sample gradient attack on the weights inside the corridor
    W = C * (1 + rho * u), u in [-1, 1] (same width in every layer as
    `certified_accuracy`). A sample counts as robust if no iterate flips the
    prediction. It is an *upper* bound on true robust accuracy (a stronger
    attack can only lower it); the certificate is the *lower* bound."""
    rng = np.random.default_rng(seed)
    centres = [l.W.copy() for l in model.layers]
    robust = 0
    for i in range(len(y)):
        xi, yi = x[i:i + 1], y[i:i + 1]
        broken = False
        starts = [[np.zeros_like(c) for c in centres],
                  [rng.choice([-1.0, 1.0], c.shape) for c in centres]]
        for u in starts:
            for _ in range(steps):
                for l, c, uu in zip(model.layers, centres, u):
                    l.W[:] = c * (1.0 + rho * uu)
                logits = model.forward(xi)
                if logits.argmax() != yi[0]:
                    broken = True
                    break
                z = logits - logits.max()
                p = np.exp(z) / np.exp(z).sum()
                p[0, yi[0]] -= 1.0
                model.zero_grad()
                model.backward(p)
                for k, (l, c) in enumerate(zip(model.layers, centres)):
                    u[k] = np.clip(u[k] + step * np.sign(l.dW * c), -1.0, 1.0)
            if broken:
                break
        robust += 0 if broken else 1
    for l, c in zip(model.layers, centres):
        l.W[:] = c
    return robust / len(y)
