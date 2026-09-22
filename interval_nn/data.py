"""Adapter pattern: any source is converted to one format (float array in
[0, 1], int labels). Only sklearn's bundled 8x8 digits are wired in; other
sources (MNIST, sensor logs) just need another `load_*` returning Dataset."""
from dataclasses import dataclass

import numpy as np


@dataclass
class Dataset:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


def load_digits(seed=0, test_share=0.25):
    from sklearn.datasets import load_digits as _load

    d = _load()
    x = d.data / 16.0
    y = d.target.astype(int)
    idx = np.random.default_rng(seed).permutation(len(y))
    n_test = int(len(y) * test_share)
    te, tr = idx[:n_test], idx[n_test:]
    return Dataset(x[tr], y[tr], x[te], y[te])


def add_gaussian_noise(x, sigma, rng):
    return x + sigma * rng.standard_normal(x.shape)


def batches(x, y, size, rng):
    idx = rng.permutation(len(y))
    for i in range(0, len(y), size):
        j = idx[i:i + size]
        yield x[j], y[j]


def load_mnist(seed=0, cache_dir=None):
    """Standard split: first 60000 images train, last 10000 test.
    Needs internet on the first call (sklearn downloads it from OpenML and
    caches it). Not exercised in the cloud sandbox: run it on your machine."""
    from sklearn.datasets import fetch_openml

    x, y = fetch_openml("mnist_784", version=1, return_X_y=True,
                        as_frame=False, data_home=cache_dir)
    x = x.astype(np.float64) / 255.0
    y = y.astype(int)
    return Dataset(x[:60000], y[:60000], x[60000:], y[60000:])


def load(name, seed=0):
    if name == "digits":
        return load_digits(seed)
    if name == "mnist":
        return load_mnist(seed)
    raise ValueError(f"unknown dataset {name!r}")
