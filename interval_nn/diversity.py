"""Strategy pattern: how the diversity coefficient gamma_l depends on depth."""
from abc import ABC, abstractmethod

import numpy as np


class DiversityStrategy(ABC):
    """Returns one gamma per layer. Values are normalised to mean 1, so
    different strategies have the same *average* corridor width and can be
    compared fairly."""

    @abstractmethod
    def raw(self, n_layers: int) -> np.ndarray:
        ...

    def gammas(self, n_layers: int) -> np.ndarray:
        g = np.asarray(self.raw(n_layers), dtype=float)
        return g / g.mean()


class ConstantDiversity(DiversityStrategy):
    def raw(self, n_layers):
        return np.ones(n_layers)


class LinearDiversity(DiversityStrategy):
    """alpha > 0: narrow near the input, wide near the output.
    alpha < 0: the opposite. alpha = 0: constant."""

    def __init__(self, alpha: float = 0.6):
        self.alpha = alpha

    def raw(self, n_layers):
        return 1.0 + self.alpha * np.linspace(-1.0, 1.0, n_layers)


class ExponentialDiversity(DiversityStrategy):
    def __init__(self, base: float = 4.0):
        self.base = base

    def raw(self, n_layers):
        return self.base ** np.linspace(0.0, 1.0, n_layers)
