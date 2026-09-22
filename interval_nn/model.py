"""MLP container and the factory that builds it (Factory pattern)."""
import numpy as np

from .diversity import DiversityStrategy
from .layers import Dense, ResDense


class MLP:
    def __init__(self, layers):
        self.layers = layers

    # ---- point pass ------------------------------------------------------
    def forward(self, x):
        self._masks = []
        a = x
        for i, layer in enumerate(self.layers):
            z = layer.forward(a)
            if i < len(self.layers) - 1:
                mask = z > 0
                self._masks.append(mask)
                a = z * mask
            else:
                a = z
        return a

    def backward(self, dlogits):
        d = dlogits
        for i in reversed(range(len(self.layers))):
            if i < len(self.layers) - 1:
                d = d * self._masks[i]
            d = self.layers[i].backward(d)

    # ---- interval pass ---------------------------------------------------
    def forward_bounds(self, x, eps, rho):
        """Returns (centre, radius) of the output logits over the whole
        corridor and the input box [x - eps, x + eps]. rho may be a single
        number (same corridor scale for every layer) or an array of one
        value per layer (used for the depth curriculum, where different
        layers ramp up their corridor on different schedules).

        Also records, in self._radii, the post-ReLU activation radius of
        every hidden layer (index 0..n_layers-2), for the optional
        interval-width regularisation term in IntervalStrategy."""
        m, e = x, np.full_like(x, eps)
        rhos = np.broadcast_to(rho, (len(self.layers),))
        self._act = []
        self._radii = []
        last = len(self.layers) - 1
        for i, layer in enumerate(self.layers):
            zm, zr = layer.forward_interval(m, e, rhos[i])
            if i == last:
                return zm, zr
            u, v = zm - zr, zm + zr
            self._act.append((u > 0, v > 0))
            lo, hi = np.maximum(u, 0.0), np.maximum(v, 0.0)
            m, e = (lo + hi) / 2.0, (hi - lo) / 2.0
            self._radii.append(e)

    def backward_bounds(self, dzm, dzr, extra_de=None):
        """extra_de, if given, is a list with one array per hidden layer
        (same indexing as self._radii): extra gradient injected into that
        layer's radius channel, on top of what flows back from the output
        loss. Used for the interval-width regularisation term."""
        for i in reversed(range(len(self.layers))):
            dm, de = self.layers[i].backward_interval(dzm, dzr)
            if i > 0:
                if extra_de is not None:
                    de = de + extra_de[i - 1]
                u_pos, v_pos = self._act[i - 1]
                du = (dm - de) / 2.0 * u_pos
                dv = (dm + de) / 2.0 * v_pos
                dzm, dzr = du + dv, dv - du

    # ---- utilities -------------------------------------------------------
    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()

    def params(self):
        for layer in self.layers:
            yield layer.W, layer.dW
            yield layer.b, layer.db

    def predict(self, x):
        return self.forward(x).argmax(axis=1)


class ModelFactory:
    @staticmethod
    def build(sizes, diversity: DiversityStrategy, seed=0, normalize="layers"):
        """normalize="layers": mean gamma over layers is 1.
        normalize="params": mean gamma weighted by the number of weights is 1,
        i.e. the *average corridor over all weights* is the same for every
        depth profile (removes the confound that late layers are small)."""
        rng = np.random.default_rng(seed)
        n_layers = len(sizes) - 1
        gammas = diversity.gammas(n_layers)
        if normalize == "params":
            n = np.array([sizes[i] * sizes[i + 1] for i in range(n_layers)])
            gammas = gammas / (gammas * n).sum() * n.sum()
        return MLP([Dense(sizes[i], sizes[i + 1], gammas[i], rng)
                    for i in range(n_layers)])

    @staticmethod
    def build_resnet(n_in, width, depth, n_out, diversity: DiversityStrategy,
                     seed=0, normalize="layers"):
        """A constant-width MLP with identity skip connections on every
        hidden layer: Dense(n_in, width), then (depth - 2) ResDense(width)
        blocks, then Dense(width, n_out). depth is the total number of
        weight layers, same convention as ModelFactory.build with
        sizes_for_depth. Needs depth >= 3 (an input and an output
        projection plus at least one residual block)."""
        assert depth >= 3
        rng = np.random.default_rng(seed)
        gammas = diversity.gammas(depth)
        if normalize == "params":
            n = np.array([n_in * width] + [width * width] * (depth - 2)
                         + [width * n_out])
            gammas = gammas / (gammas * n).sum() * n.sum()
        layers = [Dense(n_in, width, gammas[0], rng)]
        for i in range(depth - 2):
            layers.append(ResDense(width, gammas[i + 1], rng))
        layers.append(Dense(width, n_out, gammas[-1], rng))
        return MLP(layers)
