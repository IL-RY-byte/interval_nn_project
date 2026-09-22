"""Dense layer with a relative weight corridor and hand-written gradients.

Corridor: W_ij in [C_ij - s|C_ij|, C_ij + s|C_ij|], s = rho * gamma.

Interval arithmetic for one layer. Input is a box (m, e) = (centre, radius).
For a product w*x with w in [c-r, c+r], x in [m-e, m+e] the exact radius is
    |c| e + r |m| + r e.
With r = s|c| and summing over inputs:
    z_centre = C m + b
    z_radius = |C| (e + s(|m| + e))
"""
import numpy as np


class Dense:
    def __init__(self, n_in, n_out, gamma=1.0, rng=None):
        rng = rng or np.random.default_rng(0)
        self.W = rng.normal(0.0, np.sqrt(2.0 / n_in), (n_out, n_in))
        self.b = np.zeros(n_out)
        self.gamma = float(gamma)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def zero_grad(self):
        self.dW[:] = 0.0
        self.db[:] = 0.0

    # ---- ordinary (point) pass -------------------------------------------
    def forward(self, x):
        self._x = x
        return x @ self.W.T + self.b

    def backward(self, dz):
        self.dW += dz.T @ self._x
        self.db += dz.sum(axis=0)
        return dz @ self.W

    # ---- interval pass ---------------------------------------------------
    def forward_interval(self, m, e, rho):
        s = rho * self.gamma
        t = e + s * (np.abs(m) + e)
        zm = m @ self.W.T + self.b
        zr = t @ np.abs(self.W).T
        self._cache = (m, t, s)
        return zm, zr

    def backward_interval(self, dzm, dzr):
        m, t, s = self._cache
        self.dW += dzm.T @ m + (dzr.T @ t) * np.sign(self.W)
        self.db += dzm.sum(axis=0)
        dt = dzr @ np.abs(self.W)
        dm = dzm @ self.W + dt * s * np.sign(m)
        de = dt * (1.0 + s)
        return dm, de


class ResDense:
    """A Dense layer with an identity skip connection: forward computes
    a + Dense(a) instead of Dense(a) alone, so a_{l+1} = ReLU(a_l + W_l a_l
    + b_l). Requires n_in == n_out. The point why this helps at depth: the
    identity path carries the signal (and, in the interval pass, the input
    box) through the layer untouched, so the corridor only ever compounds
    through the branch W_l a_l, not through the whole depth of the network.

    Interval pass: box(m, e) + box(zm, zr) is computed by adding centres
    and radii, the standard (sound but slightly conservative) IBP rule for
    a sum of two boxes; this is exact when the two summands are
    independent and an over-approximation otherwise, same caveat IBP
    already carries for every layer."""

    def __init__(self, n, gamma=1.0, rng=None):
        self.dense = Dense(n, n, gamma, rng)

    @property
    def W(self):
        return self.dense.W

    @property
    def b(self):
        return self.dense.b

    @property
    def dW(self):
        return self.dense.dW

    @property
    def db(self):
        return self.dense.db

    @property
    def gamma(self):
        return self.dense.gamma

    @gamma.setter
    def gamma(self, value):
        self.dense.gamma = value

    def zero_grad(self):
        self.dense.zero_grad()

    def forward(self, x):
        return x + self.dense.forward(x)

    def backward(self, dz):
        return dz + self.dense.backward(dz)

    def forward_interval(self, m, e, rho):
        zm, zr = self.dense.forward_interval(m, e, rho)
        return m + zm, e + zr

    def backward_interval(self, dzm, dzr):
        dm, de = self.dense.backward_interval(dzm, dzr)
        return dzm + dm, dzr + de
