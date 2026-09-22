"""Bayes by Backprop baseline (Blundell, Cornebise, Kavukcuoglu, Wierstra,
ICML 2015). Every weight is a Gaussian q(w) = N(mu, sigma^2), with
sigma = softplus(rho), trained with the reparameterisation trick against
an analytic KL divergence to a fixed N(0, sigma_prior^2) prior.

Kept separate from interval_nn.layers and interval_nn.model, which
implement the tested certified training path. Bayes by Backprop needs a
different per weight parameterisation (mean and variance, not a single
value), so it gets its own small module instead.

Reuses softmax_ce, Adam and batches from the rest of the codebase, so the
usual evaluate.py metrics (clean accuracy, noise, quantisation, pruning)
work on it unchanged: they only touch .layers[i].W, .forward, .predict.
"""
import numpy as np

from .strategies import Adam, softmax_ce


def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class BBBDense:
    def __init__(self, n_in, n_out, rho_init=-3.0, rng=None):
        rng = rng or np.random.default_rng(0)
        self.mu = rng.normal(0.0, np.sqrt(2.0 / n_in), (n_out, n_in))
        self.rho = np.full((n_out, n_in), rho_init)
        self.b = np.zeros(n_out)
        self.dmu = np.zeros_like(self.mu)
        self.drho = np.zeros_like(self.rho)
        self.db = np.zeros_like(self.b)
        self.W = self.mu.copy()  # point weight used for forward/eval

    def zero_grad(self):
        self.dmu[:] = 0.0
        self.drho[:] = 0.0
        self.db[:] = 0.0

    def sample(self, rng):
        sigma = softplus(self.rho)
        eps = rng.standard_normal(self.mu.shape)
        self.W = self.mu + sigma * eps
        self._eps, self._sigma = eps, sigma
        return self.W

    def use_map(self):
        self.W = self.mu.copy()

    def forward(self, x):
        self._x = x
        return x @ self.W.T + self.b

    def backward(self, dz):
        dW = dz.T @ self._x
        self.dmu += dW
        if hasattr(self, "_eps"):
            self.drho += dW * self._eps * sigmoid(self.rho)
        self.db += dz.sum(axis=0)
        return dz @ self.W

    def kl_grad(self, sigma_prior):
        """d(KL)/d(mu), d(KL)/d(rho) for KL(N(mu,sigma^2) || N(0,sigma_prior^2))."""
        sigma = softplus(self.rho)
        g_mu = self.mu / sigma_prior ** 2
        g_sigma = sigma / sigma_prior ** 2 - 1.0 / sigma
        g_rho = g_sigma * sigmoid(self.rho)
        return g_mu, g_rho

    def kl(self, sigma_prior):
        sigma = softplus(self.rho)
        return float(np.sum(np.log(sigma_prior / sigma)
                             + (sigma ** 2 + self.mu ** 2) / (2 * sigma_prior ** 2)
                             - 0.5))


class BBBModel:
    """Same forward/backward contract as interval_nn.model.MLP, but every
    forward call samples fresh weights, unless sample=False, which uses
    whatever is currently in layer.W (the posterior mean after use_map())."""

    def __init__(self, layers):
        self.layers = layers

    def forward(self, x, rng=None, sample=True):
        self._masks = []
        a = x
        last = len(self.layers) - 1
        for i, layer in enumerate(self.layers):
            if sample:
                layer.sample(rng)
            z = layer.forward(a)
            if i < last:
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

    def zero_grad(self):
        for l in self.layers:
            l.zero_grad()

    def params(self):
        for l in self.layers:
            yield l.mu, l.dmu
            yield l.rho, l.drho
            yield l.b, l.db

    def predict(self, x):
        return self.forward(x, sample=False).argmax(axis=1)

    def use_map(self):
        for l in self.layers:
            l.use_map()


def train_bbb(sizes, data, seed, epochs, batch, sigma_prior=0.5, lr=3e-3,
              rho_init=-3.0):
    """Minibatch KL weighting pi_i = 2^-(i+1), from Blundell et al. 2015,
    section 3.4, instead of the naive uniform 1/n_batches. Uniform
    weighting adds up to a complexity cost far larger than the
    cross-entropy term on every step, so the optimiser only ever shrinks
    sigma and mu towards the prior and never fits the data. Tested this:
    uniform weighting held test accuracy at chance for 30 epochs.

    The batch index i counts across the whole run, not just one epoch.
    Resetting it every epoch also tested badly: it reapplies the large
    early weights each epoch, and the repeated pull towards the prior
    slowly erases what was learned. Counting globally means only the
    first few steps of the whole run get a real complexity cost, and the
    rest train on the likelihood almost undisturbed.
    """
    from .data import batches  # local import avoids a circular import

    rng = np.random.default_rng(seed + 4000)
    layers = [BBBDense(sizes[i], sizes[i + 1], rho_init, rng)
              for i in range(len(sizes) - 1)]
    model = BBBModel(layers)
    opt = Adam(lr)
    gi = 0
    for _ in range(epochs):
        for xb, yb in batches(data.x_train, data.y_train, batch, rng):
            beta = 2.0 ** (-(gi + 1))
            gi += 1
            model.zero_grad()
            logits = model.forward(xb, rng, sample=True)
            _, d = softmax_ce(logits, yb)
            model.backward(d)
            for l in model.layers:
                g_mu, g_rho = l.kl_grad(sigma_prior)
                l.dmu += beta * g_mu
                l.drho += beta * g_rho
            opt.step(model)
    model.use_map()
    return model
