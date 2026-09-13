from __future__ import annotations

import numpy as np
import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn
from pyro.infer import Predictive, SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.nn import PyroModule, PyroSample


class LagBNN(PyroModule):
    """BNN over the same lag vector used by deterministic baselines."""

    def __init__(self, d: int, hidden: int = 24):
        super().__init__()
        self.h = PyroModule[nn.Linear](d, hidden)
        self.h.weight = PyroSample(dist.Normal(0, 0.35).expand([hidden, d]).to_event(2))
        self.h.bias = PyroSample(dist.Normal(0, 0.50).expand([hidden]).to_event(1))
        self.out = PyroModule[nn.Linear](hidden, 1)
        self.out.weight = PyroSample(dist.Normal(0, 0.35).expand([1, hidden]).to_event(2))
        self.out.bias = PyroSample(dist.Normal(0, 0.50).expand([1]).to_event(1))

    def forward(self, x, y=None):
        mu = self.out(torch.tanh(self.h(x))).squeeze(-1)
        sigma = pyro.sample("sigma", dist.LogNormal(-1.1, 0.35))
        pyro.deterministic("mu", mu)
        with pyro.plate("data", x.shape[0]):
            pyro.sample("obs", dist.Normal(mu, sigma), obs=y)
        return mu


def fit_bnn(Xtr: np.ndarray, ytr: np.ndarray, steps: int = 1800):
    pyro.clear_param_store()
    model = LagBNN(Xtr.shape[1])
    guide = AutoDiagonalNormal(model)
    svi = SVI(model, guide, pyro.optim.Adam({"lr": 0.01}), loss=Trace_ELBO())
    X, y = torch.tensor(Xtr), torch.tensor(ytr)
    for _ in range(steps):
        svi.step(X, y)
    return model, guide


def bnn_predictive(model, guide, Xte: np.ndarray, draws: int = 500):
    post = Predictive(model, guide=guide, num_samples=draws, return_sites=("obs", "mu"))(torch.tensor(Xte))
    obs = np.squeeze(post["obs"].detach().cpu().numpy())
    mu = np.squeeze(post["mu"].detach().cpu().numpy())
    return (obs[:, None] if obs.ndim == 1 else obs, mu[:, None] if mu.ndim == 1 else mu)
