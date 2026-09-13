"""Mean-field Gaussian BNN: analytic KL and correctly scaled Monte Carlo ELBO.

Reworked from im_rl.ipynb cell 12. PyTorch only; no Pyro dependency. The toy
regression model assumes a known, constant observation noise standard deviation.
"""
from __future__ import annotations
import math

import torch
from torch import nn
from torch.nn import functional as F


class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_sigma=1.0):
        super().__init__()
        if not math.isfinite(prior_sigma) or prior_sigma <= 0:
            raise ValueError('prior_sigma must be finite and positive')
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features).normal_(0, 0.1))
        self.weight_rho = nn.Parameter(torch.full((out_features, in_features), -3.0))
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_rho = nn.Parameter(torch.full((out_features,), -3.0))
        self.register_buffer('prior_sigma', torch.tensor(float(prior_sigma)))

    @staticmethod
    def scale(rho):
        return F.softplus(rho) + torch.finfo(rho.dtype).eps

    def forward(self, x):
        weight = self.weight_mu + self.scale(self.weight_rho) * torch.randn_like(self.weight_mu)
        bias = self.bias_mu + self.scale(self.bias_rho) * torch.randn_like(self.bias_mu)
        return F.linear(x, weight, bias)

    def kl_divergence(self):
        """Exact KL(q || N(0, prior_sigma^2)); no last-draw cache."""
        kl = self.weight_mu.new_zeros(())
        for mu, rho in ((self.weight_mu, self.weight_rho), (self.bias_mu, self.bias_rho)):
            sigma = self.scale(rho)
            kl = kl + (torch.log(self.prior_sigma / sigma)
                       + (sigma.square() + mu.square()) / (2 * self.prior_sigma.square()) - 0.5).sum()
        return kl


class BayesianNeuralNetwork(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=20, output_dim=1, prior_sigma=1.0):
        super().__init__()
        self.hidden = BayesianLinear(input_dim, hidden_dim, prior_sigma)
        self.output = BayesianLinear(hidden_dim, output_dim, prior_sigma)

    def forward(self, x, num_samples=1):
        if num_samples < 1:
            raise ValueError('num_samples must be positive')
        return torch.stack([self.output(torch.tanh(self.hidden(x))) for _ in range(num_samples)])

    def kl_divergence(self):
        return self.hidden.kl_divergence() + self.output.kl_divergence()


def negative_elbo(draws, targets, kl, dataset_size, noise_std=0.3):
    """Mean NLL across posterior draws and rows + KL / full dataset size.

    draws: [samples, batch, outputs], targets: [batch, outputs]. Independent
    output log-likelihoods are summed. Evaluate log p(y|w) BEFORE averaging w.
    Uniform minibatches therefore give an unbiased estimate of this full-data
    normalized objective. The full KL is added once, not once per weight draw.
    """
    if dataset_size < 1 or dataset_size < targets.shape[0]:
        raise ValueError('dataset_size must cover the minibatch')
    if not math.isfinite(noise_std) or noise_std <= 0:
        raise ValueError('noise_std must be finite and positive')
    if draws.ndim != 3 or targets.ndim != 2 or draws.shape[1:] != targets.shape or draws.shape[0] < 1 or targets.shape[0] < 1:
        raise ValueError('draws and targets have incompatible shapes')
    log_prob = torch.distributions.Normal(draws, noise_std).log_prob(targets.unsqueeze(0))
    return -log_prob.sum(dim=-1).mean() + kl / dataset_size


@torch.no_grad()
def predictive_moments(model, x, num_samples=500, noise_std=0.3):
    if num_samples < 2 or not math.isfinite(noise_std) or noise_std <= 0:
        raise ValueError('need two draws and finite positive noise_std')
    draws = model(x, num_samples)
    mean = draws.mean(0)
    epistemic_variance = draws.var(0, unbiased=False)
    total_variance = epistemic_variance + noise_std**2
    return mean, epistemic_variance, total_variance


def make_data(seed=42, n_train=100, n_test=100, noise_std=0.3):
    rng = torch.Generator().manual_seed(seed)
    x = 6 * torch.rand(n_train + n_test, 1, generator=rng) - 3
    y = torch.sin(x) + noise_std * torch.randn(x.shape, generator=rng)
    return x[:n_train], y[:n_train], x[n_train:], y[n_train:]


def fit(model, x, y, steps=1000, learning_rate=0.01, num_samples=3, noise_std=0.3):
    if steps < 1:
        raise ValueError('steps must be positive')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = []
    model.train()
    for _ in range(steps):
        optimizer.zero_grad()
        draws = model(x, num_samples)
        loss = negative_elbo(draws, y, model.kl_divergence(), len(x), noise_std)
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    return history
