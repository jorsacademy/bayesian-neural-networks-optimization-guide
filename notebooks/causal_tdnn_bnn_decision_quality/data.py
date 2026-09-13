from __future__ import annotations

import random
import numpy as np
import pyro
import torch

SEED = 42
WINDOW = 28


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    pyro.set_rng_seed(seed)


def generate_demand(n: int = 720, seed: int = SEED) -> np.ndarray:
    """Synthetic demand with seasonality, autocorrelation, changing noise, and shocks."""
    rng = np.random.default_rng(seed)
    y = np.zeros(n, dtype=np.float32)
    y[:7] = 90 + rng.normal(0, 3, 7)
    for t in range(7, n):
        weekly = 8.0 * np.sin(2 * np.pi * t / 7)
        monthly = 5.0 * np.sin(2 * np.pi * t / 30)
        trend = 0.018 * t
        ar = 0.42 * (y[t - 1] - 90.0) + 0.10 * (y[t - 7] - 90.0)
        sigma = 2.5 + 1.8 * (np.sin(2 * np.pi * t / 45) + 1.0)
        shock = rng.gamma(2.0, 5.0) if rng.random() < 0.06 else 0.0
        y[t] = 90.0 + weekly + monthly + trend + ar + rng.normal(0, sigma) + shock
    return np.maximum(y, 1.0)


def make_windows(values: np.ndarray, window: int = WINDOW):
    X, y, idx = [], [], []
    for t in range(window, len(values)):
        X.append(values[t - window : t])
        y.append(values[t])
        idx.append(t)
    return np.asarray(X, np.float32), np.asarray(y, np.float32), np.asarray(idx)


def temporal_split(series: np.ndarray, window: int = WINDOW):
    """Chronological 60/20/20 split with scaling fitted on training history only."""
    n_samples = len(series) - window
    n_train, n_val = int(0.60 * n_samples), int(0.20 * n_samples)
    last_train_target = window + n_train - 1
    history = series[: last_train_target + 1]
    mean, std = float(history.mean()), float(history.std() + 1e-6)
    X, y, idx = make_windows((series - mean) / std, window)
    X = X[..., None]
    tr = slice(0, n_train)
    va = slice(n_train, n_train + n_val)
    te = slice(n_train + n_val, None)
    return X[tr], y[tr], X[va], y[va], X[te], y[te], idx[te], mean, std
