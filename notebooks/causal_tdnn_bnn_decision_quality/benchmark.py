from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from bnn import bnn_predictive, fit_bnn
from data import generate_demand, set_seed, temporal_split
from optimization import CostParameters, realized_cost, saa_quantile, solve_saa_with_pyomo
from tdnn import DEVICE, assert_causal_layer, train_tdnn


def ridge_fit(X, y, alpha=2.0):
    A = np.column_stack([np.ones(len(X)), X])
    reg = np.eye(A.shape[1], dtype=np.float32) * alpha
    reg[0, 0] = 0
    return np.linalg.solve(A.T @ A + reg, A.T @ y)


def metrics(y, pred):
    return {
        "MAE": float(np.mean(abs(y - pred))),
        "RMSE": float(np.sqrt(np.mean((y - pred) ** 2))),
        "WAPE": float(np.sum(abs(y - pred)) / np.sum(abs(y))),
    }


def main():
    set_seed()
    assert_causal_layer()
    Xtr, ytr, Xva, yva, Xte, yte, idx, mean, std = temporal_split(generate_demand())
    unscale = lambda z: np.asarray(z) * std + mean
    Xtr2, Xte2 = Xtr[..., 0], Xte[..., 0]

    beta = ridge_fit(Xtr2, ytr)
    preds = {
        "Naive": unscale(Xte2[:, -1]),
        "Ridge AR": unscale(np.column_stack([np.ones(len(Xte2)), Xte2]) @ beta),
    }

    tdnn = train_tdnn(Xtr, ytr, Xva, yva)
    tdnn.eval()
    with torch.no_grad():
        preds["Causal TDNN"] = unscale(
            tdnn(torch.tensor(Xte, device=DEVICE)).cpu().numpy()
        )

    bnn, guide = fit_bnn(Xtr2, ytr)
    obs_z, mu_z = bnn_predictive(bnn, guide, Xte2)
    samples = unscale(obs_z)
    preds["BNN mean"] = unscale(mu_z).mean(axis=0)
    y = unscale(yte)

    forecast = pd.DataFrame(
        [{"model": k, **metrics(y, v)} for k, v in preds.items()]
    ).sort_values("RMSE")

    lo, hi = np.quantile(samples, [0.05, 0.95], axis=0)
    coverage = float(np.mean((y >= lo) & (y <= hi)))

    costs = CostParameters()
    decisions = {k: np.maximum(v, 0) for k, v in preds.items()}
    decisions["BNN + SAA"] = np.maximum(saa_quantile(samples, costs), 0)
    oracle = realized_cost(y, y, costs)
    rows = []
    for name, q in decisions.items():
        c = realized_cost(q, y, costs)
        rows.append(
            {
                "decision": name,
                "avg_cost": float(c.mean()),
                "avg_regret": float((c - oracle).mean()),
                "service_level": float(np.mean(q >= y)),
            }
        )
    decision = pd.DataFrame(rows).sort_values("avg_cost")

    q_lp, status = solve_saa_with_pyomo(samples[:, 0], costs)
    q_emp = float(saa_quantile(samples[:, 0], costs))

    print(f"device={DEVICE}; untouched test={idx[0]}..{idx[-1]}")
    print("\nPoint forecast metrics\n", forecast.round(3).to_string(index=False))
    print(f"\nBNN 90% coverage={coverage:.3f}")
    print("\nDecision metrics\n", decision.round(3).to_string(index=False))
    print(f"\nHiGHS SAA check: {status}; Pyomo q={q_lp:.4f}; empirical q={q_emp:.4f}")


if __name__ == "__main__":
    main()
