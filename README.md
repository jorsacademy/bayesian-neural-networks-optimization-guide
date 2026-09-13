# Bayesian Neural Networks for Optimization and Operations Research

An applied guide to using Bayesian neural networks (BNNs) and related uncertainty-aware models in optimization, industrial engineering, and operations research.

The core architecture is:

```text
Data
  ↓
BNN / probabilistic surrogate
  ↓
Posterior predictive distribution
  ↓
Calibration and scenario generation
  ↓
Risk model
  ↓
Optimization
  ↓
Decision-quality evaluation
```

A BNN is usually **not the optimization solver**. It provides predictive uncertainty; the OR layer converts that uncertainty into decisions using Bayesian optimization, stochastic programming, CVaR, chance constraints, robust optimization, or mathematical programming.

## Main software stack

- **PyTorch** — neural-network and autodiff layer
- **Pyro** — BNNs, SVI, autoguides, hierarchical models, HMC/NUTS
- **NumPyro** — JAX-based HMC/NUTS and variational inference
- **PyMC** — structured Bayesian modeling
- **GPyTorch** — Gaussian processes and deep/variational GP models
- **BoTorch** — Bayesian optimization and acquisition functions
- **Ax** — experiment orchestration and production BO
- **Pyomo / Gurobi** — decision-optimization layer
- **laplace-torch** — post-hoc Laplace approximation for trained PyTorch models

For the detailed library-role matrix, integration patterns, and problem-to-stack recommendations, see [`TOOLS_AND_USE_CASES.md`](./TOOLS_AND_USE_CASES.md).

## Key decision formulations

Posterior predictive distribution:

\[
p(y^*\mid x^*,D)=\int p(y^*\mid x^*,w)p(w\mid D)\,dw.
\]

Sample Average Approximation:

\[
\min_x E[Q(x,\xi)]\approx\min_x\frac1S\sum_{s=1}^S Q(x,\xi_s).
\]

CVaR:

\[
\operatorname{CVaR}_\alpha(L)=\min_\eta\left[\eta+\frac{1}{1-\alpha}E[(L-\eta)^+]\right].
\]

Chance constraint:

\[
P(g(x,\xi)\le0)\ge1-\alpha.
\]

For heteroscedastic BNNs, the law of total variance gives

\[
\operatorname{Var}(Y\mid x,D)=E_w[\sigma_w^2(x)]+\operatorname{Var}_w[\mu_w(x)].
\]

## When to use what

| Problem | Strong starting point |
|---|---|
| Small-data expensive black box | GPyTorch + BoTorch |
| High-dimensional nonlinear black box | Pyro BNN + BoTorch |
| Sequential experimentation | Ax + BoTorch |
| Demand uncertainty + inventory | PyMC / NumPyro / Pyro + Pyomo/Gurobi |
| Predictive maintenance + planning | Pyro / NumPyro + Pyomo/Gurobi |
| Chance-constrained production | Posterior scenarios + Pyomo/Gurobi |
| Digital-twin optimization | PyTorch/Pyro + BoTorch/Ax |
| Existing trained neural network needing UQ | Laplace approximation |
| Multiple plants with uneven data | Hierarchical model / hierarchical BNN |

Do not default to a full BNN. For small expensive datasets, a GP is often the better first model. Deep ensembles should also be used as a practical uncertainty baseline.

## Applied examples

The `notebooks/` directory contains ten workflows:

1. `bnn_demand_inventory_optimization.ipynb` — BNN demand scenarios → SAA → Pyomo.
2. `bnn_cvar_chance_constrained_production.ipynb` — expected cost vs CVaR vs chance constraints.
3. `bnn_botorch_bayesian_optimization.ipynb` — Pyro BNN surrogate → BoTorch.
4. `heteroscedastic_bnn_uncertainty_cvar_chance.ipynb` — aleatoric/epistemic decomposition and risk-aware decisions.
5. `bayesian_last_layer_botorch.ipynb` — neural features + Bayesian last layer + BoTorch.
6. `numpyro_nuts_vs_pyro_vi_bnn.ipynb` — VI vs NUTS and downstream decision comparison.
7. `multi_output_bnn_multi_objective_botorch.ipynb` — quality/energy/cycle-time Pareto optimization with qLogEHVI.
8. `hierarchical_bnn_partial_pooling_capacity_allocation.md` — multi-plant partial pooling → stochastic capacity allocation.
9. `laplace_approximation_pytorch_stochastic_capacity.ipynb` — trained PyTorch NN → Laplace approximation → capacity scenarios → Pyomo.
10. `causal_tdnn_bnn_decision_quality/` — naive/AR/causal TDNN/BNN forecasting → SAA → forecast accuracy vs downstream decision quality.

See [`notebooks/README.md`](./notebooks/README.md) for details.

## Theory and model taxonomy

- [`BNN_FAMILIES.md`](./BNN_FAMILIES.md) — BNN family taxonomy and practical selection rules.
- [`TOOLS_AND_USE_CASES.md`](./TOOLS_AND_USE_CASES.md) — software stack, integration patterns, and IE/OR use-case matrix.
- [`THEORETICAL_FOUNDATIONS.md`](./THEORETICAL_FOUNDATIONS.md) — Bayesian inference, uncertainty decomposition, SAA, CVaR, chance constraints, VI vs NUTS, and validation.
- [`MULTI_OUTPUT_MOBO_FOUNDATIONS.md`](./MULTI_OUTPUT_MOBO_FOUNDATIONS.md) — multi-output modeling, Pareto dominance, hypervolume, qLogEHVI.
- [`LAPLACE_FOUNDATIONS.md`](./LAPLACE_FOUNDATIONS.md) — post-hoc Laplace approximation and its limitations.

## Industrial validation rule

A mathematically valid uncertainty model is not automatically valid for a specific factory or decision system. Evaluate at least:

- predictive accuracy,
- interval coverage and calibration,
- NLL/CRPS or another proper score,
- deterministic NN / ensemble / GP / classical baseline,
- scenario-count sensitivity,
- out-of-sample constraint violation,
- expected cost,
- CVaR / tail loss,
- OOD behavior,
- inference sensitivity,
- computation and update cost.

The objective is not to use the most sophisticated Bayesian model. It is to use the simplest uncertainty model that is calibrated enough to improve downstream decisions.

## Start here: a PyTorch-only BNN tutorial

[`notebooks/bnn_from_scratch_tutorial.ipynb`](notebooks/bnn_from_scratch_tutorial.ipynb)
introduces reparameterized Gaussian weights, analytic KL, correctly normalized
Monte Carlo ELBO, and epistemic versus total predictive variance. It includes
executed training and uncertainty figures and a held-out synthetic check.
The accompanying `notebooks/bnn_from_scratch.py` is a corrected adaptation of
cell 12 in `im_rl.ipynb`; it uses no Pyro or optimization solver.

```bash
python -m pip install -r requirements-tutorial.txt
python -m pytest -q
```

Open the tutorial from either the repository root or `notebooks/`. Its noise
standard deviation is assumed known. Bands are mean ±2 standard deviations,
not a guarantee of 95% coverage, and extrapolation is not validated by the
in-range test sample. CI tests the mathematics and executes the notebook.
