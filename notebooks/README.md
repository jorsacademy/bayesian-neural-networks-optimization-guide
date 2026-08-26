# Applied Notebooks and Guides

This directory contains nine applied workflows showing how Bayesian neural networks and related uncertainty-aware models connect to industrial-engineering and operations-research decisions.

## 1. BNN Demand Uncertainty + Inventory / Production Optimization

[`bnn_demand_inventory_optimization.ipynb`](./bnn_demand_inventory_optimization.ipynb)

```text
Demand data
    ↓
Pyro BNN + variational inference
    ↓
Posterior predictive demand scenarios
    ↓
Sample Average Approximation
    ↓
Pyomo + HiGHS
    ↓
Production / inventory decision
```

The BNN is the scenario generator; Pyomo is the decision-optimization layer.

## 2. BNN + CVaR + Chance-Constrained Production

[`bnn_cvar_chance_constrained_production.ipynb`](./bnn_cvar_chance_constrained_production.ipynb)

Compares three decisions under the same posterior predictive demand distribution:

- expected cost / SAA,
- CVaR95,
- 95% empirical service-level chance constraint.

The example emphasizes that uncertainty modeling and risk preference are separate modeling layers.

## 3. Pyro BNN Surrogate + BoTorch Bayesian Optimization

[`bnn_botorch_bayesian_optimization.ipynb`](./bnn_botorch_bayesian_optimization.ipynb)

```text
Initial experiments
    ↓
Pyro BNN surrogate
    ↓
Posterior samples
    ↓
BoTorch acquisition function
    ↓
Next experiment
```

Applicable to expensive physical experiments, discrete-event simulation, digital twins, FEA/CFD, and process optimization.

## 4. Heteroscedastic BNN + Aleatoric/Epistemic Decomposition

[`heteroscedastic_bnn_uncertainty_cvar_chance.ipynb`](./heteroscedastic_bnn_uncertainty_cvar_chance.ipynb)

The BNN learns both the conditional mean and input-dependent observation noise:

\[
Var(Y\mid x,D)=E_w[\sigma_w^2(x)]+Var_w[\mu_w(x)].
\]

Posterior scenarios are then used in CVaR and chance-constrained capacity decisions.

## 5. Bayesian Last Layer / Neural Linear + BoTorch

[`bayesian_last_layer_botorch.ipynb`](./bayesian_last_layer_botorch.ipynb)

This is not a full BNN. A deterministic backbone learns features and only the output layer is Bayesian. The resulting posterior samples are passed to BoTorch for sequential optimization.

## 6. NumPyro NUTS vs Pyro Variational Inference

[`numpyro_nuts_vs_pyro_vi_bnn.ipynb`](./numpyro_nuts_vs_pyro_vi_bnn.ipynb)

The same small BNN is inferred with:

- Pyro SVI + mean-field variational inference,
- NumPyro NUTS.

The comparison includes predictive coverage and downstream expected-cost / CVaR decisions, not only RMSE.

## 7. Multi-output BNN + Multi-objective Bayesian Optimization

[`multi_output_bnn_multi_objective_botorch.ipynb`](./multi_output_bnn_multi_objective_botorch.ipynb)

A shared Bayesian network models:

- quality ↑,
- energy ↓,
- cycle time ↓.

BoTorch qLogEHVI is used to improve the Pareto set and hypervolume.

## 8. Hierarchical BNN + Multi-plant Partial Pooling + Capacity Allocation

[`hierarchical_bnn_partial_pooling_capacity_allocation.md`](./hierarchical_bnn_partial_pooling_capacity_allocation.md)

```text
Multi-plant data
    ↓
Shared Bayesian NN
    +
Plant random intercept / slope
    ↓
Partial pooling posterior
    ↓
Plant capacity scenarios
    ↓
Pyomo stochastic capacity allocation
```

The guide also compares complete pooling, no pooling, and partial pooling conceptually and explains when a simpler mixed-effects model may be preferable.

## 9. Laplace Approximation for an Existing PyTorch Model

[`laplace_approximation_pytorch_stochastic_capacity.ipynb`](./laplace_approximation_pytorch_stochastic_capacity.ipynb)

```text
Trained deterministic PyTorch model
    ↓
MAP solution
    ↓
Last-layer Laplace approximation
    ↓
Predictive capacity distribution
    ↓
Scenario generation
    ↓
Pyomo stochastic planning
```

This is the pragmatic post-hoc uncertainty workflow for an already trained neural model.

## Installation

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

## Main packages

- `torch` — neural networks
- `pyro-ppl` — BNNs, SVI, hierarchical modeling
- `jax`, `numpyro` — NUTS / HMC inference
- `botorch` — Bayesian optimization and multi-objective BO
- `laplace-torch` — post-hoc Laplace approximation
- `pyomo`, `highspy` — mathematical optimization
- `numpy`, `pandas`, `matplotlib` — analysis and visualization

## Important validation note

The examples use synthetic data for pedagogy. A real industrial implementation should separately validate:

- posterior calibration,
- OOD behavior,
- scenario-count sensitivity,
- inference diagnostics,
- baseline models,
- computation budget,
- out-of-sample downstream decision quality.
