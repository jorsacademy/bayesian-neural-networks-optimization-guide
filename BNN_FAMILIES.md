# BNN Families for Optimization and Operations Research

This file separates the main Bayesian-neural-network families from closely related uncertainty methods and explains when each is useful in industrial engineering and operations research.

The goal is not to use every method. The goal is to understand which uncertainty model is appropriate for a given decision problem.

## 1. Full / layer-wise BNN with Variational Inference

**Status:** implemented in this repository.

Weights are assigned prior distributions and an approximate posterior is learned, typically with variational inference.

Typical uses:

- demand and lead-time uncertainty,
- nonlinear response surfaces,
- predictive maintenance,
- posterior scenario generation,
- stochastic programming,
- CVaR / chance constraints,
- BNN surrogates for Bayesian optimization.

Recommended tools: Pyro, NumPyro, PyMC.

## 2. HMC / NUTS BNN

**Status:** implemented as a comparative example.

Notebook: [`notebooks/numpyro_nuts_vs_pyro_vi_bnn.ipynb`](./notebooks/numpyro_nuts_vs_pyro_vi_bnn.ipynb)

The same small BNN is inferred with Pyro variational inference and NumPyro NUTS. The objective is not to declare NUTS automatically correct, but to test whether the inference method changes predictive coverage and downstream operational decisions.

NUTS can be a strong posterior benchmark for small and medium models, but its computational cost grows rapidly with network size.

## 3. SGMCMC: SGLD / SGHMC

**Status:** no dedicated example yet.

Stochastic-gradient MCMC methods use mini-batches to make MCMC-like posterior sampling more scalable.

Potential OR uses:

- large manufacturing or sensor datasets,
- higher-dimensional neural surrogates,
- posterior-sample scenario generation.

These methods are more research-oriented and require careful tuning and diagnostics.

## 4. Bayesian Last Layer / Neural Linear Model

**Status:** implemented.

Notebook: [`notebooks/bayesian_last_layer_botorch.ipynb`](./notebooks/bayesian_last_layer_botorch.ipynb)

A deterministic neural network learns features \(\phi(x)\), and only the output layer is Bayesian:

\[
f(x)=\tilde\phi(x)^\top\beta,
\qquad
\beta\mid D\sim\mathcal N(m,\Sigma).
\]

Advantages:

- much cheaper than a full BNN,
- fast posterior updates,
- easy posterior sampling,
- practical BoTorch integration,
- useful for online or sequential optimization.

Critical limitation: uncertainty in the deterministic feature extractor is not represented. A Bayesian last layer is therefore not equivalent to a full BNN and may be overconfident in OOD regions.

## 5. Laplace Approximation

**Status:** implemented.

Notebook: [`notebooks/laplace_approximation_pytorch_stochastic_capacity.ipynb`](./notebooks/laplace_approximation_pytorch_stochastic_capacity.ipynb)

A deterministic or MAP neural network is trained first. The posterior is then approximated locally around the MAP solution:

\[
p(\theta\mid D)
\approx
\mathcal N(\theta_{MAP},H^{-1}).
\]

This is especially useful when an existing production PyTorch model should be made uncertainty-aware without rebuilding it as a full BNN.

The example uses a last-layer Laplace approximation because it is computationally attractive. Full-network Laplace is possible but more expensive.

## 6. Heteroscedastic BNN

**Status:** implemented.

Notebook: [`notebooks/heteroscedastic_bnn_uncertainty_cvar_chance.ipynb`](./notebooks/heteroscedastic_bnn_uncertainty_cvar_chance.ipynb)

The model learns both the conditional mean and input-dependent observation noise:

\[
Y\mid x,w\sim\mathcal N(\mu_w(x),\sigma_w^2(x)).
\]

Using the law of total variance:

\[
Var(Y\mid x,D)
=
E_w[\sigma_w^2(x)]
+
Var_w[\mu_w(x)].
\]

This is useful when variability itself changes with the operating condition, for example:

- demand volatility under promotions,
- processing-time variability under high load,
- transportation variability under congestion,
- quality variability across process set-points.

The posterior predictive distribution can then be used in CVaR and chance-constrained decisions.

## 7. Multi-output BNN + Multi-objective Optimization

**Status:** implemented.

Notebook: [`notebooks/multi_output_bnn_multi_objective_botorch.ipynb`](./notebooks/multi_output_bnn_multi_objective_botorch.ipynb)

The example jointly models:

- quality,
- energy consumption,
- cycle time.

The optimization objectives are transformed into a common maximization convention:

\[
(quality,-energy,-cycle).
\]

BoTorch qLogEHVI is then used to select experiments that improve the Pareto set and hypervolume.

### Critical distinction

**Multi-output is not the same as multi-objective.**

- Multi-output BNN: models several random outputs.
- Multi-objective optimization: defines how the decision maker treats trade-offs among objectives.

The repository example uses a shared Bayesian representation but a diagonal residual likelihood. It is therefore not a fully correlated multivariate residual model.

## 8. Hierarchical BNN / Partial Pooling

**Status:** implemented.

Guide: [`notebooks/hierarchical_bnn_partial_pooling_capacity_allocation.md`](./notebooks/hierarchical_bnn_partial_pooling_capacity_allocation.md)

The example models multiple plants that share a common nonlinear response surface but have local deviations.

Shared BNN:

\[
g_w(x)=W_2\tanh(W_1x+b_1)+b_2.
\]

Plant-level random effects:

\[
a_j\sim\mathcal N(\mu_a,\tau_a^2),
\]

\[
b_j\sim\mathcal N(\mu_b,\tau_b^2).
\]

Prediction:

\[
\mu_{ij}=g_w(x_{ij})+a_j+b_j\widetilde{load}_{ij}.
\]

This produces partial pooling: plants are neither completely independent nor forced to be identical. A low-data plant can borrow strength from the group while retaining larger posterior uncertainty.

Typical applications:

- multi-plant capacity models,
- machine-group processing times,
- supplier lead-time and quality models,
- product-family demand models,
- regional demand and network allocation.

### Critical limitation

Not every hierarchical problem needs a BNN. If the common response is simple, a hierarchical linear model, GLMM, or mixed-effects model may be more interpretable and computationally efficient.

## 9. Bayesian RNN / LSTM / Temporal BNN

**Status:** not implemented yet.

Possible uses:

- demand forecasting,
- remaining useful life,
- energy load,
- dynamic production-state prediction.

However, probabilistic state-space models and modern probabilistic sequence models should also be considered as baselines.

## 10. Bayesian GNN

**Status:** advanced extension.

Potential OR links:

- transportation networks,
- supply-chain networks,
- power grids,
- routing surrogates,
- facility/network design.

A Bayesian GNN can represent graph structure and epistemic uncertainty simultaneously, but the inference and calibration burden is substantial.

## 11. Physics-Informed Bayesian Neural Networks

**Status:** advanced extension.

Physical equations or engineering constraints are incorporated into the model or likelihood while Bayesian uncertainty is retained.

Potential applications:

- energy systems,
- heat transfer,
- fluid mechanics,
- structural design,
- process engineering,
- FEA / CFD surrogate optimization.

## 12. Structured / Low-rank / Flow Variational Posteriors

**Status:** advanced research topic.

A mean-field posterior such as `AutoDiagonalNormal` ignores posterior correlations among weights. Richer approximations include:

- low-rank Gaussian,
- full-rank Gaussian,
- normalizing flows,
- structured variational inference.

These can matter when the downstream decision is sensitive to posterior geometry, especially for tail risk, CVaR, or chance constraints.

---

# Non-BNN baselines that should be included

The following are not strict full BNNs, but they are important uncertainty baselines:

- Deep Ensembles,
- MC Dropout,
- SWAG,
- Gaussian Processes,
- conformal prediction,
- quantile regression,
- hierarchical linear / mixed-effects models,
- classical response-surface or Bayesian regression models.

A BNN should be justified by **downstream decision quality**, not only by predictive RMSE.

# What is established mathematics vs modeling choice?

Established theory includes:

- Bayesian posterior and posterior predictive,
- HMC / NUTS,
- Bayesian linear regression,
- hierarchical priors and partial pooling,
- random effects,
- heteroscedastic likelihoods,
- law of total variance,
- Laplace approximation,
- Pareto dominance,
- hypervolume,
- CVaR,
- chance constraints,
- SAA,
- Bayesian optimization and multi-objective BO.

Modeling or inference choices include:

- Normal likelihood,
- `AutoDiagonalNormal`,
- network architecture,
- prior and hyperprior scales,
- random-intercept / random-slope structure,
- number of posterior or scenario samples,
- diagonal residual covariance,
- hypervolume reference point,
- synthetic data-generating functions.

`Mathematically valid` and `empirically valid for a particular factory` are different claims. The latter requires calibration, holdout evaluation, and out-of-sample decision testing.

# Recommended next extensions

The repository now covers the highest-priority practical families. Natural future extensions are:

1. correlated multi-output / multi-task likelihoods,
2. constrained multi-objective Bayesian optimization,
3. hierarchical BNN sensitivity with NUTS or richer VI,
4. temporal BNNs,
5. Bayesian GNNs,
6. physics-informed BNNs,
7. SGMCMC / SGLD / SGHMC.
