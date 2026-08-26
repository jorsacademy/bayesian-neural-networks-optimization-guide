# Multi-output BNNs and Multi-objective Bayesian Optimization

This note supports `notebooks/multi_output_bnn_multi_objective_botorch.ipynb`.

## 1. Multi-output modeling and multi-objective optimization are different

A multi-output probabilistic model predicts several random outputs from the same input:

\[
Y(x)=\begin{bmatrix}Y_1(x)&Y_2(x)&\cdots&Y_m(x)\end{bmatrix}^\top.
\]

Multi-objective optimization defines a decision problem with several objectives:

\[
\max_x (f_1(x),\ldots,f_m(x)).
\]

A model can be multi-output without the decision problem being multi-objective. The objectives, directions, constraints, and preferences are part of the OR layer.

## 2. Shared multi-output BNN used in the repository

The example uses a shared Bayesian hidden representation:

\[
h_w(x)=\tanh(W_1x+b_1),
\]

and a three-output Bayesian head:

\[
\mu_w(x)=W_2h_w(x)+b_2\in\mathbb R^3.
\]

The three physical outputs are:

- quality,
- energy consumption,
- cycle time.

The likelihood uses output-specific standard deviations:

\[
Y\mid x,w
\sim
\mathcal N\left(\mu_w(x),\operatorname{diag}(\sigma_1^2,\sigma_2^2,\sigma_3^2)\right).
\]

The outputs share Bayesian network parameters, so posterior function samples are statistically linked through the common representation. However, the residual covariance is diagonal; this is not a full multivariate residual model.

## 3. Objective directions

BoTorch conventionally handles maximization objectives. If the physical goals are

- maximize quality,
- minimize energy,
- minimize cycle time,

the objective vector is transformed to

\[
(quality,-energy,-cycle).
\]

This sign transformation does not change the underlying engineering problem; it only creates a common mathematical convention.

## 4. Pareto dominance

A feasible point \(a\) dominates \(b\) if it is at least as good in every objective and strictly better in at least one.

A point is Pareto optimal if no other feasible point dominates it.

Therefore, a multi-objective problem generally produces a **set of trade-off solutions** rather than a single optimum.

## 5. Hypervolume

Given a reference point \(r\), the hypervolume of a Pareto set measures the volume of objective space dominated by the set and bounded by the reference point.

A larger hypervolume generally represents a better trade-off set under the chosen reference point.

The reference point is a modeling choice and should be chosen carefully. Poor reference points can distort acquisition behavior.

## 6. Expected Hypervolume Improvement

EHVI asks how much an additional observation is expected to improve the current hypervolume.

For batch and Monte Carlo Bayesian optimization, BoTorch provides variants including qEHVI and numerically stable log-EHVI formulations.

The repository uses `qLogExpectedHypervolumeImprovement` (qLogEHVI).

The conceptual loop is:

```text
Observed process experiments
        ↓
Multi-output probabilistic surrogate
        ↓
Posterior samples in objective space
        ↓
Current Pareto set / hypervolume
        ↓
qLogEHVI
        ↓
Optimize acquisition
        ↓
Run the next physical/simulation experiment
```

## 7. Why this is useful in industrial engineering

Many process decisions are naturally multi-objective:

- quality vs energy,
- throughput vs defect rate,
- cost vs service level,
- cycle time vs reliability,
- emissions vs productivity,
- maintenance cost vs availability.

A single weighted-sum objective can hide important trade-offs. Pareto-based BO allows a decision maker to discover the trade-off frontier before committing to preferences.

## 8. Correlated outputs

The repository's shared BNN creates dependence through shared weights, but the residual likelihood is diagonal.

A richer model could use:

- multivariate Gaussian residual covariance,
- low-rank covariance structures,
- multi-task GPs,
- latent-factor probabilistic models,
- copula or structured likelihoods.

This matters when objective residual correlations are operationally important.

## 9. Constrained multi-objective BO

Real process optimization often adds feasibility constraints such as:

- defect probability below a threshold,
- temperature below a safety limit,
- power below a capacity limit,
- service-level probability above a target.

The next extension after unconstrained MOBO is therefore a constrained multi-objective acquisition strategy.

## 10. BoTorch reference

BoTorch multi-objective documentation:

- https://botorch.org/docs/multi_objective

General model interface:

- https://botorch.org/docs/models

The key point is that the acquisition function consumes a probabilistic posterior; it does not conceptually require the surrogate to be a GP if the custom model satisfies the required posterior interface.

## 11. Validation checklist

For an industrial multi-output / multi-objective study, evaluate:

- per-output predictive accuracy,
- per-output calibration,
- joint or marginal coverage,
- residual correlation misspecification,
- Pareto-front quality,
- hypervolume,
- reference-point sensitivity,
- sample efficiency / best-so-far hypervolume,
- constraint violations if constrained,
- alternative surrogate baselines such as multi-task GP and independent GPs.
