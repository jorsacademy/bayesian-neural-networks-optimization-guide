# Laplace Approximation for Neural Networks: Foundations and Limits

This note supports `notebooks/laplace_approximation_pytorch_stochastic_capacity.ipynb`.

## 1. Core idea

Suppose a neural network has parameters \(\theta\) and a MAP estimate \(\theta_{MAP}\). Laplace approximation uses a second-order local expansion of the negative log posterior around the MAP point:

\[
p(\theta\mid D)
\approx
\mathcal N(\theta_{MAP},H^{-1}),
\]

where \(H\) is the local curvature / Hessian approximation of the negative log posterior.

This converts a trained deterministic or MAP model into a local Gaussian approximation to parameter uncertainty.

## 2. Why this is useful in industry

A common practical situation is:

```text
A deterministic PyTorch model already exists
        ↓
It is accurate enough for point prediction
        ↓
A new planning problem needs uncertainty
        ↓
Retraining a full BNN is expensive or operationally undesirable
        ↓
Add a post-hoc Laplace approximation
```

This can be attractive for:

- production capacity models,
- quality prediction,
- processing-time prediction,
- predictive maintenance,
- simulation surrogates,
- digital-twin components.

## 3. Full-network, subnetwork, and last-layer Laplace

A Laplace approximation can be applied to different parameter subsets.

### Full-network Laplace

All network parameters are assigned the local Gaussian approximation.

Pros:

- represents more of the model's parameter uncertainty.

Cons:

- much more expensive,
- curvature storage and computation can dominate.

### Subnetwork Laplace

Only a selected subset of parameters is approximated.

Useful when certain layers or components are more relevant to uncertainty.

### Last-layer Laplace

Only the final layer is treated probabilistically around its MAP solution.

Pros:

- cheap,
- easy to add to an existing model,
- often practical in online systems.

Cons:

- uncertainty in the feature extractor is not represented.

The repository uses last-layer Laplace for the applied example.

## 4. Laplace is not a full BNN posterior

The approximation is local and Gaussian.

It can miss:

- multimodality,
- strong skewness,
- non-Gaussian tails,
- distant posterior modes,
- uncertainty caused by very different feature representations.

Therefore:

> `Laplace approximation gives a posterior approximation` does not mean `the full neural posterior is exactly Gaussian`.

## 5. Difference from a Bayesian last layer

The repository contains both methods because they answer slightly different practical questions.

### Bayesian last layer

A fixed neural feature representation is followed by a Bayesian linear model. Under conjugate Gaussian assumptions, the last-layer posterior can be computed analytically.

### Last-layer Laplace

A deterministic/MAP network is trained first, then local curvature around the fitted last layer is used to form a Gaussian posterior approximation.

The two can look similar computationally, but their derivations and assumptions differ.

## 6. Predictive uncertainty

For regression, a Laplace approximation can provide a predictive distribution for the latent function:

\[
f(x)\mid D.
\]

If observation noise is Gaussian with variance \(\sigma_\epsilon^2\), a simple predictive decomposition is

\[
Var(Y\mid x,D)
\approx
Var(f(x)\mid D)+\sigma_\epsilon^2.
\]

The first term represents parameter/model uncertainty under the local approximation; the second represents observation noise.

## 7. Prior precision and marginal likelihood

The strength of the Gaussian parameter prior influences the posterior covariance. A practical approach is to tune prior precision using marginal-likelihood criteria.

`laplace-torch` supports post-hoc prior-precision optimization for common workflows.

## 8. Decision integration

Once predictive samples are available, the OR integration is the same as for a BNN:

```text
Laplace predictive distribution
        ↓
Scenario generation
        ↓
SAA / CVaR / chance constraint
        ↓
Pyomo / Gurobi
        ↓
Decision
```

The applied notebook compares uncertainty-aware capacity planning with a mean-only deterministic baseline.

## 9. What must be validated

For a real application, compare at least:

- deterministic neural network,
- last-layer Laplace,
- Bayesian last layer,
- deep ensemble,
- GP where appropriate,
- full BNN where computationally feasible.

Evaluate:

- predictive coverage,
- NLL / CRPS,
- OOD uncertainty,
- expected decision cost,
- tail loss / CVaR,
- constraint violations,
- update time.

## 10. Software reference

The applied example uses `laplace-torch`:

- https://github.com/aleximmer/Laplace

The package supports several curvature structures and parameter subsets. API details can evolve, so production code should be checked against the installed version.