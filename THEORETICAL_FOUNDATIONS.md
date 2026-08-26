# Theoretical Foundations

This file separates standard probability / Bayesian / operations-research theory from practical modeling choices used throughout the repository.

## 1. Bayesian neural networks and posterior prediction

A BNN treats neural-network weights as random variables:

\[
w\sim p(w),
\qquad
p(w\mid D)\propto p(D\mid w)p(w).
\]

For a new input \(x^*\),

\[
p(y^*\mid x^*,D)
=
\int p(y^*\mid x^*,w)p(w\mid D)\,dw.
\]

In practice, this integral is approximated by posterior samples or an approximate posterior.

Pyro references:

- https://docs.pyro.ai/en/stable/nn.html
- https://docs.pyro.ai/en/stable/inference.html

## 2. Variational inference

Variational inference introduces a tractable family \(q_\phi(w)\) and minimizes

\[
KL(q_\phi(w)\Vert p(w\mid D)).
\]

Equivalently, one maximizes the evidence lower bound (ELBO):

\[
\operatorname{ELBO}
=
E_{q_\phi(w)}[\log p(D,w)-\log q_\phi(w)].
\]

Mean-field guides such as `AutoDiagonalNormal` are computationally attractive but may under-represent posterior correlation, multimodality, and tail mass.

This is an approximation method, not a theorem that the posterior is Gaussian or independent.

## 3. HMC and NUTS

Hamiltonian Monte Carlo augments the posterior with momentum variables and uses Hamiltonian dynamics to propose distant moves with high acceptance probability.

NUTS adapts the trajectory length automatically and is commonly used as a high-quality MCMC method for differentiable Bayesian models.

Important diagnostics include:

- divergences,
- effective sample size,
- multiple chains,
- \(\hat R\),
- sufficient warmup.

NUTS is not automatically a “ground-truth posterior.” It still uses finite samples and depends on convergence and model specification.

NumPyro reference:

- https://num.pyro.ai/

## 4. Heteroscedastic uncertainty

A heteroscedastic regression model allows observation variance to depend on the input:

\[
Y\mid x,w\sim\mathcal N(\mu_w(x),\sigma_w^2(x)).
\]

This is a standard probabilistic regression construction. In Bayesian deep learning, input-dependent aleatoric uncertainty and epistemic uncertainty are often modeled together.

References:

- Kendall & Gal (2017), *What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?* https://arxiv.org/abs/1703.04977
- Depeweg et al. (2018), uncertainty decomposition and risk-sensitive decision making: https://arxiv.org/abs/1710.07283

## 5. Aleatoric / epistemic decomposition

The law of total variance gives

\[
Var(Y\mid x,D)
=
E_{w\mid D}[Var(Y\mid x,w)]
+
Var_{w\mid D}(E[Y\mid x,w]).
\]

For the Gaussian heteroscedastic model,

\[
E[Y\mid x,w]=\mu_w(x),
\qquad
Var(Y\mid x,w)=\sigma_w^2(x),
\]

hence

\[
Var(Y\mid x,D)
=
E_w[\sigma_w^2(x)]
+
Var_w[\mu_w(x)].
\]

The repository estimates these terms by posterior Monte Carlo.

## 6. Bayesian last layer / neural linear model

For a fixed neural feature extractor \(\phi(x)\), a Bayesian linear output layer can be written as

\[
f(x)=\tilde\phi(x)^\top\beta,
\qquad
\beta\sim\mathcal N(0,\alpha^{-1}I).
\]

Under Gaussian observation noise,

\[
y\mid\beta\sim\mathcal N(\Phi\beta,\sigma_\epsilon^2I),
\]

the posterior is Gaussian. With \(\tau=1/\sigma_\epsilon^2\):

\[
\Sigma=(\alpha I+\tau\Phi^\top\Phi)^{-1},
\]

\[
m=\tau\Sigma\Phi^\top y,
\]

\[
\beta\mid D\sim\mathcal N(m,\Sigma).
\]

The exact Gaussian posterior applies to the **fixed-feature linear layer**, not to the full neural network.

References:

- https://arxiv.org/abs/2302.10975
- https://arxiv.org/abs/1912.06760

## 7. Hierarchical priors and partial pooling

For group-specific effects,

\[
a_j\sim\mathcal N(\mu_a,\tau_a^2).
\]

The group effects share hyperparameters and therefore borrow information from one another. This creates partial pooling.

Three useful comparisons are:

- complete pooling: ignore group differences,
- no pooling: estimate each group independently,
- partial pooling: combine shared and local information.

Hierarchical BNNs simply combine this standard hierarchical structure with a nonlinear Bayesian neural component.

## 8. Sample Average Approximation

For a stochastic program

\[
\min_x E[Q(x,\xi)],
\]

SAA replaces the expectation with a finite sample average:

\[
\min_x \frac1S\sum_{s=1}^{S}Q(x,\xi_s).
\]

In this repository, \(\xi_s\) can be generated from a posterior predictive distribution rather than from a manually assumed parametric distribution.

## 9. CVaR

For loss random variable \(L\), the Rockafellar–Uryasev representation is

\[
CVaR_\alpha(L)
=
\min_\eta
\left[
\eta+
\frac{1}{1-\alpha}E[(L-\eta)^+]
\right].
\]

This form is convenient because scenario approximations can often be written as linear or convex optimization models.

Reference:

- Rockafellar & Uryasev (2000), *Optimization of Conditional Value-at-Risk*: https://doi.org/10.21314/JOR.2000.038

## 10. Chance constraints

A chance constraint has the form

\[
P(g(x,\xi)\le0)\ge1-\alpha.
\]

A finite empirical scenario constraint is not automatically a formal probability guarantee under the true distribution.

Formal scenario-approach guarantees require assumptions and sample-complexity results; otherwise independent out-of-sample validation is necessary.

References:

- Stanford EE364A notes: https://stanford.edu/class/ee364a/lectures/chance_constr.pdf
- Campi & Garatti (2008): https://epubs.siam.org/doi/10.1137/07069821X

## 11. Bayesian optimization

Bayesian optimization combines:

1. a probabilistic surrogate,
2. an acquisition function,
3. numerical optimization of the acquisition,
4. sequential evaluation of the expensive objective.

BoTorch does not require every Monte Carlo acquisition function to use a Gaussian process. A custom model can be used if it exposes an appropriate posterior and supports reparameterized sampling when gradient-based acquisition optimization is required.

Reference:

- https://botorch.org/docs/models

## 12. Multi-objective optimization

For multiple objectives,

\[
\max_x (f_1(x),\ldots,f_m(x)),
\]

there is generally no single optimum. Pareto dominance defines the nondominated set.

Hypervolume measures the volume of objective space dominated by the Pareto set relative to a reference point. Hypervolume-improvement acquisitions such as qLogEHVI select new experiments expected to expand that dominated region.

BoTorch reference:

- https://botorch.org/docs/multi_objective

## 13. What is not standard theory?

The following are engineering/modeling choices and must be validated:

- Normal vs Student-t likelihood,
- network width and depth,
- prior scale,
- `AutoDiagonalNormal`,
- number of SVI steps,
- number of posterior samples,
- number of stochastic-programming scenarios,
- hypervolume reference point,
- diagonal residual covariance,
- synthetic data functions.

## 14. Minimum industrial validation checklist

A credible uncertainty-aware optimization study should evaluate:

1. train/validation/test or temporal holdout,
2. predictive RMSE/MAE,
3. NLL, CRPS, or another proper scoring rule,
4. predictive interval coverage,
5. calibration,
6. deterministic NN / ensemble / GP / classical baseline,
7. posterior-sample and scenario-count sensitivity,
8. out-of-sample constraint violation,
9. expected operational cost,
10. CVaR or tail loss,
11. OOD / distribution shift,
12. inference sensitivity,
13. Bayesian-optimization regret / best-so-far when applicable,
14. computation and update time.

The mathematics in this repository is standard. Whether a specific model is useful in a specific industrial system must still be demonstrated empirically.