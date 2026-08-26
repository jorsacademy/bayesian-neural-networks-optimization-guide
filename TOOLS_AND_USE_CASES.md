# Software Stack and Industrial Engineering / OR Use Cases

This guide maps probabilistic-modeling libraries to optimization roles and industrial-engineering problem classes.

## 1. Recommended architecture

A useful default architecture is

```text
historical / sensor / simulation data
        ↓
probabilistic predictive model
        ↓
posterior predictive distribution
        ↓
calibration / scenario generation
        ↓
risk model
        ↓
mathematical or Bayesian optimization
        ↓
operational decision
```

A common 2026 research stack is

\[
\boxed{
\text{PyTorch}
+
\text{Pyro / GPyTorch}
+
\text{BoTorch / Ax}
+
\text{Pyomo / Gurobi}
}
\]

with NumPyro or PyMC added when posterior fidelity, hierarchical structure, or interpretable Bayesian modeling is the priority.

---

## 2. Library roles

### PyTorch

Role: deterministic neural-network and autodiff foundation.

Use it for:

- neural feature extractors,
- deterministic baselines,
- deep ensembles,
- neural surrogates,
- differentiable decision models.

Official site: https://pytorch.org/

### Pyro

Role: probabilistic programming on PyTorch.

Strong for:

- full BNNs,
- variational inference,
- autoguides,
- HMC / NUTS,
- hierarchical BNNs,
- custom likelihoods,
- posterior predictive sampling.

Official site: https://pyro.ai/

### NumPyro

Role: JAX-based Bayesian inference.

Strong for:

- NUTS / HMC,
- high-performance posterior sampling,
- SVI,
- inference-method benchmarking,
- small/medium BNN posterior studies.

Official site: https://num.pyro.ai/

### PyMC

Role: structured Bayesian modeling.

Particularly useful for:

- hierarchical demand forecasting,
- reliability models,
- failure-rate models,
- maintenance decisions,
- interpretable Bayesian regression,
- supplier/process uncertainty.

Official site: https://www.pymc.io/

### GPyTorch

Role: scalable Gaussian-process modeling.

Use it for:

- exact GPs,
- variational GPs,
- deep kernel learning,
- deep GPs,
- small-data expensive black-box modeling.

Official site: https://gpytorch.ai/

### BoTorch

Role: Bayesian optimization and acquisition-function research.

Strong for:

- qLogEI / qLogNEI,
- constrained BO,
- multi-objective BO,
- qLogEHVI / qLogNEHVI,
- multi-fidelity optimization,
- custom probabilistic surrogate models.

Official site: https://botorch.org/

BoTorch is often the correct low-level research framework when the acquisition function or surrogate is part of the methodological contribution.

### Ax

Role: adaptive experimentation orchestration.

Useful for:

- production experiment management,
- trial lifecycle management,
- automated experiment loops,
- BoTorch-backed optimization without manually managing every acquisition detail,
- A/B tests and engineering experimentation.

Official site: https://ax.dev/

### laplace-torch

Role: post-hoc uncertainty for trained PyTorch models.

Strong when:

- a deterministic model already exists,
- full BNN retraining is undesirable,
- a last-layer or subnetwork uncertainty approximation is sufficient.

Project: https://github.com/aleximmer/Laplace

### Pyomo

Role: algebraic mathematical-programming model layer.

Use it for:

- LP / MILP / NLP,
- stochastic programs,
- scenario models,
- production planning,
- inventory,
- scheduling,
- network optimization.

Official site: https://www.pyomo.org/

### Gurobi

Role: high-performance mathematical optimization solver.

Typical use:

- MILP,
- LP/QP/QCP,
- mixed-integer nonlinear/general-constraint models,
- scheduling,
- facility location,
- network design,
- production planning.

Official site: https://www.gurobi.com/

Gurobi is not a Bayesian framework. It is usually the decision solver that receives posterior-derived scenarios, quantiles, risk coefficients, or learned surrogate constraints.

---

## 3. BNN + mathematical optimization integration patterns

### Pattern A — posterior scenarios into stochastic programming

```text
Data
 ↓
BNN
 ↓
Posterior predictive samples
 ↓
Scenario matrix
 ↓
Pyomo / Gurobi
 ↓
Stochastic decision
```

Examples:

- demand → inventory / production,
- processing times → scheduling,
- lead times → supply-chain planning,
- machine capacity → multi-plant allocation,
- failures → maintenance planning.

### Pattern B — probabilistic surrogate into Bayesian optimization

```text
Expensive experiment / simulator
        ↓
BNN or GP surrogate
        ↓
Posterior
        ↓
Acquisition function
        ↓
BoTorch / Ax
        ↓
Next experiment
```

Examples:

- process parameter tuning,
- simulation optimization,
- digital-twin calibration,
- energy settings,
- quality optimization,
- design optimization.

### Pattern C — differentiable uncertainty-aware objective

For continuous decisions:

\[
\nabla_x E_{w\mid D}[f(x,w)]
\approx
\frac{1}{S}\sum_s\nabla_x f(x,w^{(s)}).
\]

This can support differentiable optimization when the full decision pipeline is smooth.

### Pattern D — combinatorial decisions

For integer or discrete decisions, use posterior scenarios rather than direct neural gradients:

```text
BNN posterior
   ↓
scenarios
   ↓
MILP / MINLP
   ↓
schedule / route / facility / allocation decision
```

---

## 4. Problem-to-stack recommendations

| Industrial / OR problem | Recommended first stack |
|---|---|
| Expensive discrete-event simulation | GPyTorch + BoTorch |
| High-dimensional nonlinear simulation | Pyro BNN + BoTorch |
| Process parameter tuning | Ax + BoTorch |
| Sequential DOE | Ax + BoTorch |
| Multi-objective process optimization | BoTorch qLogEHVI/qLogNEHVI + probabilistic surrogate |
| Multi-fidelity simulation | GPyTorch + BoTorch |
| Demand uncertainty + inventory | PyMC/NumPyro/Pyro + Pyomo/Gurobi |
| Production planning under uncertain demand | posterior scenarios + Pyomo/Gurobi |
| Supply-chain stochastic optimization | structured Bayesian model + Pyomo/Gurobi |
| Predictive maintenance | Pyro/NumPyro + maintenance optimization model |
| Reliability optimization | PyMC/NumPyro + solver |
| Learned processing times + scheduling | PyTorch/Pyro + Gurobi/Pyomo |
| Simulation-based scheduling | BoTorch + simulator |
| Chance-constrained planning | calibrated posterior + Pyomo/Gurobi |
| CVaR planning | posterior scenarios + Pyomo/Gurobi |
| Digital-twin optimization | PyTorch/Pyro + BoTorch/Ax |
| Existing neural surrogate requiring UQ | Laplace approximation or Bayesian last layer |
| Multiple factories / machines | hierarchical model / hierarchical BNN |

---

## 5. GP vs BNN selection

### Start with a GP when

- evaluations are expensive,
- the dataset is small,
- the input dimension is moderate,
- sample efficiency is more important than representation learning,
- Bayesian optimization is the main task.

### Consider a BNN when

- data are larger,
- the response is highly nonlinear,
- the input space is high-dimensional,
- neural representations are useful,
- the probabilistic neural model can be reused beyond BO.

### Practical rule

For a simulation with only a few hundred expensive evaluations, a GP is usually the first baseline.

For a large nonlinear sensor/process dataset, a BNN or ensemble may be more natural.

---

## 6. BNN vs Deep Ensemble

A deep ensemble is not a strict BNN, but it should often be treated as a mandatory baseline.

Why:

- simple implementation,
- parallel training,
- strong empirical uncertainty performance,
- fewer probabilistic-programming assumptions.

A full BNN should demonstrate added value in:

- calibration,
- OOD behavior,
- posterior decision quality,
- sample efficiency,
- operational cost or risk.

If it does not, the additional complexity may not be justified.

---

## 7. Bayesian last layer vs Laplace vs full BNN

| Method | Bayesian scope | Cost | Typical use |
|---|---:|---:|---|
| Full BNN | many/all weights | high | research-grade uncertainty, complex posterior scenarios |
| Bayesian last layer | output layer | low | sequential optimization, fast online updates |
| Last-layer Laplace | local output-layer approximation | low | add UQ to existing trained model |
| Full-network Laplace | local approximation to many weights | medium/high | post-hoc UQ when curvature is tractable |

---

## 8. Multi-output and multi-objective systems

Industrial processes rarely have one output.

Example:

\[
x
\rightarrow
(quality, energy, cycle\ time, defect\ probability).
\]

Possible modeling choices:

- independent probabilistic models,
- shared multi-output BNN,
- multi-task GP,
- correlated likelihood model.

Possible decision choices:

- weighted sum,
- epsilon-constraint,
- Pareto optimization,
- hypervolume-based BO,
- constrained MOBO.

The probabilistic model and the preference model should remain conceptually separate.

---

## 9. Hierarchical industrial systems

Hierarchical models are natural when data are grouped by:

- plant,
- machine,
- supplier,
- product family,
- region,
- customer segment.

Partial pooling is especially useful when some groups have rich data and others have sparse data.

Do not use a hierarchical BNN merely because groups exist. If the common response is approximately linear or has a simple parametric form, a hierarchical regression / mixed-effects model is usually preferable.

---

## 10. Calibration and decision quality

The recommended evaluation ladder is

```text
point accuracy
    ↓
probabilistic calibration
    ↓
scenario quality
    ↓
expected decision cost
    ↓
tail risk / CVaR
    ↓
constraint violation
    ↓
operational robustness under shift
```

A BNN with slightly worse RMSE may still be the better decision model if its uncertainty is better calibrated and produces lower out-of-sample operational cost.

---

## 11. Distribution shift, conformal prediction, and DRO

Posterior uncertainty is not a universal defense against distribution shift.

Useful combinations include:

\[
\text{BNN prediction}
+
\text{conformal calibration}
+
\text{robust / stochastic optimization}.
\]

For high-stakes applications, consider:

- temporal holdouts,
- stress regimes,
- OOD tests,
- conformal intervals,
- ambiguity sets,
- distributionally robust optimization.

---

## 12. Value of information

The next-best experiment is itself a decision variable.

Bayesian optimization and active learning can prioritize experiments where information has high expected value.

This is important for:

- destructive tests,
- laboratory experiments,
- physical manufacturing trials,
- expensive simulation,
- digital twins,
- engineering design.

---

## 13. Practical recommendation

Use complexity only when it changes the decision.

A defensible research progression is:

```text
deterministic baseline
      ↓
probabilistic classical model / GP
      ↓
deep ensemble
      ↓
Bayesian last layer / Laplace
      ↓
full BNN if justified
      ↓
calibrated posterior scenarios
      ↓
OR decision model
      ↓
out-of-sample decision evaluation
```
