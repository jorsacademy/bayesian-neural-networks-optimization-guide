# Hierarchical BNN: Multi-plant Partial Pooling + Stochastic Capacity Allocation

This guide considers an industrial system in which several plants use the same underlying production technology but have different amounts of data and different local performance.

The workflow has two layers:

1. learn the shared nonlinear process and plant-level deviations with a **hierarchical BNN**;
2. pass posterior predictive plant-capacity scenarios to a **stochastic capacity-allocation model**.

```text
multi-plant process data
        ↓
shared Bayesian neural network
        +
plant random effects
        ↓
partial-pooling posterior
        ↓
plant capacity scenarios
        ↓
Pyomo + HiGHS
        ↓
cost / shortage-risk balanced allocation
```

The key modeling comparison is:

- **complete pooling** — ignore plant differences;
- **no pooling** — fit every plant independently;
- **partial pooling** — learn shared structure while allowing plant-specific deviations.

---

## Mathematical model

The shared nonlinear response is learned by a Bayesian neural network:

\[
h_w(x)=\tanh(W_1x+b_1),
\]

\[
g_w(x)=W_2h_w(x)+b_2.
\]

For plant \(j\), define a random intercept and random load slope:

\[
a_j\sim\mathcal N(\mu_a,\tau_a^2),
\]

\[
b_j\sim\mathcal N(\mu_b,\tau_b^2).
\]

The conditional mean for observation \(i\) in plant \(j\) is

\[
\mu_{ij}=g_w(x_{ij})+a_j+b_j\widetilde{load}_{ij}.
\]

Observation model:

\[
y_{ij}\mid x_{ij},j,w,a_j,b_j
\sim
\mathcal N(\mu_{ij},\sigma_j^2).
\]

The group-level parameters \(\mu_a,\tau_a,\mu_b,\tau_b\) are also inferred. Therefore the plant effects are not unrelated fixed coefficients: they come from shared group distributions and produce partial pooling.

A low-data plant is usually regularized more strongly toward the group structure and can retain wider posterior uncertainty.

---

## Runnable example

```python
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.nn import PyroModule, PyroSample
import pyomo.environ as pyo

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
pyro.set_rng_seed(SEED)
torch.set_default_dtype(torch.float32)

# ------------------------------------------------------------
# 1. Synthetic multi-plant data
# ------------------------------------------------------------

plant_names = ["Plant A", "Plant B", "Plant C", "Plant D"]
n_by_plant = [90, 60, 35, 12]

# True synthetic local effects; the model does not know them.
plant_intercept = np.array([-7.0, -1.5, 4.0, 9.0])
plant_load_effect = np.array([-5.0, -1.0, 3.0, 6.0])
plant_noise = np.array([4.2, 4.8, 5.2, 6.2])

rows = []
for j, n_j in enumerate(n_by_plant):
    temperature = np.random.normal(25.0 + 0.8*j, 4.5, n_j)
    load = np.random.uniform(0.40, 0.98, n_j)
    maintenance = np.random.uniform(0.25, 1.00, n_j)

    shared_mean = (
        94.0
        + 22.0 * np.tanh(2.2 * (load - 0.62))
        - 0.45 * np.abs(temperature - 23.0)
        + 9.0 * maintenance
        - 11.0 * (load - 0.82)**2
    )

    local_mean = (
        shared_mean
        + plant_intercept[j]
        + plant_load_effect[j] * (load - 0.65)
    )

    capacity = local_mean + np.random.normal(0.0, plant_noise[j], n_j)

    for i in range(n_j):
        rows.append({
            "plant_id": j,
            "plant": plant_names[j],
            "temperature": temperature[i],
            "load": load[i],
            "maintenance": maintenance[i],
            "capacity": capacity[i],
        })

df = pd.DataFrame(rows)
print(df.groupby("plant").size())

# ------------------------------------------------------------
# 2. Standardization
# ------------------------------------------------------------

features = ["temperature", "load", "maintenance"]
X_raw = df[features].to_numpy(np.float32)
y_raw = df["capacity"].to_numpy(np.float32)
plant_id_np = df["plant_id"].to_numpy(np.int64)

X_mean = X_raw.mean(axis=0, keepdims=True)
X_std = X_raw.std(axis=0, keepdims=True) + 1e-6
y_mean = float(y_raw.mean())
y_std = float(y_raw.std() + 1e-6)

X = torch.tensor((X_raw - X_mean) / X_std)
y = torch.tensor((y_raw - y_mean) / y_std)
plant_id = torch.tensor(plant_id_np, dtype=torch.long)

# ------------------------------------------------------------
# 3. Hierarchical BNN
# ------------------------------------------------------------

class HierarchicalCapacityBNN(PyroModule):
    def __init__(self, in_features, n_plants, hidden=18):
        super().__init__()
        self.n_plants = n_plants

        self.hidden = PyroModule[nn.Linear](in_features, hidden)
        self.hidden.weight = PyroSample(
            dist.Normal(0.0, 0.8).expand([hidden, in_features]).to_event(2)
        )
        self.hidden.bias = PyroSample(
            dist.Normal(0.0, 0.8).expand([hidden]).to_event(1)
        )

        self.out = PyroModule[nn.Linear](hidden, 1)
        self.out.weight = PyroSample(
            dist.Normal(0.0, 0.8).expand([1, hidden]).to_event(2)
        )
        self.out.bias = PyroSample(
            dist.Normal(0.0, 0.8).expand([1]).to_event(1)
        )

    def forward(self, X, plant_id, y=None):
        base = self.out(torch.tanh(self.hidden(X))).squeeze(-1)

        mu_a = pyro.sample("mu_a", dist.Normal(0.0, 0.30))
        tau_a = pyro.sample("tau_a", dist.HalfNormal(0.40))
        mu_b = pyro.sample("mu_b", dist.Normal(0.0, 0.25))
        tau_b = pyro.sample("tau_b", dist.HalfNormal(0.30))

        with pyro.plate("plants", self.n_plants):
            a_plant = pyro.sample("a_plant", dist.Normal(mu_a, tau_a))
            b_plant = pyro.sample("b_plant", dist.Normal(mu_b, tau_b))
            sigma_plant = pyro.sample(
                "sigma_plant", dist.LogNormal(-1.15, 0.30)
            )

        load_z = X[:, 1]
        mean = base + a_plant[plant_id] + b_plant[plant_id] * load_z
        pyro.deterministic("mu", mean)

        with pyro.plate("data", X.shape[0]):
            pyro.sample(
                "obs",
                dist.Normal(mean, sigma_plant[plant_id]),
                obs=y,
            )
        return mean

pyro.clear_param_store()
model = HierarchicalCapacityBNN(X.shape[1], len(plant_names))
guide = AutoDiagonalNormal(model)
svi = SVI(
    model,
    guide,
    pyro.optim.Adam({"lr": 0.012}),
    loss=Trace_ELBO(),
)

for step in range(3200):
    loss = svi.step(X, plant_id, y) / len(y)
    if (step + 1) % 640 == 0:
        print(f"Step {step+1}: ELBO/observation={loss:.4f}")

# ------------------------------------------------------------
# 4. Posterior plant effects
# ------------------------------------------------------------

latent = Predictive(
    model,
    guide=guide,
    num_samples=2500,
    return_sites=(
        "a_plant", "b_plant", "mu_a", "tau_a", "mu_b", "tau_b"
    ),
)(X, plant_id)

a_samples = latent["a_plant"].detach().cpu().numpy() * y_std
b_samples = latent["b_plant"].detach().cpu().numpy() * y_std

summary = []
for j, name in enumerate(plant_names):
    summary.append({
        "plant": name,
        "n": n_by_plant[j],
        "intercept_mean": a_samples[:, j].mean(),
        "intercept_q05": np.quantile(a_samples[:, j], 0.05),
        "intercept_q95": np.quantile(a_samples[:, j], 0.95),
        "load_slope_mean": b_samples[:, j].mean(),
        "load_slope_q05": np.quantile(b_samples[:, j], 0.05),
        "load_slope_q95": np.quantile(b_samples[:, j], 0.95),
    })

print(pd.DataFrame(summary))

# ------------------------------------------------------------
# 5. Future plant-capacity scenarios
# ------------------------------------------------------------

future_raw = np.array(
    [[30.0, 0.86, 0.70]] * len(plant_names), dtype=np.float32
)
future_X = torch.tensor((future_raw - X_mean) / X_std)
future_plant_id = torch.arange(len(plant_names), dtype=torch.long)

future = Predictive(
    model,
    guide=guide,
    num_samples=3000,
    return_sites=("obs", "mu"),
)(future_X, future_plant_id)

capacity_samples = (
    future["obs"].detach().cpu().numpy() * y_std + y_mean
)
capacity_samples = np.clip(capacity_samples, 0.0, None)

print(pd.DataFrame({
    "plant": plant_names,
    "mean_capacity": capacity_samples.mean(axis=0),
    "q05": np.quantile(capacity_samples, 0.05, axis=0),
    "q50": np.quantile(capacity_samples, 0.50, axis=0),
    "q95": np.quantile(capacity_samples, 0.95, axis=0),
}))

# ------------------------------------------------------------
# 6. Stochastic capacity allocation
# ------------------------------------------------------------

rng = np.random.default_rng(SEED)
S = 350
idx = rng.choice(capacity_samples.shape[0], size=S, replace=False)
scenario_capacity = capacity_samples[idx]

unit_reservation_cost = np.array([1.75, 1.95, 2.20, 1.85])
max_plan = np.array([135.0, 135.0, 135.0, 135.0])
demand = 365.0
shortage_penalty = 12.0

m = pyo.ConcreteModel()
m.J = pyo.RangeSet(0, len(plant_names)-1)
m.S = pyo.RangeSet(0, S-1)
m.x = pyo.Var(m.J, domain=pyo.NonNegativeReals)
m.delivered = pyo.Var(m.J, m.S, domain=pyo.NonNegativeReals)
m.shortage = pyo.Var(m.S, domain=pyo.NonNegativeReals)

for j in m.J:
    m.x[j].setub(float(max_plan[j]))

m.plan_link = pyo.Constraint(
    m.J, m.S, rule=lambda M,j,s: M.delivered[j,s] <= M.x[j]
)
m.capacity_link = pyo.Constraint(
    m.J, m.S,
    rule=lambda M,j,s: M.delivered[j,s] <= float(scenario_capacity[s,j])
)
m.shortage_def = pyo.Constraint(
    m.S,
    rule=lambda M,s: M.shortage[s] >= demand - sum(M.delivered[j,s] for j in M.J)
)

m.obj = pyo.Objective(
    expr=sum(unit_reservation_cost[j]*m.x[j] for j in m.J)
    + shortage_penalty*(1/S)*sum(m.shortage[s] for s in m.S),
    sense=pyo.minimize,
)

result = pyo.SolverFactory("appsi_highs").solve(m)
plan = np.array([pyo.value(m.x[j]) for j in m.J])

print(result.solver.termination_condition)
print(pd.DataFrame({
    "plant": plant_names,
    "planned_capacity": plan,
    "unit_cost": unit_reservation_cost,
}))
```

---

## Why partial pooling matters

Plant D has only 12 observations. A completely independent BNN for Plant D would have very weak statistical support. A complete-pooling model would ignore its local behavior. The hierarchical model uses the group distribution to regularize its local parameters while preserving plant-specific uncertainty.

This is the main industrial value of partial pooling: **share information without pretending that every asset or site is identical**.

---

## Decision interpretation

The posterior predictive plant capacities are treated as random scenario parameters:

\[
C_{js}\sim p(C_j\mid x,D).
\]

The allocation model chooses planned capacities \(x_j\). Realized delivery in scenario \(s\) is limited by both the plan and the uncertain plant capacity:

\[
d_{js}\le x_j,
\]

\[
d_{js}\le C_{js}.
\]

Shortage is

\[
u_s\ge D-\sum_jd_{js}.
\]

The objective trades reservation cost against expected shortage penalty:

\[
\min_x
\sum_j c_jx_j
+
\lambda\frac1S\sum_su_s.
\]

The same architecture can be extended to CVaR, service-level constraints, or multi-period capacity planning.

---

## When not to use a hierarchical BNN

A hierarchical BNN is not automatically the best model. Prefer simpler models when the nonlinear representation is unnecessary.

Strong baselines include:

- hierarchical linear regression,
- GLMM / mixed-effects models,
- hierarchical Gaussian processes,
- complete pooling,
- no pooling,
- deep ensembles with group features.

Mean-field variational inference may also under-represent posterior correlations in a hierarchical model. For a high-stakes analysis, compare against NUTS or a richer variational posterior when computationally feasible.

---

## Industrial applications

The same pattern applies to:

- multiple factories,
- machine fleets,
- supplier groups,
- product families,
- regional demand,
- hospital or service locations,
- distribution centers.

The core pattern is:

\[
\boxed{
\text{shared nonlinear structure}
+
\text{group-level partial pooling}
+
\text{posterior scenarios}
+
\text{optimization}
}
\]
