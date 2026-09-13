# Causal TDNN vs BNN: Forecast Accuracy vs Decision Quality

This workflow turns a standalone TDNN implementation into a decision-focused OR benchmark.

## Research question

Does the model with the best point forecast also produce the best operational decision?

The benchmark compares the same 28-lag demand history across:

1. naive last-value forecast,
2. ridge autoregression,
3. strictly causal dilated TDNN,
4. Bayesian neural network posterior mean,
5. BNN posterior predictive scenarios + Sample Average Approximation.

```text
historical demand
      ↓
chronological train / validation / test
      ↓
Naive / Ridge AR / Causal TDNN / BNN
      ↓
point forecasts + BNN predictive scenarios
      ↓
MAE / RMSE / WAPE
      +
production cost / regret / service level
```

## Why the TDNN is causal

The original symmetric `Conv1d` padding can let an output at time `t` depend on later sequence positions. `CausalConv1d` instead applies left padding only:

\[
\text{left padding}=(k-1)d,
\]

where `k` is kernel size and `d` is dilation. A gradient invariant in `assert_causal_layer()` verifies that an output at time `t` has zero gradient with respect to inputs after `t`.

The TDNN uses dilations 1, 2, and 4, so its receptive field expands without future leakage.

## Leakage-safe evaluation

`data.py` uses a chronological 60/20/20 train/validation/test split. Scaling statistics are fitted only on the training history. The test block is untouched until final evaluation.

The deep temporal model is not evaluated in isolation: naive and ridge autoregressive baselines are mandatory comparisons.

## Decision layer

For production quantity `q` and realized demand `D`, the single-period cost is

\[
C(q,D)=cq+h(q-D)^+ + p(D-q)^+.
\]

Deterministic models use their point forecast as the production decision. The BNN generates posterior predictive demand samples and the uncertainty-aware decision minimizes the sampled expected cost.

For this convex single-period problem, the SAA optimum is the empirical critical fractile

\[
F(q^*)=\frac{p-c}{h+p}.
\]

`optimization.py` uses this empirical quantile for repeated evaluation and independently verifies one case with an explicit Pyomo + HiGHS linear program.

## Outputs

The benchmark reports two separate scoreboards:

- prediction quality: MAE, RMSE, WAPE, and BNN 90% predictive coverage;
- decision quality: average realized cost, perfect-information regret, and service level.

This separation is deliberate: lower forecast error does not imply lower operational cost when shortage and holding costs are asymmetric.

## Run

From the repository root:

```bash
pip install -r requirements.txt
python notebooks/causal_tdnn_bnn_decision_quality/benchmark.py
```

The example is synthetic and pedagogical. A real implementation should add rolling-origin retraining, calibration diagnostics such as NLL/CRPS, scenario-count sensitivity, OOD/nonstationarity tests, and actual capacity, lead-time, lot-sizing, and service-level constraints.
