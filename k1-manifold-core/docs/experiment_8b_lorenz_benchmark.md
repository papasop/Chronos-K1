# Experiment 8-B: Lorenz Attractor Benchmark (Colab Edition)

This is an **architecture-prior** experiment, not a validation of Chronos
physics theory.

## Positioning

- Chronos theory motivates an indefinite latent metric structure.
- Exp8-B tests whether that structure helps as an architectural inductive bias
  on nonlinear chaotic dynamics (Lorenz).
- Lorenz has no physical Lorentzian light-cone interpretation in this context.

## Primary Metric and Decision Rule

- Primary: VPT (Valid Prediction Time, in Lyapunov times).
- Primary hypothesis family: Lorentz vs each control
  (`EuclideanNormalized`, `RandomMetric_1..3`) with one-sided Wilcoxon and
  Holm correction.
- Headline pass requires beating Euclidean **and** all random controls.

## Secondary Metrics (Exploratory)

- Attractor reconstruction (normalized 1D Wasserstein over marginals)
- Short-horizon rollout MSE

These are reported as exploratory and are not headline decision criteria.

## Run

In Colab/script config block:
- `RUN_MODE = "SMOKE"` for pipeline check
- `RUN_MODE = "FULL"` for benchmark run

Script:
- `benchmarks/experiment_8b_lorenz_benchmark_colab.py`

## Outputs

- `exp8b_results.csv`
- `exp8b_summary_table.csv`
- `config.json`
