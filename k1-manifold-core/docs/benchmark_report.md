# Benchmark Report

This report records research benchmarks that involve model training,
randomness, statistics, or optional ML dependencies. They are not unit tests.

## OOD Extrapolation: Light-Cone Classification

**Task.** Classify synthetic event differences as timelike or spacelike using
the Lorentz interval

```text
dt^2 - ||dx||^2
```

The benchmark trains on samples from `box=2` and tests on larger boxes.

**Models.**

- `lorentz`: explicit Lorentzian score `scale * (dt^2 - ||dx||^2) + bias`.
- `euclid_mahalanobis`: Euclidean radial quadratic baseline with positive
  weights.
- `euclid_mlp`: unstructured MLP on raw event coordinates.

**Command.**

```bash
cd k1-manifold-core
python benchmarks/ood_extrapolation.py
```

**Artifacts.**

- `results/ood_extrapolation.json`
- `results/ood_extrapolation_auc.png`

**Current results.**

| Test box | Lorentz AUC | Euclid Mahalanobis AUC | Euclid MLP AUC | Lorentz - MLP gap | Wilcoxon p |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2 | 1.0000 +/- 0.0000 | 0.7256 +/- 0.0134 | 0.9997 +/- 0.0001 | +0.0003 | 0.0020 |
| 4 | 1.0000 +/- 0.0000 | 0.7278 +/- 0.0094 | 0.9997 +/- 0.0002 | +0.0003 | 0.0020 |
| 8 | 1.0000 +/- 0.0000 | 0.7213 +/- 0.0077 | 0.9996 +/- 0.0003 | +0.0004 | 0.0020 |
| 12 | 1.0000 +/- 0.0000 | 0.7217 +/- 0.0131 | 0.9995 +/- 0.0003 | +0.0005 | 0.0020 |

**Interpretation boundary.** This benchmark supports a narrow claim: an
explicit Lorentzian inductive bias solves this synthetic light-cone
classification task exactly and maintains a small, statistically consistent
OOD edge over a strong MLP baseline, while a radial Euclidean quadratic
baseline does not capture the task geometry. It is not evidence for broad
world-model superiority.

## Experiment 5: Causal Stress Test

**Task.** Train a Euclidean latent predictor (ELP) and Chronos latent predictor
(CLP) on synthetic Lorentzian oscillator trajectories, then evaluate rollout
MSE, decoded causal violation, interval drift, and latent `K` drift under OOD
box extrapolation.

These models are JEPA-style in the narrow sense that they predict future
embeddings. They are not implementations of Meta/LeCun JEPA.

**Command.**

```bash
cd k1-manifold-core
python benchmarks/experiment_5_causal_stress_test.py
```

Use `--smoke` for a tiny CPU-friendly run and `--full` for the larger
configuration:

```bash
python benchmarks/experiment_5_causal_stress_test.py --smoke
python benchmarks/experiment_5_causal_stress_test.py --full
```

**Artifacts.**

- `results/experiment_5_ablation_stress_summary.csv`
- `results/experiment_5_ablation_stress_raw.json`
- `results/experiment_5_violation_vs_box.png`
- `results/experiment_5_mse_vs_box.png`
- `results/experiment_5_violation_by_step.png`
- `results/experiment_5_K_drift_by_step.png`

**Representative Colab result.** The run below used the default benchmark
configuration on CUDA (`n_seeds=5`, `n_train=1000`, `n_test=256`,
`epochs=100`). Chronos latent predictor keeps final rollout MSE essentially
unchanged while reducing decoded causal-violation rates by roughly an order of
magnitude across OOD boxes. The Wilcoxon p-values are not stable enough at
`n=5` to claim statistical significance; this should be read as mechanistic
benchmark evidence, not a final statistical result.

| Lambda | Test box | Euclid violation | Chronos violation | MSE improvement vs Euclid | p(MSE) | p(violation) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0 | 2 | 0.3562 | 0.0188 | +0.0910% | 0.1875 | 0.3125 |
| 0.0 | 8 | 0.3806 | 0.0276 | +0.0023% | 0.8125 | 0.4375 |
| 0.0 | 32 | 0.3764 | 0.0257 | +0.0020% | 0.4375 | 0.4375 |
| 0.1 | 2 | 0.3562 | 0.0109 | +0.0924% | 0.0625 | 0.6250 |
| 0.1 | 8 | 0.3806 | 0.0216 | +0.0022% | 1.0000 | 0.6250 |
| 0.1 | 32 | 0.3764 | 0.0181 | +0.0015% | 0.8125 | 0.8125 |
| 0.5 | 2 | 0.3562 | 0.0168 | +0.0766% | 0.0625 | 0.8125 |
| 0.5 | 8 | 0.3806 | 0.0261 | -0.0011% | 1.0000 | 0.8125 |
| 0.5 | 32 | 0.3764 | 0.0217 | +0.0015% | 0.8125 | 0.8125 |

**Interpretation boundary.** This benchmark should not be cited as evidence
that Chronos latent predictor improves world-model prediction accuracy.
Instead, it supports the narrower claim that Chronos constraints can act as a
causality-preserving world-model regularizer while maintaining comparable
rollout error on this synthetic stress test.

## Experiment 5b: Causal Mechanism Ablation

**Task.** Decompose the Chronos latent predictor into mechanism variants and
measure which parts of the constraint stack affect decoded causal consistency.

Variants:

- Euclidean latent predictor
- Chronos geometry-only latent predictor
- Chronos causal-only latent predictor
- Chronos interval-only latent predictor
- Chronos full latent predictor

These variants separate Lorentz-normalized latent-step geometry from causal,
interval, and `K=1` regularization losses.

**Command.**

```bash
cd k1-manifold-core
python benchmarks/experiment_5b_causal_mechanism_ablation.py
```

Use `--smoke` for a tiny CPU-friendly run and `--full` for the larger
configuration:

```bash
python benchmarks/experiment_5b_causal_mechanism_ablation.py --smoke
python benchmarks/experiment_5b_causal_mechanism_ablation.py --full
```

**Artifacts.**

- `results/experiment_5b_causal_mechanism_ablation_summary.csv`
- `results/experiment_5b_causal_mechanism_ablation_raw.json`
- `results/experiment_5b_violation_vs_box.png`
- `results/experiment_5b_mse_vs_box.png`
- `results/experiment_5b_effective_rank.png`
- `results/experiment_5b_violation_by_step.png`
- `results/experiment_5b_K_drift_by_step.png`

**Interpretation boundary.** Experiment 5b is a mechanism probe. It is not a
unit test, not a theorem, and not by itself a claim that Chronos improves
forecasting accuracy. Its purpose is to identify which Chronos constraints are
responsible for causal-consistency preservation before moving to stronger
dynamical baselines.
