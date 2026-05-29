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

**Task.** Train Euclidean JEPA and Chronos-JEPA on synthetic Lorentzian
oscillator trajectories, then evaluate rollout MSE, decoded causal violation,
interval drift, and latent `K` drift under OOD box extrapolation.

**Command.**

```bash
cd k1-manifold-core
python benchmarks/experiment_5_causal_stress_test.py
```

Use `--full` for the larger configuration:

```bash
python benchmarks/experiment_5_causal_stress_test.py --full
```

**Artifacts.**

- `results/experiment_5_ablation_stress_summary.csv`
- `results/experiment_5_ablation_stress_raw.json`
- `results/experiment_5_violation_vs_box.png`
- `results/experiment_5_mse_vs_box.png`
- `results/experiment_5_violation_by_step.png`
- `results/experiment_5_K_drift_by_step.png`

**Current quick result.** The quick benchmark is deliberately small
(`n_seeds=2`) and should be read as a smoke-test ablation, not a final result.
In this configuration, Chronos-JEPA does **not** improve rollout MSE and
increases decoded causal-violation rates relative to Euclidean JEPA. This is
useful negative evidence: the current latent regularizer is not yet a reliable
causality-preserving world-model route on this oscillator stress test.

| Lambda | Test box | MSE improvement vs Euclid | Violation improvement vs Euclid | p(MSE) | p(violation) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0 | 2 | -0.30% | -161.10% | 0.5000 | 1.0000 |
| 0.0 | 8 | -0.07% | -147.39% | 0.5000 | 1.0000 |
| 0.0 | 32 | -0.02% | -285.17% | 0.5000 | 1.0000 |
| 0.5 | 2 | -0.36% | -734.37% | 0.5000 | 1.0000 |
| 0.5 | 8 | -0.08% | -661.37% | 0.5000 | 1.0000 |
| 0.5 | 32 | -0.02% | -1207.78% | 0.5000 | 1.0000 |

**Interpretation boundary.** This benchmark should not be cited as evidence
that Chronos-JEPA improves world-model prediction. It is included because it
exposes a concrete failure mode and gives the project a reproducible target:
future Chronos regularizers should reduce causal violations without sacrificing
rollout accuracy on this stress test.
