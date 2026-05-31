# Benchmark Report

This report records research benchmarks that involve model training,
randomness, statistics, or optional ML dependencies. They are not unit tests.

## Experiment 6: Light-cone Geodesic Benchmark

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

**Main physics-AI result statement.** On true Lorentz geodesic data, Chronos
latent geometry reduces causal violation relative to Euclidean latent
prediction.

## Experiment 5: Oscillator Stress Test

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

For the dedicated full sanity reproduction script:

```bash
CHRONOS_DEVICE=cuda python benchmarks/experiment_5_full_sanity_reproduction.py --full
```

**Artifacts.**

- `results/experiment_5_ablation_stress_summary.csv`
- `results/experiment_5_ablation_stress_raw.json`
- `results/experiment_5_violation_vs_box.png`
- `results/experiment_5_mse_vs_box.png`
- `results/experiment_5_violation_by_step.png`
- `results/experiment_5_K_drift_by_step.png`

**Full sanity reproduction.** A reproduction of the original Experiment 5
design was run on CUDA with `n_seeds=10`, `n_train=3000`, `n_test=512`, and
`epochs=250`. At `lambda=0.1`, Chronos latent predictor keeps final rollout
MSE approximately unchanged while reducing decoded causal-violation rates.
The exact seed strategy is documented in
`docs/experiment_5_reproduction_protocol.md`.

| Setting | Euclid violation | Chronos violation | Reduction | p(violation) |
| --- | ---: | ---: | ---: | ---: |
| In-distribution, `box=2` | 0.2866 | 0.1475 | 48.5% | 0.0840 |
| OOD, `box=32` | 0.4306 | 0.3155 | about 27% | not reported |

The in-distribution Wilcoxon value is close to, but does not meet, the
conventional `p<0.05` threshold. This should be read as a primary phenomenon
benchmark with substantial effect size and OOD persistence, not as a final
statistical significance claim.

**Interpretation boundary.** This benchmark should not be cited as evidence
that Chronos latent predictor improves world-model prediction accuracy.
Instead, it supports the narrower claim that Chronos constraints can act as a
causality-preserving world-model regularizer while maintaining comparable
rollout error on this synthetic stress test.

## Experiment 5b: Fixed Ablation Diagnostic

**Task.** After fixing latent/decoded mismatch behavior, decompose the Chronos
latent predictor into mechanism variants and
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

## Experiment 7: Metric-Controlled Normalization

**Question.** Does Lorentz normalization produce a dataset-specific advantage
on timelike trajectories that Euclidean/random normalization does not?

**Design.** Compare `euclidean_baseline`, `chronos_lorentz`,
`chronos_euclidean`, and `chronos_random` on `timelike` and `spacelike`
datasets. Analyze per-seed improvement relative to dataset baseline and test
Metric x Dataset interaction via one-sided paired Wilcoxon.

**Command.**

```bash
cd k1-manifold-core
python benchmarks/experiment_7_metric_controlled_normalization.py
```

**Artifacts.**

- `results/experiment_7_metric_controlled_normalization/experiment_7_raw_results.csv`
- `results/experiment_7_metric_controlled_normalization/experiment_7_raw_results_with_improvement.csv`
- `results/experiment_7_metric_controlled_normalization/experiment_7_metric_dataset_interaction.csv`

**Headline result format.** N=30, one-sided Wilcoxon p-value on the Lorentz
metric interaction term (timelike improvement > spacelike improvement), with
Euclidean/random as controls.

**Interpretation boundary.**

- ✅ Evidence for metric-sensitive inductive bias
- ❌ Not proof of general Physics AI
- ❌ Not proof that Lorentz is always best in raw performance
