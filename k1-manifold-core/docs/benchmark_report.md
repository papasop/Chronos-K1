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
