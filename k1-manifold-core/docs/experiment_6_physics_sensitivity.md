# Experiment 6: Physics Sensitivity Benchmark

## Question

Does Chronos react differently to Lorentz-structured data than to non-Lorentz
alternatives under controlled extrapolation stress?

## Benchmark Role

- Exp6 is the **sensitivity layer** in the Physics-AI evidence ladder.
- Exp7 is the **mechanism layer** (metric-specific interaction test).

## Current Implementation

- Benchmark script: `benchmarks/ood_extrapolation.py`
- Setup: train on `box=2`, test OOD on larger boxes.
- Models:
  - `lorentz`
  - `euclid_mahalanobis`
  - `euclid_mlp`

## Run

```bash
cd k1-manifold-core
python benchmarks/ood_extrapolation.py
```

## Main Artifacts

- `results/ood_extrapolation.json`
- `results/ood_extrapolation_auc.png`

## Interpretation Boundary

- ✅ Evidence for physics sensitivity under this synthetic setup
- ❌ Not proof of general world-model superiority
- ❌ Not replacement for real-physics dataset validation
