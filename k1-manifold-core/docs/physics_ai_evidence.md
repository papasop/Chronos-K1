# Physics-AI Evidence Index

## Chronos Program Status

| Stage | Result | Status |
| --- | --- | --- |
| K1 | Framework validation / spectral prior | `BOUNDED_POSITIVE` |
| K2 | Symplectic prior on FPU-beta | `FULL_TRANSFER_CONFIRMED` |

K1 established the VPSL framework and a bounded positive spectral /
Lorentz-sensitive prior result. K2 produced the first VPSL-certified physical
structure: a symplectic prior on FPU-beta with H=200 transfer confirmed on the
graceful-baseline subset.

See:

- `../../chronos/k1/archive.md`
- `../../chronos/k2/archive.md`
- `../../chronos/vpsl/certified_structures.md`

## Evidence Level 1: Exp5

- Experiment 5 (`oscillator` stress test)
- Role: historical world-model benchmark and stress baseline
- Status: mixed

## Evidence Level 2: Exp6

- Experiment 6 (`ood_extrapolation.py`)
- Question: Does Chronos react differently to timelike vs spacelike structure?
- Role: physics sensitivity evidence
- Status: positive

## Evidence Level 3: Exp7

- Experiment 7 (`experiment_7_metric_controlled_normalization.py`)
- Question: Is the sensitivity effect specific to Lorentz normalization?
- Method: Metric x Dataset interaction with one-sided Wilcoxon
- Status: positive (`N=30`, `p=0.040` for Lorentz; Euclidean/random not significant)

## Interpretation Boundary

- ✅ Evidence for VPSL framework validation
- ✅ Evidence for physics-sensitive, metric-sensitive inductive bias in K1
- ✅ Evidence for the first VPSL-certified structure in K2
- ❌ Not proof of general Physics AI
- ❌ Not proof Chronos dominates all world-model tasks
