# Physics-AI Evidence Index

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

- ✅ Evidence for physics-sensitive, metric-sensitive inductive bias
- ❌ Not proof of general Physics AI
- ❌ Not proof Chronos dominates all world-model tasks
