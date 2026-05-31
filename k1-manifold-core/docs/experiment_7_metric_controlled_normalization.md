# Experiment 7: Metric-Controlled Normalization

## Goal

Test whether Lorentz normalization produces a dataset-specific advantage on
`timelike` trajectories that Euclidean/random normalization does not.

This is a **Metric x Dataset interaction** test. The key diagnostic is the
improvement gap between datasets, not absolute violation rate alone.

## Design

- Datasets:
  - `timelike`: geodesics with `eta(v,v) > 0`
  - `spacelike`: geodesics with `eta(v,v) < 0`
- Models:
  - `euclidean_baseline` (no normalization)
  - `chronos_lorentz`
  - `chronos_euclidean`
  - `chronos_random`
- Seeds: `N=30`
- Metric:
  - decoded rollout causal-violation rate
  - dataset-normalized improvement vs baseline

## Statistical Test

For each metric variant, compute paired per-seed differences:

- `improvement_timelike - improvement_spacelike`

Then run one-sided Wilcoxon signed-rank test:

- `H0`: timelike improvement equals spacelike improvement
- `H1`: timelike improvement is greater than spacelike improvement

Also report:

- Cohen's d on paired differences
- bootstrap 95% CI for mean paired difference

## Run

```bash
cd k1-manifold-core
python benchmarks/experiment_7_metric_controlled_normalization.py
```

## Output Artifacts

- `results/experiment_7_metric_controlled_normalization/experiment_7_raw_results.csv`
- `results/experiment_7_metric_controlled_normalization/experiment_7_raw_results_with_improvement.csv`
- `results/experiment_7_metric_controlled_normalization/experiment_7_metric_dataset_interaction.csv`

## Interpretation Boundary

- ✅ Evidence for metric-sensitive inductive bias
- ❌ Not proof of general Physics AI
- ❌ Not proof Lorentz is always best in raw performance
