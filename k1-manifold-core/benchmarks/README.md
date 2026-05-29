# Research Benchmarks

This directory contains research benchmarks that may involve training,
randomness, statistics, or optional machine-learning dependencies. They are
intentionally separate from `tests/`.

- `ood_extrapolation.py`: train light-cone classifiers on event differences
  from `box=2`, then evaluate OOD extrapolation on larger boxes. It writes
  `results/ood_extrapolation.json` and `results/ood_extrapolation_auc.png`.
- `experiment_5_causal_stress_test.py`: scan Chronos latent predictor causal
  regularizer strength on Lorentzian oscillator trajectories and write
  CSV/JSON/PNG artifacts under `results/`. The model is JEPA-style because it
  predicts future embeddings, but it is not a Meta/LeCun JEPA implementation.
  Use `--smoke` for a tiny CPU-friendly check or `--full` for the larger
  `n_seeds=10` sanity reproduction.
- `experiment_5_full_sanity_reproduction.py`: dedicated reproduction script
  for the original Experiment 5 two-model lambda scan. It records RNG seeding,
  mean/final rollout MSE, causal violation, interval drift, and paired
  Wilcoxon p-values.
- `experiment_5b_causal_mechanism_ablation.py`: decompose Experiment 5 into
  Euclidean, Chronos geometry-only, Chronos causal-only, Chronos interval-only,
  and Chronos full latent-predictor variants. It writes summary CSV, raw JSON,
  and diagnostic PNG artifacts under `results/`.

Run from `k1-manifold-core`:

```bash
python benchmarks/ood_extrapolation.py
python benchmarks/experiment_5_causal_stress_test.py
python benchmarks/experiment_5_causal_stress_test.py --smoke
python benchmarks/experiment_5_full_sanity_reproduction.py --smoke
python benchmarks/experiment_5b_causal_mechanism_ablation.py --smoke
```

These benchmarks currently require PyTorch in addition to the core scientific
Python stack. Experiments 5 and 5b also use pandas for summary CSV/tables.
