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
  The default run is the representative `n_seeds=5` configuration; use
  `--smoke` for a tiny CPU-friendly check or `--full` for the larger sweep.

Run from `k1-manifold-core`:

```bash
python benchmarks/ood_extrapolation.py
python benchmarks/experiment_5_causal_stress_test.py
python benchmarks/experiment_5_causal_stress_test.py --smoke
```

These benchmarks currently require PyTorch in addition to the core scientific
Python stack. Experiment 5 also uses pandas for its summary CSV/table.
