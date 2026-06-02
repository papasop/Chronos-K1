# Reproduce The Core Results

This page separates the historical K1 reproduction path from the newer K2
VPSL archive path.

Chronos currently contains two evidence stages:

- K1: framework / bounded result.
- K2: first VPSL-certified physical structure.

## 1. Install

From the repository root:

```bash
python -m pip install -r requirements.txt
python -m pip install -e "k1-manifold-core[dev]"
```

## 2. K1: Historical Framework Checks

Run the K1 test suite:

```bash
pytest -v k1-manifold-core
```

Expected result for the current K1 suite:

```text
collected 29 items
...
29 passed
```

This covers:

- `tests/test_information_time.py`: `dt_info = dPhi / H`.
- `tests/test_causal_cone.py`: timelike / lightlike / spacelike
  classification by `x.T @ G @ x`.
- `tests/test_k1_attractor.py`: monotone decrease of
  `V = 1/2 (K - 1)^2` and recovery toward `K = 1` under the implemented
  Law II / Law III dynamics.
- `tests/test_null_flow.py`: rank-one critical-damping null flow,
  first-integral conservation, leaf probability, and inverse-square recovery
  scaling.
- `tests/test_einstein.py`: symbolic spherical-sector reformulation checks.
- `tests/test_world_model_v01.py`: minimal latent world-model regularizer
  checks.

## 3. K1: Null-Flow Figure And Numbers

```bash
cd k1-manifold-core
python examples/demo_04_recovery_scaling.py
```

Expected output fields:

```text
rank(A_c) = 1
max |c(t)-c(0)| = O(1e-13)
ideal probability ratio = 1.0000
log-log recovery slope = -2.000000
log-log fit R^2 = 1.000000
```

The figure is written to:

```text
k1-manifold-core/examples/outputs/demo_04_recovery_scaling.png
```

## 4. K1: Latent World-Model v0.1 Benchmark

```bash
cd k1-manifold-core
python examples/benchmark_world_model_v01.py
```

Expected output fields:

```text
euclidean_linear.prediction_mse
euclidean_linear.long_horizon_rollout_mse
euclidean_linear.mean_abs_k_drift
chronos_k1_projected.prediction_mse
chronos_k1_projected.long_horizon_rollout_mse
chronos_k1_projected.mean_abs_k_drift
```

The result is written to:

```text
k1-manifold-core/results/world_model_v01.json
```

This is a toy latent regularizer benchmark. It is not a claim about large-scale
world models.

## 5. K1 Paper Mapping

| Paper object | Script / test | What to check |
| --- | --- | --- |
| Law I information time identity | `tests/test_information_time.py` | `dt_info = dPhi / H` |
| Causal cone classification | `tests/test_causal_cone.py` and `examples/demo_02_causal_cone.py` | sign of `x.T @ G @ x` |
| Proposition: critical-damping null flow is rank-one | `tests/test_null_flow.py` and `examples/demo_04_recovery_scaling.py` | `rank(A_c) = 1` |
| First integral on null-flow leaves | `tests/test_null_flow.py` | `max |c(t)-c(0)| < 1e-12` |
| Leaf probability scaling | `tests/test_null_flow.py` and `examples/demo_04_recovery_scaling.py` | `ratio = 1.0000` for the ideal `4c^2` law |
| Recovery-time scaling figure | `examples/demo_04_recovery_scaling.py` | slope near `-2`, with `R^2` near `1` |
| Spherical-sector reformulation | `tests/test_einstein.py` | Schwarzschild, Reissner-Nordstrom asymmetry, and Schwarzschild-de Sitter symbolic checks |
| Latent world-model regularizer | `tests/test_world_model_v01.py` and `examples/benchmark_world_model_v01.py` | prediction MSE, rollout MSE, and `K` drift on a toy hyperbolic dataset |

The companion paper may report finite-sample visual-fit values such as
`slope = -1.986` and `R^2 = 0.99998`. The current demo also prints the
deterministic analytic reference fit, which gives `slope = -2.000000` and
`R^2 = 1.000000`. Both correspond to the same inverse-square recovery law.

## 6. K1 Optional AI Benchmark

The OOD light-cone benchmark uses PyTorch and is intentionally not part of
pytest:

```bash
cd k1-manifold-core
python -m pip install -r requirements-benchmarks.txt
python benchmarks/ood_extrapolation.py
python benchmarks/experiment_5_causal_stress_test.py
python benchmarks/experiment_5_full_sanity_reproduction.py --full
python benchmarks/experiment_5b_causal_mechanism_ablation.py
```

For a tiny CPU-friendly smoke check of Experiment 5, run:

```bash
python benchmarks/experiment_5_causal_stress_test.py --smoke
python benchmarks/experiment_5_full_sanity_reproduction.py --smoke
python benchmarks/experiment_5b_causal_mechanism_ablation.py --smoke
```

The public diagnostic entry point is also available from the repository root:

```bash
python exp5-diagnostic/chronos_k1_complete_colab.py --smoke
```

## 7. K2: VPSL Archive

K2 is archived under:

```text
chronos/k2/
```

Current K2 files:

```text
chronos/k1/archive.md
chronos/k2/archive.md
chronos/k2/docs/chronos_k2_archive.md
chronos/k2/historical_logs/k2_0_regime_validation_log.txt
chronos/k2/historical_logs/k2_0b_refine_window_log.txt
chronos/k2/reconstruction_notes.md
chronos/k2/experiments/k2_1_symplectic_prior.py
chronos/k2/experiments/k2_1b_repair_controls.py
chronos/k2/experiments/k2_2a_transfer_h200.py
chronos/k2/results/k2_2a_summary.csv
```

K2.2-A is the current headline result:

```text
FULL_TRANSFER_CONFIRMED
```

The archived K2.2-A repository summary can be regenerated with:

```bash
python chronos/k2/experiments/k2_2a_transfer_h200.py
```

The full Colab training run remains expensive. Its claim is valid only when
the full experiment is run with the registered K2.2-A gates:

- primary metric: rollout MSE at H=200
- primary subset: graceful-baseline subset
- controls: fair energy and fair L2, re-checked at H=200
- mechanism: full `||J^T Omega J - Omega||` reduction above threshold

Full reproducibility currently starts from K2.1. The predecessor K2.0 and
K2.0-B stages are preserved as historical logs, and their source scripts are
being reconstructed.

## 8. CI

GitHub Actions runs the K1 core command on every push and pull request:

```bash
cd k1-manifold-core
python -m pip install -e ".[dev]"
pytest -v
```
