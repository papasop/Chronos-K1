# Reproduce The Core Results

This page gives the shortest path from a clean checkout to the numerical
checks used by the Chronos-K1 research prototype. The commands below do not
expand the theory claims; they reproduce the implemented tests and demos.

## 1. Install

From the repository root:

```bash
python -m pip install -r requirements.txt
python -m pip install -e "k1-manifold-core[dev]"
```

## 2. Run The Test Suite

```bash
pytest -v k1-manifold-core
```

Expected result:

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

## 3. Reproduce The Null-Flow Figure And Numbers

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

## 4. Reproduce The Latent World-Model v0.1 Benchmark

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

This is a toy latent regularizer benchmark: the Chronos-K1 variant is the same
affine transition as the Euclidean baseline, followed by a `K=1` projection.
It is not a claim about large-scale world models.

## Paper Mapping

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
`R^2 = 1.000000`. Both correspond to the same inverse-square recovery law;
the finite-sample values are numerical-plot estimates, while the deterministic
reference is the exact implemented check.

## CI

GitHub Actions runs the same core command on every push and pull request:

```bash
cd k1-manifold-core
python -m pip install -e ".[dev]"
pytest -v
```

## Optional AI Benchmark

The OOD light-cone benchmark uses PyTorch and is intentionally not part of
pytest:

```bash
cd k1-manifold-core
python -m pip install -r requirements-benchmarks.txt
python benchmarks/ood_extrapolation.py
```

It writes:

```text
k1-manifold-core/results/ood_extrapolation.json
k1-manifold-core/results/ood_extrapolation_auc.png
```
