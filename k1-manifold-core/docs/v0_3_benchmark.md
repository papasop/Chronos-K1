# v0.3 Task Benchmark

This benchmark moves Chronos-K1 from geometric self-consistency checks toward a minimal task validation.

The task is intentionally narrow: compare a Euclidean baseline against Chronos-K1 Lorentzian dynamics on a deterministic `K=1` constrained rollout task.

It does not claim broad superiority on world-modeling tasks.

## Models Compared

### Euclidean Baseline

The baseline evolves toward the Euclidean unit circle:

```text
dx/dt = -2 (||x||^2 - 1) x
```

This is a simple Euclidean normalization dynamics.

### Chronos-K1 Lorentzian Dynamics

Chronos-K1 uses the implemented Law II / Law III dynamics:

```text
xdot = (J_G - D) grad V
V = 1/2 (K - 1)^2
D = d_c I
K(x) = x.T @ G @ x
```

## Shared Initial Conditions

Both models use the same deterministic initial states:

```text
[1.2,  0.0]
[0.0,  0.8]
[1.1,  0.1]
[1.4,  0.2]
[1.3, -0.4]
```

## Metrics

All metrics are lower-is-better.

- `k_stability_tail_mae`: mean absolute `|K(t)-1|` over the late rollout tail.
- `causal_violation_rate`: fraction of rollout steps with `K(t) <= 0`.
- `long_horizon_rollout_error`: final absolute `|K(T)-1|`.

## Run

```bash
cd k1-manifold-core
python examples/benchmark_v03.py
```

This writes:

```text
results/benchmark_v03.json
```

## Interpretation Boundary

This benchmark is a constrained numerical diagnostic. It shows whether the implemented Chronos-K1 dynamics better preserves the implemented `K=1` task constraints than the selected Euclidean baseline.

It is not a general proof of physical correctness, nor a world-model benchmark.
