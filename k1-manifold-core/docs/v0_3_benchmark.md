# v0.3 Task Benchmark

This benchmark moves Chronos-K1 from geometric self-consistency checks toward a minimal task validation.

The task is intentionally narrow: compare a Euclidean gradient baseline against Chronos-K1 Lorentzian dynamics on a noisy `K=1` recovery task.

It does not claim broad superiority on world-modeling tasks.

## Models Compared

### Euclidean Gradient Baseline

The baseline uses the same potential but follows Euclidean gradient descent:

```text
xdot = -grad V
V = 1/2 (K - 1)^2
```

This is a direct baseline for recovery under the same scalar objective.

### Chronos-K1 Lorentzian Dynamics

Chronos-K1 uses the implemented Law II / Law III dynamics:

```text
xdot = (J_G - D) grad V
V = 1/2 (K - 1)^2
D = d_c I
K(x) = x.T @ G @ x
```

Both systems receive the same seeded perturbation sequence:

```text
x <- x + dt * dynamics(x) + 0.05 * N(0, I)
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

- `mean_abs_k_error_tail`: mean absolute `|K(t)-1|` over the late rollout tail.
- `mean_potential_tail`: mean `V(t)` over the late rollout tail.
- `mean_recovery_time`: first time to reach `|K(t)-1| <= 0.05`, averaged over rollouts.
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

This benchmark is a constrained numerical diagnostic. It shows whether the implemented Chronos-K1 dynamics recovers the implemented `K=1` task constraints faster and more stably than the selected Euclidean gradient baseline under shared random perturbations.

It is not a general proof of physical correctness, nor a world-model benchmark.
