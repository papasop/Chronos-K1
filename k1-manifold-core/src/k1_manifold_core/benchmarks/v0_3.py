"""v0.3 noisy K=1 recovery benchmark.

The benchmark compares two dynamics on the same K=1 recovery task:

* Euclidean gradient dynamics: ``xdot = -grad V``.
* Chronos-K1 Lorentzian dynamics: ``xdot = (J_G - D) grad V``.

Both systems receive the same seeded Gaussian perturbations. This is a
numerical task benchmark, not a broad theory claim.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np

from k1_manifold_core.dynamics.critical_damping import critical_damping_matrix
from k1_manifold_core.dynamics.symplectic_dissipative import k1_potential_gradient, vector_field


Array = np.ndarray
RHS = Callable[[Array], Array]


DEFAULT_G = np.diag([1.0, -1.0])
DEFAULT_INITIAL_STATES = np.array(
    [
        [1.2, 0.0],
        [0.0, 0.8],
        [1.1, 0.1],
        [1.4, 0.2],
        [1.3, -0.4],
    ],
    dtype=float,
)


def k_value(x: Array, G: Array = DEFAULT_G) -> float:
    """Return ``K(x)=x.T @ G @ x``."""

    x = np.asarray(x, dtype=float)
    return float(x.T @ np.asarray(G, dtype=float) @ x)


def k_potential(x: Array, G: Array = DEFAULT_G) -> float:
    """Return ``V=1/2(K-1)^2``."""

    K = k_value(x, G)
    return 0.5 * (K - 1.0) ** 2


def euclidean_gradient_rhs(x: Array, G: Array = DEFAULT_G) -> Array:
    """Euclidean baseline ``xdot = -grad V`` for the same K-potential."""

    return -k1_potential_gradient(np.asarray(x, dtype=float), np.asarray(G, dtype=float))


def chronos_lorentzian_rhs(x: Array, G: Array = DEFAULT_G) -> Array:
    """Chronos-K1 Law II/III flow for the same K-potential."""

    G = np.asarray(G, dtype=float)
    D = critical_damping_matrix(G)
    return vector_field(x, G, D, lambda y: k1_potential_gradient(y, G))


def noisy_rollout(
    rhs: RHS,
    x0: Array,
    noise: Array,
    *,
    dt: float,
) -> Array:
    """Euler rollout with additive perturbations ``x += noise[t]``."""

    x = np.asarray(x0, dtype=float).copy()
    trajectory = [x.copy()]
    for perturbation in np.asarray(noise, dtype=float):
        x = x + dt * rhs(x) + perturbation
        trajectory.append(x.copy())
    return np.vstack(trajectory)


def shared_noise(
    *,
    seed: int,
    initial_count: int,
    steps: int,
    dimension: int,
    noise_scale: float,
) -> Array:
    """Return one deterministic noise tensor shared across models."""

    rng = np.random.default_rng(seed)
    return noise_scale * rng.standard_normal((initial_count, steps, dimension))


def recovery_time(k_series: Array, *, dt: float, tolerance: float) -> float:
    """Return first time that ``|K(t)-1| <= tolerance``; inf if absent."""

    errors = np.abs(np.asarray(k_series, dtype=float) - 1.0)
    hits = np.flatnonzero(errors <= tolerance)
    if hits.size == 0:
        return float("inf")
    return float(hits[0] * dt)


def evaluate_rollouts(
    rhs: RHS,
    noise: Array,
    initial_states: Array = DEFAULT_INITIAL_STATES,
    *,
    G: Array = DEFAULT_G,
    dt: float = 1e-3,
    tail: int = 1000,
    recovery_tolerance: float = 0.05,
) -> dict[str, float]:
    """Compute noisy K-error, potential, and recovery-time metrics."""

    G = np.asarray(G, dtype=float)
    abs_k_errors = []
    potentials = []
    recovery_times = []
    final_errors = []

    for index, x0 in enumerate(np.asarray(initial_states, dtype=float)):
        trajectory = noisy_rollout(rhs, x0, noise[index], dt=dt)
        k_series = np.array([k_value(x, G) for x in trajectory])
        v_series = 0.5 * (k_series - 1.0) ** 2
        abs_error = np.abs(k_series - 1.0)

        abs_k_errors.append(float(np.mean(abs_error[-tail:])))
        potentials.append(float(np.mean(v_series[-tail:])))
        recovery_times.append(recovery_time(k_series, dt=dt, tolerance=recovery_tolerance))
        final_errors.append(float(abs_error[-1]))

    return {
        "mean_abs_k_error_tail": float(np.mean(abs_k_errors)),
        "mean_potential_tail": float(np.mean(potentials)),
        "mean_recovery_time": float(np.mean(recovery_times)),
        "long_horizon_rollout_error": float(np.mean(final_errors)),
    }


def run_benchmark_v03(
    *,
    dt: float = 1e-3,
    steps: int = 5000,
    tail: int = 500,
    noise_scale: float = 0.05,
    seed: int = 7,
    recovery_tolerance: float = 0.05,
) -> dict[str, object]:
    """Run the deterministic noisy v0.3 task benchmark."""

    noise = shared_noise(
        seed=seed,
        initial_count=len(DEFAULT_INITIAL_STATES),
        steps=steps,
        dimension=DEFAULT_INITIAL_STATES.shape[1],
        noise_scale=noise_scale,
    )
    euclidean = evaluate_rollouts(
        lambda x: euclidean_gradient_rhs(x, DEFAULT_G),
        noise,
        dt=dt,
        tail=tail,
        recovery_tolerance=recovery_tolerance,
    )
    chronos = evaluate_rollouts(
        lambda x: chronos_lorentzian_rhs(x, DEFAULT_G),
        noise,
        dt=dt,
        tail=tail,
        recovery_tolerance=recovery_tolerance,
    )
    return {
        "benchmark": "v0.3 noisy K=1 recovery",
        "description": "Euclidean xdot=-grad V vs Chronos-K1 xdot=(J_G-D)grad V under shared Gaussian perturbations.",
        "parameters": {
            "dt": dt,
            "steps": steps,
            "tail": tail,
            "noise_scale": noise_scale,
            "seed": seed,
            "recovery_tolerance": recovery_tolerance,
            "G": DEFAULT_G.tolist(),
            "initial_states": DEFAULT_INITIAL_STATES.tolist(),
        },
        "metrics": {
            "lower_is_better": [
                "mean_abs_k_error_tail",
                "mean_potential_tail",
                "mean_recovery_time",
                "long_horizon_rollout_error",
            ],
            "euclidean_gradient": euclidean,
            "chronos_k1_lorentzian": chronos,
        },
    }


def write_benchmark_v03(path: str | Path) -> dict[str, object]:
    """Run and write ``benchmark_v03.json``."""

    result = run_benchmark_v03()
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    default_path = Path(__file__).resolve().parents[3] / "results" / "benchmark_v03.json"
    data = write_benchmark_v03(default_path)
    print(json.dumps(data["metrics"], indent=2))
    print(f"saved results = {default_path}")
