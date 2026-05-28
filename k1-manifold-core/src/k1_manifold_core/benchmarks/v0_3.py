"""v0.3 constrained rollout benchmark.

The benchmark is intentionally narrow: it compares a Euclidean unit-sphere
baseline against Chronos-K1 Lorentzian dynamics on a K=1 constraint task.
It is a numerical diagnostic, not a broad claim about all world-model tasks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np

from k1_manifold_core.dynamics.critical_damping import critical_damping_matrix
from k1_manifold_core.dynamics.ode_solvers import integrate
from k1_manifold_core.dynamics.symplectic_dissipative import k1_potential_gradient, vector_field


Array = np.ndarray
RHS = Callable[[Array, float], Array]


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


def euclidean_baseline_rhs(x: Array, _t: float) -> Array:
    """Euclidean baseline flow toward the Euclidean unit circle."""

    x = np.asarray(x, dtype=float)
    radius_squared = float(x.T @ x)
    return -2.0 * (radius_squared - 1.0) * x


def chronos_lorentzian_rhs(x: Array, _t: float, G: Array = DEFAULT_G) -> Array:
    """Chronos-K1 Law II/III flow for the K=1 potential."""

    G = np.asarray(G, dtype=float)
    D = critical_damping_matrix(G)
    return vector_field(x, G, D, lambda y: k1_potential_gradient(y, G))


def rollout(rhs: RHS, x0: Array, *, dt: float, steps: int) -> Array:
    """Integrate one rollout from ``x0``."""

    return integrate(rhs, np.asarray(x0, dtype=float), dt=dt, steps=steps)


def evaluate_rollouts(
    rhs: RHS,
    initial_states: Array = DEFAULT_INITIAL_STATES,
    *,
    G: Array = DEFAULT_G,
    dt: float = 1e-3,
    steps: int = 10000,
    tail: int = 1000,
) -> dict[str, float]:
    """Compute K-stability, causal-violation, and long-horizon metrics."""

    G = np.asarray(G, dtype=float)
    k_series = []
    final_errors = []
    violation_rates = []

    for x0 in np.asarray(initial_states, dtype=float):
        trajectory = rollout(rhs, x0, dt=dt, steps=steps)
        values = np.array([k_value(x, G) for x in trajectory])
        k_series.append(values)
        final_errors.append(abs(values[-1] - 1.0))
        violation_rates.append(float(np.mean(values <= 0.0)))

    tail_errors = [float(np.mean(np.abs(values[-tail:] - 1.0))) for values in k_series]
    return {
        "k_stability_tail_mae": float(np.mean(tail_errors)),
        "causal_violation_rate": float(np.mean(violation_rates)),
        "long_horizon_rollout_error": float(np.mean(final_errors)),
    }


def run_benchmark_v03(
    *,
    dt: float = 1e-3,
    steps: int = 10000,
    tail: int = 1000,
) -> dict[str, object]:
    """Run the deterministic v0.3 task benchmark."""

    euclidean = evaluate_rollouts(euclidean_baseline_rhs, dt=dt, steps=steps, tail=tail)
    chronos = evaluate_rollouts(chronos_lorentzian_rhs, dt=dt, steps=steps, tail=tail)
    return {
        "benchmark": "v0.3 constrained K=1 rollout",
        "description": "Euclidean unit-sphere baseline vs Chronos-K1 Lorentzian dynamics on a K=1 constraint task.",
        "parameters": {
            "dt": dt,
            "steps": steps,
            "tail": tail,
            "G": DEFAULT_G.tolist(),
            "initial_states": DEFAULT_INITIAL_STATES.tolist(),
        },
        "metrics": {
            "lower_is_better": [
                "k_stability_tail_mae",
                "causal_violation_rate",
                "long_horizon_rollout_error",
            ],
            "euclidean_baseline": euclidean,
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
