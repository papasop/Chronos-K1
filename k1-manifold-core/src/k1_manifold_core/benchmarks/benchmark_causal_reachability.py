"""v0.4 causal reachability benchmark.

This benchmark compares raw Euclidean random displacements against a Chronos
causal-cone projection on the same random steps. It measures whether each
step is causally valid under ``dx.T @ G @ dx >= 0``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np


Array = np.ndarray

G = np.array([[1.0, 0.0], [0.0, -1.0]])


def interval(dx: Array, G_metric: Array = G) -> float:
    """Return the signed interval ``dx.T @ G @ dx``."""

    dx = np.asarray(dx, dtype=float)
    return float(dx.T @ np.asarray(G_metric, dtype=float) @ dx)


def causal_valid(dx: Array, G_metric: Array = G, *, atol: float = 1e-12) -> bool:
    """Return whether a displacement is inside or on the causal cone."""

    return interval(dx, G_metric) >= -atol


def causal_project(dx: Array, *, eps: float = 1e-9) -> Array:
    """Project a displacement ``(dt, dx_space)`` into the future causal cone."""

    t, x = np.asarray(dx, dtype=float)
    if t < 0.0:
        t = abs(t)
    if t * t - x * x >= 0.0:
        return np.array([t, x], dtype=float)
    x = np.sign(x) * max(abs(t) - eps, 0.0)
    return np.array([t, x], dtype=float)


def simulate_reachability(
    *,
    steps: int = 5000,
    step_sigma: float = 0.3,
    seed: int = 2026,
) -> dict[str, Any]:
    """Simulate raw Euclidean and causal-projected random walks."""

    rng = np.random.default_rng(seed)
    x_e = np.array([0.0, 0.0])
    x_c = np.array([0.0, 0.0])
    traj_e = [x_e.copy()]
    traj_c = [x_c.copy()]
    euclid_violations = 0
    chronos_violations = 0
    euclid_intervals = []
    chronos_intervals = []

    for _ in range(steps):
        dx_raw = step_sigma * rng.standard_normal(2)
        dx_e = dx_raw
        dx_c = causal_project(dx_raw)

        if not causal_valid(dx_e):
            euclid_violations += 1
        if not causal_valid(dx_c):
            chronos_violations += 1

        euclid_intervals.append(interval(dx_e))
        chronos_intervals.append(interval(dx_c))

        x_e = x_e + dx_e
        x_c = x_c + dx_c
        traj_e.append(x_e.copy())
        traj_c.append(x_c.copy())

    return {
        "traj_e": np.array(traj_e),
        "traj_c": np.array(traj_c),
        "euclid_violations": euclid_violations,
        "chronos_violations": chronos_violations,
        "euclid_intervals": np.array(euclid_intervals),
        "chronos_intervals": np.array(chronos_intervals),
        "steps": steps,
        "step_sigma": step_sigma,
        "seed": seed,
    }


def summarize_reachability(result: dict[str, Any]) -> dict[str, Any]:
    """Return JSON-serializable benchmark metrics."""

    steps = int(result["steps"])
    euclid_intervals = np.asarray(result["euclid_intervals"], dtype=float)
    chronos_intervals = np.asarray(result["chronos_intervals"], dtype=float)
    return {
        "benchmark": "v0.4 causal reachability",
        "description": "Raw Euclidean random displacements vs Chronos causal-cone projection.",
        "parameters": {
            "steps": steps,
            "step_sigma": float(result["step_sigma"]),
            "seed": int(result["seed"]),
            "G": G.tolist(),
        },
        "metrics": {
            "euclidean": {
                "violations": int(result["euclid_violations"]),
                "violation_rate": float(result["euclid_violations"] / steps),
                "mean_interval": float(np.mean(euclid_intervals)),
                "min_interval": float(np.min(euclid_intervals)),
            },
            "chronos_causal_projection": {
                "violations": int(result["chronos_violations"]),
                "violation_rate": float(result["chronos_violations"] / steps),
                "mean_interval": float(np.mean(chronos_intervals)),
                "min_interval": float(np.min(chronos_intervals)),
            },
        },
    }


def write_reachability_results(
    output_path: str | Path,
    *,
    steps: int = 5000,
    step_sigma: float = 0.3,
    seed: int = 2026,
) -> dict[str, Any]:
    """Run the benchmark and write a JSON summary."""

    result = simulate_reachability(steps=steps, step_sigma=step_sigma, seed=seed)
    summary = summarize_reachability(result)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def plot_reachability(result: dict[str, Any], output_dir: str | Path) -> tuple[Path, Path]:
    """Save trajectory and interval-distribution plots."""

    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    trajectory_path = output / "causal_reachability_trajectories.png"
    intervals_path = output / "causal_reachability_intervals.png"

    traj_e = np.asarray(result["traj_e"], dtype=float)
    traj_c = np.asarray(result["traj_c"], dtype=float)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(traj_e[:, 0], traj_e[:, 1], alpha=0.7, label="Euclidean trajectory")
    ax.plot(traj_c[:, 0], traj_c[:, 1], alpha=0.7, label="Chronos causal trajectory")
    t_max = max(float(traj_c[:, 0].max()), float(traj_e[:, 0].max()), 1.0)
    t = np.linspace(0.0, t_max, 300)
    ax.plot(t, t, "k--", linewidth=1, label="light cone")
    ax.plot(t, -t, "k--", linewidth=1)
    ax.set_xlabel("time coordinate")
    ax.set_ylabel("space coordinate")
    ax.set_title("Reachability under Lorentzian causal constraint")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(trajectory_path, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(result["euclid_intervals"], bins=50, alpha=0.6, label="Euclidean intervals")
    ax.hist(result["chronos_intervals"], bins=50, alpha=0.6, label="Chronos intervals")
    ax.axvline(0.0, color="black", linestyle="--")
    ax.set_xlabel("dx.T @ G @ dx")
    ax.set_ylabel("count")
    ax.set_title("Signed interval distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(intervals_path, dpi=160)
    plt.close(fig)

    return trajectory_path, intervals_path
