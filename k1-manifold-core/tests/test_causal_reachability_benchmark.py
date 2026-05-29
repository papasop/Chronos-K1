import numpy as np

from k1_manifold_core.benchmarks.benchmark_causal_reachability import (
    causal_project,
    causal_valid,
    interval,
    simulate_reachability,
    summarize_reachability,
)


def test_causal_project_maps_spacelike_or_past_displacements_into_future_cone():
    samples = [
        np.array([0.1, 1.0]),
        np.array([-0.1, 1.0]),
        np.array([-2.0, 0.5]),
        np.array([1.0, -3.0]),
    ]

    for sample in samples:
        projected = causal_project(sample)
        assert projected[0] >= 0.0
        assert causal_valid(projected)
        assert interval(projected) >= -1e-12


def test_causal_reachability_benchmark_reduces_violation_rate():
    result = simulate_reachability(steps=5000, step_sigma=0.3, seed=2026)
    summary = summarize_reachability(result)
    euclidean = summary["metrics"]["euclidean"]
    chronos = summary["metrics"]["chronos_causal_projection"]

    assert euclidean["violations"] > 0
    assert chronos["violations"] == 0
    assert chronos["violation_rate"] < euclidean["violation_rate"]
    assert chronos["min_interval"] >= -1e-12
