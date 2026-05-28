from k1_manifold_core.benchmarks.v0_3 import run_benchmark_v03


def test_benchmark_v03_reports_required_metrics():
    result = run_benchmark_v03(steps=3000, tail=500)
    metrics = result["metrics"]

    for model_name in ("euclidean_baseline", "chronos_k1_lorentzian"):
        model_metrics = metrics[model_name]
        assert set(model_metrics) == {
            "k_stability_tail_mae",
            "causal_violation_rate",
            "long_horizon_rollout_error",
        }
        assert all(value >= 0.0 for value in model_metrics.values())


def test_chronos_k1_outperforms_euclidean_baseline_on_k1_constraint_task():
    result = run_benchmark_v03(steps=10000, tail=1000)
    metrics = result["metrics"]
    baseline = metrics["euclidean_baseline"]
    chronos = metrics["chronos_k1_lorentzian"]

    assert chronos["k_stability_tail_mae"] < baseline["k_stability_tail_mae"]
    assert chronos["causal_violation_rate"] < baseline["causal_violation_rate"]
    assert chronos["long_horizon_rollout_error"] < baseline["long_horizon_rollout_error"]
