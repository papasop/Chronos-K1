from k1_manifold_core.benchmarks.v0_3 import run_benchmark_v03


def test_benchmark_v03_reports_required_metrics():
    result = run_benchmark_v03(steps=1000, tail=200)
    metrics = result["metrics"]

    for model_name in ("euclidean_gradient", "chronos_k1_lorentzian"):
        model_metrics = metrics[model_name]
        assert set(model_metrics) == {
            "mean_abs_k_error_tail",
            "mean_potential_tail",
            "mean_recovery_time",
            "long_horizon_rollout_error",
        }
        assert all(value >= 0.0 for value in model_metrics.values())


def test_chronos_k1_outperforms_euclidean_gradient_under_shared_noise():
    result = run_benchmark_v03(steps=5000, tail=500, noise_scale=0.05, seed=7)
    metrics = result["metrics"]
    baseline = metrics["euclidean_gradient"]
    chronos = metrics["chronos_k1_lorentzian"]

    assert chronos["mean_abs_k_error_tail"] < baseline["mean_abs_k_error_tail"]
    assert chronos["mean_potential_tail"] < baseline["mean_potential_tail"]
    assert chronos["mean_recovery_time"] < baseline["mean_recovery_time"]
    assert chronos["long_horizon_rollout_error"] < baseline["long_horizon_rollout_error"]
