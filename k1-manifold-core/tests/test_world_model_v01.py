import numpy as np

from k1_manifold_core.benchmarks.world_model_v01 import run_world_model_v01
from k1_manifold_core.world_model.datasets import make_hyperbolic_world_model_dataset
from k1_manifold_core.world_model.latent import DEFAULT_G, k_values, project_to_k1
from k1_manifold_core.world_model.loss import k_constraint_loss


def test_project_to_k1_normalizes_positive_timelike_latents():
    z = np.array([[2.0, 0.5], [1.5, -0.2]])
    projected = project_to_k1(z, DEFAULT_G)

    assert np.allclose(k_values(projected, DEFAULT_G), 1.0)


def test_hyperbolic_dataset_clean_targets_live_on_k1():
    dataset = make_hyperbolic_world_model_dataset(seed=11, target_noise=0.02)

    assert np.max(np.abs(k_values(dataset.test_targets, DEFAULT_G) - 1.0)) < 1e-12
    assert k_constraint_loss(dataset.train_targets, DEFAULT_G) > 0.0


def test_world_model_v01_reports_k1_regularizer_metrics():
    result = run_world_model_v01()
    metrics = result["metrics"]
    baseline = metrics["euclidean_linear"]
    chronos = metrics["chronos_k1_projected"]

    for key in metrics["lower_is_better"]:
        assert key in baseline
        assert key in chronos

    assert chronos["mean_abs_k_drift"] < 1e-10
    assert chronos["mean_abs_k_drift"] < baseline["mean_abs_k_drift"]
    assert chronos["prediction_mse"] < baseline["prediction_mse"]
    assert chronos["long_horizon_rollout_mse"] < baseline["long_horizon_rollout_mse"]
