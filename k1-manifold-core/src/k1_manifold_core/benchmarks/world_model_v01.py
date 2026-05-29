"""World-model v0.1 latent K=1 prediction benchmark.

This is a narrow latent-state benchmark: fit a small affine transition on a
toy K=1 sequence dataset, then compare the raw Euclidean transition with the
same transition wrapped by a K=1 projection. It is a regularizer smoke test,
not a claim about large world models.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from k1_manifold_core.world_model.datasets import make_hyperbolic_world_model_dataset
from k1_manifold_core.world_model.latent import DEFAULT_G
from k1_manifold_core.world_model.loss import mean_abs_k_drift, prediction_mse, rollout_mse
from k1_manifold_core.world_model.transition import (
    K1ProjectedTransition,
    fit_linear_transition,
    rollout_transition,
)


Array = np.ndarray


def _rollout_metrics(model, dataset) -> tuple[float, float]:
    errors = []
    drifts = []
    for z0, actions, target in zip(dataset.rollout_initials, dataset.rollout_actions, dataset.rollout_targets):
        rollout = rollout_transition(model, z0, actions)
        errors.append(rollout_mse(rollout, target))
        drifts.append(mean_abs_k_drift(rollout, DEFAULT_G))
    return float(np.mean(errors)), float(np.mean(drifts))


def run_world_model_v01(
    *,
    seed: int = 2026,
    target_noise: float = 0.015,
    steps: int = 60,
    action: float = 0.035,
) -> dict[str, object]:
    """Run the deterministic latent K=1 world-model benchmark."""

    dataset = make_hyperbolic_world_model_dataset(
        seed=seed,
        target_noise=target_noise,
        steps=steps,
        action=action,
    )
    baseline = fit_linear_transition(dataset.train_z, dataset.train_actions, dataset.train_targets)
    chronos = K1ProjectedTransition(baseline, DEFAULT_G)

    baseline_one_step = baseline.predict(dataset.test_z, dataset.test_actions)
    chronos_one_step = chronos.predict(dataset.test_z, dataset.test_actions)
    baseline_rollout_error, baseline_k_drift = _rollout_metrics(baseline, dataset)
    chronos_rollout_error, chronos_k_drift = _rollout_metrics(chronos, dataset)

    return {
        "benchmark": "world_model_v01_latent_k1_prediction",
        "description": "Affine latent transition vs the same transition with a K=1 projection regularizer on a toy hyperbolic sequence dataset.",
        "scope": "Minimal latent regularizer benchmark; not a large world-model claim.",
        "parameters": {
            "seed": seed,
            "target_noise": target_noise,
            "steps": steps,
            "action": action,
            "G": DEFAULT_G.tolist(),
        },
        "metrics": {
            "lower_is_better": [
                "prediction_mse",
                "long_horizon_rollout_mse",
                "mean_abs_k_drift",
            ],
            "euclidean_linear": {
                "prediction_mse": prediction_mse(baseline_one_step, dataset.test_targets),
                "long_horizon_rollout_mse": baseline_rollout_error,
                "mean_abs_k_drift": baseline_k_drift,
            },
            "chronos_k1_projected": {
                "prediction_mse": prediction_mse(chronos_one_step, dataset.test_targets),
                "long_horizon_rollout_mse": chronos_rollout_error,
                "mean_abs_k_drift": chronos_k_drift,
            },
        },
    }


def write_world_model_v01(path: str | Path) -> dict[str, object]:
    """Run and write ``world_model_v01.json``."""

    result = run_world_model_v01()
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    default_path = Path(__file__).resolve().parents[3] / "results" / "world_model_v01.json"
    data = write_world_model_v01(default_path)
    print(json.dumps(data["metrics"], indent=2))
    print(f"saved results = {default_path}")
