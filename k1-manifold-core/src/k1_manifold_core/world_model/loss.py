"""Loss and metric helpers for latent world-model benchmarks."""

from __future__ import annotations

import numpy as np

from k1_manifold_core.world_model.latent import DEFAULT_G, k_constraint_error


Array = np.ndarray


def prediction_mse(predicted: Array, target: Array) -> float:
    """Return mean squared one-step prediction error."""

    diff = np.asarray(predicted, dtype=float) - np.asarray(target, dtype=float)
    return float(np.mean(np.sum(diff * diff, axis=-1)))


def k_constraint_loss(z: Array, G: Array = DEFAULT_G) -> float:
    """Return mean squared ``K(z)-1`` constraint loss."""

    error = k_constraint_error(z, G)
    return float(np.mean(error * error))


def rollout_mse(rollout: Array, target_rollout: Array) -> float:
    """Return mean squared long-horizon rollout error."""

    return prediction_mse(rollout, target_rollout)


def mean_abs_k_drift(z: Array, G: Array = DEFAULT_G) -> float:
    """Return mean absolute ``K(z)-1`` over a trajectory or batch."""

    return float(np.mean(np.abs(k_constraint_error(z, G))))
