"""Latent transition models for the minimal Chronos-K1 world-model track."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from k1_manifold_core.world_model.latent import DEFAULT_G, project_to_k1


Array = np.ndarray


def _as_action_batch(actions: Array | None, count: int) -> Array:
    if actions is None:
        return np.zeros((count, 0), dtype=float)
    action_array = np.asarray(actions, dtype=float)
    if action_array.ndim == 1:
        action_array = action_array.reshape(-1, 1)
    if action_array.shape[0] != count:
        raise ValueError("actions and latent states must have the same batch size")
    return action_array


def transition_features(z: Array, actions: Array | None = None) -> Array:
    """Concatenate latent state and action features."""

    z_array = np.asarray(z, dtype=float)
    if z_array.ndim == 1:
        z_array = z_array.reshape(1, -1)
    action_array = _as_action_batch(actions, z_array.shape[0])
    return np.hstack([z_array, action_array])


@dataclass(frozen=True)
class LinearTransition:
    """Affine baseline transition ``z_next = [z, a] @ W + b``."""

    weights: Array
    bias: Array

    def predict(self, z: Array, actions: Array | None = None) -> Array:
        features = transition_features(z, actions)
        prediction = features @ self.weights + self.bias
        if np.asarray(z).ndim == 1:
            return prediction[0]
        return prediction


@dataclass(frozen=True)
class K1ProjectedTransition:
    """Wrap a transition with a ``K=1`` projection on its predicted latent state."""

    base: LinearTransition
    G: Array = DEFAULT_G

    def predict(self, z: Array, actions: Array | None = None) -> Array:
        raw = self.base.predict(z, actions)
        return project_to_k1(raw, self.G)


def fit_linear_transition(
    z: Array,
    actions: Array | None,
    targets: Array,
    *,
    ridge: float = 1e-6,
) -> LinearTransition:
    """Fit an affine one-step latent transition by ridge least squares."""

    features = transition_features(z, actions)
    targets = np.asarray(targets, dtype=float)
    if targets.ndim == 1:
        targets = targets.reshape(1, -1)
    design = np.hstack([features, np.ones((features.shape[0], 1))])
    penalty = ridge * np.eye(design.shape[1])
    penalty[-1, -1] = 0.0
    params = np.linalg.solve(design.T @ design + penalty, design.T @ targets)
    return LinearTransition(weights=params[:-1], bias=params[-1])


def rollout_transition(model: LinearTransition | K1ProjectedTransition, z0: Array, actions: Array) -> Array:
    """Roll out a one-step transition over an action sequence."""

    z = np.asarray(z0, dtype=float).copy()
    trajectory = [z.copy()]
    action_array = np.asarray(actions, dtype=float)
    if action_array.ndim == 1:
        action_array = action_array.reshape(-1, 1)
    for action in action_array:
        z = model.predict(z, action.reshape(1, -1))
        trajectory.append(z.copy())
    return np.vstack(trajectory)
