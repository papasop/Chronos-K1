"""Toy latent sequence datasets for Chronos-K1 world-model experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class WorldModelDataset:
    """Flattened one-step data plus clean rollout targets."""

    train_z: Array
    train_actions: Array
    train_targets: Array
    test_z: Array
    test_actions: Array
    test_targets: Array
    rollout_initials: Array
    rollout_actions: Array
    rollout_targets: Array


def hyperbolic_state(theta: Array) -> Array:
    """Return ``[cosh(theta), sinh(theta)]``, which satisfies ``K=1``."""

    theta = np.asarray(theta, dtype=float)
    return np.stack([np.cosh(theta), np.sinh(theta)], axis=-1)


def hyperbolic_sequence(theta0: float, actions: Array) -> Array:
    """Generate a clean K=1 sequence under additive rapidity actions."""

    action_array = np.asarray(actions, dtype=float).reshape(-1)
    theta = theta0 + np.concatenate([[0.0], np.cumsum(action_array)])
    return hyperbolic_state(theta)


def make_hyperbolic_world_model_dataset(
    *,
    train_sequences: int = 48,
    test_sequences: int = 16,
    steps: int = 60,
    action: float = 0.035,
    target_noise: float = 0.015,
    seed: int = 2026,
) -> WorldModelDataset:
    """Return a small noisy one-step dataset with clean K=1 rollout targets."""

    rng = np.random.default_rng(seed)
    sequence_count = train_sequences + test_sequences
    theta0 = rng.uniform(-0.7, 0.7, size=sequence_count)
    action_sequences = np.full((sequence_count, steps, 1), action, dtype=float)
    clean_sequences = np.array(
        [hyperbolic_sequence(value, action_sequences[index, :, 0]) for index, value in enumerate(theta0)]
    )

    z_t = clean_sequences[:, :-1, :].reshape(-1, 2)
    actions = action_sequences.reshape(-1, 1)
    clean_targets = clean_sequences[:, 1:, :].reshape(-1, 2)
    radial_noise = 1.0 + target_noise * rng.standard_normal((clean_targets.shape[0], 1))
    noisy_targets = clean_targets * radial_noise

    train_count = train_sequences * steps
    train_z = z_t[:train_count]
    train_actions = actions[:train_count]
    train_targets = noisy_targets[:train_count]

    test_z = z_t[train_count:]
    test_actions = actions[train_count:]
    test_targets = clean_targets[train_count:]

    rollout_initials = clean_sequences[train_sequences:, 0, :]
    rollout_actions = action_sequences[train_sequences:]
    rollout_targets = clean_sequences[train_sequences:]
    return WorldModelDataset(
        train_z=train_z,
        train_actions=train_actions,
        train_targets=train_targets,
        test_z=test_z,
        test_actions=test_actions,
        test_targets=test_targets,
        rollout_initials=rollout_initials,
        rollout_actions=rollout_actions,
        rollout_targets=rollout_targets,
    )
