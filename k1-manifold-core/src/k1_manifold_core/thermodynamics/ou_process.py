"""Ornstein-Uhlenbeck reductions near K=1."""

from __future__ import annotations

import numpy as np


def stationary_variance(noise_sigma: float, restoring_rate: float) -> float:
    """Return OU stationary variance ``sigma^2 / (2 kappa)``."""

    if restoring_rate <= 0.0:
        raise ValueError("restoring_rate must be positive")
    return float(noise_sigma**2 / (2.0 * restoring_rate))


def k_space_noise_amplitude(G: np.ndarray, x_star: np.ndarray, noise_sigma: float) -> float:
    """Return ``sigma_K = 2 sigma ||G x_star||``."""

    return float(2.0 * noise_sigma * np.linalg.norm(np.asarray(G, dtype=float) @ np.asarray(x_star, dtype=float)))


def k_variance_at_critical_damping(noise_sigma: float, d_c: float) -> float:
    """Return the base-point-cancelled K variance ``sigma^2 / (2 d_c)``."""

    if d_c <= 0.0:
        raise ValueError("d_c must be positive")
    return float(noise_sigma**2 / (2.0 * d_c))
