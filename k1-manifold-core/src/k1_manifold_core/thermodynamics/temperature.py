"""Effective and Tolman temperature formulas."""

from __future__ import annotations

import numpy as np


def effective_temperature(noise_sigma: float, d_c: float) -> float:
    """Return ``T_eff = sigma^2 / (2 d_c)``."""

    if d_c <= 0.0:
        raise ValueError("d_c must be positive")
    return float(noise_sigma**2 / (2.0 * d_c))


def hawking_temperature(kappa: float, *, hbar: float = 1.0) -> float:
    if kappa <= 0.0:
        raise ValueError("kappa must be positive")
    return float(hbar * kappa / (2.0 * np.pi))


def tolman_temperature(proper_distance: float, *, alpha: float = 1.0, hbar: float = 1.0) -> float:
    if proper_distance <= 0.0:
        raise ValueError("proper_distance must be positive")
    return float(alpha * hbar / (2.0 * np.pi * proper_distance))


def noise_for_tolman_identification(d_c: float, T_tol: float) -> float:
    """Return ``sigma^2`` required by ``T_eff = T_tol``."""

    if d_c <= 0.0 or T_tol < 0.0:
        raise ValueError("invalid d_c or temperature")
    return float(2.0 * d_c * T_tol)
