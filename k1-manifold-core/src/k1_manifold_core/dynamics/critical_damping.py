"""Law III critical damping selection."""

from __future__ import annotations

import numpy as np

from k1_manifold_core.geometry.symplectic import spectral_threshold


Array = np.ndarray


def critical_damping(G: Array, *, alpha: float = 1.0) -> float:
    """Return ``d_c`` for the isotropic critical damping choice."""

    return spectral_threshold(G, alpha=alpha)


def critical_damping_matrix(G: Array, *, alpha: float = 1.0) -> Array:
    """Return ``D = d_c I``."""

    return critical_damping(G, alpha=alpha) * np.eye(2)


def local_restoring_rate(G: Array, x_star: Array, *, alpha: float = 1.0) -> float:
    """Return ``kappa_K(x*) = 4 d_c x*.T G^2 x*``."""

    G = np.asarray(G, dtype=float)
    x_star = np.asarray(x_star, dtype=float)
    return float(4.0 * critical_damping(G, alpha=alpha) * x_star.T @ (G @ G) @ x_star)
