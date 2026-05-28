"""Law II dynamics: ``xdot = (J_G - D) grad V``."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from k1_manifold_core.geometry.symplectic import induced_generator


Array = np.ndarray
Gradient = Callable[[Array], Array]


def law_ii_matrix(G: Array, D: Array, *, alpha: float = 1.0) -> Array:
    """Return the Law II generator ``J_G - D``."""

    return induced_generator(G, alpha=alpha) - np.asarray(D, dtype=float)


def vector_field(x: Array, G: Array, D: Array, grad_V: Gradient, *, alpha: float = 1.0) -> Array:
    """Evaluate ``(J_G - D) grad_V(x)``."""

    return law_ii_matrix(G, D, alpha=alpha) @ np.asarray(grad_V(np.asarray(x, dtype=float)), dtype=float)


def k1_potential_gradient(x: Array, G: Array) -> Array:
    """Gradient of ``V = 1/2 (K - 1)^2`` with ``K = x.T G x``."""

    x = np.asarray(x, dtype=float)
    G = np.asarray(G, dtype=float)
    K = float(x.T @ G @ x)
    return 2.0 * (K - 1.0) * G @ x


def k_dot(x: Array, G: Array, D: Array, *, alpha: float = 1.0) -> float:
    """Time derivative of ``K`` under the K=1 potential."""

    x = np.asarray(x, dtype=float)
    xdot = vector_field(x, G, D, lambda y: k1_potential_gradient(y, G), alpha=alpha)
    return float(2.0 * x.T @ np.asarray(G, dtype=float) @ xdot)
