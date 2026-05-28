"""Symplectic generator induced by a non-degenerate 2D cost form."""

from __future__ import annotations

import numpy as np

from .lorentzian import canonical_J


Array = np.ndarray


def induced_generator(G: Array, *, alpha: float = 1.0) -> Array:
    """Return ``J_G = alpha * G^{-1} J``."""

    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    return alpha * np.linalg.inv(np.asarray(G, dtype=float)) @ canonical_J()


def spectral_threshold(G: Array, *, alpha: float = 1.0) -> float:
    """Return ``d_c = alpha * sqrt(-1 / det(G))`` for Lorentzian 2D forms."""

    det = float(np.linalg.det(np.asarray(G, dtype=float)))
    if det >= 0.0:
        raise ValueError("spectral_threshold requires det(G) < 0")
    return float(alpha * np.sqrt(-1.0 / det))


def generator_eigenvalues(G: Array, *, alpha: float = 1.0) -> Array:
    return np.linalg.eigvals(induced_generator(G, alpha=alpha))


def zero_eigenvalue_locus_invariant(G: Array, H_star: Array, *, d: float, alpha: float = 1.0) -> bool:
    """Check determinant invariance under right multiplication by invertible H_star."""

    JG = induced_generator(G, alpha=alpha)
    left = np.linalg.det((JG - d * np.eye(2)) @ np.asarray(H_star, dtype=float))
    right = np.linalg.det(JG - d * np.eye(2)) * np.linalg.det(np.asarray(H_star, dtype=float))
    return bool(np.allclose(left, right))
