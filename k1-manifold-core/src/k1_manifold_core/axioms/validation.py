"""Validation helpers for 2D signature statements."""

from __future__ import annotations

import numpy as np


Array = np.ndarray


def is_symmetric(G: Array, *, atol: float = 1e-10) -> bool:
    G = np.asarray(G, dtype=float)
    return G.shape == (2, 2) and bool(np.allclose(G, G.T, atol=atol))


def determinant(G: Array) -> float:
    return float(np.linalg.det(np.asarray(G, dtype=float)))


def is_non_degenerate(G: Array, *, atol: float = 1e-12) -> bool:
    return abs(determinant(G)) > atol


def inertia(G: Array, *, atol: float = 1e-10) -> tuple[int, int, int]:
    """Return ``(positive, negative, zero)`` eigenvalue counts."""

    values = np.linalg.eigvalsh(np.asarray(G, dtype=float))
    pos = int(np.sum(values > atol))
    neg = int(np.sum(values < -atol))
    zero = int(len(values) - pos - neg)
    return pos, neg, zero


def is_lorentzian(G: Array, *, atol: float = 1e-10) -> bool:
    return is_symmetric(G, atol=atol) and inertia(G, atol=atol) == (1, 1, 0)


def validate_det_negative(G: Array, *, atol: float = 1e-12) -> bool:
    """Check the 2D Lorentzian determinant condition ``det(G) < 0``."""

    return determinant(G) < -atol
