"""Lorentzian quadratic-form operations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from k1_manifold_core.axioms.validation import determinant, is_lorentzian


Array = np.ndarray


@dataclass(frozen=True)
class QuadraticForm2D:
    """A non-degenerate two-dimensional signed quadratic form."""

    G: Array

    def __post_init__(self) -> None:
        G = np.asarray(self.G, dtype=float)
        if G.shape != (2, 2):
            raise ValueError("QuadraticForm2D requires a 2x2 matrix")
        if not np.allclose(G, G.T):
            raise ValueError("G must be symmetric")
        object.__setattr__(self, "G", G)

    @property
    def det(self) -> float:
        return determinant(self.G)

    @property
    def inverse(self) -> Array:
        return np.linalg.inv(self.G)

    def K(self, x: Array) -> float:
        x = np.asarray(x, dtype=float)
        return float(x.T @ self.G @ x)

    def gradient_K(self, x: Array) -> Array:
        return 2.0 * self.G @ np.asarray(x, dtype=float)

    def is_lorentzian(self) -> bool:
        return is_lorentzian(self.G)

    def null_residual(self, x: Array) -> float:
        return abs(self.K(x))


def canonical_J() -> Array:
    return np.array([[0.0, 1.0], [-1.0, 0.0]])


def lorentzian_interval(G: Array, dx: Array) -> float:
    """Return the signed interval ``dx.T @ G @ dx``."""

    return QuadraticForm2D(G).K(dx)
