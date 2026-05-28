"""Axiom R/E/T helpers for two-dimensional leading cost forms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class RealizabilityWitness:
    """A small numerical witness for the realizability layer."""

    direction: Array
    cost_ratio: float


def positive_part_cost_squared(G: Array, dx: Array) -> float:
    """Return ``max(dx.T @ G @ dx, 0)``."""

    value = float(np.asarray(dx, dtype=float).T @ np.asarray(G, dtype=float) @ np.asarray(dx, dtype=float))
    return max(value, 0.0)


def cost(G: Array, dx: Array) -> float:
    """Primitive positive-part cost induced by a signed leading form."""

    return float(np.sqrt(positive_part_cost_squared(G, dx)))


def normalized_direction(dx: Array) -> Array:
    """Return the Euclidean unit direction of ``dx``."""

    dx = np.asarray(dx, dtype=float)
    norm = float(np.linalg.norm(dx))
    if norm == 0.0:
        raise ValueError("zero displacement has no normalized direction")
    return dx / norm


def axiom_t_holds(G: Array, *, min_temporal_cost: float = 0.0) -> bool:
    """Check the leading-form version of Axiom T: ``Q(e_t) > min``."""

    G = np.asarray(G, dtype=float)
    return float(G[0, 0]) > min_temporal_cost


def find_zero_threshold_witness(
    G: Array,
    *,
    samples: int = 2048,
    tolerance: float = 1e-8,
) -> Optional[RealizabilityWitness]:
    """Search forward directions with nonzero spatial part and near-zero cost ratio.

    This is a numerical proxy for Axiom R. It scans the forward unit semicircle
    and returns a direction whose positive-part cost is below ``tolerance``.
    """

    G = np.asarray(G, dtype=float)
    angles = np.linspace(-np.pi / 2.0 + 1e-6, np.pi / 2.0 - 1e-6, samples)
    best: Optional[RealizabilityWitness] = None
    for theta in angles:
        direction = np.array([np.cos(theta), np.sin(theta)], dtype=float)
        if direction[0] <= 0.0 or abs(direction[1]) <= 1e-12:
            continue
        ratio = cost(G, direction)
        if best is None or ratio < best.cost_ratio:
            best = RealizabilityWitness(direction=direction, cost_ratio=ratio)
    if best is not None and best.cost_ratio <= tolerance:
        return best
    return None


def realizes_lorentzian_signature(G: Array) -> bool:
    """Return whether R/E/T plus nondegeneracy select Lorentzian type numerically."""

    from .validation import is_lorentzian, is_non_degenerate

    return is_non_degenerate(G) and axiom_t_holds(G) and find_zero_threshold_witness(G) is not None and is_lorentzian(G)
