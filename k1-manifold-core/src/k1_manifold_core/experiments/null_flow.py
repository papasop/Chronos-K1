"""Null-flow recovery scaling utilities.

This module is a small numerical experiment around a rank-one null flow. It
checks algebraic invariance and a simple inverse-square recovery-time scaling.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


Array = np.ndarray
NULL_DIRECTION = np.array([1.0, 1.0])
LEAF_COVECTOR = np.array([1.0, -1.0])


def null_flow_matrix() -> Array:
    """Return ``A_c = n m^T`` with rank one and ``m^T n = 0``."""

    return np.outer(NULL_DIRECTION, LEAF_COVECTOR)


def leaf_coordinate(x: Array) -> float:
    """Return the conserved transverse coordinate ``c = m^T x``."""

    return float(LEAF_COVECTOR @ np.asarray(x, dtype=float))


def integrate_null_flow(x0: Array, *, dt: float = 1e-3, steps: int = 1000) -> Array:
    """Integrate ``xdot = A_c x`` with explicit Euler.

    The conserved coordinate is exact up to floating-point accumulation because
    ``m^T A_c = 0``.
    """

    A_c = null_flow_matrix()
    x = np.asarray(x0, dtype=float).copy()
    trajectory = [x.copy()]
    for _ in range(steps):
        x = x + dt * (A_c @ x)
        trajectory.append(x.copy())
    return np.vstack(trajectory)


def leaf_box_probability(radius: float, *, samples: int = 200_000, seed: int = 2026) -> float:
    """Estimate probability of a small square leaf chart of side ``2r``.

    Points are sampled uniformly from the unit square. The chart region
    ``u <= 2r`` and ``v <= 2r`` has ideal probability ``4r^2`` for
    ``0 <= r <= 0.5``.
    """

    if radius < 0.0 or radius > 0.5:
        raise ValueError("radius must lie in [0, 0.5]")
    rng = np.random.default_rng(seed)
    points = rng.random((samples, 2))
    hits = (points[:, 0] <= 2.0 * radius) & (points[:, 1] <= 2.0 * radius)
    return float(np.mean(hits))


def ideal_leaf_probability(radius: float) -> float:
    """Return the ideal small-leaf probability ``4r^2``."""

    if radius < 0.0 or radius > 0.5:
        raise ValueError("radius must lie in [0, 0.5]")
    return 4.0 * radius * radius


def recovery_time(epsilon: float, *, initial_error: float = 1.0, tolerance: float = 1e-3) -> float:
    """Return recovery time for ``ydot = -epsilon^2 y`` to reach tolerance."""

    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    if initial_error <= 0.0 or tolerance <= 0.0 or tolerance >= initial_error:
        raise ValueError("require 0 < tolerance < initial_error")
    return float(np.log(initial_error / tolerance) / (epsilon * epsilon))


def recovery_scaling_slope(epsilons: Iterable[float]) -> float:
    """Fit the log-log slope of recovery time versus epsilon."""

    eps = np.asarray(list(epsilons), dtype=float)
    times = np.array([recovery_time(value) for value in eps])
    slope, _intercept = np.polyfit(np.log(eps), np.log(times), 1)
    return float(slope)
