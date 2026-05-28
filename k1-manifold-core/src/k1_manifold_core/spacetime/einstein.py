"""Einstein-sector formula checks for static spherical metrics."""

from __future__ import annotations

from collections.abc import Callable

from .spherical import K1, K2, derivative, second_derivative


Scalar = Callable[[float], float]


def scalar_curvature(f: Scalar, r: float) -> float:
    """Return scalar curvature for ``ds^2=-f dt^2+f^-1 dr^2+r^2 dOmega^2``."""

    fp = derivative(f, r)
    fpp = second_derivative(f, r)
    return float(-fpp - 4.0 * fp / r + 2.0 * (1.0 - f(r)) / (r * r))


def ricci_theta_theta(f: Scalar, r: float) -> float:
    return float(1.0 - K2(f, r))


def ricci_tt(f: Scalar, r: float) -> float:
    return float((f(r) / (r * r)) * (K1(f, r) - K2(f, r)))


def vacuum_residual(f: Scalar, r: float) -> float:
    """Return a compact residual for the spherical-vacuum equivalence."""

    return abs(K1(f, r) - 1.0) + abs(K2(f, r) - 1.0) + abs(ricci_theta_theta(f, r)) + abs(ricci_tt(f, r))
