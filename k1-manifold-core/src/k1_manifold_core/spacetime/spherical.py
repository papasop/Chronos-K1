"""Static spherical-sector cost-structural functionals."""

from __future__ import annotations

from collections.abc import Callable


Scalar = Callable[[float], float]


def derivative(f: Scalar, r: float, *, h: float = 1e-5) -> float:
    return float((f(r + h) - f(r - h)) / (2.0 * h))


def second_derivative(f: Scalar, r: float, *, h: float = 1e-4) -> float:
    return float((f(r + h) - 2.0 * f(r) + f(r - h)) / (h * h))


def K1(f: Scalar, r: float) -> float:
    """Return ``K1 = f + 2 r f' + r^2 f'' / 2``."""

    return float(f(r) + 2.0 * r * derivative(f, r) + 0.5 * r * r * second_derivative(f, r))


def K2(f: Scalar, r: float) -> float:
    """Return ``K2 = f + r f'``."""

    return float(f(r) + r * derivative(f, r))


def schwarzschild_f(mass: float) -> Scalar:
    return lambda r: 1.0 - 2.0 * mass / r


def schwarzschild_de_sitter_f(mass: float, Lambda: float) -> Scalar:
    return lambda r: 1.0 - 2.0 * mass / r - Lambda * r * r / 3.0
