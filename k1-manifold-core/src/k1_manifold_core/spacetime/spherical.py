"""Static spherical-sector cost-structural functionals.

The formulas here are intentionally direct: callers supply ``f``, ``f'``,
and ``f''`` so tests can use symbolic differentiation instead of finite
differences.
"""

from __future__ import annotations

from typing import Any


def K1(f: Any, fp: Any, fpp: Any, r: Any) -> Any:
    """Return ``K1 = f + 2 r f' + (r^2 / 2) f''``."""

    return f + 2 * r * fp + (r**2 / 2) * fpp


def K2(f: Any, fp: Any, r: Any) -> Any:
    """Return ``K2 = f + r f' = d(r f) / dr``."""

    return f + r * fp


def ricci_components(f: Any, fp: Any, fpp: Any, r: Any) -> dict[str, Any]:
    """Return nonzero Ricci components for the static spherical ansatz.

    The metric convention is

    ``ds^2 = -f(r) dt^2 + f(r)^-1 dr^2 + r^2 dOmega^2``.

    Returned keys are ``R_tt``, ``R_rr``, and ``R_theta_theta``. The
    spherical symmetry component is ``R_phi_phi = sin(theta)^2 R_theta_theta``.
    """

    radial_block = fpp + 2 * fp / r
    return {
        "R_tt": f * radial_block / 2,
        "R_rr": -radial_block / (2 * f),
        "R_theta_theta": 1 - f - r * fp,
    }


def scalar_curvature(f: Any, fp: Any, fpp: Any, r: Any) -> Any:
    """Return scalar curvature for the static spherical ansatz."""

    return -fpp - 4 * fp / r + 2 * (1 - f) / (r**2)


def schwarzschild_f(mass: Any, r: Any) -> Any:
    """Return ``f(r)=1-2M/r``."""

    return 1 - 2 * mass / r


def reissner_nordstrom_f(mass: Any, charge: Any, r: Any) -> Any:
    """Return ``f(r)=1-2M/r+Q^2/r^2``."""

    return 1 - 2 * mass / r + charge**2 / r**2


def schwarzschild_de_sitter_f(mass: Any, Lambda: Any, r: Any) -> Any:
    """Return ``f(r)=1-2M/r-Lambda r^2/3``."""

    return 1 - 2 * mass / r - Lambda * r**2 / 3
