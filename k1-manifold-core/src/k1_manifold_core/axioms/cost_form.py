"""Construction utilities for signed leading cost forms."""

from __future__ import annotations

import numpy as np


Array = np.ndarray


def symmetric_part(matrix: Array) -> Array:
    """Return the symmetric part of a square matrix."""

    matrix = np.asarray(matrix, dtype=float)
    return 0.5 * (matrix + matrix.T)


def leading_quadratic_form(hessian: Array) -> Array:
    """Return the symmetric leading quadratic coefficient form.

    The paper uses ``G`` as the signed leading form associated with the
    second-order expansion of ``dt_info^2``. Numerically, that means retaining
    the symmetric part of the Hessian-like coefficient.
    """

    G = symmetric_part(hessian)
    if G.shape != (2, 2):
        raise ValueError("the current core supports only 2x2 leading forms")
    return G


def diagonal_cost_form(temporal_scale: float, spatial_scale: float = 1.0) -> Array:
    """Create ``diag(temporal_scale^2, -spatial_scale^2)`` in cost-sign convention."""

    if temporal_scale <= 0.0 or spatial_scale <= 0.0:
        raise ValueError("scales must be positive")
    return np.diag([temporal_scale**2, -(spatial_scale**2)]).astype(float)
