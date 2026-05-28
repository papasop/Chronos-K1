"""Small geodesic integration utilities."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


Array = np.ndarray
Christoffel = Callable[[Array], Array]


def geodesic_rhs(state: Array, christoffel: Christoffel) -> Array:
    """Return first-order geodesic RHS for ``state = [x, v]``."""

    state = np.asarray(state, dtype=float)
    n = state.size // 2
    x = state[:n]
    v = state[n:]
    gamma = np.asarray(christoffel(x), dtype=float)
    acceleration = -np.einsum("ijk,j,k->i", gamma, v, v)
    return np.concatenate([v, acceleration])


def euler_geodesic_step(state: Array, dt: float, christoffel: Christoffel) -> Array:
    """One explicit Euler step for simple local experiments."""

    return np.asarray(state, dtype=float) + dt * geodesic_rhs(state, christoffel)
