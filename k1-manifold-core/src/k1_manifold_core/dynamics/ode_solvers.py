"""Small ODE solvers for reproducible examples."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


Array = np.ndarray
RHS = Callable[[Array, float], Array]


def rk4_step(rhs: RHS, y: Array, t: float, dt: float) -> Array:
    y = np.asarray(y, dtype=float)
    k1 = rhs(y, t)
    k2 = rhs(y + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = rhs(y + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = rhs(y + dt * k3, t + dt)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def integrate(rhs: RHS, y0: Array, *, t0: float = 0.0, dt: float = 1e-2, steps: int = 1000) -> Array:
    """Return trajectory including the initial state."""

    y = np.asarray(y0, dtype=float)
    trajectory = [y.copy()]
    t = t0
    for _ in range(steps):
        y = rk4_step(rhs, y, t, dt)
        t += dt
        trajectory.append(y.copy())
    return np.vstack(trajectory)
