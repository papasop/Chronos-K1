"""Small deterministic toy simulations for S0-E1.

These simulations are deliberately tiny and stdlib-only. They are not robotics
and not physics certification; they provide moving toy trajectories from which
diagnostic extractors can measure simple signals for S0.
"""

from __future__ import annotations

import math


def pendulum_energy(theta: float, omega: float) -> float:
    return 0.5 * omega * omega + (1.0 - math.cos(theta))


def simulate_pendulum(n_steps: int = 400, dt: float = 0.05, theta0: float = 1.0, omega0: float = 0.0) -> dict:
    """Simulate a pendulum with symplectic and explicit-Euler integrators."""

    theta, omega = theta0, omega0
    energy_symplectic = [pendulum_energy(theta, omega)]
    for _ in range(n_steps):
        omega = omega - dt * math.sin(theta)
        theta = theta + dt * omega
        energy_symplectic.append(pendulum_energy(theta, omega))

    theta, omega = theta0, omega0
    energy_control = [pendulum_energy(theta, omega)]
    for _ in range(n_steps):
        theta_next = theta + dt * omega
        omega_next = omega - dt * math.sin(theta)
        theta, omega = theta_next, omega_next
        energy_control.append(pendulum_energy(theta, omega))

    return {
        "name": "pendulum",
        "energy_symplectic": energy_symplectic,
        "energy_control": energy_control,
        "dt": dt,
        "n_steps": n_steps,
    }


def simulate_contact(n_steps: int = 100, dt: float = 0.05, x0: float = 1.0, v0: float = -1.0) -> dict:
    """Simulate a 1D ball approaching a wall and record contact events."""

    x, v = x0, v0
    positions = [x]
    events = []
    for step in range(n_steps):
        x = x + dt * v
        if x <= 0.0 and v < 0.0:
            x = -x
            v = -v
            events.append(step)
        positions.append(x)
    return {
        "name": "contact",
        "positions": positions,
        "events": events,
        "dt": dt,
        "n_steps": n_steps,
    }


def simulate_object_persistence_fail(n_steps: int = 50, lose_at: int = 20) -> dict:
    """Simulate a tracked object whose id disappears partway through."""

    track_ids = [1 if step < lose_at else None for step in range(n_steps)]
    return {
        "name": "object_persistence",
        "track_ids": track_ids,
        "expected_id": 1,
        "n_steps": n_steps,
    }
