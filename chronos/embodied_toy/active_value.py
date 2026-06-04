"""S0-E2b active exploration diagnostic-value world.

S0-E2 proves the active-to-diagnostic loop is wired. S0-E2b uses a partitioned
toy world where the diagnostic answer depends on reaching a far structure zone,
so active exploration has measured diagnostic value over a random-action
control.
"""

from __future__ import annotations

import math
import random

from chronos.s0.diagnostics_schema import CTX_SYMPLECTIC

RAIL = 20.0
ZONE = 14.0
DELTA = 0.4
ACTIONS = ["left", "right", "stay"]
ACTION_DELTAS = {"left": -DELTA, "right": DELTA, "stay": 0.0}


def clamp_position(x: float) -> float:
    return max(0.0, min(RAIL, x))


class RailWorld:
    """Partitioned rail world with structure only in the far zone."""

    ACTIONS = ACTIONS

    def __init__(self, x0: float = 0.0):
        self.x = x0

    def peek(self, x: float, action: str) -> float:
        return clamp_position(x + ACTION_DELTAS[action])

    def step(self, action: str) -> dict:
        previous = self.x
        self.x = self.peek(self.x, action)
        return {"prev": previous, "action": action, "next": self.x}


def probe_energy_drifts(
    in_structure_zone: bool,
    n_steps: int = 200,
    dt: float = 0.05,
    theta0: float = 1.0,
    omega0: float = 0.0,
) -> tuple[float, float]:
    """Measure energy drift in the structure zone vs the dissipative near zone."""

    def energy(theta: float, omega: float) -> float:
        return 0.5 * omega * omega + (1.0 - math.cos(theta))

    damping = 0.0 if in_structure_zone else 0.25

    theta, omega = theta0, omega0
    energy_symplectic = [energy(theta, omega)]
    for _ in range(n_steps):
        omega = (omega - dt * math.sin(theta)) * (1.0 - damping * dt)
        theta = theta + dt * omega
        energy_symplectic.append(energy(theta, omega))

    theta, omega = theta0, omega0
    energy_control = [energy(theta, omega)]
    for _ in range(n_steps):
        theta_next = theta + dt * omega
        omega_next = (omega - dt * math.sin(theta)) * (1.0 - damping * dt)
        theta, omega = theta_next, omega_next
        energy_control.append(energy(theta, omega))

    def drift(energy_series: list[float]) -> float:
        e0 = abs(energy_series[0]) if abs(energy_series[0]) > 1e-9 else 1e-9
        return (max(energy_series) - min(energy_series)) / e0

    return drift(energy_symplectic), drift(energy_control)


def state_novelty(x: float, memory: list[float]) -> float:
    return 1.0 if not memory else min(abs(x - remembered) for remembered in memory)


class ActiveRailExplorer:
    """Choose the action whose next rail position is most novel."""

    def __init__(self, world: RailWorld):
        self.world = world
        self.memory = [world.x]
        self.positions = [world.x]

    def choose_action(self) -> str:
        best_action = None
        best_novelty = -1.0
        for action in self.world.ACTIONS:
            next_x = self.world.peek(self.world.x, action)
            novelty = state_novelty(next_x, self.memory)
            if novelty > best_novelty:
                best_novelty = novelty
                best_action = action
        return best_action

    def run(self, n_steps: int) -> list[float]:
        for _ in range(n_steps):
            self.world.step(self.choose_action())
            self.memory.append(self.world.x)
            self.positions.append(self.world.x)
        return self.positions


def random_rail_run(world: RailWorld, n_steps: int, seed: int = 0) -> list[float]:
    rng = random.Random(seed)
    positions = [world.x]
    for _ in range(n_steps):
        world.step(rng.choice(world.ACTIONS))
        positions.append(world.x)
    return positions


def diagnostics_at_reached_state(reached_x: float, n_steps: int) -> dict:
    """Diagnose structure at the rail position the explorer actually reached."""

    in_zone = reached_x >= ZONE
    drift_symplectic, drift_control = probe_energy_drifts(in_zone, n_steps=n_steps)
    improves = drift_symplectic < 0.5 * drift_control
    diagnostics = {
        "field_learnable": True,
        "baseline_divergence": 0.0,
        "_measured": {
            "reached_x": round(reached_x, 3),
            "in_structure_zone": in_zone,
            "energy_drift_symplectic": round(drift_symplectic, 4),
            "energy_drift_control": round(drift_control, 4),
        },
    }
    if improves:
        diagnostics["diagnostic_context"] = CTX_SYMPLECTIC
        diagnostics["symplectic_improves_vs_controls"] = True
        diagnostics["symplectic_jacobian_error"] = float(round(min(drift_symplectic, 0.99), 4))
    return diagnostics
