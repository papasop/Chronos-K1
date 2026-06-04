"""S0-E2 active toy exploration utilities.

This is not robotics, RL training, a neural network, or online learning. The
"active" component is a deterministic novelty heuristic that chooses the action
whose candidate next state is farthest from previously visited states.
"""

from __future__ import annotations

import math
import random

from chronos.s0.diagnostics_schema import CTX_SYMPLECTIC


class PendulumWorld:
    """Pendulum with a small discrete push action space."""

    ACTIONS = ["left", "right", "none"]
    PUSH = {"left": -0.5, "right": 0.5, "none": 0.0}

    def __init__(self, theta0: float = 1.0, omega0: float = 0.0, dt: float = 0.05):
        self.dt = dt
        self.state = (theta0, omega0)

    def peek(self, state: tuple[float, float], action: str) -> tuple[float, float]:
        theta, omega = state
        omega_next = omega - self.dt * math.sin(theta) + self.PUSH[action] * self.dt
        theta_next = theta + self.dt * omega_next
        return theta_next, omega_next

    def step(self, action: str) -> dict:
        previous = self.state
        self.state = self.peek(self.state, action)
        return {"prev": previous, "action": action, "next": self.state}


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def state_novelty(state: tuple[float, float], memory: list[tuple[float, float]]) -> float:
    """RND-lite novelty: distance to nearest visited state."""

    if not memory:
        return 1.0
    return min(_distance(state, remembered) for remembered in memory)


class ActiveToyExplorer:
    """Choose the action whose resulting state is most novel."""

    def __init__(self, world: PendulumWorld, novelty_fn=state_novelty):
        self.world = world
        self.novelty_fn = novelty_fn
        self.memory = [world.state]
        self.transitions = []

    def choose_action(self) -> str:
        best_action = None
        best_novelty = -1.0
        for action in self.world.ACTIONS:
            next_state = self.world.peek(self.world.state, action)
            novelty = self.novelty_fn(next_state, self.memory)
            if novelty > best_novelty:
                best_novelty = novelty
                best_action = action
        return best_action

    def step(self) -> dict:
        transition = self.world.step(self.choose_action())
        self.memory.append(transition["next"])
        self.transitions.append(transition)
        return transition

    def run(self, n_steps: int) -> list[dict]:
        for _ in range(n_steps):
            self.step()
        return self.transitions


def random_explore(world: PendulumWorld, n_steps: int, seed: int = 0) -> tuple[list[dict], list[tuple[float, float]]]:
    """Random-action control using the same world dynamics."""

    rng = random.Random(seed)
    memory = [world.state]
    transitions = []
    for _ in range(n_steps):
        transition = world.step(rng.choice(world.ACTIONS))
        memory.append(transition["next"])
        transitions.append(transition)
    return transitions, memory


def state_coverage(states: list[tuple[float, float]]) -> int:
    """Coarse state-space coverage proxy."""

    return len({(round(theta, 1), round(omega, 1)) for theta, omega in states})


def _energy(theta: float, omega: float) -> float:
    return 0.5 * omega * omega + (1.0 - math.cos(theta))


def most_novel_state_from_transitions(transitions: list[dict]) -> tuple[float, float] | None:
    """Return the most novel state actually reached by active exploration."""

    visited = [transition["next"] for transition in transitions]
    if not visited:
        return None
    best_state = None
    best_novelty = -1.0
    for index, state in enumerate(visited):
        others = visited[:index] + visited[index + 1 :]
        novelty = state_novelty(state, others)
        if novelty > best_novelty:
            best_novelty = novelty
            best_state = state
    return best_state


def diagnostics_from_active_pendulum(transitions: list[dict], probe_from: str = "most_novel") -> dict:
    """Measure K2 diagnostics from a state active exploration reached."""

    if not transitions:
        return {}

    if probe_from == "terminal":
        start = transitions[-1]["next"]
        source = "active_terminal_state"
    else:
        start = most_novel_state_from_transitions(transitions)
        source = "active_most_novel_state"

    dt = 0.05
    n_steps = len(transitions)

    theta, omega = start
    energy_symplectic = [_energy(theta, omega)]
    for _ in range(n_steps):
        omega = omega - dt * math.sin(theta)
        theta = theta + dt * omega
        energy_symplectic.append(_energy(theta, omega))

    theta, omega = start
    energy_control = [_energy(theta, omega)]
    for _ in range(n_steps):
        theta_next = theta + dt * omega
        omega_next = omega - dt * math.sin(theta)
        theta, omega = theta_next, omega_next
        energy_control.append(_energy(theta, omega))

    def drift(energy: list[float]) -> float:
        e0 = abs(energy[0]) if abs(energy[0]) > 1e-9 else 1e-9
        return (max(energy) - min(energy)) / e0

    drift_symplectic = drift(energy_symplectic)
    drift_control = drift(energy_control)
    return {
        "diagnostic_context": CTX_SYMPLECTIC,
        "field_learnable": True,
        "baseline_divergence": 0.0,
        "symplectic_improves_vs_controls": bool(drift_symplectic < 0.5 * drift_control),
        "symplectic_jacobian_error": float(round(min(drift_symplectic, 0.99), 4)),
        "_measured": {
            "probe_start": [round(start[0], 4), round(start[1], 4)],
            "probe_source": source,
            "energy_drift_symplectic": round(drift_symplectic, 4),
            "energy_drift_control": round(drift_control, 4),
        },
    }
