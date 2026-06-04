"""Portable mirror for Colab.

Canonical implementation lives in chronos/embodied_toy/ and chronos/s0/.

S0-E2 tests:

    active exploration -> diagnostic probe -> S0 recommendation

This is not robotics, not RL training, not a neural network, and not online
learning. Novelty is a deterministic distance-to-visited-states heuristic.
"""

from __future__ import annotations

import json
import math
import random
import sys

try:
    from chronos.embodied_toy.run_active_suite import run_active_suite

    S0_SOURCE = "chronos.embodied_toy (repo)"
except Exception:
    from dataclasses import asdict, dataclass
    from typing import Any

    K2_SYMPLECTIC = "K2_SYMPLECTIC"
    ACT_CONTINUE = "continue"
    ACT_DO_NOT_PROMOTE = "do_not_promote"

    @dataclass(frozen=True)
    class Recommendation:
        candidate_family: str
        confidence: str
        reason: str
        next_vpsl_gate: str | None
        allowed_action: str

        def to_dict(self) -> dict[str, Any]:
            return asdict(self)

    def recommend(diagnostics: dict[str, Any] | None) -> Recommendation:
        d = diagnostics or {}
        if d.get("symplectic_improves_vs_controls") is True:
            return Recommendation(
                K2_SYMPLECTIC,
                "high",
                "Symplectic mechanism diagnostics improve against controls.",
                "mechanism",
                ACT_CONTINUE,
            )
        return Recommendation("UNRESOLVED", "low", "Diagnostics present but none decisive.", None, ACT_DO_NOT_PROMOTE)

    class PendulumWorld:
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
        if not memory:
            return 1.0
        return min(_distance(state, remembered) for remembered in memory)

    class ActiveToyExplorer:
        def __init__(self, world: PendulumWorld):
            self.world = world
            self.memory = [world.state]
            self.transitions = []

        def choose_action(self) -> str:
            best_action = None
            best_novelty = -1.0
            for action in self.world.ACTIONS:
                next_state = self.world.peek(self.world.state, action)
                novelty = state_novelty(next_state, self.memory)
                if novelty > best_novelty:
                    best_action = action
                    best_novelty = novelty
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
        rng = random.Random(seed)
        memory = [world.state]
        transitions = []
        for _ in range(n_steps):
            transition = world.step(rng.choice(world.ACTIONS))
            memory.append(transition["next"])
            transitions.append(transition)
        return transitions, memory

    def state_coverage(states: list[tuple[float, float]]) -> int:
        return len({(round(theta, 1), round(omega, 1)) for theta, omega in states})

    def _energy(theta: float, omega: float) -> float:
        return 0.5 * omega * omega + (1.0 - math.cos(theta))

    def most_novel_state_from_transitions(transitions: list[dict]) -> tuple[float, float] | None:
        visited = [transition["next"] for transition in transitions]
        best_state = None
        best_novelty = -1.0
        for index, state in enumerate(visited):
            others = visited[:index] + visited[index + 1 :]
            novelty = state_novelty(state, others)
            if novelty > best_novelty:
                best_state = state
                best_novelty = novelty
        return best_state

    def diagnostics_from_active_pendulum(transitions: list[dict], probe_from: str = "most_novel") -> dict:
        if probe_from == "terminal":
            start = transitions[-1]["next"]
            source = "active_terminal_state"
        else:
            start = most_novel_state_from_transitions(transitions)
            source = "active_most_novel_state"
        dt = 0.05
        theta, omega = start
        energy_symplectic = [_energy(theta, omega)]
        for _ in range(len(transitions)):
            omega = omega - dt * math.sin(theta)
            theta = theta + dt * omega
            energy_symplectic.append(_energy(theta, omega))
        theta, omega = start
        energy_control = [_energy(theta, omega)]
        for _ in range(len(transitions)):
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
            "diagnostic_context": "K2_SYMPLECTIC",
            "field_learnable": True,
            "baseline_divergence": 0.0,
            "symplectic_improves_vs_controls": drift_symplectic < 0.5 * drift_control,
            "symplectic_jacobian_error": float(round(min(drift_symplectic, 0.99), 4)),
            "_measured": {
                "probe_start": [round(start[0], 4), round(start[1], 4)],
                "probe_source": source,
                "energy_drift_symplectic": round(drift_symplectic, 4),
                "energy_drift_control": round(drift_control, 4),
            },
        }

    def _strip_private(diagnostics: dict) -> dict:
        return {key: value for key, value in diagnostics.items() if not key.startswith("_")}

    def run_active_suite(n_steps: int = 120, *, quiet: bool = False) -> dict:
        active_world = PendulumWorld()
        explorer = ActiveToyExplorer(active_world)
        transitions = explorer.run(n_steps)
        coverage_active = state_coverage(explorer.memory)
        random_world = PendulumWorld()
        _, random_memory = random_explore(random_world, n_steps, seed=0)
        coverage_random = state_coverage(random_memory)
        diagnostics = diagnostics_from_active_pendulum(transitions)
        recommendation = recommend(_strip_private(diagnostics)).to_dict()
        result = {
            "recommendation": recommendation,
            "diagnostics": diagnostics,
            "coverage_active": coverage_active,
            "coverage_random": coverage_random,
            "coverage_ratio": round(coverage_active / max(1, coverage_random), 3),
        }
        if not quiet:
            print(f"active exploration -> {recommendation['candidate_family']} / {recommendation['allowed_action']}")
            print(f"measured: {diagnostics.get('_measured')}")
            print(
                f"state-space coverage: active={coverage_active} cells vs random={coverage_random} "
                f"cells ({result['coverage_ratio']}x)"
            )
        return result

    S0_SOURCE = "embedded fallback"


def _tests() -> int:
    result = run_active_suite(quiet=True)
    count = 0
    assert result["recommendation"]["candidate_family"] == "K2_SYMPLECTIC"
    count += 1
    assert result["recommendation"]["allowed_action"] == "continue"
    count += 1
    measured = result["diagnostics"]["_measured"]
    assert measured["energy_drift_symplectic"] < measured["energy_drift_control"]
    count += 1
    assert result["diagnostics"]["symplectic_improves_vs_controls"] is True
    count += 1
    assert result["coverage_active"] > result["coverage_random"]
    count += 1
    assert state_novelty((0.0, 0.0), []) == 1.0
    count += 1
    assert state_novelty((0.0, 0.0), [(0.0, 0.0)]) == 0.0
    count += 1
    assert state_novelty((1.0, 0.0), [(0.0, 0.0)]) > state_novelty((0.1, 0.0), [(0.0, 0.0)])
    count += 1
    rerun = run_active_suite(quiet=True)
    assert result["recommendation"] == rerun["recommendation"]
    count += 1
    assert result["coverage_active"] == rerun["coverage_active"]
    count += 1
    assert result["recommendation"]["allowed_action"] in {"continue", "archive", "do_not_promote"}
    count += 1
    world = PendulumWorld()
    explorer = ActiveToyExplorer(world)
    transitions = explorer.run(120)
    diagnostics = diagnostics_from_active_pendulum(transitions)
    expected = most_novel_state_from_transitions(transitions)
    assert diagnostics["_measured"]["probe_source"] == "active_most_novel_state"
    count += 1
    assert diagnostics["_measured"]["probe_start"] == [round(expected[0], 4), round(expected[1], 4)]
    count += 1
    assert diagnostics["_measured"]["probe_start"] != [1.0, 0.0]
    count += 1
    terminal_diagnostics = diagnostics_from_active_pendulum(transitions, probe_from="terminal")
    terminal = transitions[-1]["next"]
    assert terminal_diagnostics["_measured"]["probe_source"] == "active_terminal_state"
    count += 1
    assert terminal_diagnostics["_measured"]["probe_start"] == [round(terminal[0], 4), round(terminal[1], 4)]
    count += 1
    return count


if __name__ == "__main__":
    quiet = "--quiet" in sys.argv
    as_json = "--json" in sys.argv
    if not quiet:
        print(f"=== S0-E2 Active Toy Exploration -> Extractor -> S0 ===  (source: {S0_SOURCE})")
        print("Not robotics / not RL training / not a neural net. Novelty = distance proxy.\n")
        suite_result = run_active_suite()
        if as_json:
            print("\n=== full result ===")
            print(json.dumps(suite_result, ensure_ascii=False, indent=2))
    print("\n=== tests ===")
    print(f"ok all {_tests()} assertions passed")
