"""Portable mirror for Colab.

Canonical implementation lives in chronos/embodied_toy/ and chronos/s0/.

S0-E2b tests whether active exploration has diagnostic value:

    partitioned toy world -> active reaches structure zone -> diagnostic probe -> S0

This is not robotics, not RL training, not a neural network, and not online
learning. Novelty is a deterministic distance proxy.
"""

from __future__ import annotations

import json
import math
import random
import sys

try:
    from chronos.embodied_toy.run_active_value_suite import run_value_suite

    S0_SOURCE = "chronos.embodied_toy (repo)"
except Exception:
    from dataclasses import asdict, dataclass
    from typing import Any

    RAIL = 20.0
    ZONE = 14.0
    DELTA = 0.4
    ACTIONS = ["left", "right", "stay"]
    ACTION_DELTAS = {"left": -DELTA, "right": DELTA, "stay": 0.0}

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
                "K2_SYMPLECTIC",
                "high",
                "Symplectic mechanism diagnostics improve against controls.",
                "mechanism",
                "continue",
            )
        return Recommendation("UNRESOLVED", "low", "Diagnostics present but none decisive.", None, "do_not_promote")

    def clamp_position(x: float) -> float:
        return max(0.0, min(RAIL, x))

    class RailWorld:
        ACTIONS = ACTIONS

        def __init__(self, x0: float = 0.0):
            self.x = x0

        def peek(self, x: float, action: str) -> float:
            return clamp_position(x + ACTION_DELTAS[action])

        def step(self, action: str) -> dict:
            previous = self.x
            self.x = self.peek(self.x, action)
            return {"prev": previous, "action": action, "next": self.x}

    def probe_energy_drifts(in_structure_zone: bool, n_steps: int = 200, dt: float = 0.05) -> tuple[float, float]:
        def energy(theta: float, omega: float) -> float:
            return 0.5 * omega * omega + (1.0 - math.cos(theta))

        damping = 0.0 if in_structure_zone else 0.25
        theta, omega = 1.0, 0.0
        energy_symplectic = [energy(theta, omega)]
        for _ in range(n_steps):
            omega = (omega - dt * math.sin(theta)) * (1.0 - damping * dt)
            theta = theta + dt * omega
            energy_symplectic.append(energy(theta, omega))
        theta, omega = 1.0, 0.0
        energy_control = [energy(theta, omega)]
        for _ in range(n_steps):
            theta_next = theta + dt * omega
            omega_next = (omega - dt * math.sin(theta)) * (1.0 - damping * dt)
            theta, omega = theta_next, omega_next
            energy_control.append(energy(theta, omega))

        def drift(series: list[float]) -> float:
            e0 = abs(series[0]) if abs(series[0]) > 1e-9 else 1e-9
            return (max(series) - min(series)) / e0

        return drift(energy_symplectic), drift(energy_control)

    def state_novelty(x: float, memory: list[float]) -> float:
        return 1.0 if not memory else min(abs(x - remembered) for remembered in memory)

    class ActiveRailExplorer:
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
                    best_action = action
                    best_novelty = novelty
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
            diagnostics["diagnostic_context"] = "K2_SYMPLECTIC"
            diagnostics["symplectic_improves_vs_controls"] = True
            diagnostics["symplectic_jacobian_error"] = float(round(min(drift_symplectic, 0.99), 4))
        return diagnostics

    def _strip_private(diagnostics: dict) -> dict:
        return {key: value for key, value in diagnostics.items() if not key.startswith("_")}

    def run_value_suite(n_steps: int = 200, random_seed: int = 0, *, quiet: bool = False) -> dict:
        active_positions = ActiveRailExplorer(RailWorld()).run(n_steps)
        active_reached = max(active_positions)
        active_diagnostics = diagnostics_at_reached_state(active_reached, n_steps)
        active_recommendation = recommend(_strip_private(active_diagnostics)).to_dict()
        random_positions = random_rail_run(RailWorld(), n_steps, seed=random_seed)
        random_reached = max(random_positions)
        random_diagnostics = diagnostics_at_reached_state(random_reached, n_steps)
        random_recommendation = recommend(_strip_private(random_diagnostics)).to_dict()
        result = {
            "active": {
                "reached_x": active_reached,
                "in_zone": active_reached >= ZONE,
                "recommendation": active_recommendation,
                "measured": active_diagnostics["_measured"],
            },
            "random": {
                "reached_x": random_reached,
                "in_zone": random_reached >= ZONE,
                "recommendation": random_recommendation,
                "measured": random_diagnostics["_measured"],
            },
        }
        if not quiet:
            print(f"ZONE boundary x>={ZONE} is the structure zone (rail length {RAIL})\n")
            print(f"ACTIVE : reached x={active_reached:.1f} -> {active_recommendation['candidate_family']} / {active_recommendation['allowed_action']}")
            print(f"         measured={active_diagnostics['_measured']}")
            print(f"RANDOM : reached x={random_reached:.1f} -> {random_recommendation['candidate_family']} / {random_recommendation['allowed_action']}")
            print(f"         measured={random_diagnostics['_measured']}")
        return result

    S0_SOURCE = "embedded fallback"


def _tests() -> int:
    count = 0
    assert max(ActiveRailExplorer(RailWorld()).run(200)) >= ZONE
    count += 1
    random_far = sum(max(random_rail_run(RailWorld(), 200, seed=seed)) >= ZONE for seed in range(20))
    assert random_far <= 2
    count += 1
    result = run_value_suite(quiet=True)
    assert result["active"]["recommendation"]["candidate_family"] == "K2_SYMPLECTIC"
    count += 1
    assert result["active"]["recommendation"]["allowed_action"] == "continue"
    count += 1
    assert result["active"]["in_zone"] is True
    count += 1
    assert result["random"]["recommendation"]["candidate_family"] != "K2_SYMPLECTIC"
    count += 1
    assert result["random"]["in_zone"] is False
    count += 1
    drift_symplectic_zone, drift_control_zone = probe_energy_drifts(True, n_steps=200)
    drift_symplectic_near, drift_control_near = probe_energy_drifts(False, n_steps=200)
    assert drift_symplectic_zone < 0.5 * drift_control_zone
    count += 1
    assert not (drift_symplectic_near < 0.5 * drift_control_near)
    count += 1
    rerun = run_value_suite(quiet=True)
    assert result["active"]["recommendation"] == rerun["active"]["recommendation"]
    count += 1
    assert result["random"]["recommendation"] == rerun["random"]["recommendation"]
    count += 1
    for key in ("active", "random"):
        assert result[key]["recommendation"]["allowed_action"] in {"continue", "archive", "do_not_promote"}
        count += 1
    return count


if __name__ == "__main__":
    quiet = "--quiet" in sys.argv
    as_json = "--json" in sys.argv
    if not quiet:
        print(f"=== S0-E2b: active diagnostic value ===  (source: {S0_SOURCE})")
        print("Partitioned world: structure only in the far zone; the probe diagnoses the reached state.\n")
        suite_result = run_value_suite()
        if as_json:
            print("\n=== full result ===")
            print(json.dumps(suite_result, ensure_ascii=False, indent=2))
    print("\n=== tests ===")
    print(f"ok all {_tests()} assertions passed")
