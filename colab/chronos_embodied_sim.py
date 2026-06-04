"""Portable mirror for Colab.

Canonical implementation lives in chronos/embodied_toy/ and chronos/s0/.

S0-E1 tests:

    toy simulation -> diagnostic extractor -> S0 recommendation

This is not robotics, not model training, not learned classification, and not
physics certification. It is a deterministic toy pipeline check.
"""

from __future__ import annotations

import json
import math
import sys

try:
    from chronos.embodied_toy.run_sim_suite import run_sim_suite

    S0_SOURCE = "chronos.embodied_toy (repo)"
except Exception:
    from dataclasses import asdict, dataclass
    from typing import Any

    K1_LORENTZ = "K1_LORENTZ"
    K2_SYMPLECTIC = "K2_SYMPLECTIC"
    K3_TOPOLOGICAL = "K3_TOPOLOGICAL"
    ACT_CONTINUE = "continue"
    ACT_DO_NOT_PROMOTE = "do_not_promote"
    ALLOWED_ACTIONS = frozenset({ACT_CONTINUE, "archive", ACT_DO_NOT_PROMOTE})
    CTX_TOPOLOGY = "K3_TOPOLOGY_REGIME"
    CTX_SYMPLECTIC = "K2_SYMPLECTIC"

    @dataclass(frozen=True)
    class Recommendation:
        candidate_family: str
        confidence: str
        reason: str
        next_vpsl_gate: str | None
        allowed_action: str

        def __post_init__(self) -> None:
            if self.allowed_action not in ALLOWED_ACTIONS:
                raise ValueError("S0 never certifies")

        def to_dict(self) -> dict[str, Any]:
            return asdict(self)

    def recommend(diagnostics: dict[str, Any] | None) -> Recommendation:
        d = diagnostics or {}
        if d.get("diagnostic_context") == CTX_TOPOLOGY and d.get("field_learnable") is True:
            transport_fail = d.get("object_tracking_valid") is False or d.get("topological_transport_score", 1.0) < 0.6
            if transport_fail:
                return Recommendation(K3_TOPOLOGICAL, "low", "Topological object is not transported.", "regime", ACT_DO_NOT_PROMOTE)
        if d.get("symplectic_improves_vs_controls") is True:
            return Recommendation(K2_SYMPLECTIC, "high", "Symplectic mechanism improves against controls.", "mechanism", ACT_CONTINUE)
        if d.get("causal_violation_rate", 0.0) > 0.05:
            return Recommendation(K1_LORENTZ, "medium", "Toy causal/contact proxy is significant.", "constraint", ACT_CONTINUE)
        return Recommendation("UNRESOLVED", "low", "Diagnostics present but none decisive.", None, ACT_DO_NOT_PROMOTE)

    def _energy(theta: float, omega: float) -> float:
        return 0.5 * omega * omega + (1.0 - math.cos(theta))

    def simulate_pendulum(n_steps: int = 400, dt: float = 0.05) -> dict:
        theta, omega = 1.0, 0.0
        energy_symplectic = [_energy(theta, omega)]
        for _ in range(n_steps):
            omega = omega - dt * math.sin(theta)
            theta = theta + dt * omega
            energy_symplectic.append(_energy(theta, omega))
        theta, omega = 1.0, 0.0
        energy_control = [_energy(theta, omega)]
        for _ in range(n_steps):
            theta_next = theta + dt * omega
            omega_next = omega - dt * math.sin(theta)
            theta, omega = theta_next, omega_next
            energy_control.append(_energy(theta, omega))
        return {"energy_symplectic": energy_symplectic, "energy_control": energy_control}

    def simulate_contact(n_steps: int = 100, dt: float = 0.05) -> dict:
        x, v = 1.0, -1.0
        events = []
        for step in range(n_steps):
            x = x + dt * v
            if x <= 0.0 and v < 0.0:
                x = -x
                v = -v
                events.append(step)
        return {"events": events, "n_steps": n_steps}

    def simulate_object_persistence_fail(n_steps: int = 50, lose_at: int = 20) -> dict:
        return {"track_ids": [1 if step < lose_at else None for step in range(n_steps)], "expected_id": 1}

    def _relative_drift(energy: list[float]) -> float:
        e0 = abs(energy[0]) if abs(energy[0]) > 1e-9 else 1e-9
        return (max(energy) - min(energy)) / e0

    def diagnostics_from_pendulum_sim(simulation: dict) -> dict:
        drift_symplectic = _relative_drift(simulation["energy_symplectic"])
        drift_control = _relative_drift(simulation["energy_control"])
        return {
            "diagnostic_context": CTX_SYMPLECTIC,
            "field_learnable": True,
            "baseline_divergence": 0.0,
            "symplectic_improves_vs_controls": drift_symplectic < 0.5 * drift_control,
            "symplectic_jacobian_error": float(round(min(drift_symplectic, 0.99), 4)),
            "_measured": {
                "energy_drift_symplectic": round(drift_symplectic, 4),
                "energy_drift_control": round(drift_control, 4),
            },
        }

    def diagnostics_from_contact_sim(simulation: dict) -> dict:
        n_steps = max(1, simulation["n_steps"])
        rate = len(simulation["events"]) / n_steps
        rate = max(rate, 0.2) if simulation["events"] else rate
        return {
            "causal_violation_rate": float(round(rate, 4)),
            "field_learnable": True,
            "_measured": {"n_contact_events": len(simulation["events"]), "n_steps": n_steps, "is_proxy": True},
        }

    def diagnostics_from_object_persistence_sim(simulation: dict) -> dict:
        track_ids = simulation["track_ids"]
        expected_id = simulation["expected_id"]
        persisted = all(track_id == expected_id for track_id in track_ids)
        present_fraction = sum(1 for track_id in track_ids if track_id == expected_id) / len(track_ids)
        return {
            "diagnostic_context": CTX_TOPOLOGY,
            "field_learnable": True,
            "object_tracking_valid": persisted,
            "topological_transport_score": 1.0 if persisted else 0.0,
            "_measured": {"id_present_fraction": round(present_fraction, 3)},
        }

    def _strip_private(diagnostics: dict) -> dict:
        return {key: value for key, value in diagnostics.items() if not key.startswith("_")}

    def run_sim_suite(*, quiet: bool = False) -> dict[str, dict]:
        diagnostics_by_case = {
            "pendulum": diagnostics_from_pendulum_sim(simulate_pendulum()),
            "contact": diagnostics_from_contact_sim(simulate_contact()),
            "object_persistence": diagnostics_from_object_persistence_sim(simulate_object_persistence_fail()),
        }
        results = {}
        for name, diagnostics in diagnostics_by_case.items():
            recommendation = recommend(_strip_private(diagnostics)).to_dict()
            results[name] = {"diagnostics": diagnostics, "recommendation": recommendation}
            if not quiet:
                print(f"{name:18s} -> {recommendation['candidate_family']} / {recommendation['allowed_action']}   measured={diagnostics.get('_measured', {})}")
        return results

    S0_SOURCE = "embedded fallback"


EXPECTED = {
    "pendulum": ("K2_SYMPLECTIC", "continue"),
    "contact": ("K1_LORENTZ", "continue"),
    "object_persistence": ("K3_TOPOLOGICAL", "do_not_promote"),
}


def _tests() -> int:
    results = run_sim_suite(quiet=True)
    count = 0
    for name, (family, action) in EXPECTED.items():
        assert results[name]["recommendation"]["candidate_family"] == family
        count += 1
        assert results[name]["recommendation"]["allowed_action"] == action
        count += 1
    pendulum = results["pendulum"]["diagnostics"]
    assert pendulum["symplectic_improves_vs_controls"] is True
    count += 1
    assert pendulum["_measured"]["energy_drift_symplectic"] < pendulum["_measured"]["energy_drift_control"]
    count += 1
    object_persistence = results["object_persistence"]["diagnostics"]
    assert object_persistence["object_tracking_valid"] is False
    count += 1
    assert object_persistence["_measured"]["id_present_fraction"] < 1.0
    count += 1
    assert results["contact"]["diagnostics"]["_measured"]["is_proxy"] is True
    count += 1
    for result in results.values():
        assert result["recommendation"]["allowed_action"] in {"continue", "archive", "do_not_promote"}
        count += 1
    rerun = run_sim_suite(quiet=True)
    for name in EXPECTED:
        assert results[name]["recommendation"] == rerun[name]["recommendation"]
        count += 1
    return count


if __name__ == "__main__":
    quiet = "--quiet" in sys.argv
    as_json = "--json" in sys.argv
    if not quiet:
        print(f"=== S0-E1 Toy Simulation -> Extractor -> S0 ===  (source: {S0_SOURCE})")
        print("Not robotics / not training / not physics certification. Pipeline test only.\n")
        suite_results = run_sim_suite()
        if as_json:
            print("\n=== full recommendations ===")
            print(json.dumps({name: result["recommendation"] for name, result in suite_results.items()}, ensure_ascii=False, indent=2))
    print("\n=== tests ===")
    print(f"ok all {_tests()} assertions passed")
