"""Portable mirror for Colab.

Canonical implementation lives in chronos/embodied_toy/ and chronos/s0/.

S0-E0 tests:

    toy physical situation -> hand-written diagnostics -> S0 recommendation

This is not robotics, not model training, and not a real simulation. The
diagnostics are hand-written packets used to check whether S0 chooses the
expected physical language.
"""

from __future__ import annotations

import json
import sys

try:
    from chronos.embodied_toy.run_toy_suite import run_toy_suite

    S0_SOURCE = "chronos.embodied_toy (repo)"
except Exception:
    from dataclasses import asdict, dataclass
    from typing import Any

    K1_LORENTZ = "K1_LORENTZ"
    K2_SYMPLECTIC = "K2_SYMPLECTIC"
    K3_TOPOLOGICAL = "K3_TOPOLOGICAL"
    UNRESOLVED = "UNRESOLVED"
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
        if not d:
            return Recommendation(UNRESOLVED, "low", "No recognized diagnostics provided.", None, ACT_DO_NOT_PROMOTE)
        if d.get("diagnostic_context") == CTX_TOPOLOGY and d.get("field_learnable") is True:
            if d.get("object_tracking_valid") is False or d.get("topological_transport_score", 1.0) < 0.6:
                return Recommendation(
                    K3_TOPOLOGICAL,
                    "low",
                    "Field prediction is learnable but the topological object is not transported.",
                    "regime",
                    ACT_DO_NOT_PROMOTE,
                )
        if d.get("symplectic_improves_vs_controls") is True:
            return Recommendation(
                K2_SYMPLECTIC,
                "high",
                "Symplectic mechanism diagnostics improve against controls.",
                "mechanism",
                ACT_CONTINUE,
            )
        if d.get("causal_violation_rate", 0.0) > 0.05:
            return Recommendation(
                K1_LORENTZ,
                "medium",
                "Causal-violation diagnostics are significant.",
                "constraint",
                ACT_CONTINUE,
            )
        return Recommendation(UNRESOLVED, "low", "Diagnostics present but none decisive.", None, ACT_DO_NOT_PROMOTE)

    def _toy_worlds() -> dict[str, dict[str, Any]]:
        return {
            "pendulum": {
                "diagnostic_context": CTX_SYMPLECTIC,
                "field_learnable": True,
                "baseline_divergence": 0.0,
                "symplectic_improves_vs_controls": True,
                "symplectic_jacobian_error": 0.1,
            },
            "causal_contact": {
                "causal_violation_rate": 0.2,
                "field_learnable": True,
            },
            "vortex_fail": {
                "diagnostic_context": CTX_TOPOLOGY,
                "field_learnable": True,
                "object_tracking_valid": False,
                "topological_transport_score": 0.0,
            },
            "unknown": {},
        }

    def run_toy_suite(*, quiet: bool = False) -> dict[str, dict]:
        results = {}
        for name, diagnostics in _toy_worlds().items():
            recommendation = recommend(diagnostics).to_dict()
            results[name] = recommendation
            if not quiet:
                print(f"{name:16s} -> {recommendation['candidate_family']} / {recommendation['allowed_action']}")
        return results

    S0_SOURCE = "embedded fallback"


EXPECTED = {
    "pendulum": ("K2_SYMPLECTIC", "continue"),
    "causal_contact": ("K1_LORENTZ", "continue"),
    "vortex_fail": ("K3_TOPOLOGICAL", "do_not_promote"),
    "unknown": ("UNRESOLVED", "do_not_promote"),
}


def _tests() -> int:
    results = run_toy_suite(quiet=True)
    count = 0
    for name, (family, action) in EXPECTED.items():
        assert results[name]["candidate_family"] == family
        count += 1
        assert results[name]["allowed_action"] == action
        count += 1
    for recommendation in results.values():
        assert recommendation["allowed_action"] in {"continue", "archive", "do_not_promote"}
        count += 1
    return count


if __name__ == "__main__":
    quiet = "--quiet" in sys.argv
    if not quiet:
        print(f"=== S0-E0 Embodied Toy Suite ===  (source: {S0_SOURCE})")
        print("Hand-written toy diagnostics; not robotics; not simulation.\n")
        suite_results = run_toy_suite()
        print("\n=== full recommendations ===")
        for toy_name, toy_recommendation in suite_results.items():
            print(f"{toy_name:16s} -> {json.dumps(toy_recommendation, ensure_ascii=False)}")
    print("\n=== tests ===")
    print(f"ok all {_tests()} assertions passed")
