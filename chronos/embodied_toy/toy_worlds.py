"""Toy diagnostic packets for S0-E0 embodied structure-language tests.

These are hand-written diagnostic packets, not real simulations. They test
whether S0 chooses the expected physical language from simple physical
situation descriptions.
"""

from __future__ import annotations

from chronos.s0.diagnostics_schema import CTX_SYMPLECTIC, CTX_TOPOLOGY


def diagnostics_for_pendulum_like() -> dict:
    """Pendulum / spring-like system: should recommend K2."""

    return {
        "diagnostic_context": CTX_SYMPLECTIC,
        "field_learnable": True,
        "baseline_divergence": 0.0,
        "symplectic_improves_vs_controls": True,
        "symplectic_jacobian_error": 0.1,
    }


def diagnostics_for_causal_contact_like() -> dict:
    """Causal contact / collision-like system: should recommend K1."""

    return {
        "causal_violation_rate": 0.2,
        "field_learnable": True,
    }


def diagnostics_for_vortex_transport_fail() -> dict:
    """Topology toy: field learnable but topological object transport fails."""

    return {
        "diagnostic_context": CTX_TOPOLOGY,
        "field_learnable": True,
        "object_tracking_valid": False,
        "topological_transport_score": 0.0,
    }


def diagnostics_for_unknown() -> dict:
    """No recognizable diagnostics: should remain unresolved."""

    return {}


TOY_WORLDS = {
    "pendulum": diagnostics_for_pendulum_like,
    "causal_contact": diagnostics_for_causal_contact_like,
    "vortex_fail": diagnostics_for_vortex_transport_fail,
    "unknown": diagnostics_for_unknown,
}
