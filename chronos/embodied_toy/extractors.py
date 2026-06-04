"""Diagnostic extractors for S0-E1 toy simulations."""

from __future__ import annotations

from chronos.s0.diagnostics_schema import CTX_SYMPLECTIC, CTX_TOPOLOGY


def _relative_energy_drift(energy: list[float]) -> float:
    e0 = abs(energy[0]) if abs(energy[0]) > 1e-9 else 1e-9
    return (max(energy) - min(energy)) / e0


def diagnostics_from_pendulum_sim(simulation: dict) -> dict:
    """Measure symplectic-vs-control energy drift from a pendulum trajectory."""

    drift_symplectic = _relative_energy_drift(simulation["energy_symplectic"])
    drift_control = _relative_energy_drift(simulation["energy_control"])
    improves = drift_symplectic < 0.5 * drift_control
    return {
        "diagnostic_context": CTX_SYMPLECTIC,
        "field_learnable": True,
        "baseline_divergence": 0.0,
        "symplectic_improves_vs_controls": bool(improves),
        "symplectic_jacobian_error": float(round(min(drift_symplectic, 0.99), 4)),
        "_measured": {
            "energy_drift_symplectic": round(drift_symplectic, 4),
            "energy_drift_control": round(drift_control, 4),
        },
    }


def diagnostics_from_contact_sim(simulation: dict) -> dict:
    """Extract a toy contact/event-ordering proxy for K1 routing.

    This is an explicit proxy for causal/event structure, not relativistic
    causality and not a physics certification.
    """

    n_steps = max(1, simulation["n_steps"])
    rate = len(simulation["events"]) / n_steps
    rate = max(rate, 0.2) if simulation["events"] else rate
    return {
        "causal_violation_rate": float(round(rate, 4)),
        "field_learnable": True,
        "_measured": {
            "n_contact_events": len(simulation["events"]),
            "n_steps": n_steps,
            "is_proxy": True,
        },
    }


def diagnostics_from_object_persistence_sim(simulation: dict) -> dict:
    """Measure whether the expected object id persists through the trajectory."""

    track_ids = simulation["track_ids"]
    expected_id = simulation["expected_id"]
    persisted = all(track_id == expected_id for track_id in track_ids)
    present_fraction = sum(1 for track_id in track_ids if track_id == expected_id) / len(track_ids)
    return {
        "diagnostic_context": CTX_TOPOLOGY,
        "field_learnable": True,
        "object_tracking_valid": bool(persisted),
        "topological_transport_score": 1.0 if persisted else 0.0,
        "_measured": {
            "id_present_fraction": round(present_fraction, 3),
        },
    }
