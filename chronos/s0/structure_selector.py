"""Chronos-S0 structure selector and K3.2D verdict helpers.

S0 is not a predictor and not a certifier. It recommends which K-family should
enter VPSL validation next and at which gate. Certification happens only by
passing downstream VPSL gates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .diagnostics_schema import (
    ACT_CONTINUE,
    ACT_DO_NOT_PROMOTE,
    CAUSAL_VIOLATION_SIGNIF,
    CONF_HIGH,
    CONF_LOW,
    CONF_MED,
    GATE_CONSTRAINT,
    GATE_MECHANISM,
    GATE_REGIME,
    GAUGE_RESIDUAL_SIGNIF,
    HARD_DIV_MAX,
    K1_LORENTZ,
    K2_SYMPLECTIC,
    K3_TOPOLOGICAL,
    K4_GAUGE,
    K5_HILBERT,
    KNOWN_DIAGNOSTICS,
    PAIR_INTACT_MIN,
    POS_ERR_CEIL,
    REF_CEIL,
    SYMPLECTIC_ERR_OK,
    TOPO_TRANSPORT_OK,
    UNITARITY_ERR_SIGNIF,
    UNRESOLVED,
)


@dataclass(frozen=True)
class Recommendation:
    candidate_family: str
    confidence: str
    reason: str
    next_vpsl_gate: str | None
    allowed_action: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _present(diagnostics: dict[str, Any], key: str) -> bool:
    return key in diagnostics and diagnostics[key] is not None


def recommend(diagnostics: dict[str, Any] | None) -> Recommendation:
    """Recommend the next structure family and VPSL gate from diagnostics."""

    d = diagnostics or {}

    if not any(_present(d, key) for key in KNOWN_DIAGNOSTICS):
        return Recommendation(
            UNRESOLVED,
            CONF_LOW,
            "No recognized diagnostics provided; cannot recommend a structure family.",
            None,
            ACT_DO_NOT_PROMOTE,
        )

    field_ok = (d.get("field_learnable") is True) or (
        _present(d, "baseline_divergence") and d["baseline_divergence"] < 0.05
    )
    transport_known = _present(d, "topological_transport_score") or _present(d, "object_tracking_valid")
    transport_fail = False
    if transport_known:
        if d.get("object_tracking_valid") is False:
            transport_fail = True
        if _present(d, "topological_transport_score") and d["topological_transport_score"] < TOPO_TRANSPORT_OK:
            transport_fail = True
    if field_ok and transport_fail:
        return Recommendation(
            K3_TOPOLOGICAL,
            CONF_LOW,
            "Field prediction is learnable but the topological object is not transported. "
            "Low field error does not imply topology success; this is an unresolved topology regime.",
            GATE_REGIME,
            ACT_DO_NOT_PROMOTE,
        )

    if d.get("symplectic_improves_vs_controls") is True:
        confidence = CONF_HIGH
        if _present(d, "symplectic_jacobian_error") and d["symplectic_jacobian_error"] > SYMPLECTIC_ERR_OK:
            confidence = CONF_MED
        return Recommendation(
            K2_SYMPLECTIC,
            confidence,
            "Symplectic mechanism diagnostics improve against controls; recommend symplectic family.",
            GATE_MECHANISM,
            ACT_CONTINUE,
        )

    if (
        _present(d, "symplectic_jacobian_error")
        and d["symplectic_jacobian_error"] < SYMPLECTIC_ERR_OK
        and not _present(d, "symplectic_improves_vs_controls")
    ):
        return Recommendation(
            K2_SYMPLECTIC,
            CONF_MED,
            "Low symplectic Jacobian error suggests a symplectic structure, but no control comparison is available.",
            GATE_CONSTRAINT,
            ACT_CONTINUE,
        )

    if _present(d, "causal_violation_rate") and d["causal_violation_rate"] > CAUSAL_VIOLATION_SIGNIF:
        return Recommendation(
            K1_LORENTZ,
            CONF_MED,
            "Causal-violation / light-cone diagnostics are significant; recommend the Lorentz family.",
            GATE_CONSTRAINT,
            ACT_CONTINUE,
        )

    if _present(d, "gauge_residual") and d["gauge_residual"] > GAUGE_RESIDUAL_SIGNIF:
        return Recommendation(
            K4_GAUGE,
            CONF_LOW,
            "Non-trivial gauge residual present; gauge family is a candidate.",
            GATE_REGIME,
            ACT_CONTINUE,
        )

    if _present(d, "unitarity_error") and d["unitarity_error"] > UNITARITY_ERR_SIGNIF:
        return Recommendation(
            K5_HILBERT,
            CONF_LOW,
            "Unitarity error suggests a Hilbert-space / unitary-structure candidate.",
            GATE_REGIME,
            ACT_CONTINUE,
        )

    if _present(d, "topological_transport_score") and d["topological_transport_score"] >= TOPO_TRANSPORT_OK:
        return Recommendation(
            K3_TOPOLOGICAL,
            CONF_MED,
            "Topological object is transported; topology family is a candidate.",
            GATE_CONSTRAINT,
            ACT_CONTINUE,
        )

    return Recommendation(
        UNRESOLVED,
        CONF_LOW,
        "Diagnostics present but none decisive for a specific structure family.",
        None,
        ACT_DO_NOT_PROMOTE,
    )


def k32d_verdict(mode: str, ref_med: float, hard_frac: float, pair_frac: float, pos_med: float) -> str:
    """Return the two-layer K3.2D regime verdict."""

    pipeline_ok = (ref_med < REF_CEIL) and (hard_frac <= HARD_DIV_MAX)
    transport_ok = (pair_frac >= PAIR_INTACT_MIN) and (pos_med < POS_ERR_CEIL)

    if mode == "SMOKE":
        if not pipeline_ok:
            return "SMOKE_PIPELINE_FAIL"
        if not transport_ok:
            return "SMOKE_PIPELINE_OK_TRANSPORT_FAIL"
        return "SMOKE_TRANSPORT_OK"

    return "FULL_REGIME_VALIDATED" if (pipeline_ok and transport_ok) else "REGIME_UNRESOLVED"


def k32d_explain(verdict: str) -> str:
    explanations = {
        "SMOKE_PIPELINE_FAIL": (
            "Baseline did not learn field prediction or hard-diverged. Fix training/regime before promotion."
        ),
        "SMOKE_PIPELINE_OK_TRANSPORT_FAIL": (
            "Baseline learns field prediction but fails vortex transport. Low field MSE does not imply topology."
        ),
        "SMOKE_TRANSPORT_OK": "Pipeline learns and the vortex pair is transported; promote to FULL for N=30.",
        "FULL_REGIME_VALIDATED": "Baseline trains, vortex pair transported, position error graceful, hard-div low.",
        "REGIME_UNRESOLVED": "Regime not validated. This is not a prior test and not a rejection of 2D topology.",
    }
    return explanations.get(verdict, "unknown verdict")

