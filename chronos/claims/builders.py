"""Build ClaimRecord objects from existing Chronos result dictionaries."""

from __future__ import annotations

from typing import Any

from chronos.s0.diagnostics_schema import (
    ACT_ARCHIVE,
    ACT_CONTINUE,
    ACT_DO_NOT_PROMOTE,
    GATE_CONSTRAINT,
    GATE_MECHANISM,
    GATE_REGIME,
    GATE_TRANSFER,
    K2_SYMPLECTIC,
    K3_TOPOLOGICAL,
)

from .failure_taxonomy import (
    ACTIVE_NO_ADVANTAGE,
    PIPELINE_OK_TRANSPORT_FAIL,
    REGIME_INVALID,
)
from .schema import ClaimRecord, new_timestamp


def _timestamp(summary: dict[str, Any]) -> str:
    return str(summary.get("timestamp") or new_timestamp())


def _claim_id(prefix: str, summary: dict[str, Any]) -> str:
    return str(summary.get("claim_id") or summary.get("run_id") or f"{prefix}:{_timestamp(summary)}")


def _score(summary: dict[str, Any], key: str, default: Any = None) -> Any:
    return summary.get(key, default)


def claim_from_k3_e2b(summary: dict[str, Any]) -> ClaimRecord:
    verdict = str(summary.get("verdict", "ACTIVE_DIAGNOSTIC_VALUE_PASSED"))
    passed = verdict == "ACTIVE_DIAGNOSTIC_VALUE_PASSED"
    return ClaimRecord(
        claim_id=_claim_id("k3_e2b", summary),
        structure_family=K3_TOPOLOGICAL,
        verdict=verdict,
        evidence_level="toy_active_regime_search",
        gate=GATE_REGIME,
        diagnostics={
            "active_best_score": _score(summary, "active_best_score"),
            "random_median_best_score": _score(summary, "random_median_best_score"),
            "random_success_count_20": _score(summary, "random_success_count_20"),
        },
        controls=["random_action_search"],
        supports=["guided active search beats blind random search on a transparent toy topology landscape"]
        if passed
        else ["toy topology search ran but did not establish active advantage"],
        does_not_support=[
            "K3 prior validation",
            "proof that topological priors work",
            "Gross-Pitaevskii truth",
            "robotics or neural training",
        ],
        failure_mode=None if passed else ACTIVE_NO_ADVANTAGE,
        next_gate="K3-E2c cheap GP truth-only active search" if passed else None,
        claim_boundary="toy search landscape only; not GP truth, not K3 prior validation",
        allowed_action=ACT_CONTINUE if passed else ACT_DO_NOT_PROMOTE,
        timestamp=_timestamp(summary),
    )


def claim_from_k3_e2c(summary: dict[str, Any]) -> ClaimRecord:
    verdict = str(summary.get("verdict", "GP_ACTIVE_NO_ADVANTAGE"))
    no_advantage = verdict == "GP_ACTIVE_NO_ADVANTAGE"
    return ClaimRecord(
        claim_id=_claim_id("k3_e2c", summary),
        structure_family=K3_TOPOLOGICAL,
        verdict=verdict,
        evidence_level="cheap_gp_truth_active_search",
        gate=GATE_REGIME,
        diagnostics={
            "active_best_score": _score(summary, "active_best_score"),
            "random_median_best_score": _score(summary, "random_median_best_score"),
            "random_success_frac_20": _score(summary, "random_success_frac_20"),
        },
        controls=["random_action_search"],
        supports=["cheap GP evaluator swap ran and showed the landscape was too trivial to separate active from random"],
        does_not_support=[
            "active diagnostic value in the cheap-GP E2c setting",
            "K3 prior validation",
            "full-resolution GP",
            "proof that topological priors work",
        ],
        failure_mode=ACTIVE_NO_ADVANTAGE if no_advantage else None,
        next_gate=None if no_advantage else "K3-E2d discriminating GP truth-only active search",
        claim_boundary="cheap real-GP truth, but too trivial to establish active advantage; not prior validation",
        allowed_action=ACT_DO_NOT_PROMOTE if no_advantage else ACT_CONTINUE,
        timestamp=_timestamp(summary),
    )


def claim_from_k3_e2d(summary: dict[str, Any]) -> ClaimRecord:
    verdict = str(summary.get("verdict", "GP_ACTIVE_DIAGNOSTIC_VALUE_PASSED"))
    passed = verdict == "GP_ACTIVE_DIAGNOSTIC_VALUE_PASSED"
    return ClaimRecord(
        claim_id=_claim_id("k3_e2d", summary),
        structure_family=K3_TOPOLOGICAL,
        verdict=verdict,
        evidence_level="gp_truth_active_search",
        gate=GATE_REGIME,
        diagnostics={
            "active_best_score": _score(summary, "active_best_score"),
            "active_found_transport_ok": _score(summary, "active_found_transport_ok"),
            "random_median_best_score": _score(summary, "random_median_best_score"),
            "random_success_count_20": _score(summary, "random_success_count_20"),
        },
        controls=["random_action_search"],
        supports=["guided active search finds a cheap GP vortex-position regime better than random control"]
        if passed
        else ["cheap GP active search ran but did not establish active diagnostic value"],
        does_not_support=[
            "CNN learnability",
            "K3 prior validation",
            "certification",
            "full-resolution GP",
        ],
        failure_mode=None if passed else ACTIVE_NO_ADVANTAGE,
        next_gate="K3.2D.0 CNN baseline regime validation" if passed else None,
        claim_boundary="truth-level cheap GP only; not CNN validation, not K3 prior validation",
        allowed_action=ACT_CONTINUE if passed else ACT_DO_NOT_PROMOTE,
        timestamp=_timestamp(summary),
    )


def claim_from_k3_2d_0_summary(summary: dict[str, Any]) -> ClaimRecord:
    verdict = str(summary.get("verdict", "REGIME_UNRESOLVED"))
    pipeline_ok = bool(summary.get("pipeline_ok"))
    transport_ok = bool(summary.get("transport_ok"))
    passed = verdict in {"SMOKE_TRANSPORT_OK", "FULL_REGIME_VALIDATED"} and pipeline_ok and transport_ok
    failure_mode = None
    if pipeline_ok and not transport_ok:
        failure_mode = PIPELINE_OK_TRANSPORT_FAIL
    elif not passed:
        failure_mode = REGIME_INVALID
    return ClaimRecord(
        claim_id=_claim_id("k3_2d_0", summary),
        structure_family=K3_TOPOLOGICAL,
        verdict=verdict,
        evidence_level="cnn_baseline_regime_validation",
        gate=GATE_REGIME,
        diagnostics={
            "pipeline_ok": pipeline_ok,
            "transport_ok": transport_ok,
            "ref_med": _score(summary, "ref_med"),
            "pos_med": _score(summary, "pos_med"),
            "pair_frac": _score(summary, "pair_frac"),
            "hard_frac": _score(summary, "hard_frac"),
        },
        controls=["baseline_only_no_prior"],
        supports=["CNN baseline regime is learnable and transports the vortex pair"]
        if passed
        else ["K3.2D.0 regime validation ran without testing any prior"],
        does_not_support=[
            "K3 prior validation",
            "certification",
            "claim that field prediction alone transports topological objects",
        ],
        failure_mode=failure_mode,
        next_gate="K3.2D.1 topological prior test" if passed else None,
        claim_boundary="baseline-only CNN regime validation; no topology prior tested",
        allowed_action=ACT_CONTINUE if passed else ACT_DO_NOT_PROMOTE,
        timestamp=_timestamp(summary),
    )


def claim_from_k2_summary(summary: dict[str, Any]) -> ClaimRecord:
    verdict = str(summary.get("verdict", "FULL_TRANSFER_CONFIRMED"))
    certified = "CERTIFIED" in verdict.upper()
    evidence_level = "vpsl_certified_structure" if certified else "vpsl_transfer_confirmed"
    gate = GATE_TRANSFER if certified or "TRANSFER" in verdict.upper() else GATE_MECHANISM
    return ClaimRecord(
        claim_id=_claim_id("k2", summary),
        structure_family=K2_SYMPLECTIC,
        verdict=verdict,
        evidence_level=evidence_level,
        gate=gate,
        diagnostics={
            "energy_drift": _score(summary, "energy_drift"),
            "symplectic_jacobian_error": _score(summary, "symplectic_jacobian_error"),
            "transfer_horizon": _score(summary, "transfer_horizon"),
        },
        controls=["energy_control", "l2_control", "wrong_omega_control"],
        supports=["symplectic structure survives VPSL transfer/mechanism controls"],
        does_not_support=[
            "universal claim about all physical systems",
            "new K-family beyond K2",
            "changing S0 recommendations",
        ],
        failure_mode=None,
        next_gate=None,
        claim_boundary="K2 claim is bounded to the archived VPSL transfer/mechanism evidence",
        allowed_action=ACT_ARCHIVE if summary.get("archived") else ACT_CONTINUE,
        timestamp=_timestamp(summary),
    )


__all__ = [
    "claim_from_k2_summary",
    "claim_from_k3_2d_0_summary",
    "claim_from_k3_e2b",
    "claim_from_k3_e2c",
    "claim_from_k3_e2d",
]
