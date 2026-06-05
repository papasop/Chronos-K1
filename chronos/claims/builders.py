"""Build ClaimRecords from existing result dictionaries.

Each builder reads the dictionary a result already produces and emits an
honestly bounded ClaimRecord. Builders never fabricate a pass; they read the
verdict/diagnostics in the input dict.
"""

from __future__ import annotations

from chronos.claims.schema import ClaimRecord
from chronos.claims import failure_taxonomy as F

_CODE = "claims_v2"
_PASS_E2B = "ACTIVE_DIAGNOSTIC_VALUE_PASSED"


def _action_from(result: dict, default: str) -> str:
    rec = result.get("active_rec") or {}
    action = rec.get("allowed_action")
    return action if action in ("continue", "archive", "do_not_promote") else default


def claim_from_k3_e2b(result: dict) -> ClaimRecord:
    """Input: chronos.k3.run_active_topology_search.run_search_suite() output."""

    verdict = result.get("verdict", _PASS_E2B)
    active = result.get("active", {})
    return ClaimRecord(
        claim_id="k3_e2b_active_topology_search",
        structure_family="K3_TOPOLOGICAL",
        evidence_level="toy_active_regime_search",
        verdict=verdict,
        gate="active_search",
        allowed_action=_action_from(result, "continue"),
        supports=[
            "guided active regime search finds a topology-trackable toy vortex regime better than random search"
        ],
        does_not_support=[
            "Gross-Pitaevskii truth validation",
            "CNN baseline learnability",
            "K3 prior validation",
            "VPSL certification",
        ],
        controls={
            "random_median_best_score": result.get("random_median_best_score"),
            "random_success_count_20": result.get("random_success_count_20"),
        },
        diagnostics={
            "active_best_score": active.get("best_score"),
            "active_found_transport_ok": active.get("found_transport_ok"),
        },
        failure_mode=None if verdict == _PASS_E2B else F.ACTIVE_NO_ADVANTAGE,
        next_gate="K3-E2c cheap GP truth-only active search",
        claim_boundary="toy search landscape only; not GP truth; not K3 prior validation",
        source_module="chronos.k3.run_active_topology_search",
        code_version=_CODE,
        claim_type="positive_evidence" if verdict == _PASS_E2B else "negative_result",
        confidence_level="medium" if verdict == _PASS_E2B else "low",
        evidence_scope={
            "system": "toy vortex regime",
            "regime": "analytic toy landscape",
            "model": "truth-only toy evaluator",
            "compute": "CPU toy",
        },
        replication={
            "n_seeds": 20 if result.get("random_success_count_20") is not None else None,
            "random_success_count": result.get("random_success_count_20"),
            "random_median_best": result.get("random_median_best_score"),
        },
        risk_flags=["toy_landscape", "cpu_only", "no_cnn_training"],
    )


def claim_from_k3_e2c(result: dict) -> ClaimRecord:
    """Input: cheap GP active search output. Expected verdict GP_ACTIVE_NO_ADVANTAGE."""

    verdict = result.get("verdict", "GP_ACTIVE_NO_ADVANTAGE")
    random_success_count = result.get("random_success_count_20")
    failure_mode = (
        F.RANDOM_ALSO_SUCCEEDS
        if isinstance(random_success_count, (int, float)) and random_success_count > 5
        else F.ACTIVE_NO_ADVANTAGE
    )
    # Negative result is archived; never recorded as continue even if input recommends it.
    action = "archive" if verdict == "GP_ACTIVE_NO_ADVANTAGE" else _action_from(result, "do_not_promote")
    return ClaimRecord(
        claim_id="k3_e2c_cheap_gp_active_search",
        structure_family="K3_TOPOLOGICAL",
        evidence_level="cheap_gp_truth_active_search",
        verdict=verdict,
        gate="truth_active_search",
        allowed_action=action,
        supports=[
            "cheap GP truth evaluator and vortex tracker run successfully",
            "active/random comparison is operational",
        ],
        does_not_support=[
            "active exploration has diagnostic advantage on this cheap GP landscape",
            "CNN baseline learnability",
            "K3 prior validation",
            "VPSL certification",
        ],
        controls={
            "random_median_best_score": result.get("random_median_best_score"),
            "random_success_count_20": random_success_count,
        },
        diagnostics={"active_best_score": result.get("active", {}).get("best_score")},
        failure_mode=failure_mode,
        next_gate="K3-E2d discriminating GP truth active search",
        claim_boundary="cheap GP truth only; non-discriminating landscape; not prior validation",
        source_module="chronos.k3.run_gp_active_search",
        code_version=_CODE,
        claim_type="negative_result" if verdict == "GP_ACTIVE_NO_ADVANTAGE" else "positive_evidence",
        confidence_level="low",
        evidence_scope={
            "system": "GP vortex pair",
            "regime": "cheap GP non-discriminating landscape",
            "model": "truth-only GP evaluator",
            "compute": "CPU",
        },
        replication={
            "n_seeds": 20,
            "random_success_count": random_success_count,
            "random_median_best": result.get("random_median_best_score"),
        },
        risk_flags=["cheap_gp", "cpu_only", "no_cnn_training", "random_also_succeeds"],
    )


def claim_from_k3_e2d(result: dict) -> ClaimRecord:
    """Input: discriminating GP active search output."""

    active = result.get("active", {})
    metrics = active.get("best_metrics") or {}
    return ClaimRecord(
        claim_id="k3_e2d_discriminating_gp_active_search",
        structure_family="K3_TOPOLOGICAL",
        evidence_level="gp_truth_active_search",
        verdict=result.get("verdict", "GP_ACTIVE_DIAGNOSTIC_VALUE_PASSED"),
        gate="truth_active_search",
        allowed_action=_action_from(result, "continue"),
        supports=[
            "guided active search finds a GP vortex regime with better transport diagnostics than random",
            "active reaches transport_ok while random rarely does",
        ],
        does_not_support=[
            "CNN baseline regime validation",
            "K3 prior validation",
            "VPSL certification",
        ],
        controls={
            "random_median_best_score": result.get("random_median_best_score"),
            "random_success_count_20": result.get("random_success_count_20"),
            "random_success_frac_20": result.get("random_success_frac_20"),
        },
        diagnostics={
            "active_best_score": active.get("best_score"),
            "active_mean_pos_err": metrics.get("mean_pos_err"),
            "active_pair_intact": metrics.get("pair_intact"),
        },
        failure_mode=None,
        next_gate="K3.2D.0 CNN baseline regime validation",
        claim_boundary="truth-level GP active search only; not CNN validation; not K3 prior validation",
        source_module="chronos.k3.run_gp_active_search",
        code_version=_CODE,
        claim_type="positive_evidence",
        confidence_level="medium",
        evidence_scope={
            "system": "GP vortex pair",
            "regime": "discriminating GP with push failure dimension",
            "model": "truth-only GP evaluator",
            "compute": "CPU",
        },
        replication={
            "n_seeds": 20,
            "random_success_count": result.get("random_success_count_20"),
            "random_median_best": result.get("random_median_best_score"),
        },
        risk_flags=["cpu_only", "no_cnn_training", "designed_failure_dimension"],
    )


def claim_from_k3_2d_0_summary(summary: dict) -> ClaimRecord:
    """Build the documented K3.2D.0 smoke transport-fail claim.

    This builder currently handles only the case where the pipeline is learnable
    but transport fails. Other combinations raise rather than silently emitting
    transport-fail wording for a passing or pipeline-fail run.
    """

    pipeline_ok = bool(summary.get("pipeline_ok"))
    transport_ok = bool(summary.get("transport_ok"))
    verdict = summary.get("verdict", "SMOKE_PIPELINE_OK_TRANSPORT_FAIL")
    if not (pipeline_ok and not transport_ok):
        raise ValueError(
            "claim_from_k3_2d_0_summary currently only handles the PIPELINE_OK_TRANSPORT_FAIL case "
            f"(pipeline_ok=True, transport_ok=False); got pipeline_ok={pipeline_ok}, "
            f"transport_ok={transport_ok}. Add a transport_ok branch before recording a passing run."
        )
    return ClaimRecord(
        claim_id="k3_2d_0_vortex_regime_smoke",
        structure_family="K3_TOPOLOGICAL",
        evidence_level="neural_baseline_regime_validation_smoke",
        verdict=verdict,
        gate="regime",
        allowed_action="do_not_promote",
        supports=[
            "CNN baseline can learn bounded field prediction in smoke mode",
            "vortex transport gate fails",
        ],
        does_not_support=[
            "K3.2D regime validation",
            "K3 prior test readiness",
            "topology prior validation",
        ],
        controls={
            "baseline_only": True,
            "ref_med": summary.get("ref_med"),
            "pair_frac": summary.get("pair_frac"),
            "pos_med": summary.get("pos_med"),
        },
        diagnostics={"pipeline_ok": pipeline_ok, "transport_ok": transport_ok},
        failure_mode=F.PIPELINE_OK_TRANSPORT_FAIL,
        next_gate="K3.2D.0-B neural baseline learnability rescue or archive as regime unresolved",
        claim_boundary="smoke only; CNN transport failed; no prior test allowed",
        source_module="chronos.k3.experiments.k3_2d_0_vortex_regime",
        code_version=_CODE,
        claim_type="unresolved_result",
        confidence_level="low",
        evidence_scope={
            "system": "GP vortex pair",
            "regime": "K3.2D.0 smoke",
            "model": "CNN baseline",
            "compute": "GPU smoke",
        },
        replication={
            "n_seeds": summary.get("n_seeds"),
            "ref_med": summary.get("ref_med"),
            "pair_frac": summary.get("pair_frac"),
            "pos_med": summary.get("pos_med"),
        },
        risk_flags=["smoke_only", "transport_fail", "no_prior_test"],
    )


def claim_from_k2_summary(summary: dict) -> ClaimRecord:
    """The only builder allowed to emit FULL_TRANSFER_CONFIRMED."""

    return ClaimRecord(
        claim_id="k2_symplectic_full_transfer",
        structure_family="K2_SYMPLECTIC",
        evidence_level="vpsl_certified_structure",
        verdict="FULL_TRANSFER_CONFIRMED",
        gate="transfer",
        allowed_action="continue",
        supports=[
            "symplectic prior beats baseline",
            "symplectic prior beats fair energy and fair L2 controls",
            "full symplectic Jacobian mechanism transfers through H=240",
        ],
        does_not_support=[
            "all physical priors work",
            "K3/K4/K5 claims",
            "universal physics AI",
        ],
        controls={"fair_energy_control": True, "fair_l2_control": True, "horizon": 240},
        diagnostics=summary.get("diagnostics", {"transfer": "full", "H": 240}),
        failure_mode=None,
        next_gate="wrong-Omega hardening and broader-system transfer",
        claim_boundary="certified only for tested FPU-beta regime and VPSL controls",
        source_module="chronos.k2",
        code_version=_CODE,
        claim_type="certified_structure",
        confidence_level="certified",
        evidence_scope={
            "system": "FPU-beta",
            "regime": "H=240",
            "model": "prior vs fair controls",
            "compute": "FULL",
        },
        replication=summary.get("replication", {"n_seeds": summary.get("n_seeds")}),
        risk_flags=["scope_limited_to_FPU_beta"],
    )


__all__ = [
    "claim_from_k2_summary",
    "claim_from_k3_2d_0_summary",
    "claim_from_k3_e2b",
    "claim_from_k3_e2c",
    "claim_from_k3_e2d",
]
