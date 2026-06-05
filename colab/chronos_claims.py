"""chronos_claims.py - Chronos Claim Denominator Layer v2 (single self-contained file).

Canonical implementation lives in ``chronos/claims/``. This file is a
pure-stdlib standalone mirror for ClaimRecord scientific denominators only. It
includes v2 audit fields (claim_type, confidence_level, evidence_scope,
replication, risk_flags). It does not learn, certify, change S0
recommendations, run experiments, or define new K-families.

Run:
    python colab/chronos_claims.py
    python colab/chronos_claims.py --quiet
"""

from __future__ import annotations

import datetime
import json
import os
import sys
import tempfile
from dataclasses import asdict, dataclass, field, replace
from typing import Any

ACT_CONTINUE = "continue"
ACT_ARCHIVE = "archive"
ACT_DO_NOT_PROMOTE = "do_not_promote"
ALLOWED_ACTIONS = frozenset({ACT_CONTINUE, ACT_ARCHIVE, ACT_DO_NOT_PROMOTE})

K2_SYMPLECTIC = "K2_SYMPLECTIC"
K3_TOPOLOGICAL = "K3_TOPOLOGICAL"

CLAIM_ACTIVE = "active"
CLAIM_SUPERSEDED = "superseded"
CLAIM_ARCHIVED = "archived"
ALLOWED_CLAIM_STATUSES = frozenset({CLAIM_ACTIVE, CLAIM_SUPERSEDED, CLAIM_ARCHIVED})
BANNED_ALLOWED_ACTIONS = frozenset({"certified", "promote", "proved", "validated"})
CLAIM_TYPES = frozenset(
    {
        "positive_evidence",
        "negative_result",
        "unresolved_result",
        "handoff",
        "certified_structure",
        "boundary_note",
    }
)
CONFIDENCE_LEVELS = frozenset({"low", "medium", "high", "certified"})

PIPELINE_OK_TRANSPORT_FAIL = "PIPELINE_OK_TRANSPORT_FAIL"
ACTIVE_NO_ADVANTAGE = "ACTIVE_NO_ADVANTAGE"
RANDOM_ALSO_SUCCEEDS = "RANDOM_ALSO_SUCCEEDS"
TRUTH_STABLE_NEURAL_UNLEARNABLE = "TRUTH_STABLE_NEURAL_UNLEARNABLE"
PENDING_GPU_VALIDATION = "PENDING_GPU_VALIDATION"
CONTROL_DEGENERATE = "CONTROL_DEGENERATE"
REGIME_INVALID = "REGIME_INVALID"
MECHANISM_DECAYS = "MECHANISM_DECAYS"
DIAGNOSTICS_INSUFFICIENT = "DIAGNOSTICS_INSUFFICIENT"
PRIOR_NO_EFFECT = "PRIOR_NO_EFFECT"
KNOWN_FAILURE_MODES = frozenset(
    {
        PIPELINE_OK_TRANSPORT_FAIL,
        ACTIVE_NO_ADVANTAGE,
        RANDOM_ALSO_SUCCEEDS,
        TRUTH_STABLE_NEURAL_UNLEARNABLE,
        PENDING_GPU_VALIDATION,
        CONTROL_DEGENERATE,
        REGIME_INVALID,
        MECHANISM_DECAYS,
        DIAGNOSTICS_INSUFFICIENT,
        PRIOR_NO_EFFECT,
    }
)

_CODE = "claims_v2"
_PASS_E2B = "ACTIVE_DIAGNOSTIC_VALUE_PASSED"


def new_timestamp() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def is_known_failure_mode(mode: str | None) -> bool:
    return mode is None or mode in KNOWN_FAILURE_MODES


def _require_nonempty(value: Any, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


@dataclass(frozen=True)
class ClaimRecord:
    claim_id: str
    structure_family: str
    verdict: str
    evidence_level: str
    gate: str
    diagnostics: dict[str, Any]
    controls: dict[str, Any]
    supports: list[str]
    does_not_support: list[str]
    failure_mode: str | None
    next_gate: str | None
    claim_boundary: str
    allowed_action: str
    timestamp: str = field(default_factory=new_timestamp)
    claim_status: str = CLAIM_ACTIVE
    source_module: str = ""
    code_version: str = "unversioned"
    superseded_by: str | None = None
    superseded_reason: str | None = None
    claim_type: str = "positive_evidence"
    confidence_level: str = "medium"
    evidence_scope: dict[str, Any] = field(default_factory=dict)
    replication: dict[str, Any] = field(default_factory=dict)
    risk_flags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        for field_name in (
            "claim_id",
            "structure_family",
            "verdict",
            "evidence_level",
            "gate",
            "claim_boundary",
            "timestamp",
            "source_module",
        ):
            _require_nonempty(getattr(self, field_name), field_name)
        if self.allowed_action not in ALLOWED_ACTIONS or self.allowed_action in BANNED_ALLOWED_ACTIONS:
            raise ValueError(f"allowed_action must be one of {sorted(ALLOWED_ACTIONS)}; got {self.allowed_action!r}")
        if self.claim_status not in ALLOWED_CLAIM_STATUSES:
            raise ValueError(f"claim_status must be one of {sorted(ALLOWED_CLAIM_STATUSES)}")
        if not self.supports:
            raise ValueError("supports must be non-empty")
        if not self.does_not_support:
            raise ValueError("does_not_support must be non-empty")
        if not isinstance(self.controls, dict):
            raise TypeError("controls must be a dict")
        if not isinstance(self.diagnostics, dict):
            raise TypeError("diagnostics must be a dict")
        if not is_known_failure_mode(self.failure_mode):
            raise ValueError(f"unknown failure_mode: {self.failure_mode!r}")
        if "CERTIFIED" in self.verdict.upper():
            if self.gate != "transfer" or self.evidence_level != "vpsl_certified_structure":
                raise ValueError("CERTIFIED verdicts require transfer gate and vpsl_certified_structure evidence")
        if self.claim_type not in CLAIM_TYPES:
            raise ValueError(f"claim_type must be one of {sorted(CLAIM_TYPES)}; got {self.claim_type!r}")
        if self.confidence_level not in CONFIDENCE_LEVELS:
            raise ValueError(
                f"confidence_level must be one of {sorted(CONFIDENCE_LEVELS)}; got {self.confidence_level!r}"
            )
        if self.confidence_level == "certified" and self.evidence_level != "vpsl_certified_structure":
            raise ValueError("confidence_level='certified' is reserved for vpsl_certified_structure evidence")
        if not isinstance(self.evidence_scope, dict):
            raise TypeError("evidence_scope must be a dict")
        if not isinstance(self.replication, dict):
            raise TypeError("replication must be a dict")
        if not isinstance(self.risk_flags, list):
            raise TypeError("risk_flags must be a list")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ClaimRecord":
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        return cls(**{key: value for key, value in payload.items() if key in known})


def _action_from(result: dict, default: str) -> str:
    rec = result.get("active_rec") or {}
    action = rec.get("allowed_action")
    return action if action in ALLOWED_ACTIONS else default


def claim_from_k3_e2b(result: dict) -> ClaimRecord:
    verdict = result.get("verdict", _PASS_E2B)
    active = result.get("active", {})
    return ClaimRecord(
        claim_id="k3_e2b_active_topology_search",
        structure_family=K3_TOPOLOGICAL,
        evidence_level="toy_active_regime_search",
        verdict=verdict,
        gate="active_search",
        allowed_action=_action_from(result, ACT_CONTINUE),
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
        failure_mode=None if verdict == _PASS_E2B else ACTIVE_NO_ADVANTAGE,
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
    verdict = result.get("verdict", "GP_ACTIVE_NO_ADVANTAGE")
    random_success_count = result.get("random_success_count_20")
    failure_mode = (
        RANDOM_ALSO_SUCCEEDS
        if isinstance(random_success_count, (int, float)) and random_success_count > 5
        else ACTIVE_NO_ADVANTAGE
    )
    action = ACT_ARCHIVE if verdict == "GP_ACTIVE_NO_ADVANTAGE" else _action_from(result, ACT_DO_NOT_PROMOTE)
    return ClaimRecord(
        claim_id="k3_e2c_cheap_gp_active_search",
        structure_family=K3_TOPOLOGICAL,
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
    active = result.get("active", {})
    metrics = active.get("best_metrics") or {}
    return ClaimRecord(
        claim_id="k3_e2d_discriminating_gp_active_search",
        structure_family=K3_TOPOLOGICAL,
        evidence_level="gp_truth_active_search",
        verdict=result.get("verdict", "GP_ACTIVE_DIAGNOSTIC_VALUE_PASSED"),
        gate="truth_active_search",
        allowed_action=_action_from(result, ACT_CONTINUE),
        supports=[
            "guided active search finds a GP vortex regime with better transport diagnostics than random",
            "active reaches transport_ok while random rarely does",
        ],
        does_not_support=["CNN baseline regime validation", "K3 prior validation", "VPSL certification"],
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
    pipeline_ok = bool(summary.get("pipeline_ok"))
    transport_ok = bool(summary.get("transport_ok"))
    verdict = summary.get("verdict", "SMOKE_PIPELINE_OK_TRANSPORT_FAIL")
    if not (pipeline_ok and not transport_ok):
        raise ValueError(
            "claim_from_k3_2d_0_summary currently only handles the PIPELINE_OK_TRANSPORT_FAIL case "
            f"(pipeline_ok=True, transport_ok=False); got pipeline_ok={pipeline_ok}, transport_ok={transport_ok}"
        )
    return ClaimRecord(
        claim_id="k3_2d_0_vortex_regime_smoke",
        structure_family=K3_TOPOLOGICAL,
        evidence_level="neural_baseline_regime_validation_smoke",
        verdict=verdict,
        gate="regime",
        allowed_action=ACT_DO_NOT_PROMOTE,
        supports=["CNN baseline can learn bounded field prediction in smoke mode", "vortex transport gate fails"],
        does_not_support=["K3.2D regime validation", "K3 prior test readiness", "topology prior validation"],
        controls={
            "baseline_only": True,
            "ref_med": summary.get("ref_med"),
            "pair_frac": summary.get("pair_frac"),
            "pos_med": summary.get("pos_med"),
        },
        diagnostics={"pipeline_ok": pipeline_ok, "transport_ok": transport_ok},
        failure_mode=PIPELINE_OK_TRANSPORT_FAIL,
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
    return ClaimRecord(
        claim_id="k2_symplectic_full_transfer",
        structure_family=K2_SYMPLECTIC,
        evidence_level="vpsl_certified_structure",
        verdict="FULL_TRANSFER_CONFIRMED",
        gate="transfer",
        allowed_action=ACT_CONTINUE,
        supports=[
            "symplectic prior beats baseline",
            "symplectic prior beats fair energy and fair L2 controls",
            "full symplectic Jacobian mechanism transfers through H=240",
        ],
        does_not_support=["all physical priors work", "K3/K4/K5 claims", "universal physics AI"],
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


def claim_from_language_grounding_summary(result: dict) -> ClaimRecord:
    passed = bool(result.get("passed"))
    n_assertions = result.get("n_assertions")
    levels = result.get("levels", ["L1", "L2", "L4"])
    return ClaimRecord(
        claim_id="L_VPSL_GROUNDED_LANGUAGE_L1_L2_L4_TOY_MVP",
        structure_family="LANGUAGE_GROUNDING",
        evidence_level="toy_bounded_positive",
        verdict="PASSED_TOY_MVP" if passed else "TOY_MVP_FAILED",
        gate="toy_mechanism",
        allowed_action=ACT_CONTINUE if passed else ACT_DO_NOT_PROMOTE,
        supports=[
            "no-LLM grounded utterance generation from verified semantic claims",
            "L1 not-visible negation",
            "L1 contrastive correction",
            "L2 causal explanation only with causal evidence",
            "L2 correlation is not treated as cause",
            "L4 single-object pronoun reference",
            "does_not_support preserved in unsupported why-questions",
        ],
        does_not_support=[
            "general language understanding",
            "open-domain conversation",
            "LLM-level fluency",
            "autonomous robot intelligence",
            "causal discovery",
            "ambiguous multi-object reference",
            "real-world robot deployment",
        ],
        controls={
            "no_llm": True,
            "no_torch": True,
            "stdlib_only": True,
            "levels_covered": levels,
        },
        diagnostics={"n_assertions": n_assertions, "passed": passed},
        failure_mode=None if passed else DIAGNOSTICS_INSUFFICIENT,
        next_gate="L3 quantifier + L5 temporal + ambiguous reference controls",
        claim_boundary=(
            "toy no-LLM grounded language MVP using hand-authored semantic claims, "
            "controlled examples, and stdlib-only surface realization; not a general language model"
        ),
        source_module="chronos.language_grounding",
        code_version=_CODE,
        claim_type="positive_evidence" if passed else "negative_result",
        confidence_level="medium" if passed else "low",
        evidence_scope={
            "system": "hand-authored semantic claims",
            "regime": "controlled toy examples",
            "model": "template realizer (no LLM)",
            "compute": "CPU stdlib",
        },
        replication={"n_assertions": n_assertions, "deterministic": True},
        risk_flags=[
            "toy_examples",
            "hand_authored_claims",
            "no_real_robot",
            "no_llm_baseline_comparison",
        ],
    )


def _count_by(claims: list[ClaimRecord], attr: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for claim in claims:
        value = getattr(claim, attr)
        key = "NONE" if value is None else str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def summarize_claims(claims: list[ClaimRecord]) -> dict[str, Any]:
    latest: dict[str, tuple[int, ClaimRecord]] = {}
    for index, claim in enumerate(claims):
        current = latest.get(claim.structure_family)
        if current is None or (claim.timestamp, index) >= (current[1].timestamp, current[0]):
            latest[claim.structure_family] = (index, claim)
    return {
        "count_total": len(claims),
        "count_by_structure_family": _count_by(claims, "structure_family"),
        "count_by_evidence_level": _count_by(claims, "evidence_level"),
        "count_by_verdict": _count_by(claims, "verdict"),
        "count_by_allowed_action": _count_by(claims, "allowed_action"),
        "count_by_failure_mode": _count_by(claims, "failure_mode"),
        "count_by_claim_status": _count_by(claims, "claim_status"),
        "count_by_claim_type": _count_by(claims, "claim_type"),
        "count_by_confidence_level": _count_by(claims, "confidence_level"),
        "latest_claim_by_structure_family": {family: claim.to_dict() for family, (_idx, claim) in latest.items()},
    }


def claims_requiring_next_gate(claims: list[ClaimRecord]) -> list[ClaimRecord]:
    return [claim for claim in claims if claim.allowed_action == ACT_CONTINUE and claim.next_gate]


def claims_with_risk_flag(claims: list[ClaimRecord], flag: str) -> list[ClaimRecord]:
    return [claim for claim in claims if flag in claim.risk_flags]


def human_readable_summary(claim: ClaimRecord) -> str:
    support = claim.supports[0] if claim.supports else "a result"
    does_not = ", ".join(claim.does_not_support[:3]) if claim.does_not_support else "nothing further"
    strength = {
        "low": "weak",
        "medium": "moderate",
        "high": "strong",
        "certified": "VPSL-certified",
    }.get(claim.confidence_level, claim.confidence_level)
    next_gate = f"; next gate: {claim.next_gate}" if claim.next_gate else ""
    return (
        f"{claim.claim_id} ({claim.claim_type}, {strength} evidence): {support}. "
        f"It does NOT establish: {does_not}. Action: {claim.allowed_action}{next_gate}."
    )


def supersede_claim(claims: list[ClaimRecord], claim_id: str, reason: str | None = None) -> list[ClaimRecord]:
    out = []
    found = False
    for claim in claims:
        if claim.claim_id == claim_id:
            found = True
            out.append(replace(claim, claim_status=CLAIM_SUPERSEDED, superseded_reason=reason))
        else:
            out.append(claim)
    if not found:
        raise ValueError(f"claim_id not found: {claim_id!r}")
    return out


def append_claim(claim: ClaimRecord, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(claim.to_dict(), ensure_ascii=False) + "\n")


def write_claims(claims: list[ClaimRecord], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for claim in claims:
            handle.write(json.dumps(claim.to_dict(), ensure_ascii=False) + "\n")


def load_claims(path: str) -> list[ClaimRecord]:
    if not os.path.exists(path):
        return []
    claims = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                claims.append(ClaimRecord.from_dict(json.loads(line)))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid claim JSONL at {path}:{line_number}: {exc}") from exc
    return claims


def _tests() -> int:
    count = 0

    def check(condition: bool) -> None:
        nonlocal count
        assert condition
        count += 1

    c = claim_from_k3_e2c({"verdict": "GP_ACTIVE_NO_ADVANTAGE", "active_rec": {"allowed_action": "continue"}})
    check(c.allowed_action == ACT_ARCHIVE)
    check(c.failure_mode == ACTIVE_NO_ADVANTAGE)
    c = claim_from_k3_e2c({"verdict": "GP_ACTIVE_NO_ADVANTAGE", "random_success_count_20": 18})
    check(c.failure_mode == RANDOM_ALSO_SUCCEEDS)
    c = claim_from_k3_e2d({"active": {"best_score": 0.97, "best_metrics": {"mean_pos_err": 0.31}}})
    check(c.next_gate == "K3.2D.0 CNN baseline regime validation")
    check(c.diagnostics["active_mean_pos_err"] == 0.31)
    check(c.claim_type == "positive_evidence")
    c = claim_from_k3_2d_0_summary({"pipeline_ok": True, "transport_ok": False})
    check(c.failure_mode == PIPELINE_OK_TRANSPORT_FAIL)
    check(c.claim_type == "unresolved_result")
    try:
        claim_from_k3_2d_0_summary({"pipeline_ok": True, "transport_ok": True})
        check(False)
    except ValueError:
        check(True)
    try:
        ClaimRecord(
            claim_id="bad",
            structure_family=K3_TOPOLOGICAL,
            verdict="X",
            evidence_level="x",
            gate="regime",
            diagnostics={},
            controls={},
            supports=["x"],
            does_not_support=["y"],
            failure_mode="MADE_UP",
            next_gate=None,
            claim_boundary="x",
            allowed_action=ACT_CONTINUE,
            source_module="test",
        )
        check(False)
    except ValueError:
        check(True)
    try:
        ClaimRecord(
            claim_id="bad2",
            structure_family=K3_TOPOLOGICAL,
            verdict="X",
            evidence_level="x",
            gate="regime",
            diagnostics={},
            controls={},
            supports=["x"],
            does_not_support=["y"],
            failure_mode=None,
            next_gate=None,
            claim_boundary="x",
            allowed_action=ACT_CONTINUE,
            source_module="test",
            confidence_level="certified",
        )
        check(False)
    except ValueError:
        check(True)
    check(
        ClaimRecord(
            claim_id="high-but-blocked",
            structure_family=K3_TOPOLOGICAL,
            verdict="X",
            evidence_level="x",
            gate="regime",
            diagnostics={},
            controls={},
            supports=["x"],
            does_not_support=["y"],
            failure_mode=None,
            next_gate=None,
            claim_boundary="x",
            allowed_action=ACT_DO_NOT_PROMOTE,
            source_module="test",
            confidence_level="high",
        ).allowed_action == ACT_DO_NOT_PROMOTE
    )
    check(is_known_failure_mode(MECHANISM_DECAYS))
    check(is_known_failure_mode(DIAGNOSTICS_INSUFFICIENT))
    check(is_known_failure_mode(PRIOR_NO_EFFECT))
    with tempfile.TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "claims.jsonl")
        append_claim(claim_from_k2_summary({}), path)
        check(load_claims(path)[0].source_module == "chronos.k2")
    summary = summarize_claims([claim_from_k3_e2b({}), claim_from_k3_e2c({}), claim_from_k2_summary({})])
    check(summary["count_by_allowed_action"][ACT_ARCHIVE] == 1)
    check("count_by_claim_type" in summary and "count_by_confidence_level" in summary)
    check(len(claims_requiring_next_gate([claim_from_k3_e2d({}), claim_from_k3_e2c({})])) == 1)
    check(claims_with_risk_flag([claim_from_k3_2d_0_summary({"pipeline_ok": True, "transport_ok": False})], "transport_fail"))
    check("does NOT establish" in human_readable_summary(claim_from_k3_e2c({})))
    physical_claims = [claim_from_k2_summary({}), claim_from_k3_e2c({"random_success_count_20": 18}), claim_from_k3_e2d({})]
    language_claim = claim_from_language_grounding_summary({"passed": True, "n_assertions": 16, "levels": ["L1", "L2", "L4"]})
    mixed = summarize_claims(physical_claims + [language_claim])
    check(mixed["count_total"] == 4)
    check(mixed["count_by_claim_type"]["positive_evidence"] == 2)
    check(language_claim.confidence_level == "medium" and language_claim.allowed_action == ACT_CONTINUE)
    check("general language understanding" in language_claim.does_not_support)
    check("L2 correlation is not treated as cause" in language_claim.supports)
    check("does NOT establish" in human_readable_summary(language_claim))
    return count


if __name__ == "__main__":
    quiet = "--quiet" in sys.argv
    if not quiet:
        claims = [
            claim_from_k3_e2c({}),
            claim_from_k3_e2d({}),
            claim_from_k2_summary({}),
            claim_from_language_grounding_summary({"passed": True, "n_assertions": 16, "levels": ["L1", "L2", "L4"]}),
        ]
        print("=== chronos_claims portable mirror ===")
        print(json.dumps(summarize_claims(claims)["count_by_claim_type"], ensure_ascii=False))
        for claim in claims:
            print("  -", human_readable_summary(claim))
    print("\n=== tests ===")
    print(f"  ok all {_tests()} assertions passed")
