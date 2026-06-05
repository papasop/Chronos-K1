"""Chronos pure-stdlib core: S0 selector + M0 memory + claim denominator.

This is a single-file portable mirror for bare Colab / standalone use. In the
repository, prefer the packages under ``chronos.s0``, ``chronos.memory``, and
``chronos.claims`` as the canonical source of truth.

No torch, no numpy, no GP/CNN training, no self-evolution.

Run:
    python colab/chronos_core.py
    python colab/chronos_core.py --quiet
"""

from __future__ import annotations

import datetime
import json
import os
import sys
import tempfile
from dataclasses import asdict, dataclass, field, replace
from typing import Any


# ======================================================================================
# S0 selector
# ======================================================================================

K1_LORENTZ = "K1_LORENTZ"
K2_SYMPLECTIC = "K2_SYMPLECTIC"
K3_TOPOLOGICAL = "K3_TOPOLOGICAL"
K4_GAUGE = "K4_GAUGE"
K5_HILBERT = "K5_HILBERT"
UNRESOLVED = "UNRESOLVED"

GATE_REGIME = "regime"
GATE_CONSTRAINT = "constraint"
GATE_MECHANISM = "mechanism"
GATE_TRANSFER = "transfer"

ACT_CONTINUE = "continue"
ACT_ARCHIVE = "archive"
ACT_DO_NOT_PROMOTE = "do_not_promote"
ALLOWED_ACTIONS = frozenset({ACT_CONTINUE, ACT_ARCHIVE, ACT_DO_NOT_PROMOTE})

CONF_LOW = "low"
CONF_MED = "medium"
CONF_HIGH = "high"

CTX_TOPOLOGY = "K3_TOPOLOGY_REGIME"
CTX_SYMPLECTIC = "K2_SYMPLECTIC"

TOPO_TRANSPORT_OK = 0.6
SYMPLECTIC_ERR_OK = 0.2
CAUSAL_VIOLATION_SIGNIF = 0.05
GAUGE_RESIDUAL_SIGNIF = 0.1
UNITARITY_ERR_SIGNIF = 0.1

KNOWN_DIAGNOSTICS = {
    "energy_drift",
    "symplectic_jacobian_error",
    "symplectic_improves_vs_controls",
    "causal_violation_rate",
    "topological_transport_score",
    "object_tracking_valid",
    "gauge_residual",
    "unitarity_error",
    "baseline_divergence",
    "field_learnable",
    "diagnostic_context",
}


@dataclass(frozen=True)
class Recommendation:
    candidate_family: str
    confidence: str
    reason: str
    next_vpsl_gate: str | None
    allowed_action: str

    def __post_init__(self) -> None:
        if self.allowed_action not in ALLOWED_ACTIONS:
            raise ValueError(
                f"allowed_action must be one of {sorted(ALLOWED_ACTIONS)} "
                f"(S0 never certifies); got {self.allowed_action!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _present(diagnostics: dict[str, Any], key: str) -> bool:
    return key in diagnostics and diagnostics[key] is not None


def _check_score01(diagnostics: dict[str, Any], key: str) -> None:
    if _present(diagnostics, key) and not 0.0 <= diagnostics[key] <= 1.0:
        raise ValueError(f"{key} must be in [0,1], got {diagnostics[key]!r}")


def recommend(diagnostics: dict[str, Any] | None) -> Recommendation:
    """Recommend a K-family and next VPSL gate. S0 never certifies."""

    d = diagnostics or {}
    if not any(_present(d, key) for key in KNOWN_DIAGNOSTICS):
        return Recommendation(
            UNRESOLVED,
            CONF_LOW,
            "No recognized diagnostics provided; cannot recommend a structure family.",
            None,
            ACT_DO_NOT_PROMOTE,
        )

    _check_score01(d, "topological_transport_score")
    field_ok = d.get("field_learnable") is True
    transport_known = _present(d, "topological_transport_score") or _present(d, "object_tracking_valid")
    transport_fail = False
    if transport_known:
        if d.get("object_tracking_valid") is False:
            transport_fail = True
        if _present(d, "topological_transport_score") and d["topological_transport_score"] < TOPO_TRANSPORT_OK:
            transport_fail = True

    if field_ok and transport_fail and d.get("diagnostic_context") == CTX_TOPOLOGY:
        return Recommendation(
            K3_TOPOLOGICAL,
            CONF_LOW,
            "Field prediction is learnable but the topological object is not transported "
            "in a topology-regime test. Low field error does not imply topology success.",
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
        return Recommendation(K4_GAUGE, CONF_LOW, "Non-trivial gauge residual present.", GATE_REGIME, ACT_CONTINUE)

    if _present(d, "unitarity_error") and d["unitarity_error"] > UNITARITY_ERR_SIGNIF:
        return Recommendation(K5_HILBERT, CONF_LOW, "Unitarity error suggests a Hilbert-space candidate.", GATE_REGIME, ACT_CONTINUE)

    if _present(d, "topological_transport_score") and d["topological_transport_score"] >= TOPO_TRANSPORT_OK:
        return Recommendation(
            K3_TOPOLOGICAL,
            CONF_MED,
            "Topological object is transported; topology family is a candidate.",
            GATE_CONSTRAINT,
            ACT_CONTINUE,
        )

    if field_ok and transport_fail:
        return Recommendation(
            K3_TOPOLOGICAL,
            CONF_LOW,
            "Field prediction is learnable but the topological object is not transported; treat as unresolved.",
            GATE_REGIME,
            ACT_DO_NOT_PROMOTE,
        )

    return Recommendation(UNRESOLVED, CONF_LOW, "Diagnostics present but none decisive.", None, ACT_DO_NOT_PROMOTE)


# ======================================================================================
# M0 memory logging
# ======================================================================================


def new_timestamp() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


@dataclass
class MemoryEvent:
    timestamp: str
    run_id: str
    module: str
    experiment_kind: str
    verdict: str
    candidate_family: str
    allowed_action: str
    score: float | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    claim_boundary: str = ""
    code_version: str = "unversioned"

    def __post_init__(self) -> None:
        if self.allowed_action not in ALLOWED_ACTIONS:
            raise ValueError(f"allowed_action must be one of {sorted(ALLOWED_ACTIONS)}; got {self.allowed_action!r}")
        if not isinstance(self.claim_boundary, str) or not self.claim_boundary.strip():
            raise ValueError("claim_boundary must be a non-empty string")
        if self.score is not None:
            self.score = float(self.score)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MemoryEvent":
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        return cls(**{key: value for key, value in payload.items() if key in known})


def append_event(event: MemoryEvent, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")


def load_events(path: str) -> list[MemoryEvent]:
    if not os.path.exists(path):
        return []
    events = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                events.append(MemoryEvent.from_dict(json.loads(line)))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid memory JSONL at {path}:{line_number}: {exc}") from exc
    return events


def _count_by(items: list[Any], attr: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        key = getattr(item, attr)
        key = "NONE" if key is None else str(key)
        counts[key] = counts.get(key, 0) + 1
    return counts


def summarize_events(events: list[MemoryEvent]) -> dict[str, Any]:
    latest_by_module: dict[str, MemoryEvent] = {}
    for event in events:
        previous = latest_by_module.get(event.module)
        if previous is None or event.timestamp >= previous.timestamp:
            latest_by_module[event.module] = event
    return {
        "count_total": len(events),
        "count_by_module": _count_by(events, "module"),
        "count_by_verdict": _count_by(events, "verdict"),
        "count_by_candidate_family": _count_by(events, "candidate_family"),
        "latest_event_by_module": {key: value.to_dict() for key, value in latest_by_module.items()},
    }


# ======================================================================================
# Claim denominator
# ======================================================================================

CLAIM_ACTIVE = "active"
CLAIM_SUPERSEDED = "superseded"
CLAIM_ARCHIVED = "archived"
ALLOWED_CLAIM_STATUSES = frozenset({CLAIM_ACTIVE, CLAIM_SUPERSEDED, CLAIM_ARCHIVED})
BANNED_ALLOWED_ACTIONS = frozenset({"certified", "promote", "proved", "validated"})

PIPELINE_OK_TRANSPORT_FAIL = "PIPELINE_OK_TRANSPORT_FAIL"
ACTIVE_NO_ADVANTAGE = "ACTIVE_NO_ADVANTAGE"
RANDOM_ALSO_SUCCEEDS = "RANDOM_ALSO_SUCCEEDS"
TRUTH_STABLE_NEURAL_UNLEARNABLE = "TRUTH_STABLE_NEURAL_UNLEARNABLE"
PENDING_GPU_VALIDATION = "PENDING_GPU_VALIDATION"
CONTROL_DEGENERATE = "CONTROL_DEGENERATE"
REGIME_INVALID = "REGIME_INVALID"
KNOWN_FAILURE_MODES = frozenset(
    {
        PIPELINE_OK_TRANSPORT_FAIL,
        ACTIVE_NO_ADVANTAGE,
        RANDOM_ALSO_SUCCEEDS,
        TRUTH_STABLE_NEURAL_UNLEARNABLE,
        PENDING_GPU_VALIDATION,
        CONTROL_DEGENERATE,
        REGIME_INVALID,
    }
)


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
    controls: list[str]
    supports: list[str]
    does_not_support: list[str]
    failure_mode: str | None
    next_gate: str | None
    claim_boundary: str
    allowed_action: str
    timestamp: str = field(default_factory=new_timestamp)
    claim_status: str = CLAIM_ACTIVE
    superseded_by: str | None = None
    superseded_reason: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("claim_id", "structure_family", "verdict", "claim_boundary", "timestamp"):
            _require_nonempty(getattr(self, field_name), field_name)
        if self.allowed_action not in ALLOWED_ACTIONS or self.allowed_action in BANNED_ALLOWED_ACTIONS:
            raise ValueError(f"allowed_action must be one of {sorted(ALLOWED_ACTIONS)}; got {self.allowed_action!r}")
        if self.claim_status not in ALLOWED_CLAIM_STATUSES:
            raise ValueError(f"claim_status must be one of {sorted(ALLOWED_CLAIM_STATUSES)}")
        if not self.supports:
            raise ValueError("supports must be non-empty")
        if not self.does_not_support:
            raise ValueError("does_not_support must be non-empty")
        if not is_known_failure_mode(self.failure_mode):
            raise ValueError(f"unknown failure_mode: {self.failure_mode!r}")
        if "CERTIFIED" in self.verdict.upper():
            if self.gate != GATE_TRANSFER or self.evidence_level != "vpsl_certified_structure":
                raise ValueError("CERTIFIED verdicts require transfer gate and vpsl_certified_structure evidence")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ClaimRecord":
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        return cls(**{key: value for key, value in payload.items() if key in known})


def claim_from_k3_e2b(summary: dict[str, Any]) -> ClaimRecord:
    verdict = str(summary.get("verdict", "ACTIVE_DIAGNOSTIC_VALUE_PASSED"))
    passed = verdict == "ACTIVE_DIAGNOSTIC_VALUE_PASSED"
    return ClaimRecord(
        claim_id=str(summary.get("claim_id") or summary.get("run_id") or f"k3_e2b:{new_timestamp()}"),
        structure_family=K3_TOPOLOGICAL,
        verdict=verdict,
        evidence_level="toy_active_regime_search",
        gate=GATE_REGIME,
        diagnostics={"active_best_score": summary.get("active_best_score")},
        controls=["random_action_search"],
        supports=["guided active search beats blind random search on a transparent toy topology landscape"]
        if passed
        else ["toy topology search ran but did not establish active advantage"],
        does_not_support=["K3 prior validation", "GP truth", "CNN learnability", "certification"],
        failure_mode=None if passed else ACTIVE_NO_ADVANTAGE,
        next_gate="K3-E2c cheap GP truth-only active search" if passed else None,
        claim_boundary="toy search landscape only; not GP truth, not K3 prior validation",
        allowed_action=ACT_CONTINUE if passed else ACT_DO_NOT_PROMOTE,
    )


def claim_from_k3_e2c(summary: dict[str, Any]) -> ClaimRecord:
    verdict = str(summary.get("verdict", "GP_ACTIVE_NO_ADVANTAGE"))
    no_advantage = verdict == "GP_ACTIVE_NO_ADVANTAGE"
    return ClaimRecord(
        claim_id=str(summary.get("claim_id") or summary.get("run_id") or f"k3_e2c:{new_timestamp()}"),
        structure_family=K3_TOPOLOGICAL,
        verdict=verdict,
        evidence_level="cheap_gp_truth_active_search",
        gate=GATE_REGIME,
        diagnostics={"active_best_score": summary.get("active_best_score")},
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
    )


def claim_from_k3_e2d(summary: dict[str, Any]) -> ClaimRecord:
    verdict = str(summary.get("verdict", "GP_ACTIVE_DIAGNOSTIC_VALUE_PASSED"))
    passed = verdict == "GP_ACTIVE_DIAGNOSTIC_VALUE_PASSED"
    return ClaimRecord(
        claim_id=str(summary.get("claim_id") or summary.get("run_id") or f"k3_e2d:{new_timestamp()}"),
        structure_family=K3_TOPOLOGICAL,
        verdict=verdict,
        evidence_level="gp_truth_active_search",
        gate=GATE_REGIME,
        diagnostics={"active_best_score": summary.get("active_best_score")},
        controls=["random_action_search"],
        supports=["guided active search finds a cheap GP vortex-position regime better than random control"]
        if passed
        else ["cheap GP active search ran but did not establish active diagnostic value"],
        does_not_support=["CNN learnability", "K3 prior validation", "certification", "full-resolution GP"],
        failure_mode=None if passed else ACTIVE_NO_ADVANTAGE,
        next_gate="K3.2D.0 CNN baseline regime validation" if passed else None,
        claim_boundary="truth-level cheap GP only; not CNN validation, not K3 prior validation",
        allowed_action=ACT_CONTINUE if passed else ACT_DO_NOT_PROMOTE,
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
        claim_id=str(summary.get("claim_id") or summary.get("run_id") or f"k3_2d_0:{new_timestamp()}"),
        structure_family=K3_TOPOLOGICAL,
        verdict=verdict,
        evidence_level="cnn_baseline_regime_validation",
        gate=GATE_REGIME,
        diagnostics={"pipeline_ok": pipeline_ok, "transport_ok": transport_ok},
        controls=["baseline_only_no_prior"],
        supports=["CNN baseline regime is learnable and transports the vortex pair"]
        if passed
        else ["K3.2D.0 regime validation ran without testing any prior"],
        does_not_support=["K3 prior validation", "certification", "field prediction alone implies topology"],
        failure_mode=failure_mode,
        next_gate="K3.2D.1 topological prior test" if passed else None,
        claim_boundary="baseline-only CNN regime validation; no topology prior tested",
        allowed_action=ACT_CONTINUE if passed else ACT_DO_NOT_PROMOTE,
    )


def claim_from_k2_summary(summary: dict[str, Any]) -> ClaimRecord:
    verdict = str(summary.get("verdict", "FULL_TRANSFER_CONFIRMED"))
    certified = "CERTIFIED" in verdict.upper()
    return ClaimRecord(
        claim_id=str(summary.get("claim_id") or summary.get("run_id") or f"k2:{new_timestamp()}"),
        structure_family=K2_SYMPLECTIC,
        verdict=verdict,
        evidence_level="vpsl_certified_structure" if certified else "vpsl_transfer_confirmed",
        gate=GATE_TRANSFER if certified or "TRANSFER" in verdict.upper() else GATE_MECHANISM,
        diagnostics={"transfer_horizon": summary.get("transfer_horizon")},
        controls=["energy_control", "l2_control", "wrong_omega_control"],
        supports=["symplectic structure survives VPSL transfer/mechanism controls"],
        does_not_support=["universal claim about all physical systems", "new K-family beyond K2"],
        failure_mode=None,
        next_gate=None,
        claim_boundary="K2 claim is bounded to the archived VPSL transfer/mechanism evidence",
        allowed_action=ACT_ARCHIVE if summary.get("archived") else ACT_CONTINUE,
    )


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
        "latest_claim_by_structure_family": {family: claim.to_dict() for family, (_idx, claim) in latest.items()},
    }


def claims_requiring_next_gate(claims: list[ClaimRecord]) -> list[ClaimRecord]:
    return [claim for claim in claims if claim.allowed_action == ACT_CONTINUE and claim.next_gate]


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


# ======================================================================================
# Self-tests
# ======================================================================================


def _memory_event(**overrides: Any) -> MemoryEvent:
    base = {
        "timestamp": new_timestamp(),
        "run_id": "r1",
        "module": "k3.active",
        "experiment_kind": "k3_e2d",
        "verdict": "GP_ACTIVE_DIAGNOSTIC_VALUE_PASSED",
        "candidate_family": K3_TOPOLOGICAL,
        "allowed_action": ACT_CONTINUE,
        "score": 1.0,
        "payload": {"a": 1},
        "claim_boundary": "toy only",
        "code_version": "v1",
    }
    base.update(overrides)
    return MemoryEvent(**base)


def _claim(**overrides: Any) -> ClaimRecord:
    base = {
        "claim_id": "c1",
        "structure_family": K3_TOPOLOGICAL,
        "verdict": "GP_ACTIVE_DIAGNOSTIC_VALUE_PASSED",
        "evidence_level": "gp_truth_active_search",
        "gate": GATE_REGIME,
        "diagnostics": {"score": 1.0},
        "controls": ["random_action_search"],
        "supports": ["active search finds a GP vortex regime better than random"],
        "does_not_support": ["K3 prior validation"],
        "failure_mode": None,
        "next_gate": "K3.2D.0 CNN baseline regime validation",
        "claim_boundary": "truth-level GP only; not prior validation",
        "allowed_action": ACT_CONTINUE,
        "timestamp": "2026-06-05T00:00:00+00:00",
    }
    base.update(overrides)
    return ClaimRecord(**base)


def _tests_s0() -> int:
    n = 0

    def check(condition: bool) -> None:
        nonlocal n
        assert condition
        n += 1

    check(recommend({}).candidate_family == UNRESOLVED)
    check(recommend({}).allowed_action == ACT_DO_NOT_PROMOTE)
    rec = recommend({"field_learnable": True, "object_tracking_valid": False, "diagnostic_context": CTX_TOPOLOGY})
    check(rec.candidate_family == K3_TOPOLOGICAL)
    check(rec.allowed_action == ACT_DO_NOT_PROMOTE)
    rec = recommend({"field_learnable": True, "object_tracking_valid": False, "symplectic_improves_vs_controls": True})
    check(rec.candidate_family == K2_SYMPLECTIC)
    rec = recommend({"symplectic_improves_vs_controls": True, "symplectic_jacobian_error": 0.1})
    check(rec.candidate_family == K2_SYMPLECTIC and rec.allowed_action == ACT_CONTINUE)
    rec = recommend({"causal_violation_rate": 0.2})
    check(rec.candidate_family == K1_LORENTZ)
    rec = recommend({"topological_transport_score": 0.8, "object_tracking_valid": True})
    check(rec.candidate_family == K3_TOPOLOGICAL)
    try:
        recommend({"topological_transport_score": 3.0})
        check(False)
    except ValueError:
        check(True)
    try:
        Recommendation(K2_SYMPLECTIC, CONF_HIGH, "x", GATE_TRANSFER, "certified")
        check(False)
    except ValueError:
        check(True)
    return n


def _tests_memory() -> int:
    n = 0

    def check(condition: bool) -> None:
        nonlocal n
        assert condition
        n += 1

    with tempfile.TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "events.jsonl")
        append_event(_memory_event(run_id="a"), path)
        append_event(_memory_event(run_id="b", allowed_action=ACT_DO_NOT_PROMOTE), path)
        loaded = load_events(path)
        check(len(loaded) == 2)
        check(loaded[0].run_id == "a" and loaded[1].run_id == "b")
        summary = summarize_events(loaded)
        check(summary["count_total"] == 2)
        check(summary["count_by_module"] == {"k3.active": 2})
    for bad in ("", "   ", None):
        try:
            _memory_event(claim_boundary=bad)
            check(False)
        except ValueError:
            check(True)
    return n


def _tests_claims() -> int:
    n = 0

    def check(condition: bool) -> None:
        nonlocal n
        assert condition
        n += 1

    claim = _claim()
    check(ClaimRecord.from_dict(claim.to_dict()) == claim)
    for action in ("certified", "promote", "proved", "validated"):
        try:
            _claim(allowed_action=action)
            check(False)
        except ValueError:
            check(True)
    check(_claim(verdict="FULL_REGIME_VALIDATED").verdict == "FULL_REGIME_VALIDATED")
    try:
        _claim(failure_mode="MADE_UP")
        check(False)
    except ValueError:
        check(True)
    try:
        _claim(verdict="K2_CERTIFIED", gate=GATE_REGIME, evidence_level="gp_truth_active_search")
        check(False)
    except ValueError:
        check(True)
    check(_claim(verdict="K2_CERTIFIED", gate=GATE_TRANSFER, evidence_level="vpsl_certified_structure").verdict)
    e2c = claim_from_k3_e2c({"verdict": "GP_ACTIVE_NO_ADVANTAGE"})
    check(e2c.allowed_action == ACT_DO_NOT_PROMOTE)
    check(e2c.next_gate is None)
    e2d = claim_from_k3_e2d({"verdict": "GP_ACTIVE_DIAGNOSTIC_VALUE_PASSED"})
    check(e2d.next_gate == "K3.2D.0 CNN baseline regime validation")
    k32 = claim_from_k3_2d_0_summary({"pipeline_ok": True, "transport_ok": False})
    check(k32.failure_mode == PIPELINE_OK_TRANSPORT_FAIL and k32.allowed_action == ACT_DO_NOT_PROMOTE)
    k2 = claim_from_k2_summary({"verdict": "K2_VPSL_CERTIFIED"})
    check(k2.evidence_level == "vpsl_certified_structure" and k2.gate == GATE_TRANSFER)
    claims = [e2c, e2d, k32, k2]
    summary = summarize_claims(claims)
    check(summary["count_total"] == 4)
    check(summary["count_by_allowed_action"][ACT_DO_NOT_PROMOTE] == 2)
    check(len(claims_requiring_next_gate(claims)) == 1)
    with tempfile.TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "claims.jsonl")
        append_claim(e2d, path)
        check(load_claims(path)[0] == e2d)
    return n


if __name__ == "__main__":
    quiet = "--quiet" in sys.argv
    if not quiet:
        print("=== chronos_core: S0 + memory + claims (pure stdlib) ===")
        print("[S0]", recommend({"symplectic_improves_vs_controls": True}).to_dict())
        claims = [
            claim_from_k3_e2c({"verdict": "GP_ACTIVE_NO_ADVANTAGE"}),
            claim_from_k3_e2d({"verdict": "GP_ACTIVE_DIAGNOSTIC_VALUE_PASSED"}),
        ]
        print("[claims]", summarize_claims(claims)["count_by_allowed_action"])
    n_s0 = _tests_s0()
    n_mem = _tests_memory()
    n_claims = _tests_claims()
    print("\n=== tests ===")
    print(f"  S0:     ok {n_s0} assertions")
    print(f"  memory: ok {n_mem} assertions")
    print(f"  claims: ok {n_claims} assertions")
    print(f"  TOTAL:  ok all {n_s0 + n_mem + n_claims} assertions passed")
