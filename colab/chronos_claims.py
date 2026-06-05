"""Portable Chronos Claims mirror for Colab.

Canonical implementation lives in ``chronos/claims/``. This file is a
pure-stdlib standalone mirror for ClaimRecord scientific denominators only. It
does not learn, certify, change S0 recommendations, run experiments, or define
new K-families.

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

_CODE = "claims_v1"
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

    def __post_init__(self) -> None:
        for field_name in (
            "claim_id",
            "structure_family",
            "verdict",
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
    c = claim_from_k3_2d_0_summary({"pipeline_ok": True, "transport_ok": False})
    check(c.failure_mode == PIPELINE_OK_TRANSPORT_FAIL)
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
    with tempfile.TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "claims.jsonl")
        append_claim(claim_from_k2_summary({}), path)
        check(load_claims(path)[0].source_module == "chronos.k2")
    summary = summarize_claims([claim_from_k3_e2b({}), claim_from_k3_e2c({}), claim_from_k2_summary({})])
    check(summary["count_by_allowed_action"][ACT_ARCHIVE] == 1)
    check(len(claims_requiring_next_gate([claim_from_k3_e2d({}), claim_from_k3_e2c({})])) == 1)
    return count


if __name__ == "__main__":
    quiet = "--quiet" in sys.argv
    if not quiet:
        claims = [claim_from_k3_e2c({}), claim_from_k3_e2d({}), claim_from_k2_summary({})]
        print("=== chronos_claims portable mirror ===")
        print(json.dumps(summarize_claims(claims)["count_by_allowed_action"], ensure_ascii=False))
    print("\n=== tests ===")
    print(f"  ok all {_tests()} assertions passed")
