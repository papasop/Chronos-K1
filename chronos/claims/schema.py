"""Scientific claim denominator records for Chronos.

ClaimRecord is an audit object, not a recommender or certifier. It records what
an experiment supports, what it does not support, the gate/evidence context, and
the boundary under which a K-word is meaningful.
"""

from __future__ import annotations

import datetime as _datetime
from dataclasses import asdict, dataclass, field
from typing import Any

from chronos.s0.diagnostics_schema import (
    ACT_ARCHIVE,
    ACT_CONTINUE,
    ACT_DO_NOT_PROMOTE,
    ALLOWED_ACTIONS,
)

from .failure_taxonomy import is_known_failure_mode

CLAIM_ACTIVE = "active"
CLAIM_SUPERSEDED = "superseded"
CLAIM_ARCHIVED = "archived"
ALLOWED_CLAIM_STATUSES = frozenset({CLAIM_ACTIVE, CLAIM_SUPERSEDED, CLAIM_ARCHIVED})

BANNED_ALLOWED_ACTIONS = frozenset({"certified", "promote", "proved", "validated"})
CERTIFIED_EVIDENCE = "vpsl_certified_structure"
CERTIFIED_GATE = "transfer"

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


def new_timestamp() -> str:
    """Return a UTC ISO-8601 timestamp."""

    return _datetime.datetime.now(_datetime.timezone.utc).isoformat()


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
            raise ValueError(
                f"allowed_action must be one of {sorted(ALLOWED_ACTIONS)} "
                f"(no certification/promote/proved/validated action); got {self.allowed_action!r}"
            )

        if self.claim_status not in ALLOWED_CLAIM_STATUSES:
            raise ValueError(
                f"claim_status must be one of {sorted(ALLOWED_CLAIM_STATUSES)}, got {self.claim_status!r}"
            )

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
            if self.gate != CERTIFIED_GATE or self.evidence_level != CERTIFIED_EVIDENCE:
                raise ValueError(
                    f"CERTIFIED verdicts require gate={CERTIFIED_GATE!r} and "
                    f"evidence_level={CERTIFIED_EVIDENCE!r}"
                )

        if self.claim_type not in CLAIM_TYPES:
            raise ValueError(f"claim_type must be one of {sorted(CLAIM_TYPES)}; got {self.claim_type!r}")
        if self.confidence_level not in CONFIDENCE_LEVELS:
            raise ValueError(
                f"confidence_level must be one of {sorted(CONFIDENCE_LEVELS)}; got {self.confidence_level!r}"
            )
        if self.confidence_level == "certified" and self.evidence_level != CERTIFIED_EVIDENCE:
            raise ValueError(
                "confidence_level='certified' is reserved for vpsl_certified_structure evidence; "
                "it describes evidence strength and never authorizes an action"
            )
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


__all__ = [
    "ACT_ARCHIVE",
    "ACT_CONTINUE",
    "ACT_DO_NOT_PROMOTE",
    "ALLOWED_CLAIM_STATUSES",
    "BANNED_ALLOWED_ACTIONS",
    "CERTIFIED_EVIDENCE",
    "CERTIFIED_GATE",
    "CLAIM_ACTIVE",
    "CLAIM_ARCHIVED",
    "CLAIM_SUPERSEDED",
    "CLAIM_TYPES",
    "CONFIDENCE_LEVELS",
    "ClaimRecord",
    "new_timestamp",
]
