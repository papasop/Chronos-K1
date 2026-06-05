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

CLAIM_ACTIVE = "active"
CLAIM_SUPERSEDED = "superseded"
CLAIM_ARCHIVED = "archived"
ALLOWED_CLAIM_STATUSES = frozenset({CLAIM_ACTIVE, CLAIM_SUPERSEDED, CLAIM_ARCHIVED})

BANNED_ALLOWED_ACTIONS = frozenset({"certified", "promote", "proved", "validated"})


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

        if "CERTIFIED" in self.verdict.upper():
            if self.gate != "transfer" or self.evidence_level != "vpsl_certified_structure":
                raise ValueError(
                    "CERTIFIED verdicts require gate='transfer' and "
                    "evidence_level='vpsl_certified_structure'"
                )

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
    "CLAIM_ACTIVE",
    "CLAIM_ARCHIVED",
    "CLAIM_SUPERSEDED",
    "ClaimRecord",
    "new_timestamp",
]
