"""S0-M0 append-only memory logging.

S0-M0 is a logging layer, not a learning layer. It records what Chronos ran,
what verdict or recommendation was emitted, and the claim boundary for that
record. It does not change S0 recommendations and does not feed back into
selector behavior.
"""

from __future__ import annotations

import datetime
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from chronos.s0.diagnostics_schema import ALLOWED_ACTIONS


@dataclass
class MemoryEvent:
    timestamp: str
    run_id: str
    module: str
    experiment_kind: str
    verdict: str
    candidate_family: str
    allowed_action: str
    score: Optional[float] = None
    payload: dict[str, Any] = field(default_factory=dict)
    claim_boundary: str = ""
    code_version: str = "unversioned"

    def __post_init__(self) -> None:
        if self.allowed_action not in ALLOWED_ACTIONS:
            raise ValueError(
                f"allowed_action must be one of {sorted(ALLOWED_ACTIONS)} "
                f"(S0 never certifies); got {self.allowed_action!r}"
            )
        if not isinstance(self.claim_boundary, str) or not self.claim_boundary.strip():
            raise ValueError(
                "claim_boundary must be a non-empty string: every recorded result must "
                "state what it does and does not establish (e.g. 'toy only; not GP truth')."
            )
        if self.score is not None:
            try:
                self.score = float(self.score)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"score must be a number or None; got {self.score!r}") from exc

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryEvent":
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        return cls(**{key: value for key, value in data.items() if key in known})


def new_timestamp() -> str:
    """Return a UTC ISO-8601 timestamp."""

    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def append_event(event: MemoryEvent, path: str) -> None:
    """Append a validated event to a JSONL file."""

    if not isinstance(event, MemoryEvent):
        raise TypeError(f"append_event expects a MemoryEvent, got {type(event).__name__}")
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")


def load_events(path: str) -> list[MemoryEvent]:
    """Load validated events from a JSONL file. Missing files return an empty list."""

    if not os.path.exists(path):
        return []
    events = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"malformed JSONL at {path}:{line_number}: {exc}") from exc
            try:
                events.append(MemoryEvent.from_dict(payload))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid event record at {path}:{line_number}: {exc}") from exc
    return events


def _count_by(events: list[MemoryEvent], attr: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in events:
        key = getattr(event, attr)
        counts[key] = counts.get(key, 0) + 1
    return counts


def summarize_events(events: list[MemoryEvent]) -> dict[str, Any]:
    """Return read-only descriptive counts. This does not learn or influence S0."""

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
        "latest_event_by_module": {module: event.to_dict() for module, event in latest_by_module.items()},
    }
