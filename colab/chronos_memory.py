"""Portable S0-M0 memory logging mirror for Colab.

Canonical implementation lives in chronos/memory/. This mirror is standalone:
pure stdlib, JSONL on disk, no cloud, no server, and no feedback into S0.
"""

from __future__ import annotations

import datetime
import json
import os
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

ALLOWED_ACTIONS = frozenset({"continue", "archive", "do_not_promote"})


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
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def append_event(event: MemoryEvent, path: str) -> None:
    if not isinstance(event, MemoryEvent):
        raise TypeError(f"append_event expects a MemoryEvent, got {type(event).__name__}")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")


def load_events(path: str) -> list[MemoryEvent]:
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


def _event(**kwargs) -> MemoryEvent:
    base = {
        "timestamp": new_timestamp(),
        "run_id": "r1",
        "module": "k3.active",
        "experiment_kind": "k3_e2b",
        "verdict": "ACTIVE_DIAGNOSTIC_VALUE_PASSED",
        "candidate_family": "K3_TOPOLOGICAL",
        "allowed_action": "continue",
        "score": 1.0,
        "payload": {"a": 1},
        "claim_boundary": "toy only",
        "code_version": "v1",
    }
    base.update(kwargs)
    return MemoryEvent(**base)


def _tests() -> int:
    count = 0

    def check(condition):
        nonlocal count
        assert condition
        count += 1

    with tempfile.TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "events.jsonl")
        first = _event(run_id="a")
        second = _event(run_id="b", allowed_action="do_not_promote", verdict="X")
        append_event(first, path)
        append_event(second, path)
        loaded = load_events(path)
        check(len(loaded) == 2)
        check(loaded[0].run_id == "a" and loaded[1].run_id == "b")
        check(loaded[0].to_dict() == first.to_dict())
        check(loaded[1].allowed_action == "do_not_promote")

    for bad in ("", "   ", None):
        try:
            _event(claim_boundary=bad)
            check(False)
        except ValueError:
            check(True)

    for bad in ("certified", "certify", "promote", "PROMOTE", "done"):
        try:
            _event(allowed_action=bad)
            check(False)
        except ValueError:
            check(True)
    for action in ALLOWED_ACTIONS:
        check(_event(allowed_action=action).allowed_action == action)

    try:
        _event(score="not-a-number")
        check(False)
    except ValueError:
        check(True)
    check(_event(score=None).score is None)

    events = [
        _event(module="k3.active", verdict="PASS", candidate_family="K3_TOPOLOGICAL", timestamp="2026-01-01T00:00:00+00:00"),
        _event(module="k3.active", verdict="PASS", candidate_family="K3_TOPOLOGICAL", timestamp="2026-01-02T00:00:00+00:00"),
        _event(module="s0.e1", verdict="FAIL", candidate_family="K2_SYMPLECTIC", timestamp="2026-01-03T00:00:00+00:00"),
    ]
    summary = summarize_events(events)
    check(summary["count_total"] == 3)
    check(summary["count_by_module"] == {"k3.active": 2, "s0.e1": 1})
    check(summary["count_by_verdict"] == {"PASS": 2, "FAIL": 1})
    check(summary["count_by_candidate_family"] == {"K3_TOPOLOGICAL": 2, "K2_SYMPLECTIC": 1})
    check(summary["latest_event_by_module"]["k3.active"]["timestamp"] == "2026-01-02T00:00:00+00:00")
    check(summary["latest_event_by_module"]["s0.e1"]["timestamp"] == "2026-01-03T00:00:00+00:00")

    with tempfile.TemporaryDirectory() as temp_dir:
        check(load_events(os.path.join(temp_dir, "missing.jsonl")) == [])

    k3 = _event(module="k3.active", claim_boundary="toy search landscape only; NOT GP truth, NOT K3 prior validation.")
    check("toy" in k3.claim_boundary.lower() and "not gp truth" in k3.claim_boundary.lower())

    with tempfile.TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "bad.jsonl")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write('{"not":"valid-event but valid-json"}\n{ broken json\n')
        try:
            load_events(path)
            check(False)
        except ValueError:
            check(True)

    return count


if __name__ == "__main__":
    quiet = "--quiet" in sys.argv
    if not quiet:
        print("=== S0-M0 Memory Logging Layer ===")
        print("logging only; not a learning layer; no feedback into S0\n")
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "events.jsonl")
            append_event(_event(run_id="demo-1", claim_boundary="toy landscape only; not GP truth"), path)
            append_event(
                _event(
                    run_id="demo-2",
                    module="s0.e1",
                    verdict="K2_MEASURED",
                    candidate_family="K2_SYMPLECTIC",
                    claim_boundary="toy pendulum sim; symplectic measured, not certified",
                ),
                path,
            )
            summary = summarize_events(load_events(path))
            print(f"  wrote 2 events to {path}")
            print(
                "  "
                + json.dumps(
                    {
                        "count_total": summary["count_total"],
                        "count_by_module": summary["count_by_module"],
                        "count_by_verdict": summary["count_by_verdict"],
                        "count_by_candidate_family": summary["count_by_candidate_family"],
                    },
                    ensure_ascii=False,
                )
            )
    print("\n=== tests ===")
    print(f"  ok all {_tests()} assertions passed")
