"""Read-only replay and persistence helpers for ClaimRecord objects."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from typing import Any

from chronos.s0.diagnostics_schema import ACT_CONTINUE

from .schema import CLAIM_SUPERSEDED, ClaimRecord


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
        "latest_claim_by_structure_family": {
            family: claim.to_dict() for family, (_index, claim) in latest.items()
        },
    }


def claims_requiring_next_gate(claims: list[ClaimRecord]) -> list[ClaimRecord]:
    return [claim for claim in claims if claim.allowed_action == ACT_CONTINUE and claim.next_gate]


def supersede_claim(claims: list[ClaimRecord], claim_id: str, reason: str | None = None) -> list[ClaimRecord]:
    out = []
    found = False
    for claim in claims:
        if claim.claim_id == claim_id:
            found = True
            out.append(
                replace(
                    claim,
                    claim_status=CLAIM_SUPERSEDED,
                    superseded_reason=reason,
                )
            )
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
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"malformed JSONL at {path}:{line_number}: {exc}") from exc
            try:
                claims.append(ClaimRecord.from_dict(payload))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid claim record at {path}:{line_number}: {exc}") from exc
    return claims


__all__ = [
    "append_claim",
    "claims_requiring_next_gate",
    "load_claims",
    "summarize_claims",
    "supersede_claim",
    "write_claims",
]
