"""Y20 debate-boundary structures.

Y20 supplies objection / response / required-gate grammar. It records bounded
argument structure only; it does not change physics verdicts or certify
metaphysical claims.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from chronos.y30.core import (
    _check_confidence,
    _check_nonempty,
    _check_str_list,
    _default_boundary,
    _ensure,
    _require_evidence,
)

TOY_BOUNDARY = "argument-structure only; not proof of metaphysical idealism or external-world (non)existence"

OBJECTION_TYPES = frozenset(
    {
        "spatiotemporal_determination",
        "intersubjective_agreement",
        "causal_efficacy",
        "dream_waking_distinction",
        "seed_continuity",
        "no_object_no_cognition",
    }
)

Y20_OBJECTION_TYPES = frozenset(
    {
        "spatiotemporal_determination",
        "shared_appearance",
        "functional_efficacy",
        "dream_waking_distinction",
        "seed_continuity",
        "external_object_boundary",
    }
)

RESPONSE_STRATEGIES = frozenset(
    {
        "dream_analogy",
        "shared_karma",
        "functional_efficacy",
        "dependent_designation",
        "waking_as_special_dream",
        "seed_continuity_model",
    }
)

GLOBAL_DOES_NOT_SUPPORT = [
    "external world nonexistence is proven",
    "metaphysical idealism is proven",
    "mind-only is established as ultimate truth",
    "Buddhist doctrine is proven",
    "the external realist is refuted as a matter of fact",
]


@dataclass
class Y20Objection:
    objection_id: str
    objection_type: str
    realist_claim: str
    evidence: list[str] = field(default_factory=list)
    does_not_support: list[str] = field(default_factory=list)
    claim_boundary: str = ""

    def __post_init__(self) -> None:
        if self.objection_type not in OBJECTION_TYPES:
            raise ValueError(f"unknown objection_type {self.objection_type!r}; expected one of {sorted(OBJECTION_TYPES)}")
        _check_nonempty(self.objection_id, "Y20Objection.objection_id")
        _check_nonempty(self.realist_claim, "Y20Objection.realist_claim")
        _check_str_list(self.evidence, "evidence")
        self.does_not_support = _ensure(
            self.does_not_support,
            [
                "external objects are proven to exist independently",
                "the realist position is established as a matter of fact",
            ],
        )
        self.claim_boundary = _default_boundary(
            self.claim_boundary,
            default="records the external-realist objection; does not itself prove realism",
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Y20Response:
    response_id: str
    objection_id: str
    strategy: str
    response_claim: str
    evidence: list[str] = field(default_factory=list)
    confidence: float = 0.0
    supports: list[str] = field(default_factory=list)
    does_not_support: list[str] = field(default_factory=list)
    claim_boundary: str = ""

    def __post_init__(self) -> None:
        if self.strategy not in RESPONSE_STRATEGIES:
            raise ValueError(f"unknown strategy {self.strategy!r}; expected one of {sorted(RESPONSE_STRATEGIES)}")
        _check_nonempty(self.response_id, "Y20Response.response_id")
        _check_nonempty(self.objection_id, "Y20Response.objection_id")
        _check_nonempty(self.response_claim, "Y20Response.response_claim")
        _check_confidence(self.confidence, "Y20Response.confidence")
        _check_str_list(self.supports, "supports")
        _require_evidence(self.evidence, "Y20Response")
        self.does_not_support = _ensure(
            self.does_not_support,
            GLOBAL_DOES_NOT_SUPPORT + ["the response is a metaphysical proof"],
        )
        self.claim_boundary = _default_boundary(self.claim_boundary, default=TOY_BOUNDARY)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExternalObjectBoundary:
    boundary_id: str
    claim: str = "appearance does not require asserting an independent external object"
    supports: list[str] = field(default_factory=list)
    does_not_support: list[str] = field(default_factory=list)
    claim_boundary: str = ""

    def __post_init__(self) -> None:
        _check_nonempty(self.boundary_id, "ExternalObjectBoundary.boundary_id")
        _check_nonempty(self.claim, "ExternalObjectBoundary.claim")
        _check_str_list(self.supports, "supports")
        self.does_not_support = _ensure(
            self.does_not_support,
            GLOBAL_DOES_NOT_SUPPORT + ["scientific realism is refuted"],
        )
        self.claim_boundary = _default_boundary(
            self.claim_boundary,
            default=(
                "expresses only that appearance need not assert an independent external object; "
                "does not prove external-world nonexistence"
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def debate_claim_record(
    objection: Y20Objection, response: Y20Response, boundary: ExternalObjectBoundary
) -> dict[str, Any]:
    """Build a ClaimRecord-like audit for one bounded debate exchange."""

    if response.objection_id != objection.objection_id:
        raise ValueError("Y20Response.objection_id must match the Y20Objection.objection_id")
    return {
        "claim_id": f"debate_{objection.objection_id}",
        "structure_family": "Y20_DEBATE_STRUCTURE",
        "evidence_level": "toy_argument_structure",
        "verdict": "Y20_DEBATE_STRUCTURE_OK",
        "gate": "argument_structure",
        "allowed_action": "continue",
        "claim_type": "debate_exchange",
        "objection": objection.to_dict(),
        "response": response.to_dict(),
        "boundary": boundary.to_dict(),
        "supports": [
            "the objection is recorded as an external-realist position",
            "the response answers it as a bounded argument move",
            "the boundary states what the exchange does NOT establish",
        ],
        "does_not_support": list(GLOBAL_DOES_NOT_SUPPORT) + ["scientific realism is refuted"],
        "controls": {
            "no_llm": True,
            "no_torch": True,
            "template_realizer": True,
            "metaphysical_certification": False,
            "religious_truth_claim": False,
        },
        "next_gate": "Y20-L1: objection taxonomy coverage",
        "claim_boundary": (
            "Y20-Core models the argument structure of external-object debate. A well-formed exchange "
            "does not prove idealism, does not prove external-world nonexistence, and does not certify "
            "any metaphysical thesis."
        ),
    }
