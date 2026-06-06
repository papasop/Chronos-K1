"""Standard Y20 objection library."""

from __future__ import annotations

from typing import Any

from chronos.y20.core import (
    ExternalObjectBoundary,
    Y20Objection,
    Y20Response,
    debate_claim_record,
)
from chronos.y20.realizer import (
    realize_external_object_boundary,
    realize_objection,
    realize_response,
)

STANDARD = {
    "Y20-O1": ("spatiotemporal_determination", "dream_analogy", "Y20-O2: shared appearance"),
    "Y20-O2": ("intersubjective_agreement", "shared_karma", "Y20-O3: functional efficacy"),
    "Y20-O3": ("causal_efficacy", "functional_efficacy", "Y20-O4: dream / waking distinction"),
    "Y20-O4": ("dream_waking_distinction", "waking_as_special_dream", "Y20-O5: seed / latent continuity"),
    "Y20-O5": ("seed_continuity", "seed_continuity_model", "Y20-O6: external-object boundary"),
}

STANDARD_OBJECTION_IDS = ("Y20-O1", "Y20-O2", "Y20-O3", "Y20-O4", "Y20-O5", "Y20-O6")


def build_standard_objection(oid: str, evidence: list[str] | None = None) -> dict[str, Any]:
    """Build one standard Y20 objection entry in bounded audit format."""

    if oid == "Y20-O6":
        return build_external_object_boundary_rule()
    if oid not in STANDARD:
        raise ValueError(f"unknown standard objection id {oid!r}; expected one of {list(STANDARD_OBJECTION_IDS)}")
    objection_type, strategy, next_gate = STANDARD[oid]
    ev = evidence or ["argument_basis"]
    obj = Y20Objection(f"{oid}_obj", objection_type, f"external-realist objection: {objection_type}")
    resp = Y20Response(
        f"{oid}_resp",
        f"{oid}_obj",
        strategy,
        f"Yogacara response via {strategy}",
        evidence=ev,
        confidence=0.0,
    )
    boundary = ExternalObjectBoundary(f"{oid}_boundary")
    return {
        "id": oid,
        "objection": realize_objection(obj),
        "response": realize_response(resp),
        "boundary": realize_external_object_boundary(boundary),
        "does_not_support": list(resp.does_not_support),
        "next_gate": next_gate,
        "_record": debate_claim_record(obj, resp, boundary),
    }


def build_external_object_boundary_rule() -> dict[str, Any]:
    """O6: the explicit may-say / may-not-say boundary rule."""

    boundary = ExternalObjectBoundary("Y20-O6_boundary")
    return {
        "id": "Y20-O6",
        "objection": "边界问题：什么可以说，什么不能说？",
        "response": "可以说：显现不必预设一个独立外境作为认识的前提（外境不必预设）。",
        "boundary": realize_external_object_boundary(boundary),
        "may_say": ["显现不必预设独立外境", "「外境」可作为依条件的假名"],
        "may_not_say": ["外境不存在已被证明", "唯心论已被证明为终极真理", "科学实在论已被驳倒"],
        "does_not_support": list(boundary.does_not_support),
        "next_gate": "Y20-L1: objection taxonomy coverage",
    }


def build_all_standard_objections() -> list[dict[str, Any]]:
    return [build_standard_objection(oid) for oid in STANDARD_OBJECTION_IDS]
