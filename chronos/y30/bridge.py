"""Y30 ClaimRecord-style bridge helpers."""

from __future__ import annotations

import copy
from typing import Any

from chronos.y30.core import BANNED_SUBSTRINGS, AppearanceEvent, DependentConditions

GLOBAL_DOES_NOT_SUPPORT = [
    "Buddhist doctrine is proven",
    "external world nonexistence is proven",
    "ultimate reality is certified",
    "Yogacara replaces physics",
    "general consciousness theory",
    "open-domain philosophical conversation",
    "LLM-level language understanding",
]

CLAIM_BOUNDARY = (
    "Y30-Core is a no-LLM toy cognitive substrate for structuring appearance, dependent conditions, "
    "projection, self-grasping diagnostics, seed-trace analogy, and three-nature analysis. It does not "
    "prove Buddhist doctrine, does not prove external-world nonexistence, does not certify ultimate "
    "reality, and does not replace physical validation."
)


def claim_from_y30_core_summary(summary: dict[str, Any]) -> dict[str, Any]:
    utterances = summary.get("utterances", [])
    clean = True
    offending = None
    for utterance in utterances:
        for bad in BANNED_SUBSTRINGS:
            if bad in utterance:
                clean = False
                offending = bad
                break
        if not clean:
            break

    passed = bool(
        clean
        and summary.get("has_appearance", False)
        and summary.get("has_object_construction", False)
        and summary.get("has_projection_or_grasping", False)
        and summary.get("has_three_nature", False)
    )
    risk_flags = []
    if not clean:
        risk_flags.append(f"overclaim_substring:{offending}")
    if not summary.get("has_three_nature", False):
        risk_flags.append("missing_three_nature")
    if not summary.get("has_projection_or_grasping", False):
        risk_flags.append("missing_projection_or_grasping")

    return {
        "claim_id": summary.get("claim_id", "y30_core_demo"),
        "structure_family": "Y30_CORE_COGNITIVE_SUBSTRATE",
        "evidence_level": "toy_cognitive_structure",
        "verdict": "Y30_CORE_TOY_MVP_PASSED" if passed else "Y30_CORE_TOY_MVP_FAILED",
        "gate": "cognitive_substrate",
        "allowed_action": "continue" if passed else "do_not_promote",
        "claim_type": "cognitive_substrate_structure",
        "confidence_level": "toy_mvp",
        "supports": [
            "appearance-first cognitive substrate",
            "object construction is separated from independent-object assertion",
            "projection/self-grasping diagnostics carry does_not_support boundaries",
            "Y30 structures can generate bounded utterances without LLM",
        ],
        "does_not_support": list(GLOBAL_DOES_NOT_SUPPORT),
        "controls": {
            "no_llm": True,
            "no_torch": True,
            "no_model_training": True,
            "template_realizer": True,
            "religious_truth_claim": False,
            "metaphysical_certification": False,
        },
        "diagnostics": {
            "n_utterances": len(utterances),
            "has_appearance": summary.get("has_appearance", False),
            "has_object_construction": summary.get("has_object_construction", False),
            "has_projection_or_grasping": summary.get("has_projection_or_grasping", False),
            "has_three_nature": summary.get("has_three_nature", False),
            "all_utterances_clean": clean,
        },
        "risk_flags": risk_flags,
        "next_gate": "Y30-L1: Appearance vs Object Boundary",
        "claim_boundary": CLAIM_BOUNDARY,
    }


def attach_y30_context(
    claim: dict[str, Any],
    appearance_event: AppearanceEvent,
    dependent_conditions: DependentConditions | None = None,
) -> dict[str, Any]:
    out = copy.deepcopy(claim)
    out["y30_context"] = {
        "appearance": appearance_event.appearance,
        "appearance_id": appearance_event.event_id,
        "appearance_has_evidence": appearance_event.is_affirmable,
        "dependent_conditions": list(dependent_conditions.conditions) if dependent_conditions is not None else [],
        "conditions_unresolved": dependent_conditions is None or not dependent_conditions.conditions,
        "blocked_overclaims": [
            "the appearance has independent self-nature",
            "the conditions are unconditioned / self-existent",
            "Y30 framing changes the physical verdict",
        ],
        "note": "Y30 cognitive-substrate context only; does not change the claim's truth or verdict.",
    }
    return out
