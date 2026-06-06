"""Y20 <-> Y30 bridge helpers."""

from __future__ import annotations

from typing import Any

from chronos.y20.core import Y20Objection
from chronos.y30.core import AppearanceEvent, DependentConditions, ProjectionClaim, SeedTrace


def attach_y30_to_objection(
    objection: Y20Objection,
    appearance: AppearanceEvent,
    dependent_conditions: DependentConditions,
    seed_trace: SeedTrace | None = None,
    projection_warning: ProjectionClaim | None = None,
) -> dict[str, Any]:
    """Attach Y30 cognitive context to a Y20 objection without making a metaphysical proof."""

    return {
        "objection_id": objection.objection_id,
        "objection_type": objection.objection_type,
        "y30_context": {
            "namespace": "Y30",
            "appearance": appearance.to_dict(),
            "dependent_conditions": dependent_conditions.to_dict(),
            "dependent_conditions_status": dependent_conditions.status,
            "seed_trace": seed_trace.to_dict() if seed_trace is not None else None,
            "projection_warning": projection_warning.to_dict() if projection_warning is not None else None,
            "is_metaphysical_proof": False,
        },
        "note": (
            "Y30 supplies cognitive structure; Y20 supplies debate structure. The fixed appearance "
            "is modeled as in-cognition condition-continuity; this argues a model and does NOT "
            "prove that no external object exists."
        ),
    }
