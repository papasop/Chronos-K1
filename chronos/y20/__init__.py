"""Y20 objection/response debate-boundary layer for Chronos."""

from .bridge import attach_y30_to_objection
from .core import (
    OBJECTION_TYPES,
    RESPONSE_STRATEGIES,
    Y20_OBJECTION_TYPES,
    ExternalObjectBoundary,
    Y20Objection,
    Y20Response,
    debate_claim_record,
)
from .library import (
    STANDARD_OBJECTION_IDS,
    build_all_standard_objections,
    build_external_object_boundary_rule,
    build_standard_objection,
)
from .physics_audit import PHYSICS_OBJECTION_IDS, physics_self_audit_objection
from .realizer import realize_external_object_boundary, realize_objection, realize_response

__all__ = [
    "ExternalObjectBoundary",
    "OBJECTION_TYPES",
    "PHYSICS_OBJECTION_IDS",
    "RESPONSE_STRATEGIES",
    "STANDARD_OBJECTION_IDS",
    "Y20Objection",
    "Y20Response",
    "Y20_OBJECTION_TYPES",
    "attach_y30_to_objection",
    "build_all_standard_objections",
    "build_external_object_boundary_rule",
    "build_standard_objection",
    "debate_claim_record",
    "physics_self_audit_objection",
    "realize_external_object_boundary",
    "realize_objection",
    "realize_response",
]
