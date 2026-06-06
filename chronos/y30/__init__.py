"""Y30 cognitive-boundary substrate for Chronos.

Y30 records appearance, dependence, construction, projection, seed-trace, and
unknown-boundary structures. It is no-LLM, stdlib-only, and does not certify
metaphysical or physical truth.
"""

from .core import (
    BANNED_SUBSTRINGS,
    TOY_BOUNDARY,
    UNKNOWN_TYPES,
    PROJECTION_TYPES,
    AppearanceEvent,
    DependentConditions,
    ObjectConstructionClaim,
    ProjectionClaim,
    SeedTrace,
    SelfGraspingSignal,
    ThreeNatureAnalysis,
    UnknownBoundary,
)

__all__ = [
    "AppearanceEvent",
    "BANNED_SUBSTRINGS",
    "DependentConditions",
    "ObjectConstructionClaim",
    "PROJECTION_TYPES",
    "ProjectionClaim",
    "SeedTrace",
    "SelfGraspingSignal",
    "TOY_BOUNDARY",
    "ThreeNatureAnalysis",
    "UNKNOWN_TYPES",
    "UnknownBoundary",
]
