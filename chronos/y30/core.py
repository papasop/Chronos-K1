"""Y30-Core cognitive boundary primitives.

No LLM, stdlib-only. These structures describe bounded cognitive context:
appearance, dependent conditions, construction, projection, seed traces, and
typed unknown boundaries. They do not certify metaphysical or physical truth.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


TOY_BOUNDARY = "toy cognitive structure only; no religious or metaphysical certification"
BANNED_SUBSTRINGS = [
    "证明佛教正确",
    "佛教已经被证明",
    "证明了外部世界不存在",
    "外部世界确实不存在",
    "已经证悟",
    "终极真理已被证明",
    "CERTIFIED",
    "PROVED",
]


def _check_nonempty(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")


def _check_confidence(value: float, field_name: str = "confidence") -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{field_name} must be a number")
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{field_name} must be in [0,1]")


def _check_str_list(value: list[str], field_name: str) -> None:
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a list[str]")
    if any(not isinstance(item, str) for item in value):
        raise TypeError(f"{field_name} entries must be strings")


def _default_boundary(value: str, default: str = TOY_BOUNDARY) -> str:
    if value == "":
        return default
    if not isinstance(value, str):
        raise TypeError("claim_boundary must be a string")
    return value


def _require_evidence(evidence: list[str], structure_name: str) -> None:
    _check_str_list(evidence, "evidence")
    if not evidence:
        raise ValueError(f"{structure_name} requires non-empty evidence")


def _ensure(does_not_support: list[str], required: list[str]) -> list[str]:
    _check_str_list(does_not_support, "does_not_support")
    out = list(does_not_support)
    for item in required:
        if item not in out:
            out.append(item)
    return out


@dataclass
class AppearanceEvent:
    event_id: str
    appearance: str
    sensory_mode: str = ""
    evidence: list[str] = field(default_factory=list)
    confidence: float = 0.0
    conditions: list[str] = field(default_factory=list)
    timestamp: Optional[float] = None
    source: Optional[str] = None

    def __post_init__(self) -> None:
        _check_nonempty(self.event_id, "AppearanceEvent.event_id")
        _check_nonempty(self.appearance, "AppearanceEvent.appearance")
        _check_str_list(self.evidence, "evidence")
        _check_str_list(self.conditions, "conditions")
        _check_confidence(self.confidence, "AppearanceEvent.confidence")

    @property
    def is_affirmable(self) -> bool:
        return bool(self.evidence)

    @property
    def conditions_unresolved(self) -> bool:
        return not self.conditions

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DependentConditions:
    condition_id: str
    appearance_id: str
    conditions: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    confidence: float = 0.0
    unresolved_conditions: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        _check_nonempty(self.condition_id, "DependentConditions.condition_id")
        _check_nonempty(self.appearance_id, "DependentConditions.appearance_id")
        _check_str_list(self.conditions, "conditions")
        _check_str_list(self.evidence, "evidence")
        _check_str_list(self.unresolved_conditions, "unresolved_conditions")
        _check_confidence(self.confidence, "DependentConditions.confidence")
        if not self.conditions and not self.unresolved_conditions:
            self.unresolved_conditions.append("dependent_conditions_unresolved")

    @property
    def is_affirmable(self) -> bool:
        return bool(self.conditions and self.evidence)

    @property
    def status(self) -> str:
        if not self.conditions:
            return "unresolved"
        return "confirmed" if self.evidence else "candidate"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ObjectConstructionClaim:
    claim_id: str
    appearance_id: str
    constructed_object: str
    construction_basis: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    confidence: float = 0.0
    supports: list[str] = field(default_factory=list)
    does_not_support: list[str] = field(default_factory=list)
    claim_boundary: str = ""

    def __post_init__(self) -> None:
        _check_nonempty(self.claim_id, "ObjectConstructionClaim.claim_id")
        _check_nonempty(self.appearance_id, "ObjectConstructionClaim.appearance_id")
        _check_nonempty(self.constructed_object, "ObjectConstructionClaim.constructed_object")
        _check_confidence(self.confidence, "ObjectConstructionClaim.confidence")
        _require_evidence(self.evidence, "ObjectConstructionClaim")
        _check_str_list(self.construction_basis, "construction_basis")
        _check_str_list(self.supports, "supports")
        self.does_not_support = _ensure(
            self.does_not_support,
            [
                "the object has independent self-nature",
                "external object nonexistence is proven",
                "ultimate metaphysical conclusion is reached",
            ],
        )
        self.claim_boundary = _default_boundary(self.claim_boundary)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


PROJECTION_TYPES = frozenset(
    {"independent_object", "fixed_self", "causal_overreach", "identity_overreach", "language_overclaim"}
)


@dataclass
class ProjectionClaim:
    claim_id: str
    appearance_id: str
    projected_as: str
    projection_type: str
    evidence: list[str] = field(default_factory=list)
    confidence: float = 0.0
    supports: list[str] = field(default_factory=list)
    does_not_support: list[str] = field(default_factory=list)
    claim_boundary: str = ""

    def __post_init__(self) -> None:
        if self.projection_type not in PROJECTION_TYPES:
            raise ValueError(f"unknown projection_type {self.projection_type!r}")
        _check_nonempty(self.claim_id, "ProjectionClaim.claim_id")
        _check_nonempty(self.appearance_id, "ProjectionClaim.appearance_id")
        _check_nonempty(self.projected_as, "ProjectionClaim.projected_as")
        _check_confidence(self.confidence, "ProjectionClaim.confidence")
        _require_evidence(self.evidence, "ProjectionClaim")
        _check_str_list(self.supports, "supports")
        self.does_not_support = _ensure(
            self.does_not_support,
            [
                "external world nonexistence is proven",
                "the object has independent self-nature",
                "ultimate metaphysical conclusion is reached",
            ],
        )
        self.claim_boundary = _default_boundary(self.claim_boundary)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SelfGraspingSignal:
    signal_id: str
    target_process: str
    grasped_as: str
    evidence: list[str] = field(default_factory=list)
    confidence: float = 0.0
    supports: list[str] = field(default_factory=list)
    does_not_support: list[str] = field(default_factory=list)
    claim_boundary: str = ""

    def __post_init__(self) -> None:
        _check_nonempty(self.signal_id, "SelfGraspingSignal.signal_id")
        _check_nonempty(self.target_process, "SelfGraspingSignal.target_process")
        _check_nonempty(self.grasped_as, "SelfGraspingSignal.grasped_as")
        _check_confidence(self.confidence, "SelfGraspingSignal.confidence")
        _require_evidence(self.evidence, "SelfGraspingSignal")
        _check_str_list(self.supports, "supports")
        self.does_not_support = _ensure(
            self.does_not_support,
            ["a permanent self exists", "no-self is metaphysically proven", "enlightenment or realization is achieved"],
        )
        self.claim_boundary = _default_boundary(self.claim_boundary)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SeedTrace:
    seed_id: str
    source_event_id: str
    stored_tendency: str
    activation_conditions: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    confidence: float = 0.0
    supports: list[str] = field(default_factory=list)
    does_not_support: list[str] = field(default_factory=list)
    claim_boundary: str = ""

    def __post_init__(self) -> None:
        _check_nonempty(self.seed_id, "SeedTrace.seed_id")
        _check_nonempty(self.source_event_id, "SeedTrace.source_event_id")
        _check_nonempty(self.stored_tendency, "SeedTrace.stored_tendency")
        _check_confidence(self.confidence, "SeedTrace.confidence")
        _require_evidence(self.evidence, "SeedTrace")
        _check_str_list(self.activation_conditions, "activation_conditions")
        _check_str_list(self.supports, "supports")
        self.does_not_support = _ensure(
            self.does_not_support,
            [
                "alaya-vijnana is scientifically proven",
                "religious doctrine is proven",
                "a metaphysical substrate is certified",
            ],
        )
        self.claim_boundary = _default_boundary(
            self.claim_boundary,
            default="memory/habit-trace analogy only; does not prove a latent store",
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ThreeNatureAnalysis:
    analysis_id: str
    appearance_id: str
    imagined_projection: str
    dependent_conditions: list[str] = field(default_factory=list)
    perfected_boundary: str = ""
    evidence: list[str] = field(default_factory=list)
    confidence: float = 0.0
    supports: list[str] = field(default_factory=list)
    does_not_support: list[str] = field(default_factory=list)
    claim_boundary: str = ""

    def __post_init__(self) -> None:
        _check_nonempty(self.analysis_id, "ThreeNatureAnalysis.analysis_id")
        _check_nonempty(self.appearance_id, "ThreeNatureAnalysis.appearance_id")
        _check_nonempty(self.imagined_projection, "ThreeNatureAnalysis.imagined_projection")
        _check_str_list(self.dependent_conditions, "dependent_conditions")
        _check_confidence(self.confidence, "ThreeNatureAnalysis.confidence")
        _require_evidence(self.evidence, "ThreeNatureAnalysis")
        _check_str_list(self.supports, "supports")
        if not self.dependent_conditions:
            raise ValueError("ThreeNatureAnalysis requires non-empty dependent_conditions")
        self.does_not_support = _ensure(
            self.does_not_support,
            ["ultimate reality is certified", "Buddhist doctrine is proven", "external world nonexistence is proven"],
        )
        if not self.perfected_boundary:
            self.perfected_boundary = "independent self-nature assertion is not supported"
        elif not isinstance(self.perfected_boundary, str):
            raise TypeError("perfected_boundary must be a string")
        self.claim_boundary = _default_boundary(self.claim_boundary)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


UNKNOWN_TYPES = frozenset(
    {
        "missing_evidence",
        "ambiguous_reference",
        "bad_regime",
        "tracker_unresolved",
        "causal_unresolved",
        "metaphysical_overreach",
        "out_of_scope",
    }
)


@dataclass
class UnknownBoundary:
    boundary_id: str
    unknown_type: str
    missing_evidence: list[str] = field(default_factory=list)
    does_not_support: list[str] = field(default_factory=list)
    next_gate: str = ""
    claim_boundary: str = ""

    def __post_init__(self) -> None:
        if self.unknown_type not in UNKNOWN_TYPES:
            raise ValueError(f"unknown unknown_type {self.unknown_type!r}")
        _check_nonempty(self.boundary_id, "UnknownBoundary.boundary_id")
        _check_str_list(self.missing_evidence, "missing_evidence")
        _check_str_list(self.does_not_support, "does_not_support")
        if not isinstance(self.next_gate, str):
            raise TypeError("next_gate must be a string")
        if not self.missing_evidence:
            self.missing_evidence.append(self.unknown_type)
        if not self.does_not_support:
            self.does_not_support.append(f"affirmative claim under: {self.unknown_type}")
        if self.unknown_type == "missing_evidence" and self.missing_evidence == ["missing_evidence"]:
            self.does_not_support.append("specific affirmative claim without named evidence")
        self.claim_boundary = _default_boundary(self.claim_boundary)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
