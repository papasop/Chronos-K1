"""Y30 functional stack structures.

These are cognitive-layer records, not metaphysical entities. Activation means
functional strength only; it is not truth, certification, or physics evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from chronos.y30.core import (
    AppearanceEvent,
    SeedTrace,
    SelfGraspingSignal,
    UnknownBoundary,
    _check_confidence,
    _check_nonempty,
    _check_str_list,
    _default_boundary,
    _ensure,
    _require_evidence,
)

NOT_AN_ENTITY = "this consciousness layer exists as a metaphysical entity"
GLOBAL_LAYER_DOES_NOT_SUPPORT = [
    "Buddhist doctrine is proven",
    "external world nonexistence is proven",
    "ultimate reality is certified",
    NOT_AN_ENTITY,
]

SENSE_BASES = frozenset({"eye", "ear", "nose", "tongue", "body"})
ATTACHMENT_TYPES = frozenset({"mine", "me", "observer", "controller", "identity"})
LAYERS = frozenset({"sense", "mano", "manas", "alaya"})
ALLOWED_LAYER_ACTIONS = frozenset({"bounded_utterance_only", "record_only", "flag_for_review"})


@dataclass
class SenseConsciousnessEvent:
    event_id: str
    sense_base: str
    appearance_id: str
    raw_feature: str
    evidence: list[str] = field(default_factory=list)
    confidence: float = 0.0
    does_not_support: list[str] = field(default_factory=list)
    claim_boundary: str = "sense-channel appearance only"

    def __post_init__(self) -> None:
        if self.sense_base not in SENSE_BASES:
            raise ValueError(f"unknown sense_base {self.sense_base!r}; expected one of {sorted(SENSE_BASES)}")
        _check_nonempty(self.event_id, "SenseConsciousnessEvent.event_id")
        _check_nonempty(self.appearance_id, "SenseConsciousnessEvent.appearance_id")
        _check_nonempty(self.raw_feature, "SenseConsciousnessEvent.raw_feature")
        _check_confidence(self.confidence, "SenseConsciousnessEvent.confidence")
        _require_evidence(self.evidence, "SenseConsciousnessEvent")
        self.does_not_support = _ensure(
            self.does_not_support,
            GLOBAL_LAYER_DOES_NOT_SUPPORT + ["the object has been confirmed"],
        )
        self.claim_boundary = _default_boundary(self.claim_boundary, "sense-channel appearance only")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ManoConstruction:
    claim_id: str
    input_appearances: list[str]
    constructed_object: str
    concepts: list[str] = field(default_factory=list)
    relations: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    confidence: float = 0.0
    supports: list[str] = field(default_factory=list)
    does_not_support: list[str] = field(default_factory=list)
    claim_boundary: str = ""

    def __post_init__(self) -> None:
        _check_nonempty(self.claim_id, "ManoConstruction.claim_id")
        _check_nonempty(self.constructed_object, "ManoConstruction.constructed_object")
        _check_str_list(self.input_appearances, "input_appearances")
        _check_str_list(self.concepts, "concepts")
        _check_str_list(self.relations, "relations")
        _check_str_list(self.supports, "supports")
        _check_confidence(self.confidence, "ManoConstruction.confidence")
        _require_evidence(self.evidence, "ManoConstruction")
        if not self.input_appearances:
            raise ValueError("ManoConstruction requires non-empty input_appearances")
        self.does_not_support = _ensure(
            self.does_not_support,
            GLOBAL_LAYER_DOES_NOT_SUPPORT + ["the object has independent self-nature"],
        )
        self.claim_boundary = _default_boundary(self.claim_boundary)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ManasAttachment:
    signal_id: str
    target_claim_id: str
    attachment_type: str
    evidence: list[str] = field(default_factory=list)
    confidence: float = 0.0
    does_not_support: list[str] = field(default_factory=list)
    claim_boundary: str = ""

    def __post_init__(self) -> None:
        if self.attachment_type not in ATTACHMENT_TYPES:
            raise ValueError(f"unknown attachment_type {self.attachment_type!r}; expected one of {sorted(ATTACHMENT_TYPES)}")
        _check_nonempty(self.signal_id, "ManasAttachment.signal_id")
        _check_nonempty(self.target_claim_id, "ManasAttachment.target_claim_id")
        _check_confidence(self.confidence, "ManasAttachment.confidence")
        _require_evidence(self.evidence, "ManasAttachment")
        self.does_not_support = _ensure(
            self.does_not_support,
            GLOBAL_LAYER_DOES_NOT_SUPPORT + ["a permanent self exists", "no-self is metaphysically proven"],
        )
        self.claim_boundary = _default_boundary(self.claim_boundary)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AlayaSeedStore:
    store_id: str
    seeds: list[SeedTrace] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    does_not_support: list[str] = field(default_factory=list)
    claim_boundary: str = ""

    def __post_init__(self) -> None:
        _check_nonempty(self.store_id, "AlayaSeedStore.store_id")
        if not isinstance(self.seeds, list) or not all(isinstance(seed, SeedTrace) for seed in self.seeds):
            raise TypeError("AlayaSeedStore.seeds must be a list[SeedTrace]")
        _require_evidence(self.evidence, "AlayaSeedStore")
        self.does_not_support = _ensure(
            self.does_not_support,
            GLOBAL_LAYER_DOES_NOT_SUPPORT
            + [
                "alaya-vijnana is scientifically proven",
                "a cosmic consciousness exists",
                "a soul or self entity exists",
                "a metaphysical storehouse is certified",
            ],
        )
        self.claim_boundary = _default_boundary(
            self.claim_boundary,
            default=(
                "memory/habit-trace store analogy only; does NOT prove the alaya-vijnana as a "
                "metaphysical entity, cosmic consciousness, or soul"
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ConsciousnessFlow:
    flow_id: str
    appearances: list[AppearanceEvent] = field(default_factory=list)
    sense_events: list[SenseConsciousnessEvent] = field(default_factory=list)
    mano_constructions: list[ManoConstruction] = field(default_factory=list)
    manas_signals: list[SelfGraspingSignal] = field(default_factory=list)
    seed_traces: list[SeedTrace] = field(default_factory=list)
    unknown_boundaries: list[UnknownBoundary] = field(default_factory=list)
    claim_boundary: str = ""

    def __post_init__(self) -> None:
        _check_nonempty(self.flow_id, "ConsciousnessFlow.flow_id")
        typed_lists = [
            ("appearances", AppearanceEvent),
            ("sense_events", SenseConsciousnessEvent),
            ("mano_constructions", ManoConstruction),
            ("manas_signals", SelfGraspingSignal),
            ("seed_traces", SeedTrace),
            ("unknown_boundaries", UnknownBoundary),
        ]
        for name, typ in typed_lists:
            value = getattr(self, name)
            if not isinstance(value, list) or not all(isinstance(item, typ) for item in value):
                raise TypeError(f"ConsciousnessFlow.{name} must be a list[{typ.__name__}]")
        self.claim_boundary = _default_boundary(
            self.claim_boundary,
            default="records a functional cognition chain; no layer is a metaphysical entity",
        )

    def chain_summary(self) -> list[str]:
        out = []
        if self.appearances:
            out.append(f"appearancex{len(self.appearances)}")
        if self.sense_events:
            out.append(f"sensex{len(self.sense_events)}")
        if self.mano_constructions:
            out.append(f"mano_constructionx{len(self.mano_constructions)}")
        if self.manas_signals:
            out.append(f"manas_self_graspingx{len(self.manas_signals)}")
        if self.seed_traces:
            out.append(f"seed_tracex{len(self.seed_traces)}")
        if self.unknown_boundaries:
            out.append(f"unknown_boundaryx{len(self.unknown_boundaries)}")
        return out

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LayerActivation:
    layer: str
    activation: float
    evidence_weight: float = 0.0
    risk_weight: float = 0.0
    allowed_action: str = "bounded_utterance_only"
    claim_boundary: str = ""

    def __post_init__(self) -> None:
        if self.layer not in LAYERS:
            raise ValueError(f"unknown layer {self.layer!r}; expected one of {sorted(LAYERS)}")
        if self.allowed_action not in ALLOWED_LAYER_ACTIONS:
            raise ValueError(f"unknown allowed_action {self.allowed_action!r}; expected one of {sorted(ALLOWED_LAYER_ACTIONS)}")
        _check_confidence(self.activation, "LayerActivation.activation")
        _check_confidence(self.evidence_weight, "LayerActivation.evidence_weight")
        _check_confidence(self.risk_weight, "LayerActivation.risk_weight")
        self.claim_boundary = _default_boundary(
            self.claim_boundary,
            default="activation indicates layer strength, NOT truth or metaphysical existence",
        )

    @property
    def requires_strict_output(self) -> bool:
        return self.risk_weight >= 0.8

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
