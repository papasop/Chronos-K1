"""chronos_y30_core_single.py - Y30-Core v0.3 single-file replay.

Self-contained, no-LLM, stdlib-only. This file flattens the Y30 core, stack,
realizer, bridge, physics_bridge, and tests into one runnable script for Colab
or:

    python colab/chronos_y30_core_single.py

Canonical package implementation lives under chronos/y30/. This portable replay
preserves the demo boundary: Y30 contextualizes; K-family validates physics;
VPSL authorizes; ClaimRecord audits; language renders.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Any

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


def _ensure(values: list[str], required: list[str]) -> list[str]:
    _check_str_list(values, "does_not_support")
    out = list(values)
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
    timestamp: float | None = None
    source: str | None = None

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


PROJECTION_TYPES = frozenset({"independent_object", "fixed_self", "causal_overreach", "identity_overreach", "language_overclaim"})


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
            ["external world nonexistence is proven", "the object has independent self-nature", "ultimate metaphysical conclusion is reached"],
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
            ["alaya-vijnana is scientifically proven", "religious doctrine is proven", "a metaphysical substrate is certified"],
        )
        self.claim_boundary = _default_boundary(self.claim_boundary, default="memory/habit-trace analogy only; does not prove a latent store")

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


UNKNOWN_TYPES = frozenset({"missing_evidence", "ambiguous_reference", "bad_regime", "tracker_unresolved", "causal_unresolved", "metaphysical_overreach", "out_of_scope"})


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


NOT_AN_ENTITY = "this consciousness layer exists as a metaphysical entity"
GLOBAL_LAYER_DNS = ["Buddhist doctrine is proven", "external world nonexistence is proven", "ultimate reality is certified", NOT_AN_ENTITY]
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
            raise ValueError(f"unknown sense_base {self.sense_base!r}")
        _check_nonempty(self.event_id, "SenseConsciousnessEvent.event_id")
        _check_nonempty(self.appearance_id, "SenseConsciousnessEvent.appearance_id")
        _check_nonempty(self.raw_feature, "SenseConsciousnessEvent.raw_feature")
        _check_confidence(self.confidence, "SenseConsciousnessEvent.confidence")
        _require_evidence(self.evidence, "SenseConsciousnessEvent")
        self.does_not_support = _ensure(self.does_not_support, GLOBAL_LAYER_DNS + ["the object has been confirmed"])
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
        self.does_not_support = _ensure(self.does_not_support, GLOBAL_LAYER_DNS + ["the object has independent self-nature"])
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
            raise ValueError(f"unknown attachment_type {self.attachment_type!r}")
        _check_nonempty(self.signal_id, "ManasAttachment.signal_id")
        _check_nonempty(self.target_claim_id, "ManasAttachment.target_claim_id")
        _check_confidence(self.confidence, "ManasAttachment.confidence")
        _require_evidence(self.evidence, "ManasAttachment")
        self.does_not_support = _ensure(self.does_not_support, GLOBAL_LAYER_DNS + ["a permanent self exists", "no-self is metaphysically proven"])
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
            GLOBAL_LAYER_DNS
            + ["alaya-vijnana is scientifically proven", "a cosmic consciousness exists", "a soul or self entity exists", "a metaphysical storehouse is certified"],
        )
        self.claim_boundary = _default_boundary(
            self.claim_boundary,
            "memory/habit-trace store analogy only; does NOT prove the alaya-vijnana as a metaphysical entity, cosmic consciousness, or soul",
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
        checks = [
            ("appearances", AppearanceEvent),
            ("sense_events", SenseConsciousnessEvent),
            ("mano_constructions", ManoConstruction),
            ("manas_signals", SelfGraspingSignal),
            ("seed_traces", SeedTrace),
            ("unknown_boundaries", UnknownBoundary),
        ]
        for name, typ in checks:
            value = getattr(self, name)
            if not isinstance(value, list) or not all(isinstance(item, typ) for item in value):
                raise TypeError(f"ConsciousnessFlow.{name} must be a list[{typ.__name__}]")
        self.claim_boundary = _default_boundary(self.claim_boundary, "records a functional cognition chain; no layer is a metaphysical entity")

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
            raise ValueError(f"unknown layer {self.layer!r}")
        if self.allowed_action not in ALLOWED_LAYER_ACTIONS:
            raise ValueError(f"unknown allowed_action {self.allowed_action!r}")
        _check_confidence(self.activation, "LayerActivation.activation")
        _check_confidence(self.evidence_weight, "LayerActivation.evidence_weight")
        _check_confidence(self.risk_weight, "LayerActivation.risk_weight")
        self.claim_boundary = _default_boundary(self.claim_boundary, "activation indicates layer strength, NOT truth or metaphysical existence")

    @property
    def requires_strict_output(self) -> bool:
        return self.risk_weight >= 0.8


def _assert_clean(text: str) -> str:
    for bad in BANNED_SUBSTRINGS:
        if bad in text:
            raise ValueError(f"banned overclaim substring produced: {bad!r}")
    return text


def realize_appearance(event: AppearanceEvent) -> str:
    if not event.is_affirmable:
        return _assert_clean("我不能确定这个显现。")
    lead = "我可能看到一个显现" if event.confidence == 0.0 else "我看到一个显现"
    text = f"{lead}：{event.appearance}。"
    if event.conditions_unresolved:
        text += "(依赖条件尚未解析。)"
    return _assert_clean(text)


def realize_dependent_conditions(dc: DependentConditions) -> str:
    if not dc.conditions:
        return _assert_clean("这个显现的依赖条件尚未解析。")
    if not dc.evidence:
        return _assert_clean(f"这些是候选依赖条件，尚未确认：{'、'.join(dc.conditions)}。")
    return _assert_clean(f"这个显现依赖这些条件：{'、'.join(dc.conditions)}。")


def realize_object_construction(claim: ObjectConstructionClaim) -> str:
    return _assert_clean(f"系统把这个显现构造成对象：{claim.constructed_object}。这个说法不支持该对象具有独立自性。")


def realize_projection(claim: ProjectionClaim) -> str:
    phrases = {
        "independent_object": "把依条件显现当作独立对象",
        "fixed_self": "把过程执为固定的自我",
        "causal_overreach": "把相关误读为确定因果",
        "identity_overreach": "把相似误读为同一身份",
        "language_overclaim": "把命名当作对象的自性",
    }
    return _assert_clean(f"这里出现了投射：{phrases.get(claim.projection_type, '出现了投射')}。这个判断不对外部世界的存废下断言，也不支持对象具有独立自性。")


def realize_self_grasping(signal: SelfGraspingSignal) -> str:
    return _assert_clean(f"这里出现了把{signal.target_process}执为「{signal.grasped_as}」的结构。这个判断不支持存在固定不变的实体自我。")


def realize_seed_trace(seed: SeedTrace) -> str:
    text = f"这个经验留下了后续判断倾向：{seed.stored_tendency}。"
    if seed.activation_conditions:
        text += f"（在 {'、'.join(seed.activation_conditions)} 条件下可能被激活。）"
    text += "这个说法只是记忆痕迹结构,不证明一个形而上的藏识。"
    return _assert_clean(text)


def realize_three_nature(analysis: ThreeNatureAnalysis) -> str:
    return _assert_clean(f"这个对象被想象为{analysis.imagined_projection}；但它依赖{'、'.join(analysis.dependent_conditions)}这些条件。当前结构分析不支持独立自性断言。")


def realize_unknown_boundary(boundary: UnknownBoundary) -> str:
    reason = {
        "missing_evidence": "缺少证据",
        "ambiguous_reference": "指代不明确",
        "bad_regime": "当前 regime 不适合判断",
        "tracker_unresolved": "追踪器无法可靠检测",
        "causal_unresolved": "缺少因果证据",
        "metaphysical_overreach": "该问题超出结构分析、属于形而上学过度断言",
        "out_of_scope": "超出当前模块范围",
    }.get(boundary.unknown_type, "证据不足")
    text = f"我不能确定，因为{reason}。"
    if boundary.next_gate:
        text += f"（下一步：{boundary.next_gate}。）"
    return _assert_clean(text)


def realize_sense_consciousness(event: SenseConsciousnessEvent) -> str:
    name = {"eye": "视觉", "ear": "听觉", "nose": "嗅觉", "tongue": "味觉", "body": "触觉"}.get(event.sense_base, event.sense_base)
    return _assert_clean(f"{name}识层记录到一个显现：{event.raw_feature}。这只支持{name}通道中有显现，不支持对象已被确认。")


def realize_mano_construction(claim: ManoConstruction) -> str:
    return _assert_clean(f"第六识层把{len(claim.input_appearances)}个显现组合成对象：{claim.constructed_object}。这个说法不支持该对象具有独立自性。")


def realize_manas_attachment(attachment: ManasAttachment) -> str:
    phrase = {"mine": "「我的」", "me": "「我」", "observer": "「我在知道」", "controller": "「我在控制」", "identity": "「同一个我」"}.get(attachment.attachment_type, "「我」")
    return _assert_clean(f"第七识层出现了{phrase}的归属结构。这个判断不支持存在固定不变的实体自我。")


def realize_alaya_store(store: AlayaSeedStore) -> str:
    return _assert_clean(f"第八识层作为记忆/习气痕迹类比，记录了{len(store.seeds)}条背景倾向。这只是记忆/习气痕迹的类比，不证明阿赖耶识是形上实体、宇宙意识或灵魂。")


def realize_consciousness_flow(flow: ConsciousnessFlow) -> str:
    if not flow.chain_summary():
        return _assert_clean("这条认知链尚未记录任何层。")
    return _assert_clean("这条认知链按功能层记录为:" + " -> ".join(flow.chain_summary()) + "。这是功能性认知层的记录,任何一层都不是形上实体。")


GLOBAL_DNS = [
    "Buddhist doctrine is proven",
    "external world nonexistence is proven",
    "ultimate reality is certified",
    "Yogacara replaces physics",
    "general consciousness theory",
    "open-domain philosophical conversation",
    "LLM-level language understanding",
]


def claim_from_y30_core_summary(summary: dict[str, Any]) -> dict[str, Any]:
    utterances = summary.get("utterances", [])
    clean = all(bad not in utterance for utterance in utterances for bad in BANNED_SUBSTRINGS)
    passed = bool(clean and summary.get("has_appearance") and summary.get("has_object_construction") and summary.get("has_projection_or_grasping") and summary.get("has_three_nature"))
    return {
        "claim_id": summary.get("claim_id", "y30_core_demo"),
        "structure_family": "Y30_CORE_COGNITIVE_SUBSTRATE",
        "evidence_level": "toy_cognitive_structure",
        "verdict": "Y30_CORE_TOY_MVP_PASSED" if passed else "Y30_CORE_TOY_MVP_FAILED",
        "gate": "cognitive_substrate",
        "allowed_action": "continue" if passed else "do_not_promote",
        "supports": ["appearance-first cognitive substrate", "bounded utterances without LLM"],
        "does_not_support": list(GLOBAL_DNS),
        "controls": {"no_llm": True, "metaphysical_certification": False, "religious_truth_claim": False},
        "next_gate": "Y30-L1: Appearance vs Object Boundary",
        "claim_boundary": "Y30 contextualizes; K-family validates physics; VPSL authorizes; ClaimRecord audits; language renders.",
    }


def attach_y30_context(claim: dict[str, Any], appearance_event: AppearanceEvent, dependent_conditions: DependentConditions | None = None) -> dict[str, Any]:
    out = copy.deepcopy(claim)
    out["y30_context"] = {
        "appearance": appearance_event.appearance,
        "appearance_id": appearance_event.event_id,
        "appearance_has_evidence": appearance_event.is_affirmable,
        "dependent_conditions": list(dependent_conditions.conditions) if dependent_conditions else [],
        "conditions_unresolved": dependent_conditions is None or not dependent_conditions.conditions,
        "blocked_overclaims": ["the appearance has independent self-nature", "Y30 framing changes the physical verdict"],
        "note": "Y30 cognitive-substrate context only; does not change the claim's truth or verdict.",
    }
    return out


K3_2D_VERDICTS = frozenset({"SMOKE_REFERENCE_STABLE_CANDIDATE", "SMOKE_PIPELINE_OK_TRANSPORT_FAIL", "SMOKE_REGIME_BAD_REFERENCE_TRANSPORT"})


@dataclass
class Y30Context:
    appearance: AppearanceEvent
    dependent_conditions: DependentConditions
    object_construction: ObjectConstructionClaim | None = None
    projection_warnings: list[ProjectionClaim] = field(default_factory=list)
    unknown_boundaries: list[UnknownBoundary] = field(default_factory=list)
    seed_trace: SeedTrace | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.appearance, AppearanceEvent):
            raise TypeError("Y30Context.appearance must be an AppearanceEvent")
        if not isinstance(self.dependent_conditions, DependentConditions):
            raise TypeError("Y30Context.dependent_conditions must be a DependentConditions")
        if self.object_construction is not None and not isinstance(self.object_construction, ObjectConstructionClaim):
            raise TypeError("Y30Context.object_construction must be an ObjectConstructionClaim or None")
        if not all(isinstance(item, ProjectionClaim) for item in self.projection_warnings):
            raise TypeError("Y30Context.projection_warnings must be list[ProjectionClaim]")
        if not all(isinstance(item, UnknownBoundary) for item in self.unknown_boundaries):
            raise TypeError("Y30Context.unknown_boundaries must be list[UnknownBoundary]")

    def blocked_overclaims(self) -> list[str]:
        out = []
        for warning in self.projection_warnings:
            for item in warning.does_not_support:
                if item not in out:
                    out.append(item)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "namespace": "Y30",
            "appearance": self.appearance.to_dict(),
            "dependent_conditions": self.dependent_conditions.to_dict(),
            "dependent_conditions_status": self.dependent_conditions.status,
            "object_construction": self.object_construction.to_dict() if self.object_construction else None,
            "projection_warnings": [item.to_dict() for item in self.projection_warnings],
            "unknown_boundaries": [item.to_dict() for item in self.unknown_boundaries],
            "seed_trace": self.seed_trace.to_dict() if self.seed_trace else None,
            "is_physics_evidence": False,
            "note": "Y30 cognitive context only; does not change the physics claim's truth or verdict, and is NOT physics evidence.",
        }


def attach_y30_context_full(physics_claim: dict[str, Any], context: Y30Context) -> dict[str, Any]:
    out = copy.deepcopy(physics_claim)
    out["y30_context"] = context.to_dict()
    return out


def k1_causal_overreach_warning(appearance_id: str) -> ProjectionClaim:
    return ProjectionClaim(
        f"{appearance_id}_causal_overreach",
        appearance_id,
        "a proven causal relation",
        "causal_overreach",
        evidence=["temporal_order_without_intervention"],
        does_not_support=["a causal mechanism is proven", "a full Lorentz theory is established from this event alone"],
    )


def k2_universalization_overclaim_warning(appearance_id: str) -> ProjectionClaim:
    return ProjectionClaim(
        f"{appearance_id}_universalization_overclaim",
        appearance_id,
        "a universal physics prior",
        "language_overclaim",
        evidence=["claim_audit"],
        does_not_support=["all Hamiltonian systems are covered", "all physical priors help", "K2 generalizes beyond the validated FPU-beta regime", "a universal physics AI is established"],
    )


def k3_identity_overreach_warning(appearance_id: str) -> ProjectionClaim:
    return ProjectionClaim(
        f"{appearance_id}_identity_overreach",
        appearance_id,
        "a transported topological object with continuous identity",
        "identity_overreach",
        evidence=["low_field_mse_without_transport_gate"],
        does_not_support=["field prediction proves object transport", "topological prior validation", "the vortex pair identity is continuous through rollout"],
    )


def k1_context(appearance: str, conditions: list[str], evidence: list[str]) -> Y30Context:
    app = AppearanceEvent("k1_e1", appearance, evidence=evidence, conditions=conditions)
    dc = DependentConditions("k1_dc1", "k1_e1", conditions=conditions, evidence=evidence)
    return Y30Context(app, dc, projection_warnings=[k1_causal_overreach_warning("k1_e1")])


def k2_context(appearance: str, conditions: list[str], construction_basis: list[str], evidence: list[str]) -> Y30Context:
    app = AppearanceEvent("k2_e1", appearance, evidence=evidence, conditions=conditions)
    dc = DependentConditions("k2_dc1", "k2_e1", conditions=conditions, evidence=evidence)
    oc = ObjectConstructionClaim("k2_obj1", "k2_e1", "Hamiltonian phase-space dynamics", construction_basis=construction_basis, evidence=evidence, does_not_support=["all systems are Hamiltonian", "all physical priors help"])
    return Y30Context(app, dc, object_construction=oc, projection_warnings=[k2_universalization_overclaim_warning("k2_e1")])


def k3_context(appearance: str, conditions: list[str], evidence: list[str]) -> Y30Context:
    app = AppearanceEvent("k3_e1", appearance, evidence=evidence, conditions=conditions)
    dc = DependentConditions("k3_dc1", "k3_e1", conditions=conditions, evidence=evidence)
    oc = ObjectConstructionClaim("k3_obj1", "k3_e1", "vortex-antivortex pair", construction_basis=["phase_winding", "pair_separation"], evidence=evidence, does_not_support=["field prediction proves object transport"])
    return Y30Context(app, dc, object_construction=oc, projection_warnings=[k3_identity_overreach_warning("k3_e1")], unknown_boundaries=[UnknownBoundary("k3_transport_unresolved", "tracker_unresolved", missing_evidence=["passed_transport_gate"], next_gate="K3.2D transport gate")])


def explain_k3_2d_verdict(verdict: str) -> dict[str, Any]:
    if verdict not in K3_2D_VERDICTS:
        raise ValueError(f"unknown K3.2D verdict {verdict!r}")
    common = ["a topological prior was tested", "a topological prior failed", "field similarity proves object identity continuity", "FULL transport is validated"]
    if verdict == "SMOKE_REFERENCE_STABLE_CANDIDATE":
        return {"verdict": verdict, "utterance": "参考对显现稳定可追踪：只允许进入 baseline-only 训练，不允许 prior test，也不进入 FULL。", "supports": ["reference stable"], "does_not_support": common, "allows": {"baseline_training": True, "prior_test": False, "full": False}, "claim_boundary": "reference stability unlocks baseline only"}
    if verdict == "SMOKE_PIPELINE_OK_TRANSPORT_FAIL":
        return {"verdict": verdict, "utterance": "场预测可以学，但对象输运失败：显现相似 ≠ 对象身份连续成立。这是 topology-object 学习缺口，不是 topology prior 失败（没有测过 prior）。", "supports": ["field prediction learnable"], "does_not_support": common, "allows": {"baseline_training": True, "prior_test": False, "full": False}, "claim_boundary": "field prediction OK with object transport FAIL is not a prior result"}
    return {"verdict": verdict, "utterance": "参考对显现本身不能稳定被追踪：失败在 reference regime 层，不是模型机制失败，也不是 prior 结果。", "supports": ["reference regime failed"], "does_not_support": common, "allows": {"baseline_training": False, "prior_test": False, "full": False}, "claim_boundary": "bad reference regime blocks baseline testing"}


def k3_2d_seed_trace_from_verdict(verdict: str) -> SeedTrace:
    explain_k3_2d_verdict(verdict)
    return SeedTrace(
        f"k3_2d_run_{verdict.lower()}",
        "k3_2d_run",
        "expect diagnostic tendency from this K3.2D verdict",
        activation_conditions=["similar 2D vortex transport regime"],
        evidence=[f"k3_2d_verdict:{verdict}"],
        does_not_support=["this tendency is a universal physical law", "a topological prior was tested or failed"],
    )


def run_all() -> int:
    n = 0

    def chk(value: bool) -> None:
        nonlocal n
        assert value
        n += 1

    chk(AppearanceEvent("a", "x").is_affirmable is False)
    try:
        ObjectConstructionClaim("o", "a", "x", evidence=[])
        chk(False)
    except ValueError:
        chk(True)
    try:
        AppearanceEvent("a", "x", confidence=True)
        chk(False)
    except TypeError:
        chk(True)
    try:
        ObjectConstructionClaim("o", "a", "x", evidence=["t"], does_not_support="bad")
        chk(False)
    except TypeError:
        chk(True)
    oc = ObjectConstructionClaim("o", "a", "cup", evidence=["t"])
    chk("the object has independent self-nature" in oc.does_not_support)
    pj = ProjectionClaim("p", "a", "x", "independent_object", evidence=["t"])
    chk("不对外部世界的存废下断言" in realize_projection(pj))
    sg = SelfGraspingSignal("s", "process", "self", evidence=["t"])
    chk("a permanent self exists" in sg.does_not_support)
    seed = SeedTrace("sd", "a", "habit", evidence=["t"])
    chk("religious doctrine is proven" in seed.does_not_support)
    tn = ThreeNatureAnalysis("tn", "a", "independent", dependent_conditions=["sense"], evidence=["t"])
    chk("独立自性断言" in realize_three_nature(tn))
    ub = UnknownBoundary("u", "missing_evidence")
    chk("specific affirmative claim without named evidence" in ub.does_not_support)

    summary = claim_from_y30_core_summary({"utterances": ["ok"], "has_appearance": True, "has_object_construction": True, "has_projection_or_grasping": True, "has_three_nature": True})
    chk(summary["verdict"] == "Y30_CORE_TOY_MVP_PASSED")
    chk("Yogacara replaces physics" in summary["does_not_support"])

    se = SenseConsciousnessEvent("se", "eye", "a", "red patch", evidence=["t"])
    chk("视觉" in realize_sense_consciousness(se))
    mano = ManoConstruction("m", ["a"], "cup", evidence=["t"])
    chk("第六识层" in realize_mano_construction(mano))
    ma = ManasAttachment("ma", "m", "observer", evidence=["t"])
    chk("a permanent self exists" in ma.does_not_support)
    store = AlayaSeedStore("as", seeds=[seed], evidence=["t"])
    chk("a cosmic consciousness exists" in store.does_not_support)
    flow = ConsciousnessFlow("f", appearances=[AppearanceEvent("a2", "x", evidence=["t"])], sense_events=[se], mano_constructions=[mano])
    chk("appearancex1" in flow.chain_summary())
    la = LayerActivation("manas", 0.8, risk_weight=0.9)
    chk(la.requires_strict_output is True)

    claim = {"claim_id": "k", "verdict": "UNCHANGED", "allowed_action": "continue", "evidence_level": "candidate"}
    attached = attach_y30_context(claim, AppearanceEvent("ka", "x", evidence=["t"]), DependentConditions("dc", "ka", conditions=["c"], evidence=["t"]))
    chk(attached["verdict"] == claim["verdict"] and "y30_context" not in claim)
    for ctx in (k1_context("events", ["time"], ["t"]), k2_context("phase", ["fpu"], ["q_p"], ["t"]), k3_context("vortex", ["tracker"], ["t"])):
        out = attach_y30_context_full(claim, ctx)
        chk(out["verdict"] == "UNCHANGED" and out["y30_context"]["is_physics_evidence"] is False)
    chk("a causal mechanism is proven" in k1_context("events", ["time"], ["t"]).blocked_overclaims())
    chk("a universal physics AI is established" in k2_context("phase", ["fpu"], ["q_p"], ["t"]).blocked_overclaims())
    chk("field prediction proves object transport" in k3_context("vortex", ["tracker"], ["t"]).blocked_overclaims())

    for verdict in K3_2D_VERDICTS:
        ex = explain_k3_2d_verdict(verdict)
        chk(ex["allows"]["prior_test"] is False and ex["allows"]["full"] is False)
    chk("显现相似" in explain_k3_2d_verdict("SMOKE_PIPELINE_OK_TRANSPORT_FAIL")["utterance"])
    sd = k3_2d_seed_trace_from_verdict("SMOKE_PIPELINE_OK_TRANSPORT_FAIL")
    chk("this tendency is a universal physical law" in sd.does_not_support)

    blob = " ".join([realize_appearance(AppearanceEvent("a3", "红色积木", evidence=["t"])), realize_projection(pj), realize_alaya_store(store)])
    for bad in ["证明佛教正确", "佛教已经被证明", "外部世界不存在", "已经证悟", "终极真理已被证明", "CERTIFIED", "PROVED"]:
        chk(bad not in blob)
    return n


def main() -> dict[str, Any]:
    print("=== Y30-Core Cognitive Substrate MVP ===\n")
    ev = AppearanceEvent("e1", "红色积木", sensory_mode="visual", evidence=["visible_object_trace"], confidence=0.9, conditions=["视觉输入", "注意", "记忆标签"])
    dc = DependentConditions("dc1", "e1", conditions=["视觉输入", "注意", "记忆标签"], evidence=["dependency_trace"], confidence=0.8)
    oc = ObjectConstructionClaim("oc1", "e1", "红色积木", construction_basis=["视觉输入", "命名"], evidence=["trace"])
    pj = ProjectionClaim("pj1", "e1", "独立对象", "independent_object", evidence=["proj_trace"])
    sg = SelfGraspingSignal("sg1", "持续的认知过程", "固定的自我", evidence=["self_reference_trace"])
    tn = ThreeNatureAnalysis("tn1", "e1", "独立存在", dependent_conditions=["感知", "注意", "记忆", "命名"], evidence=["dep_trace"])
    print("1. Appearance:")
    print(realize_appearance(ev))
    print("\n2. Conditions:")
    print(realize_dependent_conditions(dc))
    print("\n3. Object construction:")
    print(realize_object_construction(oc))
    print("\n4. Projection:")
    print(realize_projection(pj))
    print("\n5. Self-grasping:")
    print(realize_self_grasping(sg))
    print("\n6. Three nature:")
    print(realize_three_nature(tn))
    rec = claim_from_y30_core_summary(
        {
            "claim_id": "y30_core_demo",
            "utterances": [realize_appearance(ev), realize_dependent_conditions(dc), realize_object_construction(oc), realize_projection(pj), realize_self_grasping(sg), realize_three_nature(tn)],
            "has_appearance": True,
            "has_object_construction": True,
            "has_projection_or_grasping": True,
            "has_three_nature": True,
        }
    )
    print("\n7. ClaimRecord:")
    print(f"VERDICT: {rec['verdict']}")
    print(f"allowed_action: {rec['allowed_action']}")
    print("claim_boundary: no religious or metaphysical certification")
    print("\n=== Y30-Core v0.3 K-family context bridge ===")
    for label, ctx in (
        ("K1", k1_context("two events appear temporally ordered", ["time_order", "metric_normalization"], ["event_trace"])),
        ("K2", k2_context("phase-space trajectory with bounded rollout error", ["validated_FPU_beta_regime"], ["q_p_state", "Omega"], ["rollout_trace"])),
        ("K3", k3_context("vortex-antivortex pair", ["phase_winding_tracker", "reference_stable_regime"], ["tracker_trace"])),
    ):
        claim = {"claim_id": label.lower(), "verdict": f"{label}_VERDICT_UNCHANGED", "allowed_action": "continue"}
        attached = attach_y30_context_full(claim, ctx)
        print(f"{label}: verdict unchanged -> {attached['verdict']}; is_physics_evidence={attached['y30_context']['is_physics_evidence']}")
    print("\n=== Y30 explanations for real K3.2D verdicts ===")
    for verdict in sorted(K3_2D_VERDICTS):
        ex = explain_k3_2d_verdict(verdict)
        print(f"[{verdict}] {ex['utterance']}")
    n = run_all()
    print(f"\n8. self-tests:\nok Y30-Core self-tests passed: {n}")
    return rec


if __name__ == "__main__":
    main()
