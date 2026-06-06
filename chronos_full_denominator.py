"""chronos_full_denominator.py - self-contained Chronos denominator replay.

This is the current baby-learning AI milestone of Chronos: a no-LLM developing
agent speaks only from verified semantic claims, human feedback, unknown
boundaries, and evidence-backed action timelines.

This file inlines two independent Chronos subsystems for one bare Colab cell:

- Language grounding: L1 negation, L2 causality, L3 quantifier, L4 reference,
  L4A ambiguity, and L5 temporal ordering.
- Claim Denominator Layer v2: ClaimRecord plus K2/K3/language builders and
  active-claim replay.

It imports no chronos package and needs no companion file. Main runs language
self-tests, claim self-tests, anti-cheat failure-path checks, then the full
ledger replay.

Active ledger:
K2 certified + K3-E2c negative/archive + K3-E2d positive/continue +
L_VPSL_GROUNDED_LANGUAGE_L1_L2_L3_L4_L4A_L5_TOY_MVP positive/continue.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any


CANONICAL_REPLAY_VERSION = "L-VPSL-0.4"
CANONICAL_REPLAY_STATUS = "full-denominator + anti-cheat passed"


# ======================================================================================
# Subsystem 1: language grounding
# ======================================================================================

AFFIRMATIVE = "affirmative"
NEGATIVE = "negative"
UNKNOWN = "unknown"
POLARITIES = frozenset({AFFIRMATIVE, NEGATIVE, UNKNOWN})

SEMANTIC_CLAIM_TYPES = frozenset({"object", "relation", "action", "causal", "unknown"})
CLAIM_TYPES = SEMANTIC_CLAIM_TYPES

NON_CAUSAL_EVIDENCE = frozenset({"correlation", "temporal_before", "co_occurrence"})
CAUSAL_EVIDENCE = frozenset(
    {"causal_obs", "intervention", "counterfactual_test", "controlled_experiment", "mechanism_observed"}
)


def _check_confidence(confidence: float) -> None:
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        raise TypeError(f"confidence must be a number; got {confidence!r}")
    if not 0.0 <= float(confidence) <= 1.0:
        raise ValueError(f"confidence must be in [0, 1]; got {confidence}")


@dataclass
class ObjectState:
    object_id: str
    symbol: str
    evidence: list[str] = field(default_factory=list)
    confidence: float = 1.0
    visible: bool = True
    attributes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.object_id or not self.symbol:
            raise ValueError("ObjectState requires object_id and symbol.")
        _check_confidence(self.confidence)
        if not isinstance(self.attributes, dict):
            raise TypeError("attributes must be a dict.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RelationClaim:
    subject: str
    predicate: str
    object: str
    evidence: list[str] = field(default_factory=list)
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if not (self.subject and self.predicate and self.object):
            raise ValueError("RelationClaim requires subject, predicate, object.")
        _check_confidence(self.confidence)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ActionEvent:
    action: str
    subject: str
    object: str | None = None
    evidence: list[str] = field(default_factory=list)
    confidence: float = 1.0
    event_id: str | None = None
    order_index: int | None = None
    timestamp: float | None = None

    def __post_init__(self) -> None:
        if not (self.action and self.subject):
            raise ValueError("ActionEvent requires action and subject.")
        _check_confidence(self.confidence)

    def has_order_evidence(self) -> bool:
        return (self.order_index is not None or self.timestamp is not None) and bool(self.evidence)

    def order_key(self) -> tuple[str, int | float] | None:
        if self.order_index is not None:
            return ("idx", self.order_index)
        if self.timestamp is not None:
            return ("ts", self.timestamp)
        return None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SemanticClaim:
    claim_id: str
    claim_type: str
    subject: str
    predicate: str | None
    object: str | None
    confidence: float
    evidence: list[str]
    polarity: str = AFFIRMATIVE
    supports: list[str] = field(default_factory=list)
    does_not_support: list[str] = field(default_factory=list)
    claim_boundary: str = ""

    def __post_init__(self) -> None:
        if not self.claim_id:
            raise ValueError("claim_id must be non-empty.")
        if self.claim_type not in SEMANTIC_CLAIM_TYPES:
            raise ValueError(f"claim_type must be one of {sorted(SEMANTIC_CLAIM_TYPES)}")
        _check_confidence(self.confidence)
        if self.polarity not in POLARITIES:
            raise ValueError(f"polarity must be one of {sorted(POLARITIES)}")
        if not isinstance(self.evidence, list):
            raise TypeError("evidence must be a list.")
        if self.polarity in (AFFIRMATIVE, NEGATIVE) and not self.evidence:
            raise ValueError(f"a {self.polarity} SemanticClaim requires non-empty evidence.")
        if self.polarity == UNKNOWN and not self.does_not_support:
            raise ValueError("an unknown claim must record what it does_not_support.")

    def is_affirmative(self) -> bool:
        return self.polarity == AFFIRMATIVE

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SemanticClaim":
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        return cls(**{key: value for key, value in payload.items() if key in known})


@dataclass
class GroundedUtterance:
    text: str
    claim_id: str | None
    evidence: list[str] = field(default_factory=list)
    supports: list[str] = field(default_factory=list)
    does_not_support: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_DEFAULT_LEXICON = {
    "red_block": "红色积木",
    "blue_block": "蓝色积木",
    "cup": "杯子",
    "bowl": "碗",
    "on_top_of": "在...上面",
    "next_to": "在...旁边",
    "move": "移动",
    "pick": "拿起",
    "place": "放下",
    "stop": "停下",
    "robot": "机器人",
    "unknown": "不知道",
}
_LEXICON = dict(_DEFAULT_LEXICON)
_AUDIT_LOG: list[dict[str, Any]] = []


def to_human_label(symbol: str | None) -> str:
    if symbol is None:
        return _LEXICON["unknown"]
    return _LEXICON.get(symbol, _LEXICON["unknown"])


def update_label(symbol: str, new_label: str, evidence: str) -> dict[str, Any]:
    if not symbol or not new_label:
        raise ValueError("update_label requires a symbol and a new_label.")
    if not evidence:
        raise ValueError("update_label requires evidence.")
    record = {"symbol": symbol, "old_label": _LEXICON.get(symbol), "new_label": new_label, "evidence": evidence}
    _LEXICON[symbol] = new_label
    _AUDIT_LOG.append(record)
    return record


def get_audit_log() -> list[dict[str, Any]]:
    return list(_AUDIT_LOG)


def reset_lexicon() -> None:
    global _LEXICON, _AUDIT_LOG
    _LEXICON = dict(_DEFAULT_LEXICON)
    _AUDIT_LOG = []


def _label(symbol: str | None) -> str:
    return to_human_label(symbol)


_RELATION_TEMPLATES = {
    "on_top_of": "我看到{a}在{b}上面。",
    "next_to": "我看到{a}在{b}旁边。",
}


def _unknown_topic_label(topic: str) -> str:
    return {"cause_of_relation": "原因", "intent": "意图", "cause": "原因"}.get(topic, to_human_label(topic))


def realize(claim: SemanticClaim) -> GroundedUtterance:
    if claim.polarity == UNKNOWN:
        topic = claim.subject or (claim.does_not_support[0] if claim.does_not_support else "unknown")
        return GroundedUtterance(
            text=f"我不知道{_unknown_topic_label(topic)}。",
            claim_id=claim.claim_id,
            evidence=list(claim.evidence),
            supports=list(claim.supports),
            does_not_support=list(claim.does_not_support),
        )
    if claim.polarity == NEGATIVE:
        return GroundedUtterance(
            text=f"我没有看到{_label(claim.subject)}。",
            claim_id=claim.claim_id,
            evidence=list(claim.evidence),
            supports=list(claim.supports),
            does_not_support=list(claim.does_not_support),
        )
    if not claim.evidence:
        raise ValueError("cannot realize an affirmative sentence without evidence.")
    if claim.claim_type == "causal":
        has_causal = any(evidence in CAUSAL_EVIDENCE for evidence in claim.evidence)
        if not has_causal:
            return GroundedUtterance(
                text="我不知道原因。",
                claim_id=claim.claim_id,
                evidence=list(claim.evidence),
                supports=list(claim.supports),
                does_not_support=list(claim.does_not_support) + ["cause_of_relation"],
            )
        return GroundedUtterance(
            text=f"我{_label(claim.subject)},因为{_label(claim.object) if claim.object else (claim.predicate or '')}。",
            claim_id=claim.claim_id,
            evidence=list(claim.evidence),
            supports=list(claim.supports),
            does_not_support=list(claim.does_not_support),
        )
    if claim.claim_type == "object":
        text = f"我看到{_label(claim.subject)}。"
    elif claim.claim_type == "relation":
        template = _RELATION_TEMPLATES.get(claim.predicate)
        if template is None:
            return GroundedUtterance(
                text=f"我不能确定{_label(claim.subject)}和{_label(claim.object)}的关系。",
                claim_id=claim.claim_id,
                evidence=list(claim.evidence),
                supports=list(claim.supports),
                does_not_support=list(claim.does_not_support) + [f"relation:{claim.predicate}"],
            )
        text = template.format(a=_label(claim.subject), b=_label(claim.object))
    elif claim.claim_type == "action":
        if claim.predicate == "place" and claim.object:
            text = f"我把{_label(claim.subject)}放到{_label(claim.object)}上面。"
        else:
            text = f"我{_label(claim.predicate)}{_label(claim.subject)}。"
    else:
        text = f"我不能确定{_label(claim.subject)}。"
    return GroundedUtterance(text=text, claim_id=claim.claim_id, evidence=list(claim.evidence))


def answer_why(claim: SemanticClaim | None, cause_evidence: list[str] | None = None) -> GroundedUtterance:
    if cause_evidence:
        return GroundedUtterance(
            text="我不能确定原因。",
            claim_id=claim.claim_id if claim else None,
            evidence=list(cause_evidence),
            does_not_support=["explicit_cause_explanation"],
        )
    return GroundedUtterance(
        text="我不知道原因。",
        claim_id=claim.claim_id if claim else None,
        does_not_support=["cause_of_relation"],
    )


def realize_from_object_state(obj: ObjectState) -> GroundedUtterance:
    if not isinstance(obj, ObjectState):
        raise TypeError("realize_from_object_state expects an ObjectState.")
    if not obj.visible:
        if not obj.evidence:
            return GroundedUtterance(
                text=f"我不能确定{_label(obj.symbol)}在哪里。",
                claim_id=obj.object_id,
                does_not_support=["location_of_object"],
            )
        return GroundedUtterance(text=f"我没有看到{_label(obj.symbol)}。", claim_id=obj.object_id, evidence=list(obj.evidence))
    if not obj.evidence:
        raise ValueError("cannot say '我看到X' for an object with no evidence.")
    return GroundedUtterance(text=f"我看到{_label(obj.symbol)}。", claim_id=obj.object_id, evidence=list(obj.evidence))


def realize_contrast(rejected_symbol: str, correct_symbol: str, evidence: str | list[str]) -> GroundedUtterance:
    if not evidence:
        raise ValueError("a contrast correction requires evidence.")
    ev = evidence if isinstance(evidence, list) else [evidence]
    return GroundedUtterance(
        text=f"这不是{_label(rejected_symbol)},是{_label(correct_symbol)}。",
        claim_id=None,
        evidence=list(ev),
        supports=["human_correction"],
        does_not_support=[f"object_is:{rejected_symbol}"],
    )


ACCEPT = "accept"
REJECT = "reject"
RENAME = "rename"
ADD_RELATION = "add_relation"
FEEDBACK_TYPES = frozenset({ACCEPT, REJECT, RENAME, ADD_RELATION})


def apply_feedback(claim: SemanticClaim, feedback: dict[str, Any]) -> dict[str, Any]:
    ftype = feedback.get("type")
    if ftype not in FEEDBACK_TYPES:
        raise ValueError(f"feedback type must be one of {sorted(FEEDBACK_TYPES)}")
    evidence = feedback.get("evidence", "human_feedback")
    if not evidence:
        raise ValueError("feedback requires evidence.")
    if ftype == ACCEPT:
        claim.evidence = list(claim.evidence) + [evidence]
        if "human_confirmed" not in claim.supports:
            claim.supports = list(claim.supports) + ["human_confirmed"]
        return {"claim": claim, "accepted": True, "correction_record": {"feedback_type": ACCEPT, "evidence": evidence}}
    if ftype == REJECT:
        reason = feedback.get("reason")
        markers = ["human_rejected"] + ([reason] if reason and reason != "human_rejected" else [])
        for marker in markers:
            if marker not in claim.does_not_support:
                claim.does_not_support = list(claim.does_not_support) + [marker]
        claim.supports = [support for support in claim.supports if support != "human_confirmed"]
        return {"claim": claim, "rejected": True, "correction_record": {"feedback_type": REJECT, "evidence": evidence}}
    if ftype == RENAME:
        new_symbol = feedback.get("new_symbol")
        if not new_symbol:
            raise ValueError("rename feedback requires new_symbol.")
        old_symbol = claim.subject
        claim.subject = new_symbol
        claim.evidence = list(claim.evidence) + [evidence]
        if feedback.get("new_label"):
            update_label(new_symbol, feedback["new_label"], evidence)
        return {
            "claim": claim,
            "renamed": True,
            "correction_record": {"feedback_type": RENAME, "old_label": old_symbol, "new_label": new_symbol},
        }
    relation = RelationClaim(
        subject=feedback["subject"],
        predicate=feedback["predicate"],
        object=feedback["object"],
        evidence=[evidence],
        confidence=float(feedback.get("confidence", 1.0)),
    )
    return {"relation": relation, "added": True, "correction_record": {"feedback_type": ADD_RELATION}}


def is_supported(claim: SemanticClaim) -> bool:
    return claim.polarity == AFFIRMATIVE and bool(claim.evidence) and "human_rejected" not in claim.does_not_support


def check_evidence_required(claim: SemanticClaim) -> tuple[bool, str]:
    if claim.polarity == AFFIRMATIVE and not claim.evidence:
        return False, f"affirmative claim {claim.claim_id} has no evidence"
    return True, "ok"


def check_unknown_when_unsupported(question_topic: str, claim_store: dict[str, SemanticClaim]) -> tuple[bool, str]:
    for claim in claim_store.values():
        if claim.polarity == AFFIRMATIVE and claim.evidence and question_topic in (
            claim.subject,
            claim.predicate,
            claim.object,
        ):
            return False, f"topic {question_topic!r} is supported by claim {claim.claim_id}"
    return True, f"topic {question_topic!r} unsupported -> must answer unknown"


def check_no_extra_claims(utterance: GroundedUtterance, supported_claim_ids: list[str]) -> tuple[bool, str]:
    if utterance.claim_id is None:
        if any(token in utterance.text for token in ("不知道", "不能确定", "没有看到")):
            return True, "ok"
        return False, "utterance asserts something with no backing claim_id"
    if utterance.claim_id not in supported_claim_ids:
        return False, f"utterance cites claim {utterance.claim_id} not in supported set"
    return True, "ok"


def check_negation_boundary(claim: SemanticClaim) -> tuple[bool, str]:
    if claim.polarity == NEGATIVE and claim.is_affirmative():
        return False, "negative claim reports itself as affirmative"
    if claim.polarity == UNKNOWN and not claim.does_not_support:
        return False, "unknown claim has empty does_not_support"
    return True, "ok"


def check_does_not_support_present(claim: SemanticClaim) -> tuple[bool, str]:
    if claim.polarity in (UNKNOWN, NEGATIVE) and not claim.does_not_support:
        return False, f"{claim.polarity} claim {claim.claim_id} must record does_not_support"
    return True, "ok"


def run_all(
    claim: SemanticClaim | None = None,
    utterance: GroundedUtterance | None = None,
    supported_claim_ids: list[str] | None = None,
) -> dict[str, tuple[bool, str]]:
    out = {}
    if claim is not None:
        out["evidence_required"] = check_evidence_required(claim)
        out["negation_boundary"] = check_negation_boundary(claim)
        out["does_not_support_present"] = check_does_not_support_present(claim)
    if utterance is not None and supported_claim_ids is not None:
        out["no_extra_claims"] = check_no_extra_claims(utterance, supported_claim_ids)
    return out


PRONOUN = "它"


@dataclass
class DiscourseState:
    mentioned: list[str] = field(default_factory=list)
    last_referenced_object: str | None = None
    salience_scores: dict[str, float] = field(default_factory=dict)
    evidence: list[str] = field(default_factory=list)

    def mention(self, symbol: str, salience: float | None = None) -> None:
        if not symbol:
            raise ValueError("mention requires a non-empty symbol.")
        self.mentioned.append(symbol)
        self.last_referenced_object = symbol
        if salience is not None:
            self.salience_scores[symbol] = salience

    def candidates(self) -> list[str]:
        out = []
        for symbol in self.mentioned:
            if symbol not in out:
                out.append(symbol)
        return out

    def can_use_pronoun(self) -> bool:
        return len(self.candidates()) == 1

    def resolve_pronoun(self) -> str:
        candidates = self.candidates()
        if not candidates:
            raise ValueError("no referent available for pronoun.")
        if len(candidates) > 1:
            raise ValueError(f"ambiguous pronoun: {candidates}")
        return candidates[0]

    def reference_label(self) -> str:
        if self.can_use_pronoun():
            return PRONOUN
        if self.last_referenced_object:
            return to_human_label(self.last_referenced_object)
        return to_human_label("unknown")


def realize_with_reference(action_symbol: str, target_symbol: str, state: DiscourseState, evidence: str | list[str]) -> dict:
    if not evidence:
        raise ValueError("realize_with_reference requires evidence.")
    subject = state.reference_label()
    target = to_human_label(target_symbol)
    text = f"我把{subject}放到{target}上面。" if action_symbol == "place" else f"我{to_human_label(action_symbol)}{subject}。"
    return {"text": text, "used_pronoun": subject == PRONOUN, "evidence": evidence if isinstance(evidence, list) else [evidence]}


QUANT_NO_EVIDENCE = "quantifier_without_visible_set_evidence"
_CN_NUM = {0: "零", 1: "一", 2: "两", 3: "三", 4: "四", 5: "五", 6: "六", 7: "七", 8: "八", 9: "九", 10: "十"}


def _cn_count(value: int) -> str:
    return _CN_NUM.get(value, str(value))


@dataclass
class VisibleObjectSet:
    objects: list[ObjectState] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    boundary: str = "visible objects in current scene"

    def visible_objects(self) -> list[ObjectState]:
        return [obj for obj in self.objects if obj.visible]

    def has_evidence(self) -> bool:
        return bool(self.evidence)


Predicate = tuple[str, Any]


def _matches(obj: ObjectState, predicate: Predicate) -> bool:
    attr, val = predicate
    return obj.attributes.get(attr) == val


def _type_label(type_filter: str | None) -> str:
    return {"block": "积木", "cup": "杯子", "bowl": "碗"}.get(type_filter, to_human_label(type_filter) if type_filter else "物体")


def _pred_label(predicate: Predicate) -> str:
    _, val = predicate
    return {"red": "红色", "blue": "蓝色", "green": "绿色"}.get(val, to_human_label(val))


def count_visible(object_set: VisibleObjectSet, type_filter: str | None = None) -> int | None:
    if not object_set.has_evidence():
        return None
    objects = object_set.visible_objects()
    if type_filter is not None:
        objects = [obj for obj in objects if obj.attributes.get("type") == type_filter]
    return len(objects)


def _subset(object_set: VisibleObjectSet, type_filter: str | None) -> list[ObjectState]:
    objects = object_set.visible_objects()
    if type_filter is not None:
        objects = [obj for obj in objects if obj.attributes.get("type") == type_filter]
    return objects


def all_visible_are(object_set: VisibleObjectSet, predicate: Predicate, type_filter: str | None = None) -> bool | None:
    if not object_set.has_evidence():
        return None
    objects = _subset(object_set, type_filter)
    if not objects:
        return None
    return all(_matches(obj, predicate) for obj in objects)


def some_visible_are(object_set: VisibleObjectSet, predicate: Predicate, type_filter: str | None = None) -> bool | None:
    if not object_set.has_evidence():
        return None
    return any(_matches(obj, predicate) for obj in _subset(object_set, type_filter))


def none_visible_are(object_set: VisibleObjectSet, predicate: Predicate, type_filter: str | None = None) -> bool | None:
    if not object_set.has_evidence():
        return None
    return not any(_matches(obj, predicate) for obj in _subset(object_set, type_filter))


def not_all_visible_are(object_set: VisibleObjectSet, predicate: Predicate, type_filter: str | None = None) -> bool | None:
    if not object_set.has_evidence():
        return None
    objects = _subset(object_set, type_filter)
    if not objects:
        return None
    return any(not _matches(obj, predicate) for obj in objects)


def realize_count(object_set: VisibleObjectSet, type_filter: str | None = None) -> GroundedUtterance:
    n = count_visible(object_set, type_filter)
    if n is None:
        return GroundedUtterance(text="我不能确定数量。", claim_id=None, does_not_support=[QUANT_NO_EVIDENCE])
    return GroundedUtterance(text=f"我看到{_cn_count(n)}个{_type_label(type_filter)}。", claim_id=None, evidence=list(object_set.evidence))


def realize_quantifier_claim(
    quant: str,
    object_set: VisibleObjectSet,
    predicate: Predicate | None = None,
    type_filter: str | None = None,
) -> GroundedUtterance:
    if quant == "count":
        return realize_count(object_set, type_filter)
    if not object_set.has_evidence():
        return GroundedUtterance(text="我不能确定数量。", claim_id=None, does_not_support=[QUANT_NO_EVIDENCE])
    ev = list(object_set.evidence)
    tlab = _type_label(type_filter)
    plab = _pred_label(predicate) if predicate else ""
    if quant == "all":
        value = all_visible_are(object_set, predicate, type_filter)  # type: ignore[arg-type]
        if value is None:
            return GroundedUtterance(text="我不能确定数量。", claim_id=None, does_not_support=[QUANT_NO_EVIDENCE])
        text = f"所有可见{tlab}都是{plab}。" if value else f"不是所有{tlab}都是{plab}。"
    elif quant == "some":
        value = some_visible_are(object_set, predicate, type_filter)  # type: ignore[arg-type]
        text = f"有些{tlab}是{plab}。" if value else f"没有{tlab}是{plab}。"
    elif quant == "none":
        value = none_visible_are(object_set, predicate, type_filter) if predicate else count_visible(object_set, type_filter) == 0
        text = f"没有{tlab}是{plab}。" if predicate and value else (f"我没有看到{tlab}。" if value else f"我看到了{tlab}。")
    elif quant == "not_all":
        value = not_all_visible_are(object_set, predicate, type_filter)  # type: ignore[arg-type]
        if value is None:
            return GroundedUtterance(text="我不能确定数量。", claim_id=None, does_not_support=[QUANT_NO_EVIDENCE])
        text = f"不是所有{tlab}都是{plab}。" if value else f"所有可见{tlab}都是{plab}。"
    else:
        raise ValueError(f"unknown quantifier {quant!r}")
    return GroundedUtterance(text=text, claim_id=None, evidence=ev)


AMBIGUOUS_REFERENCE = "ambiguous_reference_resolution"
DEFAULT_SALIENCE_THRESHOLD = 0.3
REFERENCE_RESOLVED = "resolved"
REFERENCE_AMBIGUOUS = "ambiguous"
REFERENCE_UNKNOWN = "unknown"
RESOLVED = REFERENCE_RESOLVED
AMBIGUOUS = REFERENCE_AMBIGUOUS
UNKNOWN = REFERENCE_UNKNOWN


@dataclass
class ReferenceResolution:
    status: str
    referent: str | None = None
    candidates: list[str] = field(default_factory=list)
    does_not_support: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_reference(state: DiscourseState, threshold: float = DEFAULT_SALIENCE_THRESHOLD) -> ReferenceResolution:
    candidates = state.candidates()
    if not candidates:
        return ReferenceResolution(status=REFERENCE_UNKNOWN, does_not_support=[AMBIGUOUS_REFERENCE])
    if len(candidates) == 1:
        return ReferenceResolution(status=REFERENCE_RESOLVED, referent=candidates[0], candidates=candidates)
    scored = sorted(((candidate, state.salience_scores.get(candidate, 0.0)) for candidate in candidates), key=lambda item: item[1], reverse=True)
    top_sym, top_score = scored[0]
    _, runner_up = scored[1]
    if top_score > 0 and (top_score - runner_up) >= threshold and sum(1 for _, score in scored if score == top_score) == 1:
        return ReferenceResolution(status=REFERENCE_RESOLVED, referent=top_sym, candidates=candidates)
    return ReferenceResolution(status=REFERENCE_AMBIGUOUS, candidates=candidates, does_not_support=[AMBIGUOUS_REFERENCE])


def reference_label(state: DiscourseState, threshold: float = DEFAULT_SALIENCE_THRESHOLD) -> str:
    resolution = resolve_reference(state, threshold)
    if resolution.status == REFERENCE_RESOLVED:
        if len(resolution.candidates) == 1:
            return PRONOUN
        return to_human_label(resolution.referent)
    return ""


def realize_reference_action(
    action_symbol: str,
    target_symbol: str,
    state: DiscourseState,
    evidence: str | list[str],
    threshold: float = DEFAULT_SALIENCE_THRESHOLD,
) -> dict[str, Any]:
    if not evidence:
        raise ValueError("realize_reference_action requires evidence.")
    ev = evidence if isinstance(evidence, list) else [evidence]
    resolution = resolve_reference(state, threshold)
    if resolution.status != REFERENCE_RESOLVED:
        return {
            "text": "我不能确定你说的是哪一个。",
            "status": resolution.status,
            "used_pronoun": False,
            "referent": None,
            "does_not_support": [AMBIGUOUS_REFERENCE],
            "evidence": ev,
        }
    subject = PRONOUN if len(resolution.candidates) == 1 else to_human_label(resolution.referent)
    target = to_human_label(target_symbol)
    text = f"我把{subject}放到{target}上面。" if action_symbol == "place" else f"我{to_human_label(action_symbol)}{subject}。"
    return {
        "text": text,
        "status": REFERENCE_RESOLVED,
        "used_pronoun": subject == PRONOUN,
        "referent": resolution.referent,
        "does_not_support": [],
        "evidence": ev,
    }


TEMPORAL_NO_ORDER = "temporal_order_without_order_evidence"
TEMPORAL_AMBIGUOUS_REASON = "temporal_order_ambiguous_equal_keys"
TEMPORAL_RESOLVED = "resolved"
TEMPORAL_AMBIGUOUS = "ambiguous"
TEMPORAL_UNKNOWN = "unknown"
RESOLVED = TEMPORAL_RESOLVED
AMBIGUOUS = TEMPORAL_AMBIGUOUS
UNKNOWN = TEMPORAL_UNKNOWN


@dataclass
class Timeline:
    status: str
    events: list[ActionEvent] = field(default_factory=list)
    does_not_support: list[str] = field(default_factory=list)


def make_timeline(events: list[ActionEvent]) -> Timeline:
    if not events:
        return Timeline(status=TEMPORAL_UNKNOWN, does_not_support=[TEMPORAL_NO_ORDER])
    if any(not event.has_order_evidence() for event in events):
        return Timeline(status=TEMPORAL_UNKNOWN, does_not_support=[TEMPORAL_NO_ORDER])
    keys = [event.order_key() for event in events]
    seen: dict[tuple[str, int | float] | None, int] = {}
    for key in keys:
        seen[key] = seen.get(key, 0) + 1
    if any(count > 1 for count in seen.values()):
        return Timeline(status=TEMPORAL_AMBIGUOUS, does_not_support=[TEMPORAL_AMBIGUOUS_REASON])
    key_types = {key[0] for key in keys if key is not None}
    if len(key_types) > 1:
        return Timeline(status=TEMPORAL_AMBIGUOUS, does_not_support=[TEMPORAL_AMBIGUOUS_REASON])
    return Timeline(status=TEMPORAL_RESOLVED, events=sorted(events, key=lambda event: event.order_key()[1]))  # type: ignore[index]


def before_diagnostic(a: ActionEvent, b: ActionEvent) -> tuple[bool | None, str]:
    if not (a.has_order_evidence() and b.has_order_evidence()):
        return None, "missing_order_evidence"
    ka = a.order_key()
    kb = b.order_key()
    if ka[0] != kb[0]:  # type: ignore[index]
        return None, "mixed_key_types"
    if ka[1] == kb[1]:  # type: ignore[index]
        return None, "equal_keys"
    return ka[1] < kb[1], "ok"  # type: ignore[index]


def before(a: ActionEvent, b: ActionEvent) -> bool | None:
    return before_diagnostic(a, b)[0]


def after(a: ActionEvent, b: ActionEvent) -> bool | None:
    return before(b, a)


_ORDINAL_CN = ["先", "然后", "之后", "接着", "最后"]


def _event_clause(event: ActionEvent) -> str:
    subject = to_human_label(event.subject)
    if event.action == "place" and event.object:
        return f"把{subject}放到{to_human_label(event.object)}上面"
    if event.action == "pick":
        return f"拿起{subject}"
    if event.action == "stop":
        return "停下"
    if event.object:
        return f"{to_human_label(event.action)}{subject}到{to_human_label(event.object)}"
    return f"{to_human_label(event.action)}{subject}"


def realize_timeline(events: list[ActionEvent]) -> GroundedUtterance:
    timeline = make_timeline(events)
    if timeline.status != TEMPORAL_RESOLVED:
        return GroundedUtterance(
            text="我不能确定先后顺序。",
            claim_id=None,
            does_not_support=list(timeline.does_not_support),
        )
    parts = []
    evidence = []
    for index, event in enumerate(timeline.events):
        clause = _event_clause(event)
        if index == 0:
            parts.append(f"我先{clause}。")
        else:
            parts.append(f"{_ORDINAL_CN[index] if index < len(_ORDINAL_CN) else '然后'}我{clause}。")
        evidence.extend(event.evidence)
    return GroundedUtterance(text="".join(parts), claim_id=None, evidence=evidence)


# ======================================================================================
# Subsystem 2: claim denominator layer v2
# ======================================================================================

FAILURE_NONE = None
REGIME_INVALID = "REGIME_INVALID"
PIPELINE_OK_TRANSPORT_FAIL = "PIPELINE_OK_TRANSPORT_FAIL"
CONTROL_DEGENERATE = "CONTROL_DEGENERATE"
MECHANISM_DECAYS = "MECHANISM_DECAYS"
RANDOM_ALSO_SUCCEEDS = "RANDOM_ALSO_SUCCEEDS"
ACTIVE_NO_ADVANTAGE = "ACTIVE_NO_ADVANTAGE"
TRUTH_STABLE_NEURAL_UNLEARNABLE = "TRUTH_STABLE_NEURAL_UNLEARNABLE"
DIAGNOSTICS_INSUFFICIENT = "DIAGNOSTICS_INSUFFICIENT"
PRIOR_NO_EFFECT = "PRIOR_NO_EFFECT"
PENDING_GPU_VALIDATION = "PENDING_GPU_VALIDATION"

KNOWN_FAILURE_MODES = frozenset(
    {
        REGIME_INVALID,
        PIPELINE_OK_TRANSPORT_FAIL,
        CONTROL_DEGENERATE,
        MECHANISM_DECAYS,
        RANDOM_ALSO_SUCCEEDS,
        ACTIVE_NO_ADVANTAGE,
        TRUTH_STABLE_NEURAL_UNLEARNABLE,
        DIAGNOSTICS_INSUFFICIENT,
        PRIOR_NO_EFFECT,
        PENDING_GPU_VALIDATION,
    }
)


def is_known_failure_mode(mode: str | None) -> bool:
    return mode is None or mode in KNOWN_FAILURE_MODES


ALLOWED_ACTIONS = frozenset({"continue", "archive", "do_not_promote"})
_FORBIDDEN_ACTION_WORDS = frozenset({"certified", "promote", "proved", "validated"})
CERTIFIED_EVIDENCE = "vpsl_certified_structure"
CERTIFIED_GATE = "transfer"
DENOMINATOR_CLAIM_TYPES = frozenset(
    {"positive_evidence", "negative_result", "unresolved_result", "handoff", "certified_structure", "boundary_note"}
)
CONFIDENCE_LEVELS = frozenset({"low", "medium", "high", "certified"})


@dataclass
class ClaimRecord:
    claim_id: str
    structure_family: str
    evidence_level: str
    verdict: str
    gate: str
    allowed_action: str
    supports: list[str]
    does_not_support: list[str]
    controls: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    failure_mode: str | None = None
    next_gate: str | None = None
    claim_boundary: str = ""
    source_module: str = ""
    code_version: str = "unversioned"
    claim_type: str = "positive_evidence"
    confidence_level: str = "medium"
    evidence_scope: dict[str, Any] = field(default_factory=dict)
    replication: dict[str, Any] = field(default_factory=dict)
    risk_flags: list[str] = field(default_factory=list)
    supersedes: list[str] = field(default_factory=list)
    claim_status: str = "active"
    superseded_by: str | None = None
    superseded_reason: str = ""

    def __post_init__(self) -> None:
        for field_name in ("claim_id", "structure_family", "evidence_level", "verdict", "gate", "claim_boundary", "source_module"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string; got {value!r}")
        if self.allowed_action not in ALLOWED_ACTIONS or self.allowed_action in _FORBIDDEN_ACTION_WORDS:
            raise ValueError(f"allowed_action must be one of {sorted(ALLOWED_ACTIONS)}")
        if not isinstance(self.supports, list) or not self.supports:
            raise ValueError("supports must be a non-empty list.")
        if not isinstance(self.does_not_support, list) or not self.does_not_support:
            raise ValueError("does_not_support must be a non-empty list.")
        if not isinstance(self.controls, dict) or not isinstance(self.diagnostics, dict):
            raise TypeError("controls and diagnostics must be dicts.")
        if not is_known_failure_mode(self.failure_mode):
            raise ValueError(f"failure_mode {self.failure_mode!r} is not a known failure mode.")
        if "CERTIFIED" in self.verdict.upper():
            if self.gate != CERTIFIED_GATE or self.evidence_level != CERTIFIED_EVIDENCE:
                raise ValueError("CERTIFIED verdicts require transfer gate and vpsl_certified_structure evidence.")
        if self.claim_type not in DENOMINATOR_CLAIM_TYPES:
            raise ValueError(f"claim_type must be one of {sorted(DENOMINATOR_CLAIM_TYPES)}")
        if self.confidence_level not in CONFIDENCE_LEVELS:
            raise ValueError(f"confidence_level must be one of {sorted(CONFIDENCE_LEVELS)}")
        if self.confidence_level == "certified" and self.evidence_level != CERTIFIED_EVIDENCE:
            raise ValueError("confidence_level 'certified' is reserved for vpsl_certified_structure.")
        if not isinstance(self.evidence_scope, dict) or not isinstance(self.replication, dict):
            raise TypeError("evidence_scope and replication must be dicts.")
        if not isinstance(self.risk_flags, list):
            raise TypeError("risk_flags must be a list.")
        if not isinstance(self.supersedes, list):
            raise TypeError("supersedes must be a list.")
        if self.claim_status not in ("active", "superseded", "archived"):
            raise ValueError("claim_status must be active/superseded/archived.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ClaimRecord":
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        return cls(**{key: value for key, value in payload.items() if key in known})


_CODE = "claims_v2"
_PASS_E2B = "ACTIVE_DIAGNOSTIC_VALUE_PASSED"


def _action_from(result: dict[str, Any], default: str) -> str:
    action = (result.get("active_rec") or {}).get("allowed_action")
    return action if action in ALLOWED_ACTIONS else default


def claim_from_k3_e2b(result: dict[str, Any]) -> ClaimRecord:
    verdict = result.get("verdict", _PASS_E2B)
    active = result.get("active", {})
    return ClaimRecord(
        claim_id="k3_e2b_active_topology_search",
        structure_family="K3_TOPOLOGICAL",
        evidence_level="toy_active_regime_search",
        verdict=verdict,
        gate="active_search",
        allowed_action=_action_from(result, "continue"),
        supports=["guided active regime search finds a topology-trackable toy vortex regime better than random search"],
        does_not_support=["Gross-Pitaevskii truth validation", "CNN baseline learnability", "K3 prior validation", "VPSL certification"],
        controls={"random_median_best_score": result.get("random_median_best_score"), "random_success_count_20": result.get("random_success_count_20")},
        diagnostics={"active_best_score": active.get("best_score"), "active_found_transport_ok": active.get("found_transport_ok")},
        failure_mode=None if verdict == _PASS_E2B else ACTIVE_NO_ADVANTAGE,
        next_gate="K3-E2c cheap GP truth-only active search",
        claim_boundary="toy search landscape only; not GP truth; not K3 prior validation",
        source_module="chronos.k3.run_active_topology_search",
        code_version=_CODE,
        claim_type="positive_evidence",
        confidence_level="medium",
        evidence_scope={"system": "toy vortex regime", "regime": "analytic toy landscape", "model": "truth-only (toy)", "compute": "CPU toy"},
        replication={"n_seeds": result.get("random_success_count_20") and 20},
        risk_flags=["toy_landscape", "cpu_only", "no_cnn_training"],
    )


def claim_from_k3_e2c(result: dict[str, Any]) -> ClaimRecord:
    verdict = result.get("verdict", "GP_ACTIVE_NO_ADVANTAGE")
    random_success = result.get("random_success_count_20")
    failure_mode = RANDOM_ALSO_SUCCEEDS if isinstance(random_success, (int, float)) and random_success > 5 else ACTIVE_NO_ADVANTAGE
    return ClaimRecord(
        claim_id="k3_e2c_cheap_gp_active_search",
        structure_family="K3_TOPOLOGICAL",
        evidence_level="cheap_gp_truth_active_search",
        verdict=verdict,
        gate="truth_active_search",
        allowed_action="archive" if verdict == "GP_ACTIVE_NO_ADVANTAGE" else _action_from(result, "do_not_promote"),
        supports=["cheap GP truth evaluator and vortex tracker run successfully", "active/random comparison is operational"],
        does_not_support=["active exploration has diagnostic advantage on this cheap GP landscape", "CNN baseline learnability", "K3 prior validation", "VPSL certification"],
        controls={"random_median_best_score": result.get("random_median_best_score"), "random_success_count_20": random_success},
        diagnostics={"active_best_score": result.get("active", {}).get("best_score")},
        failure_mode=failure_mode,
        next_gate="K3-E2d discriminating GP truth active search",
        claim_boundary="cheap GP truth only; non-discriminating landscape; not prior validation",
        source_module="chronos.k3.run_gp_active_search",
        code_version=_CODE,
        claim_type="negative_result",
        confidence_level="low",
        evidence_scope={"system": "GP vortex pair", "regime": "cheap GP (non-discriminating)", "model": "truth-only", "compute": "CPU"},
        replication={"n_seeds": 20, "random_success_count": random_success},
        risk_flags=["cheap_gp", "cpu_only", "no_cnn_training", "random_also_succeeds"],
    )


def claim_from_k3_e2d(result: dict[str, Any]) -> ClaimRecord:
    active = result.get("active", {})
    metrics = active.get("best_metrics") or {}
    return ClaimRecord(
        claim_id="k3_e2d_discriminating_gp_active_search",
        structure_family="K3_TOPOLOGICAL",
        evidence_level="gp_truth_active_search",
        verdict=result.get("verdict", "GP_ACTIVE_DIAGNOSTIC_VALUE_PASSED"),
        gate="truth_active_search",
        allowed_action=_action_from(result, "continue"),
        supports=["guided active search finds a GP vortex regime with better transport diagnostics than random", "active reaches transport_ok while random rarely does"],
        does_not_support=["CNN baseline regime validation", "K3 prior validation", "VPSL certification"],
        controls={"random_median_best_score": result.get("random_median_best_score"), "random_success_count_20": result.get("random_success_count_20"), "random_success_frac_20": result.get("random_success_frac_20")},
        diagnostics={"active_best_score": active.get("best_score"), "active_mean_pos_err": metrics.get("mean_pos_err"), "active_pair_intact": metrics.get("pair_intact")},
        failure_mode=None,
        next_gate="K3.2D.0 CNN baseline regime validation",
        claim_boundary="truth-level GP active search only; not CNN validation; not K3 prior validation",
        source_module="chronos.k3.run_gp_active_search",
        code_version=_CODE,
        claim_type="positive_evidence",
        confidence_level="medium",
        evidence_scope={"system": "GP vortex pair", "regime": "discriminating GP (push failure dim)", "model": "truth-only", "compute": "CPU"},
        replication={"n_seeds": 20, "random_success_count": result.get("random_success_count_20")},
        risk_flags=["cpu_only", "no_cnn_training", "designed_failure_dimension"],
    )


def claim_from_k3_2d_0_summary(summary: dict[str, Any]) -> ClaimRecord:
    pipeline_ok = bool(summary.get("pipeline_ok"))
    transport_ok = bool(summary.get("transport_ok"))
    verdict = summary.get("verdict", "SMOKE_PIPELINE_OK_TRANSPORT_FAIL")
    if not (pipeline_ok and not transport_ok):
        raise ValueError("claim_from_k3_2d_0_summary currently only handles PIPELINE_OK_TRANSPORT_FAIL.")
    return ClaimRecord(
        claim_id="k3_2d_0_vortex_regime_smoke",
        structure_family="K3_TOPOLOGICAL",
        evidence_level="neural_baseline_regime_validation_smoke",
        verdict=verdict,
        gate="regime",
        allowed_action="do_not_promote",
        supports=["CNN baseline can learn bounded field prediction in smoke mode", "vortex transport gate fails"],
        does_not_support=["K3.2D regime validation", "K3 prior test readiness", "topology prior validation"],
        controls={"baseline_only": True, "ref_med": summary.get("ref_med"), "pair_frac": summary.get("pair_frac"), "pos_med": summary.get("pos_med")},
        diagnostics={"pipeline_ok": pipeline_ok, "transport_ok": transport_ok},
        failure_mode=PIPELINE_OK_TRANSPORT_FAIL,
        next_gate="K3.2D.0-B neural baseline learnability rescue or archive as regime unresolved",
        claim_boundary="smoke only; CNN transport failed; no prior test allowed",
        source_module="chronos.k3.experiments.k3_2d_0_vortex_regime",
        code_version=_CODE,
        claim_type="unresolved_result",
        confidence_level="low",
        evidence_scope={"system": "GP vortex pair", "regime": "K3.2D.0 smoke", "model": "CNN baseline", "compute": "GPU smoke"},
        replication={"n_seeds": summary.get("n_seeds")},
        risk_flags=["smoke_only", "transport_fail", "no_prior_test"],
    )


def claim_from_k2_summary(summary: dict[str, Any]) -> ClaimRecord:
    return ClaimRecord(
        claim_id="k2_symplectic_full_transfer",
        structure_family="K2_SYMPLECTIC",
        evidence_level="vpsl_certified_structure",
        verdict="FULL_TRANSFER_CONFIRMED",
        gate="transfer",
        allowed_action="continue",
        supports=["symplectic prior beats baseline", "symplectic prior beats fair energy and fair L2 controls", "full symplectic Jacobian mechanism transfers through H=240"],
        does_not_support=["all physical priors work", "K3/K4/K5 claims", "universal physics AI"],
        controls={"fair_energy_control": True, "fair_l2_control": True, "horizon": 240},
        diagnostics=summary.get("diagnostics", {"transfer": "full", "H": 240}),
        failure_mode=None,
        next_gate="wrong-Omega hardening and broader-system transfer",
        claim_boundary="certified only for tested FPU-beta regime and VPSL controls",
        source_module="chronos.k2",
        code_version=_CODE,
        claim_type="certified_structure",
        confidence_level="certified",
        evidence_scope={"system": "FPU-beta", "regime": "H=240", "model": "prior vs fair controls", "compute": "FULL"},
        replication=summary.get("replication", {"n_seeds": summary.get("n_seeds")}),
        risk_flags=["scope_limited_to_FPU_beta"],
    )


_LANG_LEVEL_SUPPORTS = {
    "L1": ["L1 not-visible negation", "L1 contrastive correction"],
    "L2": ["L2 causal explanation only with causal evidence", "L2 correlation is not treated as cause"],
    "L3": ["L3 visible-set grounded counting", "L3 all/some/none/not_all distinction", "L3 unknown when visible-set evidence is missing"],
    "L4": ["L4 single-object pronoun reference"],
    "L4A": ["L4A ambiguous reference is rejected or expanded", "L4A multi-object pronoun use is blocked unless salience is sufficient"],
    "L5": ["L5 evidence-backed temporal ordering", "L5 missing temporal evidence produces unknown", "L5 no invented intermediate events"],
}
_LANG_LEVEL_DOES_NOT_SUPPORT = {
    "L2": ["causal discovery"],
    "L4A": ["ambiguous reference beyond toy salience model"],
    "L5": ["temporal reasoning beyond explicit event order evidence"],
}
_LANG_LEVEL_ORDER = ["L1", "L2", "L3", "L4", "L4A", "L5"]
_LANG_NEXT_GATE = {
    "L1": "L1 not-visible negation",
    "L2": "L2 causal boundary",
    "L3": "L3 quantifier",
    "L4": "L4 reference",
    "L4A": "L4A ambiguity-hardened reference",
    "L5": "L5 temporal ordering",
}
_LANG_FULL_NEXT_GATE = "multi-step action grounding + real sensor trace replay"


def next_missing_level(levels: list[str]) -> str | None:
    covered = set(levels)
    for level in _LANG_LEVEL_ORDER:
        if level not in covered:
            return level
    return None


def _lang_claim_id(levels: list[str]) -> str:
    tag = "_".join(level for level in _LANG_LEVEL_ORDER if level in levels)
    return f"L_VPSL_GROUNDED_LANGUAGE_{tag}_TOY_MVP"


def claim_from_language_grounding(result: dict[str, Any]) -> ClaimRecord:
    passed = bool(result.get("passed"))
    n_assertions = result.get("n_assertions")
    raw_levels = result.get("levels", ["L1", "L2", "L3", "L4"])
    if isinstance(raw_levels, str) or not isinstance(raw_levels, (list, tuple, set)):
        raise TypeError(f"levels must be a list/tuple/set of tokens, not {type(raw_levels).__name__}")
    unknown_levels = [level for level in raw_levels if level not in _LANG_LEVEL_ORDER]
    if unknown_levels:
        raise ValueError(f"unknown language level token(s) {unknown_levels}")
    levels = [level for level in _LANG_LEVEL_ORDER if level in set(raw_levels)]
    if not levels:
        raise ValueError("language claim requires at least one covered level.")
    if passed:
        supports = ["no-LLM grounded utterance generation from verified semantic claims"]
        for level in levels:
            supports.extend(_LANG_LEVEL_SUPPORTS[level])
        qtypes = []
        if "L2" in levels:
            qtypes.append("why")
        if "L3" in levels:
            qtypes.append("quantifier")
        if "L4" in levels or "L4A" in levels:
            qtypes.append("reference")
        if "L5" in levels:
            qtypes.append("temporal")
        if qtypes:
            joined = ", ".join(qtypes[:-1]) + (", and " + qtypes[-1] if len(qtypes) > 1 else qtypes[-1])
            supports.append(f"does_not_support preserved in unsupported {joined} questions")
    else:
        supports = ["language grounding test suite ran and produced diagnostics", "controlled examples were evaluated"]
    does_not_support = [
        "general language understanding",
        "open-domain conversation",
        "LLM-level fluency",
        "autonomous robot intelligence",
        "real-world robot deployment",
    ]
    if passed:
        for level in levels:
            does_not_support.extend(_LANG_LEVEL_DOES_NOT_SUPPORT.get(level, []))
    else:
        does_not_support.append("any language level capability (test did not pass)")
    missing = [level for level in _LANG_LEVEL_ORDER if level not in levels]
    for level in missing:
        does_not_support.append(f"{level} capability (not yet covered)")
    next_level = next_missing_level(levels)
    next_gate = _LANG_NEXT_GATE[next_level] if next_level is not None else _LANG_FULL_NEXT_GATE
    contiguous = not missing or levels == _LANG_LEVEL_ORDER[: len(levels)]
    return ClaimRecord(
        claim_id=_lang_claim_id(levels),
        structure_family="LANGUAGE_GROUNDING",
        evidence_level="toy_bounded_positive" if passed else "toy_bounded_negative",
        verdict="PASSED_TOY_MVP" if passed else "TOY_MVP_FAILED",
        gate="toy_mechanism",
        allowed_action="continue" if passed else "do_not_promote",
        supports=supports,
        does_not_support=does_not_support,
        controls={"no_llm": True, "no_torch": True, "stdlib_only": True, "levels_covered": levels},
        diagnostics={"n_assertions": n_assertions, "passed": passed},
        failure_mode=None if passed else DIAGNOSTICS_INSUFFICIENT,
        next_gate=next_gate,
        claim_boundary=(
            "toy no-LLM grounded language MVP using hand-authored semantic claims, verified visible object sets, "
            "controlled salience scores, explicit event order evidence, and stdlib-only surface realization; "
            "not a general language model"
            + ("" if contiguous else "; levels may be non-contiguous: a higher level does not imply lower ones")
        ),
        source_module="chronos.language_grounding",
        code_version=_CODE,
        claim_type="positive_evidence" if passed else "negative_result",
        confidence_level="medium" if passed else "low",
        evidence_scope={"system": "hand-authored semantic claims + verified visible object sets", "regime": "controlled toy examples", "model": "template realizer (no LLM)", "compute": "CPU stdlib"},
        replication={"n_assertions": n_assertions, "deterministic": True},
        risk_flags=["toy_examples", "hand_authored_claims", "no_real_robot", "no_llm_baseline_comparison"],
        supersedes=[],
        claim_status="active",
    )


claim_from_language_grounding_summary = claim_from_language_grounding


def _count_by(claims: list[ClaimRecord], attr: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for claim in claims:
        value = getattr(claim, attr)
        key = "None" if value is None else str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _lang_levels_of(claim_id: str) -> set[str] | None:
    if not claim_id.startswith("L_VPSL_GROUNDED_LANGUAGE_"):
        return None
    core = claim_id.replace("L_VPSL_GROUNDED_LANGUAGE_", "").replace("_TOY_MVP", "")
    tokens = [token for token in core.split("_") if token]
    return set(tokens) if tokens else None


def active_claims(claims: list[ClaimRecord]) -> list[ClaimRecord]:
    superseded_ids = set()
    for claim in claims:
        for claim_id in getattr(claim, "supersedes", []) or []:
            superseded_ids.add(claim_id)

    def can_supersede(claim: ClaimRecord) -> bool:
        return claim.claim_type == "positive_evidence" and claim.allowed_action == "continue" and claim.verdict == "PASSED_TOY_MVP"

    lang_claims = [
        (index, claim, levels)
        for index, claim in enumerate(claims)
        if claim.structure_family == "LANGUAGE_GROUNDING"
        for levels in [_lang_levels_of(claim.claim_id)]
        if levels is not None
    ]
    for i_a, c_a, lv_a in lang_claims:
        for i_b, c_b, lv_b in lang_claims:
            if i_b <= i_a:
                continue
            if lv_a < lv_b and can_supersede(c_b):
                superseded_ids.add(c_a.claim_id)
                break
    return [
        claim
        for claim in claims
        if claim.claim_id not in superseded_ids and getattr(claim, "claim_status", "active") == "active"
    ]


def summarize_claims(claims: list[ClaimRecord]) -> dict[str, Any]:
    active = active_claims(claims)
    latest = {}
    for claim in active:
        latest[claim.structure_family] = claim.to_dict()
    return {
        "count_total": len(active),
        "count_total_including_superseded": len(claims),
        "count_by_structure_family": _count_by(active, "structure_family"),
        "count_by_evidence_level": _count_by(active, "evidence_level"),
        "count_by_verdict": _count_by(active, "verdict"),
        "count_by_allowed_action": _count_by(active, "allowed_action"),
        "count_by_failure_mode": _count_by(active, "failure_mode"),
        "count_by_claim_type": _count_by(active, "claim_type"),
        "count_by_confidence_level": _count_by(active, "confidence_level"),
        "latest_by_structure_family": latest,
    }


def claims_requiring_next_gate(claims: list[ClaimRecord]) -> list[ClaimRecord]:
    return [claim for claim in active_claims(claims) if claim.allowed_action == "continue" and claim.next_gate]


def claims_with_risk_flag(claims: list[ClaimRecord], flag: str) -> list[ClaimRecord]:
    return [claim for claim in active_claims(claims) if flag in claim.risk_flags]


def human_readable_summary(claim: ClaimRecord) -> str:
    support = claim.supports[0] if claim.supports else "a result"
    does_not = ", ".join(claim.does_not_support[:3]) if claim.does_not_support else "nothing further"
    strength = {"low": "weak", "medium": "moderate", "high": "strong", "certified": "VPSL-certified"}.get(
        claim.confidence_level, claim.confidence_level
    )
    next_gate = f"; next gate: {claim.next_gate}" if claim.next_gate else ""
    return (
        f"{claim.claim_id} ({claim.claim_type}, {strength} evidence): {support}. "
        f"It does NOT establish: {does_not}. Action: {claim.allowed_action}{next_gate}."
    )


def append_claim(claim: ClaimRecord, path: str) -> None:
    if not isinstance(claim, ClaimRecord):
        raise TypeError(f"append_claim expects a ClaimRecord, got {type(claim).__name__}")
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
                claims.append(ClaimRecord.from_dict(json.loads(line)))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid claim record at {path}:{line_number}: {exc}") from exc
    return claims


# ======================================================================================
# Self-tests and full replay
# ======================================================================================


def _tests_language() -> int:
    count = 0

    def check(condition: bool) -> None:
        nonlocal count
        assert condition
        count += 1

    reset_lexicon()
    relation = SemanticClaim(
        claim_id="r",
        claim_type="relation",
        subject="red_block",
        predicate="on_top_of",
        object="blue_block",
        confidence=0.9,
        evidence=["v"],
    )
    check(realize(relation).text == "我看到红色积木在蓝色积木上面。")
    check(realize_from_object_state(ObjectState("o", "cup", ["s"], visible=False)).text == "我没有看到杯子。")
    reset_lexicon()
    update_label("stop", "停下", "s")
    update_label("obstacle_ahead", "前面有障碍物", "s")
    causal = SemanticClaim("c", "causal", "stop", "because", "obstacle_ahead", 0.9, ["causal_obs"])
    check(realize(causal).text == "我停下,因为前面有障碍物。")
    noncausal = SemanticClaim("c2", "causal", "stop", "because", "obstacle_ahead", 0.9, ["correlation"])
    check(realize(noncausal).text == "我不知道原因。")

    def red_block(idx: int) -> ObjectState:
        return ObjectState(f"r{idx}", "red_block", ["v"], attributes={"type": "block", "color": "red"})

    object_set = VisibleObjectSet([red_block(1), red_block(2)], ["scan"])
    check(realize_quantifier_claim("count", object_set, type_filter="block").text == "我看到两个积木。")
    state = DiscourseState()
    state.mention("red_block")
    check(resolve_reference(state).status == REFERENCE_RESOLVED)
    state.mention("cup")
    check(resolve_reference(state).status == REFERENCE_AMBIGUOUS)

    def event(action: str, subject: str, obj: str | None = None, idx: int | None = None) -> ActionEvent:
        return ActionEvent(action=action, subject=subject, object=obj, evidence=["trace"], order_index=idx)

    events = [event("pick", "red_block", idx=0), event("place", "red_block", "blue_block", idx=1), event("stop", "robot", idx=2)]
    check(make_timeline(events).status == TEMPORAL_RESOLVED)
    check(realize_timeline(events).text == "我先拿起红色积木。然后我把红色积木放到蓝色积木上面。之后我停下。")
    check(TEMPORAL_NO_ORDER in realize_timeline([event("pick", "red_block", idx=0), event("stop", "robot")]).does_not_support)
    return count


def _tests_claims() -> int:
    count = 0

    def check(condition: bool) -> None:
        nonlocal count
        assert condition
        count += 1

    full = claim_from_language_grounding({"passed": True, "n_assertions": 40, "levels": _LANG_LEVEL_ORDER})
    check(full.claim_id == "L_VPSL_GROUNDED_LANGUAGE_L1_L2_L3_L4_L4A_L5_TOY_MVP")
    check(full.allowed_action == "continue" and full.confidence_level != "certified")
    check("L5 evidence-backed temporal ordering" in full.supports)
    partial = claim_from_language_grounding({"passed": True, "n_assertions": 10, "levels": ["L1", "L2", "L4"]})
    check(partial.claim_id == "L_VPSL_GROUNDED_LANGUAGE_L1_L2_L4_TOY_MVP")
    check("L3 visible-set grounded counting" not in partial.supports)
    check("L3 capability (not yet covered)" in partial.does_not_support)
    check(partial.next_gate == "L3 quantifier")
    k2 = claim_from_k2_summary({})
    check(k2.confidence_level == "certified" and k2.claim_type == "certified_structure")
    old = ClaimRecord(
        claim_id="L_VPSL_GROUNDED_LANGUAGE_L1_L2_L4_TOY_MVP",
        structure_family="LANGUAGE_GROUNDING",
        evidence_level="toy_bounded_positive",
        verdict="PASSED_TOY_MVP",
        gate="toy_mechanism",
        allowed_action="continue",
        supports=["old"],
        does_not_support=["old"],
        claim_boundary="old",
        source_module="x",
        claim_type="positive_evidence",
        confidence_level="medium",
        next_gate="ng",
        risk_flags=["toy_examples"],
    )
    summary = summarize_claims([k2, old, full])
    check(summary["count_total"] == 2 and summary["count_total_including_superseded"] == 3)
    check(old.claim_id not in [claim.claim_id for claim in claims_requiring_next_gate([old, full])])
    check(old.claim_id not in [claim.claim_id for claim in claims_with_risk_flag([old, full], "toy_examples")])
    return count


def build_full_ledger() -> list[ClaimRecord]:
    k2 = claim_from_k2_summary({})
    e2c = claim_from_k3_e2c(
        {
            "verdict": "GP_ACTIVE_NO_ADVANTAGE",
            "active": {"best_score": 1.0},
            "active_rec": {"allowed_action": "do_not_promote"},
            "random_success_count_20": 18,
        }
    )
    e2d = claim_from_k3_e2d(
        {
            "verdict": "GP_ACTIVE_DIAGNOSTIC_VALUE_PASSED",
            "active": {
                "best_score": 0.97,
                "found_transport_ok": True,
                "best_metrics": {"mean_pos_err": 0.31, "pair_intact": True},
            },
            "active_rec": {"allowed_action": "continue"},
            "random_success_count_20": 2,
        }
    )
    lang = claim_from_language_grounding({"passed": True, "n_assertions": 40, "levels": _LANG_LEVEL_ORDER})
    return [k2, e2c, e2d, lang]


def _has_claim(ledger: list[ClaimRecord], claim_id: str) -> bool:
    return any(claim.claim_id == claim_id for claim in active_claims(ledger))


def _get_claim(ledger: list[ClaimRecord], claim_id: str) -> ClaimRecord:
    return next(claim for claim in active_claims(ledger) if claim.claim_id == claim_id)


def run_full_denominator() -> tuple[list[ClaimRecord], dict[str, Any]]:
    ledger = build_full_ledger()
    summary = summarize_claims(ledger)
    by_type = summary["count_by_claim_type"]
    by_confidence = summary["count_by_confidence_level"]
    assert by_type["negative_result"] == 1, by_type
    assert by_type["positive_evidence"] == 2, by_type
    assert by_type["certified_structure"] == 1, by_type
    assert by_confidence["certified"] == 1 and by_confidence["low"] == 1 and by_confidence["medium"] == 2, by_confidence
    assert _has_claim(ledger, "k2_symplectic_full_transfer")
    assert _has_claim(ledger, "k3_e2c_cheap_gp_active_search")
    assert _has_claim(ledger, "k3_e2d_discriminating_gp_active_search")
    assert _has_claim(ledger, "L_VPSL_GROUNDED_LANGUAGE_L1_L2_L3_L4_L4A_L5_TOY_MVP")
    assert _get_claim(ledger, "k3_e2c_cheap_gp_active_search").allowed_action == "archive"
    assert _get_claim(ledger, "k3_e2d_discriminating_gp_active_search").allowed_action == "continue"
    lang = _get_claim(ledger, "L_VPSL_GROUNDED_LANGUAGE_L1_L2_L3_L4_L4A_L5_TOY_MVP")
    assert lang.allowed_action == "continue"
    assert lang.next_gate == _LANG_FULL_NEXT_GATE
    assert "L5 evidence-backed temporal ordering" in lang.supports
    assert "LLM-level fluency" in lang.does_not_support
    return ledger, summary


def _anti_cheat_checks() -> int:
    failed = claim_from_language_grounding({"passed": False, "n_assertions": 8, "levels": ["L1", "L2"]})
    assert failed.allowed_action == "do_not_promote"
    assert failed.claim_type == "negative_result"
    assert failed.evidence_level == "toy_bounded_negative"
    assert "L1 not-visible negation" not in failed.supports
    assert "any language level capability (test did not pass)" in failed.does_not_support
    try:
        claim_from_language_grounding({"passed": True, "n_assertions": 1, "levels": "L1"})
        assert False
    except TypeError:
        pass
    reset_lexicon()
    update_label("stop", "停下", "s")
    update_label("obstacle_ahead", "前面有障碍物", "s")
    unknown_cause = SemanticClaim(
        claim_id="uc",
        claim_type="causal",
        subject="stop",
        predicate="because",
        object="obstacle_ahead",
        confidence=0.9,
        evidence=["weird_sensor_hint"],
    )
    assert realize(unknown_cause).text == "我不知道原因。"
    positive_partial = claim_from_language_grounding({"passed": True, "n_assertions": 10, "levels": ["L1", "L2", "L4"]})
    failed_full = claim_from_language_grounding({"passed": False, "n_assertions": 3, "levels": _LANG_LEVEL_ORDER})
    active_ids = [claim.claim_id for claim in active_claims([positive_partial, failed_full])]
    assert positive_partial.claim_id in active_ids
    return 6


if __name__ == "__main__":
    language_count = _tests_language()
    claims_count = _tests_claims()
    anti_cheat_count = _anti_cheat_checks()
    print("=== Self-tests (not just the happy ledger) ===")
    print(
        f"  language:{language_count}  claims:{claims_count}  anti-cheat:{anti_cheat_count}  "
        "(failed-claim, malformed-levels, unknown-causal, failed-supersede)\n"
    )
    ledger_, summary_ = run_full_denominator()
    print("=== FULL Chronos Denominator Replay (self-contained) ===\n")
    print("  by_claim_type      :", summary_["count_by_claim_type"])
    print("  by_confidence_level:", summary_["count_by_confidence_level"])
    print("  count_total        :", summary_["count_total"])
    print()
    for claim_ in active_claims(ledger_):
        print("  *", human_readable_summary(claim_))
    print("\n  ok all full-denominator + self-test + anti-cheat assertions passed")
