"""chronos_y20_core_single.py - Y20-Core single-file Colab replay.

Self-contained, no-LLM, stdlib-only. This file flattens the minimal Y30
cognitive substrate and the Y20 objection / response / debate-boundary layer
into one runnable script for Colab or:

    python colab/chronos_y20_core_single.py

Canonical package implementation lives under chronos/y30/ and chronos/y20/.
This replay preserves the portable demonstration shape: Y30 supplies cognitive
structure; Y20 supplies debate structure; the K-family + VPSL gates decide the
physics. A Y20 objection answers by naming the required gate, never by changing
a physics verdict.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

TOY_BOUNDARY = "toy cognitive/debate structure only; no metaphysical certification"
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


def _ensure(values: list[str], required: list[str]) -> list[str]:
    _check_str_list(values, "does_not_support")
    out = list(values)
    for item in required:
        if item not in out:
            out.append(item)
    return out


def _require_evidence(evidence: list[str], name: str) -> None:
    _check_str_list(evidence, "evidence")
    if not evidence:
        raise ValueError(f"{name} requires non-empty evidence")


@dataclass
class AppearanceEvent:
    event_id: str
    appearance: str
    evidence: list[str] = field(default_factory=list)
    confidence: float = 0.0
    conditions: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        _check_nonempty(self.event_id, "event_id")
        _check_nonempty(self.appearance, "appearance")
        _check_str_list(self.evidence, "evidence")
        _check_str_list(self.conditions, "conditions")
        _check_confidence(self.confidence)

    @property
    def is_affirmable(self) -> bool:
        return bool(self.evidence)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DependentConditions:
    condition_id: str
    appearance_id: str
    conditions: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    unresolved_conditions: list[str] = field(default_factory=list)
    confidence: float = 0.0

    def __post_init__(self) -> None:
        _check_nonempty(self.condition_id, "condition_id")
        _check_nonempty(self.appearance_id, "appearance_id")
        _check_str_list(self.conditions, "conditions")
        _check_str_list(self.evidence, "evidence")
        _check_str_list(self.unresolved_conditions, "unresolved_conditions")
        _check_confidence(self.confidence)
        if not self.conditions and not self.unresolved_conditions:
            self.unresolved_conditions.append("dependent_conditions_unresolved")

    @property
    def status(self) -> str:
        if not self.conditions:
            return "unresolved"
        return "confirmed" if self.evidence else "candidate"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SeedTrace:
    seed_id: str
    source_event_id: str
    stored_tendency: str
    evidence: list[str] = field(default_factory=list)
    does_not_support: list[str] = field(default_factory=list)
    claim_boundary: str = ""

    def __post_init__(self) -> None:
        _check_nonempty(self.seed_id, "seed_id")
        _check_nonempty(self.source_event_id, "source_event_id")
        _check_nonempty(self.stored_tendency, "stored_tendency")
        _require_evidence(self.evidence, "SeedTrace")
        self.does_not_support = _ensure(
            self.does_not_support,
            ["alaya-vijnana is scientifically proven", "religious doctrine is proven"],
        )
        self.claim_boundary = self.claim_boundary or "memory/habit-trace analogy only"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
GLOBAL_DNS = [
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
            raise ValueError(f"unknown objection_type {self.objection_type!r}")
        _check_nonempty(self.objection_id, "objection_id")
        _check_nonempty(self.realist_claim, "realist_claim")
        _check_str_list(self.evidence, "evidence")
        self.does_not_support = _ensure(
            self.does_not_support,
            ["external objects are proven to exist independently", "the realist position is established as a matter of fact"],
        )
        self.claim_boundary = self.claim_boundary or "records the external-realist objection; does not prove realism"

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
            raise ValueError(f"unknown strategy {self.strategy!r}")
        _check_nonempty(self.response_id, "response_id")
        _check_nonempty(self.objection_id, "objection_id")
        _check_nonempty(self.response_claim, "response_claim")
        _check_confidence(self.confidence)
        _require_evidence(self.evidence, "Y20Response")
        _check_str_list(self.supports, "supports")
        self.does_not_support = _ensure(self.does_not_support, GLOBAL_DNS + ["the response is a metaphysical proof"])
        self.claim_boundary = self.claim_boundary or TOY_BOUNDARY

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExternalObjectBoundary:
    boundary_id: str
    claim: str = "appearance does not require asserting an independent external object"
    does_not_support: list[str] = field(default_factory=list)
    claim_boundary: str = ""

    def __post_init__(self) -> None:
        _check_nonempty(self.boundary_id, "boundary_id")
        self.does_not_support = _ensure(self.does_not_support, GLOBAL_DNS + ["scientific realism is refuted"])
        self.claim_boundary = self.claim_boundary or "argument structure only; no external-world nonexistence proof"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


OBJECTION_PHRASE = {
    "spatiotemporal_determination": "如果只是识，为什么显现固定在特定的地点和时间？",
    "intersubjective_agreement": "如果只是识，为什么多个观察者会看到一致的对象？",
    "causal_efficacy": "如果只是识，为什么对象能产生真实的作用？",
    "dream_waking_distinction": "如果都是识，梦境和醒觉经验又有什么差别？",
    "seed_continuity": "如果只是识，显现的连续性如何解释？",
    "no_object_no_cognition": "认识必须有一个外部对象作为所缘，否则认识无法成立。",
}
RESPONSE_PHRASE = {
    "dream_analogy": "以梦为喻：梦中显现也有地点、时间和作用，却不需要外部实体",
    "shared_karma": "以共业为喻：多个观察者的一致，可由共同的业缘条件解释",
    "functional_efficacy": "表象本身可以有功能作用，不需要外部实体作为支撑",
    "dependent_designation": "「对象」是依条件的假名安立，不是被发现的自性实体",
    "waking_as_special_dream": "醒觉与梦的差别在于稳定性和连续性，而非是否需要外部实体",
    "seed_continuity_model": "连续性可建模为潜在种子/习气的条件相续，而非一个形上基体",
}


def _assert_clean(text: str) -> str:
    for bad in BANNED_SUBSTRINGS:
        if bad in text:
            raise ValueError(f"banned overclaim substring produced: {bad!r}")
    return text


def realize_objection(obj: Y20Objection) -> str:
    return _assert_clean(f"外境实有论者诘难：{OBJECTION_PHRASE.get(obj.objection_type, obj.realist_claim)}")


def realize_response(resp: Y20Response) -> str:
    phrase = RESPONSE_PHRASE.get(resp.strategy, resp.response_claim)
    return _assert_clean(f"唯识回应：{phrase}。这是论证结构上的回应，不证明外境的存废，也不证明唯心论为终极真理。")


def realize_external_object_boundary(boundary: ExternalObjectBoundary) -> str:
    return _assert_clean("边界说明：显现不需要预设一个独立的外部对象；但这只是论证结构，不否定科学实在论，也不断言外境的存废。")


def debate_claim_record(obj: Y20Objection, resp: Y20Response, boundary: ExternalObjectBoundary) -> dict[str, Any]:
    if resp.objection_id != obj.objection_id:
        raise ValueError("response.objection_id must match objection.objection_id")
    return {
        "claim_id": f"debate_{obj.objection_id}",
        "structure_family": "Y20_DEBATE_STRUCTURE",
        "verdict": "Y20_DEBATE_STRUCTURE_OK",
        "allowed_action": "continue",
        "supports": ["objection recorded", "bounded response recorded", "boundary recorded"],
        "does_not_support": list(GLOBAL_DNS) + ["scientific realism is refuted"],
        "controls": {"no_llm": True, "metaphysical_certification": False},
        "next_gate": "Y20-L1: objection taxonomy coverage",
        "claim_boundary": "argument structure only; no metaphysical certification",
    }


STANDARD = {
    "Y20-O1": ("spatiotemporal_determination", "dream_analogy", "Y20-O2: shared appearance"),
    "Y20-O2": ("intersubjective_agreement", "shared_karma", "Y20-O3: functional efficacy"),
    "Y20-O3": ("causal_efficacy", "functional_efficacy", "Y20-O4: dream / waking distinction"),
    "Y20-O4": ("dream_waking_distinction", "waking_as_special_dream", "Y20-O5: seed / latent continuity"),
    "Y20-O5": ("seed_continuity", "seed_continuity_model", "Y20-O6: external-object boundary"),
}
STANDARD_OBJECTION_IDS = ("Y20-O1", "Y20-O2", "Y20-O3", "Y20-O4", "Y20-O5", "Y20-O6")


def build_external_object_boundary_rule() -> dict[str, Any]:
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


def build_standard_objection(oid: str) -> dict[str, Any]:
    if oid == "Y20-O6":
        return build_external_object_boundary_rule()
    if oid not in STANDARD:
        raise ValueError(f"unknown standard objection id {oid!r}")
    otype, strategy, next_gate = STANDARD[oid]
    obj = Y20Objection(f"{oid}_obj", otype, f"external-realist objection: {otype}")
    resp = Y20Response(f"{oid}_resp", f"{oid}_obj", strategy, f"response via {strategy}", evidence=["argument_basis"])
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


def build_all_standard_objections() -> list[dict[str, Any]]:
    return [build_standard_objection(oid) for oid in STANDARD_OBJECTION_IDS]


def attach_y30_to_objection(obj: Y20Objection, app: AppearanceEvent, dep: DependentConditions, seed_trace: SeedTrace | None = None) -> dict[str, Any]:
    return {
        "objection_id": obj.objection_id,
        "objection_type": obj.objection_type,
        "y30_context": {
            "appearance": app.to_dict(),
            "dependent_conditions": dep.to_dict(),
            "dependent_conditions_status": dep.status,
            "seed_trace": seed_trace.to_dict() if seed_trace else None,
            "is_metaphysical_proof": False,
        },
        "note": "Y30 supplies cognitive structure; Y20 supplies debate structure. This does NOT prove that no external object exists.",
    }


PHYSICS_OBJECTIONS = {
    "k3_field_vs_object": ("K3.2D transport gate", ["field prediction proves object transport"]),
    "k1_order_vs_cause": ("causal / metric gate", ["temporal order proves a causal mechanism"]),
    "k2_local_vs_universal": ("transfer gate beyond the validated regime", ["a narrow validated success establishes a universal physics law"]),
}
PHYSICS_OBJECTION_IDS = tuple(PHYSICS_OBJECTIONS)


def physics_self_audit_objection(pid: str) -> dict[str, Any]:
    if pid not in PHYSICS_OBJECTIONS:
        raise ValueError(f"unknown physics objection id {pid!r}")
    gate, dns = PHYSICS_OBJECTIONS[pid]
    return {
        "id": pid,
        "required_gate": gate,
        "next_gate": gate,
        "does_not_support": list(dns),
        "resolves_claim": False,
        "note": "Y20 answers by naming the required gate, not by changing a physics verdict.",
    }


def run_all() -> int:
    n = 0

    def chk(value: bool) -> None:
        nonlocal n
        assert value
        n += 1

    try:
        Y20Objection("o", "bad", "claim")
        chk(False)
    except ValueError:
        chk(True)
    try:
        Y20Response("r", "o", "dream_analogy", "claim", evidence=[])
        chk(False)
    except ValueError:
        chk(True)
    try:
        Y20Response("r", "o", "bad", "claim", evidence=["t"])
        chk(False)
    except ValueError:
        chk(True)

    obj = Y20Objection("o1", "intersubjective_agreement", "many agree")
    resp = Y20Response("r1", "o1", "shared_karma", "common conditions", evidence=["t"])
    boundary = ExternalObjectBoundary("b1")
    record = debate_claim_record(obj, resp, boundary)
    chk(record["verdict"] == "Y20_DEBATE_STRUCTURE_OK")
    chk(record["controls"]["metaphysical_certification"] is False)
    chk("external world nonexistence is proven" in record["does_not_support"])
    chk("不证明唯心论为终极真理" in realize_response(resp))
    chk("不否定科学实在论" in realize_external_object_boundary(boundary))
    blob = " ".join(e["response"] + e["boundary"] for e in build_all_standard_objections())
    chk(all(bad not in blob for bad in BANNED_SUBSTRINGS))
    chk(len(build_all_standard_objections()) == 6)
    chk(any("不必预设" in s for s in build_external_object_boundary_rule()["may_say"]))

    app = AppearanceEvent("ap", "固定显现", evidence=["trace"])
    dep = DependentConditions("dc", "ap", conditions=["time"], evidence=["trace"])
    seed = SeedTrace("sd", "ap", "expectation", evidence=["trace"])
    bridged = attach_y30_to_objection(Y20Objection("o2", "spatiotemporal_determination", "fixed"), app, dep, seed)
    chk(bridged["y30_context"]["is_metaphysical_proof"] is False)
    chk(bridged["y30_context"]["dependent_conditions_status"] == "confirmed")

    chk(Y20_OBJECTION_TYPES == {"spatiotemporal_determination", "shared_appearance", "functional_efficacy", "dream_waking_distinction", "seed_continuity", "external_object_boundary"})
    for pid in PHYSICS_OBJECTION_IDS:
        audit = physics_self_audit_objection(pid)
        chk(audit["resolves_claim"] is False and audit["required_gate"] == audit["next_gate"])
    chk("field prediction proves object transport" in physics_self_audit_objection("k3_field_vs_object")["does_not_support"])
    try:
        physics_self_audit_objection("k9_unknown")
        chk(False)
    except ValueError:
        chk(True)
    return n


def main() -> dict[str, Any]:
    print("=== Y20-Core Objection / Response / Debate Boundary MVP ===\n")
    obj = Y20Objection("o1", "spatiotemporal_determination", "external objects explain stable place and time", evidence=["place_time_regularity"])
    resp = Y20Response("r1", "o1", "dream_analogy", "dream-like representation can have place/time without external substance", evidence=["dream_phenomenology"], confidence=0.7)
    boundary = ExternalObjectBoundary("b1")
    print("1. Objection:")
    print("   " + realize_objection(obj))
    print("\n2. Response:")
    print("   " + realize_response(resp))
    print("\n3. Boundary:")
    print("   " + realize_external_object_boundary(boundary))
    record = debate_claim_record(obj, resp, boundary)
    print("\n4. DebateClaimRecord:")
    print(f"   VERDICT: {record['verdict']}")
    print(f"   allowed_action: {record['allowed_action']}")
    print("   claim_boundary: argument structure only; no metaphysical certification")
    print("\n5. Standard objection library (O1-O6):")
    for entry in build_all_standard_objections():
        print(f"   [{entry['id']}] {entry['objection']}")
        print(f"        next: {entry['next_gate']}")
    print("\n6. Y20-style physics self-audit (objection -> required gate):")
    for pid in PHYSICS_OBJECTION_IDS:
        audit = physics_self_audit_objection(pid)
        print(f"   [{pid}] required_gate={audit['required_gate']}, resolves_claim={audit['resolves_claim']}")
    n = run_all()
    print(f"\n7. self-tests:\nok Y20-Core self-tests passed: {n}")
    return record


if __name__ == "__main__":
    main()
