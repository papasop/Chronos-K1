"""Y30 bounded utterance realizers."""

from __future__ import annotations

from chronos.y30.core import (
    BANNED_SUBSTRINGS,
    AppearanceEvent,
    DependentConditions,
    ObjectConstructionClaim,
    ProjectionClaim,
    SeedTrace,
    SelfGraspingSignal,
    ThreeNatureAnalysis,
    UnknownBoundary,
)
from chronos.y30.stack import (
    AlayaSeedStore,
    ConsciousnessFlow,
    ManasAttachment,
    ManoConstruction,
    SenseConsciousnessEvent,
)

PROJECTION_PHRASE = {
    "independent_object": "把依条件显现当作独立对象",
    "fixed_self": "把过程执为固定的自我",
    "causal_overreach": "把相关误读为确定因果",
    "identity_overreach": "把相似误读为同一身份",
    "language_overclaim": "把命名当作对象的自性",
}

UNKNOWN_PHRASE = {
    "missing_evidence": "缺少证据",
    "ambiguous_reference": "指代不明确",
    "bad_regime": "当前 regime 不适合判断",
    "tracker_unresolved": "追踪器无法可靠检测",
    "causal_unresolved": "缺少因果证据",
    "metaphysical_overreach": "该问题超出结构分析、属于形而上学过度断言",
    "out_of_scope": "超出当前模块范围",
}

SENSE_NAME = {"eye": "视觉", "ear": "听觉", "nose": "嗅觉", "tongue": "味觉", "body": "触觉"}
ATTACH_PHRASE = {
    "mine": "「我的」",
    "me": "「我」",
    "observer": "「我在知道」",
    "controller": "「我在控制」",
    "identity": "「同一个我」",
}


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
        text = f"这些是候选依赖条件，尚未确认：{'、'.join(dc.conditions)}。"
        if dc.unresolved_conditions:
            text += f"（仍有未解析条件：{'、'.join(dc.unresolved_conditions)}。）"
        return _assert_clean(text)
    text = f"这个显现依赖这些条件：{'、'.join(dc.conditions)}。"
    if dc.unresolved_conditions:
        text += f"（仍有未解析条件：{'、'.join(dc.unresolved_conditions)}。）"
    return _assert_clean(text)


def realize_object_construction(claim: ObjectConstructionClaim) -> str:
    return _assert_clean(
        f"系统把这个显现构造成对象：{claim.constructed_object}。"
        "这个说法不支持该对象具有独立自性。"
    )


def realize_projection(claim: ProjectionClaim) -> str:
    phrase = PROJECTION_PHRASE.get(claim.projection_type, "出现了投射")
    return _assert_clean(
        f"这里出现了投射：{phrase}。这个判断不对外部世界的存废下断言，也不支持对象具有独立自性。"
    )


def realize_self_grasping(signal: SelfGraspingSignal) -> str:
    return _assert_clean(
        f"这里出现了把{signal.target_process}执为「{signal.grasped_as}」的结构。"
        "这个判断不支持存在固定不变的实体自我。"
    )


def realize_seed_trace(seed: SeedTrace) -> str:
    text = f"这个经验留下了后续判断倾向：{seed.stored_tendency}。"
    if seed.activation_conditions:
        text += f"（在 {'、'.join(seed.activation_conditions)} 条件下可能被激活。）"
    text += "这个说法只是记忆痕迹结构,不证明一个形而上的藏识。"
    return _assert_clean(text)


def realize_three_nature(analysis: ThreeNatureAnalysis) -> str:
    return _assert_clean(
        f"这个对象被想象为{analysis.imagined_projection}；"
        f"但它依赖{'、'.join(analysis.dependent_conditions)}这些条件。"
        "当前结构分析不支持独立自性断言。"
    )


def realize_unknown_boundary(boundary: UnknownBoundary) -> str:
    reason = UNKNOWN_PHRASE.get(boundary.unknown_type, "证据不足")
    text = f"我不能确定，因为{reason}。"
    if boundary.next_gate:
        text += f"（下一步：{boundary.next_gate}。）"
    return _assert_clean(text)


def realize_sense_consciousness(event: SenseConsciousnessEvent) -> str:
    name = SENSE_NAME.get(event.sense_base, event.sense_base)
    return _assert_clean(
        f"{name}识层记录到一个显现：{event.raw_feature}。"
        f"这只支持{name}通道中有显现，不支持对象已被确认。"
    )


def realize_mano_construction(claim: ManoConstruction) -> str:
    return _assert_clean(
        f"第六识层把{len(claim.input_appearances)}个显现组合成对象：{claim.constructed_object}。"
        "这个说法不支持该对象具有独立自性。"
    )


def realize_manas_attachment(attachment: ManasAttachment) -> str:
    phrase = ATTACH_PHRASE.get(attachment.attachment_type, "「我」")
    return _assert_clean(f"第七识层出现了{phrase}的归属结构。这个判断不支持存在固定不变的实体自我。")


def realize_alaya_store(store: AlayaSeedStore) -> str:
    return _assert_clean(
        f"第八识层作为记忆/习气痕迹类比，记录了{len(store.seeds)}条背景倾向。"
        "这只是记忆/习气痕迹的类比，不证明阿赖耶识是形上实体、宇宙意识或灵魂。"
    )


def realize_consciousness_flow(flow: ConsciousnessFlow) -> str:
    chain = flow.chain_summary()
    if not chain:
        return _assert_clean("这条认知链尚未记录任何层。")
    return _assert_clean("这条认知链按功能层记录为:" + " -> ".join(chain) + "。这是功能性认知层的记录,任何一层都不是形上实体。")
