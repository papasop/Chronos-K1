"""Template realizers for Y20 debate structures."""

from __future__ import annotations

from chronos.y20.core import ExternalObjectBoundary, Y20Objection, Y20Response
from chronos.y30.core import BANNED_SUBSTRINGS

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


def realize_objection(objection: Y20Objection) -> str:
    phrase = OBJECTION_PHRASE.get(objection.objection_type, objection.realist_claim)
    return _assert_clean(f"外境实有论者诘难：{phrase}")


def realize_response(response: Y20Response) -> str:
    if not response.evidence:
        return _assert_clean("我不能确定这个回应是否成立。")
    phrase = RESPONSE_PHRASE.get(response.strategy, response.response_claim)
    return _assert_clean(
        f"唯识回应：{phrase}。这是论证结构上的回应，不证明外境的存废，也不证明唯心论为终极真理。"
    )


def realize_external_object_boundary(boundary: ExternalObjectBoundary) -> str:
    return _assert_clean(
        "边界说明：显现不需要预设一个独立的外部对象；"
        "但这只是论证结构，不否定科学实在论，也不断言外境的存废。"
    )
