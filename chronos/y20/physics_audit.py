"""Y20-style objections applied to K-family physics claims."""

from __future__ import annotations

from typing import Any

PHYSICS_OBJECTIONS = {
    "k3_field_vs_object": {
        "objection": "如果 K3 只是 field prediction,为什么能说存在一个被输运的 topological object?",
        "response": "不能。必须通过 object transport gate;low field MSE 不支持对象身份连续。",
        "boundary": "low field MSE 不支持 object identity;未过 transport gate 不允许 prior test。",
        "required_gate": "K3.2D transport gate",
        "does_not_support": [
            "field prediction proves object transport",
            "a topological object exists merely because field MSE is low",
            "a topological prior was tested or validated",
        ],
    },
    "k1_order_vs_cause": {
        "objection": "如果 K1 只看到时间先后,为什么能说存在因果机制?",
        "response": "不能。必须有 intervention / 反事实证据;时间先后不支持因果机制已成立。",
        "boundary": "temporal order 不支持 proven causation;需要 causal / metric gate。",
        "required_gate": "causal / metric gate",
        "does_not_support": [
            "temporal order proves a causal mechanism",
            "a full Lorentz theory is established from order alone",
        ],
    },
    "k2_local_vs_universal": {
        "objection": "如果 K2 在 FPU-beta 上成功,为什么能说所有物理都该用 symplectic prior?",
        "response": "不能。成功被限定在已验证 regime;局部成功不支持普遍规律。",
        "boundary": "validated-regime success 不支持 universal law;不得外推到未验证 regime。",
        "required_gate": "transfer gate beyond the validated regime",
        "does_not_support": [
            "a narrow validated success establishes a universal physics law",
            "K2 generalizes beyond the validated FPU-beta regime",
            "a universal physics AI is established",
        ],
    },
}

PHYSICS_OBJECTION_IDS = tuple(PHYSICS_OBJECTIONS)


def physics_self_audit_objection(objection_id: str) -> dict[str, Any]:
    """Return a bounded physics self-audit objection.

    The answer names the required gate. It never resolves or upgrades the
    underlying K-family physics claim.
    """

    if objection_id not in PHYSICS_OBJECTIONS:
        raise ValueError(f"unknown physics objection id {objection_id!r}; expected one of {list(PHYSICS_OBJECTION_IDS)}")
    entry = PHYSICS_OBJECTIONS[objection_id]
    return {
        "id": objection_id,
        "objection": entry["objection"],
        "response": entry["response"],
        "boundary": entry["boundary"],
        "required_gate": entry["required_gate"],
        "does_not_support": list(entry["does_not_support"]),
        "resolves_claim": False,
        "note": (
            "Y20-style self-audit: the objection is answered by naming the required gate, not by "
            "changing the physics verdict. Y20 supplies debate structure; the K-family + VPSL "
            "gates decide the physics."
        ),
        "next_gate": entry["required_gate"],
    }
