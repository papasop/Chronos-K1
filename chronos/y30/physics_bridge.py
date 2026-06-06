"""Y30 context bridges for K-family claims.

Y30 contextualizes physics claims. It is not physics evidence and never changes
verdict, allowed_action, evidence_level, or diagnostics.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from chronos.y30.core import (
    AppearanceEvent,
    DependentConditions,
    ObjectConstructionClaim,
    ProjectionClaim,
    SeedTrace,
    UnknownBoundary,
)

K3_2D_VERDICTS = frozenset(
    {
        "SMOKE_REFERENCE_STABLE_CANDIDATE",
        "SMOKE_PIPELINE_OK_TRANSPORT_FAIL",
        "SMOKE_REGIME_BAD_REFERENCE_TRANSPORT",
    }
)


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
        if self.seed_trace is not None and not isinstance(self.seed_trace, SeedTrace):
            raise TypeError("Y30Context.seed_trace must be a SeedTrace or None")

    def blocked_overclaims(self) -> list[str]:
        blocked = []
        for warning in self.projection_warnings:
            for item in warning.does_not_support:
                if item not in blocked:
                    blocked.append(item)
        return blocked

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
            "note": (
                "Y30 cognitive context only; does not change the physics claim's truth or verdict, "
                "and is NOT physics evidence."
            ),
        }


def attach_y30_context_full(physics_claim: dict[str, Any], context: Y30Context) -> dict[str, Any]:
    out = copy.deepcopy(physics_claim)
    out["y30_context"] = context.to_dict()
    return out


def k1_causal_overreach_warning(appearance_id: str, evidence: list[str] | None = None) -> ProjectionClaim:
    return ProjectionClaim(
        claim_id=f"{appearance_id}_causal_overreach",
        appearance_id=appearance_id,
        projected_as="a proven causal relation",
        projection_type="causal_overreach",
        evidence=evidence or ["temporal_order_without_intervention"],
        supports=["temporal ordering may invite a causal interpretation"],
        does_not_support=["a causal mechanism is proven", "a full Lorentz theory is established from this event alone"],
    )


def k1_causal_unknown_boundary(boundary_id: str = "k1_causal_unresolved") -> UnknownBoundary:
    return UnknownBoundary(
        boundary_id=boundary_id,
        unknown_type="causal_unresolved",
        missing_evidence=["intervention_evidence", "counterfactual_test"],
        next_gate="causal / metric gate",
    )


def k2_universalization_overclaim_warning(appearance_id: str, evidence: list[str] | None = None) -> ProjectionClaim:
    return ProjectionClaim(
        claim_id=f"{appearance_id}_universalization_overclaim",
        appearance_id=appearance_id,
        projected_as="a universal physics prior",
        projection_type="language_overclaim",
        evidence=evidence or ["claim_audit"],
        supports=["a narrow validated success can be over-read as a universal law"],
        does_not_support=[
            "all Hamiltonian systems are covered",
            "all physical priors help",
            "K2 generalizes beyond the validated FPU-beta regime",
            "a universal physics AI is established",
        ],
    )


def k3_identity_overreach_warning(appearance_id: str, evidence: list[str] | None = None) -> ProjectionClaim:
    return ProjectionClaim(
        claim_id=f"{appearance_id}_identity_overreach",
        appearance_id=appearance_id,
        projected_as="a transported topological object with continuous identity",
        projection_type="identity_overreach",
        evidence=evidence or ["low_field_mse_without_transport_gate"],
        supports=["low field-prediction error can be over-read as object transport"],
        does_not_support=[
            "field prediction proves object transport",
            "topological prior validation",
            "the vortex pair identity is continuous through rollout",
        ],
    )


def k3_transport_unknown_boundary(boundary_id: str = "k3_transport_unresolved") -> UnknownBoundary:
    return UnknownBoundary(
        boundary_id=boundary_id,
        unknown_type="tracker_unresolved",
        missing_evidence=["passed_transport_gate", "pair_identity_continuity"],
        next_gate="K3.2D transport gate (then FULL before any prior test)",
    )


def k1_context(event_appearance: str, conditions: list[str], evidence: list[str], confidence: float = 0.0) -> Y30Context:
    app = AppearanceEvent("k1_e1", event_appearance, sensory_mode="simulation_trace", evidence=evidence, confidence=confidence, conditions=conditions)
    dc = DependentConditions("k1_dc1", "k1_e1", conditions=conditions, evidence=evidence)
    return Y30Context(app, dc, projection_warnings=[k1_causal_overreach_warning("k1_e1")], unknown_boundaries=[k1_causal_unknown_boundary()])


def k2_context(
    trajectory_appearance: str,
    conditions: list[str],
    construction_basis: list[str],
    evidence: list[str],
    confidence: float = 0.0,
) -> Y30Context:
    app = AppearanceEvent("k2_e1", trajectory_appearance, sensory_mode="simulation_trace", evidence=evidence, confidence=confidence, conditions=conditions)
    dc = DependentConditions("k2_dc1", "k2_e1", conditions=conditions, evidence=evidence)
    oc = ObjectConstructionClaim(
        "k2_obj1",
        "k2_e1",
        "Hamiltonian phase-space dynamics",
        construction_basis=construction_basis,
        evidence=evidence,
        supports=["the observed trajectory is represented as phase-space dynamics"],
        does_not_support=["all systems are Hamiltonian", "all physical priors help"],
    )
    return Y30Context(app, dc, object_construction=oc, projection_warnings=[k2_universalization_overclaim_warning("k2_e1")])


def k3_context(
    appearance: str,
    conditions: list[str],
    evidence: list[str],
    confidence: float = 0.0,
    constructed_object: str = "vortex-antivortex pair",
    object_evidence: list[str] | None = None,
) -> Y30Context:
    app = AppearanceEvent("k3_e1", appearance, sensory_mode="field_tracker", evidence=evidence, confidence=confidence, conditions=conditions)
    dc = DependentConditions("k3_dc1", "k3_e1", conditions=conditions, evidence=evidence)
    oc = ObjectConstructionClaim(
        "k3_obj1",
        "k3_e1",
        constructed_object,
        construction_basis=["phase_winding", "pair_separation"],
        evidence=object_evidence or evidence,
        supports=["the tracked pair is represented as a topological object"],
        does_not_support=["the object has independent self-nature", "field prediction proves object transport"],
    )
    return Y30Context(
        app,
        dc,
        object_construction=oc,
        projection_warnings=[k3_identity_overreach_warning("k3_e1")],
        unknown_boundaries=[k3_transport_unknown_boundary()],
    )


def explain_k3_2d_verdict(verdict: str) -> dict[str, Any]:
    if verdict not in K3_2D_VERDICTS:
        raise ValueError(f"unknown K3.2D verdict {verdict!r}; expected one of {sorted(K3_2D_VERDICTS)}")
    common = [
        "a topological prior was tested",
        "a topological prior failed",
        "field similarity proves object identity continuity",
        "FULL transport is validated",
    ]
    if verdict == "SMOKE_REFERENCE_STABLE_CANDIDATE":
        return {
            "verdict": verdict,
            "utterance": "参考对显现稳定可追踪：只允许进入 baseline-only 训练，不允许 prior test，也不进入 FULL。",
            "supports": ["the reference vortex-antivortex pair is stably trackable", "baseline-only pipeline testing may proceed"],
            "does_not_support": common + ["the regime is physically rich"],
            "allows": {"baseline_training": True, "prior_test": False, "full": False},
            "claim_boundary": "reference stability is a regime-gate precondition only; it unlocks baseline testing, never a prior test.",
        }
    if verdict == "SMOKE_PIPELINE_OK_TRANSPORT_FAIL":
        return {
            "verdict": verdict,
            "utterance": "场预测可以学（beat persistence），但对象输运失败：显现相似 ≠ 对象身份连续成立。这是 topology-object 学习缺口，不是 topology prior 失败（没有测过 prior）。",
            "supports": ["field prediction is learnable and beats persistence", "this is a genuine topology-object learning gap"],
            "does_not_support": common + ["the baseline transports the object as a topological object"],
            "allows": {"baseline_training": True, "prior_test": False, "full": False},
            "claim_boundary": "field prediction OK with object transport FAIL is a learning gap, not a prior result; archive as REGIME_UNRESOLVED.",
        }
    return {
        "verdict": verdict,
        "utterance": "参考对显现本身不能稳定被追踪：失败在 reference regime 层，不是模型机制失败，也不是 prior 结果。先修 regime（见 0c）。",
        "supports": ["the failure is localized to the reference regime layer"],
        "does_not_support": common + ["the model mechanism failed", "the reference regime is usable as-is"],
        "allows": {"baseline_training": False, "prior_test": False, "full": False},
        "claim_boundary": "a bad reference regime blocks baseline testing; fix the regime before any judgement about the model or a prior.",
    }


def k3_2d_seed_trace_from_verdict(verdict: str, source_event_id: str = "k3_2d_run") -> SeedTrace:
    explain_k3_2d_verdict(verdict)
    tendencies = {
        "SMOKE_REFERENCE_STABLE_CANDIDATE": "prefer baseline-only testing in this regime; do not jump to prior",
        "SMOKE_PIPELINE_OK_TRANSPORT_FAIL": "expect a topology-object gap here; field MSE is not enough",
        "SMOKE_REGIME_BAD_REFERENCE_TRANSPORT": "fix the reference regime before trusting any model result here",
    }
    return SeedTrace(
        seed_id=f"{source_event_id}_{verdict.lower()}",
        source_event_id=source_event_id,
        stored_tendency=tendencies[verdict],
        activation_conditions=["similar 2D vortex transport regime"],
        evidence=[f"k3_2d_verdict:{verdict}"],
        does_not_support=["this tendency is a universal physical law", "a topological prior was tested or failed"],
    )
