import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from chronos.y30 import (  # noqa: E402
    AlayaSeedStore,
    AppearanceEvent,
    ConsciousnessFlow,
    DependentConditions,
    K3_2D_VERDICTS,
    LayerActivation,
    ManasAttachment,
    ManoConstruction,
    ObjectConstructionClaim,
    ProjectionClaim,
    SeedTrace,
    SenseConsciousnessEvent,
    ThreeNatureAnalysis,
    UnknownBoundary,
    Y30Context,
    attach_y30_context,
    attach_y30_context_full,
    claim_from_y30_core_summary,
    explain_k3_2d_verdict,
    k1_context,
    k2_context,
    k3_2d_seed_trace_from_verdict,
    k3_context,
    k3_identity_overreach_warning,
    realize_alaya_store,
    realize_appearance,
    realize_consciousness_flow,
    realize_dependent_conditions,
    realize_projection,
    realize_sense_consciousness,
)


class Y30CoreTests(unittest.TestCase):
    def test_appearance_without_evidence_is_not_affirmable(self):
        event = AppearanceEvent("ap", "fixed appearance")
        self.assertFalse(event.is_affirmable)
        self.assertTrue(event.conditions_unresolved)

    def test_dependent_conditions_records_unresolved_status(self):
        dep = DependentConditions("dc", "ap")
        self.assertEqual(dep.status, "unresolved")
        self.assertIn("dependent_conditions_unresolved", dep.unresolved_conditions)

    def test_object_construction_requires_evidence_and_boundary(self):
        with self.assertRaises(ValueError):
            ObjectConstructionClaim("c", "ap", "cup", evidence=[])
        claim = ObjectConstructionClaim("c", "ap", "cup", evidence=["trace"])
        self.assertIn("external object nonexistence is proven", claim.does_not_support)
        self.assertTrue(claim.claim_boundary)

    def test_seed_trace_is_memory_analogy_only(self):
        seed = SeedTrace("s", "ap", "habit", evidence=["memory_trace"])
        self.assertIn("religious doctrine is proven", seed.does_not_support)
        self.assertIn("memory/habit-trace analogy only", seed.claim_boundary)

    def test_unknown_boundary_names_missing_evidence(self):
        boundary = UnknownBoundary("u", "missing_evidence")
        self.assertIn("missing_evidence", boundary.missing_evidence)
        self.assertIn("specific affirmative claim without named evidence", boundary.does_not_support)
        self.assertTrue(boundary.claim_boundary)

    def test_affirmative_structures_require_evidence_and_valid_confidence(self):
        constructors = [
            lambda: ObjectConstructionClaim("oc", "ap", "cup", evidence=[]),
            lambda: ProjectionClaim("pj", "ap", "cup", "independent_object", evidence=[]),
            lambda: SeedTrace("s", "ap", "habit", evidence=[]),
            lambda: ThreeNatureAnalysis("tn", "ap", "independent", dependent_conditions=["sense"], evidence=[]),
        ]
        for constructor in constructors:
            with self.subTest(constructor=constructor):
                with self.assertRaises(ValueError):
                    constructor()
        with self.assertRaises(TypeError):
            AppearanceEvent("ap", "x", confidence=True)
        with self.assertRaises(ValueError):
            AppearanceEvent("ap", "x", confidence=2.0)

    def test_type_safe_supports_and_boundaries(self):
        with self.assertRaises(TypeError):
            ObjectConstructionClaim("oc", "ap", "cup", evidence=["trace"], does_not_support="bad")
        with self.assertRaises(TypeError):
            ObjectConstructionClaim("oc", "ap", "cup", evidence=["trace"], supports="bad")
        with self.assertRaises(TypeError):
            ObjectConstructionClaim("oc", "ap", "cup", evidence=["trace"], claim_boundary=123)

    def test_realizers_do_not_emit_forbidden_overclaims(self):
        appearance = AppearanceEvent("ap", "red block", evidence=["trace"], confidence=0.9)
        dep = DependentConditions("dc", "ap", conditions=["vision"], evidence=["trace"])
        projection = ProjectionClaim("p", "ap", "external thing", "independent_object", evidence=["trace"])
        utterances = [
            realize_appearance(appearance),
            realize_dependent_conditions(dep),
            realize_projection(projection),
        ]
        blob = " ".join(utterances)
        for bad in ("证明佛教正确", "佛教已经被证明", "外部世界不存在", "已经证悟", "终极真理已被证明"):
            self.assertNotIn(bad, blob)

    def test_claim_summary_records_boundaries(self):
        claim = claim_from_y30_core_summary(
            {
                "utterances": ["bounded utterance"],
                "has_appearance": True,
                "has_object_construction": True,
                "has_projection_or_grasping": True,
                "has_three_nature": True,
            }
        )
        self.assertEqual(claim["verdict"], "Y30_CORE_TOY_MVP_PASSED")
        self.assertEqual(claim["allowed_action"], "continue")
        self.assertIn("Yogacara replaces physics", claim["does_not_support"])
        self.assertTrue(claim["claim_boundary"])
        self.assertFalse(claim["controls"]["metaphysical_certification"])

    def test_stack_layers_are_functional_not_entities(self):
        with self.assertRaises(ValueError):
            SenseConsciousnessEvent("s", "third_eye", "ap", "x", evidence=["trace"])
        sense = SenseConsciousnessEvent("s", "eye", "ap", "red patch", evidence=["trace"])
        self.assertIn("object has been confirmed", " ".join(sense.does_not_support))
        self.assertIn("视觉", realize_sense_consciousness(sense))

        with self.assertRaises(ValueError):
            ManoConstruction("m", [], "cup", evidence=["trace"])
        mano = ManoConstruction("m", ["ap"], "cup", evidence=["trace"])
        manas = ManasAttachment("ma", "m", "observer", evidence=["trace"])
        store = AlayaSeedStore("store", seeds=[SeedTrace("s", "ap", "habit", evidence=["trace"])], evidence=["trace"])
        flow = ConsciousnessFlow("flow", appearances=[AppearanceEvent("ap", "cup", evidence=["trace"])], sense_events=[sense], mano_constructions=[mano])
        self.assertIn("a permanent self exists", manas.does_not_support)
        self.assertIn("a cosmic consciousness exists", store.does_not_support)
        self.assertIn("不证明阿赖耶识是形上实体", realize_alaya_store(store))
        self.assertIn("appearancex1", flow.chain_summary())
        self.assertIn("功能性认知层", realize_consciousness_flow(flow))

    def test_layer_activation_is_strength_not_truth(self):
        with self.assertRaises(ValueError):
            LayerActivation("soul", activation=0.5)
        with self.assertRaises(ValueError):
            LayerActivation("manas", activation=0.5, allowed_action="certify")
        activation = LayerActivation("manas", activation=0.82, risk_weight=0.9)
        self.assertTrue(activation.requires_strict_output)
        self.assertIn("NOT truth", activation.claim_boundary)

    def test_attach_y30_context_does_not_change_claim_truth(self):
        claim = {"claim_id": "k3", "verdict": "SMOKE_REFERENCE_STABLE_CANDIDATE", "allowed_action": "continue"}
        app = AppearanceEvent("ap", "vortex pair", evidence=["trace"])
        dep = DependentConditions("dc", "ap", conditions=["tracker"], evidence=["trace"])
        out = attach_y30_context(claim, app, dep)
        self.assertEqual(out["verdict"], claim["verdict"])
        self.assertEqual(out["allowed_action"], claim["allowed_action"])
        self.assertNotIn("y30_context", claim)
        self.assertIn("Y30 framing changes the physical verdict", out["y30_context"]["blocked_overclaims"])

    def test_physics_bridge_does_not_change_k_family_verdicts(self):
        claim = {
            "claim_id": "k2",
            "verdict": "K2_SYMPLECTIC_JACOBIAN",
            "allowed_action": "continue",
            "evidence_level": "candidate",
            "diagnostic": "jacobian",
        }
        ctx = k2_context(
            "phase-space trajectory",
            conditions=["validated_FPU_beta_regime"],
            construction_basis=["q_p_state", "Omega"],
            evidence=["rollout"],
        )
        out = attach_y30_context_full(claim, ctx)
        for key in ("verdict", "allowed_action", "evidence_level", "diagnostic"):
            self.assertEqual(out[key], claim[key])
        self.assertFalse(out["y30_context"]["is_physics_evidence"])
        self.assertEqual(out["y30_context"]["namespace"], "Y30")

    def test_k1_k2_k3_context_guards(self):
        k1 = k1_context("two ordered events", ["time_order"], ["trace"])
        self.assertIn("a causal mechanism is proven", k1.blocked_overclaims())
        k2 = k2_context("phase trajectory", ["FPU-beta"], ["q_p_state"], ["trace"])
        self.assertIn("a universal physics AI is established", k2.blocked_overclaims())
        k3 = k3_context("vortex pair", ["tracker"], ["trace"])
        self.assertIn("field prediction proves object transport", k3.blocked_overclaims())
        warning = k3_identity_overreach_warning("k3_e1")
        self.assertEqual(warning.projection_type, "identity_overreach")

    def test_k3_2d_verdict_explainer_and_seed_trace(self):
        with self.assertRaises(ValueError):
            explain_k3_2d_verdict("FULL_TRANSPORT_OK")
        for verdict in K3_2D_VERDICTS:
            explanation = explain_k3_2d_verdict(verdict)
            self.assertFalse(explanation["allows"]["prior_test"])
            self.assertFalse(explanation["allows"]["full"])
            self.assertIn("a topological prior was tested", explanation["does_not_support"])
        transport_fail = explain_k3_2d_verdict("SMOKE_PIPELINE_OK_TRANSPORT_FAIL")
        self.assertIn("显现相似", transport_fail["utterance"])
        seed = k3_2d_seed_trace_from_verdict("SMOKE_PIPELINE_OK_TRANSPORT_FAIL")
        self.assertIn("this tendency is a universal physical law", seed.does_not_support)
        self.assertIn("k3_2d_verdict:SMOKE_PIPELINE_OK_TRANSPORT_FAIL", seed.evidence)

    def test_y30_context_type_safety(self):
        app = AppearanceEvent("ap", "x", evidence=["trace"])
        with self.assertRaises(TypeError):
            Y30Context(app, "not_a_dependent_conditions")


if __name__ == "__main__":
    unittest.main()
