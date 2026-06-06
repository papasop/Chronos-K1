import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from chronos.y20 import (  # noqa: E402
    STANDARD_OBJECTION_IDS,
    Y20_OBJECTION_TYPES,
    ExternalObjectBoundary,
    Y20Objection,
    Y20Response,
    attach_y30_to_objection,
    build_all_standard_objections,
    build_external_object_boundary_rule,
    build_standard_objection,
    debate_claim_record,
    realize_external_object_boundary,
    realize_objection,
    realize_response,
)
from chronos.y30 import AppearanceEvent, DependentConditions, SeedTrace  # noqa: E402

FORBIDDEN = [
    "证明佛教正确",
    "佛教已经被证明",
    "外部世界不存在",
    "已经证悟",
    "终极真理已被证明",
    "CERTIFIED",
    "PROVED",
]


class Y20CoreTests(unittest.TestCase):
    def test_objection_type_validated(self):
        with self.assertRaises(ValueError):
            Y20Objection("o", "not_a_type", "claim")
        obj = Y20Objection("o", "causal_efficacy", "objects have real effects")
        self.assertIn("external objects are proven to exist independently", obj.does_not_support)

    def test_response_requires_evidence_and_valid_strategy(self):
        with self.assertRaises(ValueError):
            Y20Response("r", "o", "dream_analogy", "claim", evidence=[])
        with self.assertRaises(ValueError):
            Y20Response("r", "o", "not_a_strategy", "claim", evidence=["t"])

    def test_response_blocks_metaphysical_proof(self):
        response = Y20Response("r", "o", "dream_analogy", "dream-like representation", evidence=["t"])
        for item in (
            "external world nonexistence is proven",
            "metaphysical idealism is proven",
            "mind-only is established as ultimate truth",
            "the response is a metaphysical proof",
        ):
            self.assertIn(item, response.does_not_support)
        utterance = realize_response(response)
        self.assertIn("不证明外境的存废", utterance)
        self.assertIn("不证明唯心论为终极真理", utterance)
        for bad in FORBIDDEN:
            self.assertNotIn(bad, utterance)

    def test_external_object_boundary(self):
        boundary = ExternalObjectBoundary("b")
        self.assertIn("scientific realism is refuted", boundary.does_not_support)
        utterance = realize_external_object_boundary(boundary)
        self.assertIn("不否定科学实在论", utterance)
        self.assertIn("不断言外境的存废", utterance)

    def test_debate_claim_record_boundaries(self):
        obj = Y20Objection("o1", "intersubjective_agreement", "many observers agree")
        resp = Y20Response("r1", "o1", "shared_karma", "common karmic conditioning", evidence=["t"])
        boundary = ExternalObjectBoundary("b1")
        record = debate_claim_record(obj, resp, boundary)
        self.assertEqual(record["verdict"], "Y20_DEBATE_STRUCTURE_OK")
        self.assertEqual(record["allowed_action"], "continue")
        self.assertFalse(record["controls"]["metaphysical_certification"])
        self.assertIn("external world nonexistence is proven", record["does_not_support"])
        self.assertTrue(record["claim_boundary"])
        with self.assertRaises(ValueError):
            debate_claim_record(obj, Y20Response("r2", "other", "shared_karma", "x", evidence=["t"]), boundary)

    def test_no_forbidden_phrases(self):
        objections = [
            Y20Objection(f"o{i}", kind, "claim")
            for i, kind in enumerate(
                [
                    "spatiotemporal_determination",
                    "intersubjective_agreement",
                    "causal_efficacy",
                    "no_object_no_cognition",
                ]
            )
        ]
        responses = [
            Y20Response(f"r{i}", f"o{i}", strategy, "claim", evidence=["t"])
            for i, strategy in enumerate(
                ["dream_analogy", "shared_karma", "functional_efficacy", "dependent_designation"]
            )
        ]
        blob = " ".join(
            [realize_objection(o) for o in objections]
            + [realize_response(r) for r in responses]
            + [realize_external_object_boundary(ExternalObjectBoundary("b"))]
        )
        for bad in FORBIDDEN:
            self.assertNotIn(bad, blob)

    def test_standard_objection_library(self):
        self.assertEqual(STANDARD_OBJECTION_IDS, ("Y20-O1", "Y20-O2", "Y20-O3", "Y20-O4", "Y20-O5", "Y20-O6"))
        entries = build_all_standard_objections()
        self.assertEqual(len(entries), 6)
        for entry in entries:
            for key in ("id", "objection", "boundary", "does_not_support", "next_gate"):
                self.assertIn(key, entry)
                self.assertTrue(entry[key])
            self.assertIn("external world nonexistence is proven", entry["does_not_support"])
        with self.assertRaises(ValueError):
            build_standard_objection("Y20-O99")

    def test_o6_may_say_vs_may_not_say(self):
        o6 = build_external_object_boundary_rule()
        self.assertTrue(any("不必预设" in item for item in o6["may_say"]))
        self.assertTrue(any("外境不存在已被证明" in item for item in o6["may_not_say"]))
        self.assertTrue(any("唯心论已被证明" in item for item in o6["may_not_say"]))

    def test_y20_to_y30_bridge_not_metaphysical_proof(self):
        obj = Y20Objection("o1", "spatiotemporal_determination", "fixed place/time")
        appearance = AppearanceEvent("ap", "固定显现", evidence=["trace"])
        dep = DependentConditions("dc", "ap", conditions=["time", "coordinate"], evidence=["trace"])
        seed = SeedTrace("sd", "ap", "expectation", evidence=["trace"])
        bridged = attach_y30_to_objection(obj, appearance, dep, seed_trace=seed)
        self.assertFalse(bridged["y30_context"]["is_metaphysical_proof"])
        self.assertEqual(bridged["y30_context"]["dependent_conditions_status"], "confirmed")
        self.assertIn("does NOT", bridged["note"])

    def test_canonical_objection_taxonomy(self):
        self.assertEqual(
            Y20_OBJECTION_TYPES,
            {
                "spatiotemporal_determination",
                "shared_appearance",
                "functional_efficacy",
                "dream_waking_distinction",
                "seed_continuity",
                "external_object_boundary",
            },
        )


if __name__ == "__main__":
    unittest.main()
