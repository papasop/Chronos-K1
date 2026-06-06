import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from chronos.y20 import PHYSICS_OBJECTION_IDS, physics_self_audit_objection  # noqa: E402


class Y20PhysicsAuditTests(unittest.TestCase):
    def test_physics_self_audit_objections_name_required_gates(self):
        self.assertEqual(
            set(PHYSICS_OBJECTION_IDS),
            {"k3_field_vs_object", "k1_order_vs_cause", "k2_local_vs_universal"},
        )
        for objection_id in PHYSICS_OBJECTION_IDS:
            audit = physics_self_audit_objection(objection_id)
            self.assertFalse(audit["resolves_claim"])
            self.assertTrue(audit["required_gate"])
            self.assertEqual(audit["next_gate"], audit["required_gate"])

    def test_k3_field_prediction_does_not_prove_object_transport(self):
        audit = physics_self_audit_objection("k3_field_vs_object")
        self.assertIn("field prediction proves object transport", audit["does_not_support"])
        self.assertIn("transport gate", audit["required_gate"])

    def test_k1_temporal_order_does_not_prove_causal_mechanism(self):
        audit = physics_self_audit_objection("k1_order_vs_cause")
        self.assertIn("temporal order proves a causal mechanism", audit["does_not_support"])
        self.assertIn("causal", audit["required_gate"])

    def test_k2_local_success_does_not_prove_universal_prior(self):
        audit = physics_self_audit_objection("k2_local_vs_universal")
        self.assertIn("a narrow validated success establishes a universal physics law", audit["does_not_support"])
        self.assertIn("beyond the validated regime", audit["required_gate"])

    def test_unknown_physics_objection_raises(self):
        with self.assertRaises(ValueError):
            physics_self_audit_objection("k9_unknown")


if __name__ == "__main__":
    unittest.main()
