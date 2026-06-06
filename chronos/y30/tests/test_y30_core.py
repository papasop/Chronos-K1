import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from chronos.y30 import (  # noqa: E402
    AppearanceEvent,
    DependentConditions,
    ObjectConstructionClaim,
    SeedTrace,
    UnknownBoundary,
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


if __name__ == "__main__":
    unittest.main()
