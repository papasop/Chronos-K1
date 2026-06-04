import unittest

from chronos.embodied_toy.run_toy_suite import run_toy_suite
from chronos.s0.diagnostics_schema import (
    ACT_ARCHIVE,
    ACT_CONTINUE,
    ACT_DO_NOT_PROMOTE,
    K1_LORENTZ,
    K2_SYMPLECTIC,
    K3_TOPOLOGICAL,
    UNRESOLVED,
)


class EmbodiedToyTests(unittest.TestCase):
    def test_toy_suite_expected_language_choices(self):
        results = run_toy_suite(quiet=True)

        self.assertEqual(results["pendulum"]["candidate_family"], K2_SYMPLECTIC)
        self.assertEqual(results["pendulum"]["allowed_action"], ACT_CONTINUE)

        self.assertEqual(results["causal_contact"]["candidate_family"], K1_LORENTZ)
        self.assertEqual(results["causal_contact"]["allowed_action"], ACT_CONTINUE)

        self.assertEqual(results["vortex_fail"]["candidate_family"], K3_TOPOLOGICAL)
        self.assertEqual(results["vortex_fail"]["allowed_action"], ACT_DO_NOT_PROMOTE)

        self.assertEqual(results["unknown"]["candidate_family"], UNRESOLVED)
        self.assertEqual(results["unknown"]["allowed_action"], ACT_DO_NOT_PROMOTE)

    def test_s0_never_certifies_on_toy_suite(self):
        results = run_toy_suite(quiet=True)
        allowed = {ACT_CONTINUE, ACT_ARCHIVE, ACT_DO_NOT_PROMOTE}
        for name, recommendation in results.items():
            with self.subTest(name=name):
                self.assertIn(recommendation["allowed_action"], allowed)


if __name__ == "__main__":
    unittest.main()
