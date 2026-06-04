import unittest

from chronos.embodied_toy.run_sim_suite import run_sim_suite
from chronos.s0.diagnostics_schema import (
    ACT_ARCHIVE,
    ACT_CONTINUE,
    ACT_DO_NOT_PROMOTE,
    K1_LORENTZ,
    K2_SYMPLECTIC,
    K3_TOPOLOGICAL,
)


class EmbodiedSimTests(unittest.TestCase):
    def test_sim_suite_expected_language_choices(self):
        results = run_sim_suite(quiet=True)

        self.assertEqual(results["pendulum"]["recommendation"]["candidate_family"], K2_SYMPLECTIC)
        self.assertEqual(results["pendulum"]["recommendation"]["allowed_action"], ACT_CONTINUE)

        self.assertEqual(results["contact"]["recommendation"]["candidate_family"], K1_LORENTZ)
        self.assertEqual(results["contact"]["recommendation"]["allowed_action"], ACT_CONTINUE)

        self.assertEqual(results["object_persistence"]["recommendation"]["candidate_family"], K3_TOPOLOGICAL)
        self.assertEqual(results["object_persistence"]["recommendation"]["allowed_action"], ACT_DO_NOT_PROMOTE)

    def test_s0_never_certifies_on_sim_suite(self):
        results = run_sim_suite(quiet=True)
        allowed = {ACT_CONTINUE, ACT_ARCHIVE, ACT_DO_NOT_PROMOTE}
        for name, result in results.items():
            with self.subTest(name=name):
                self.assertIn(result["recommendation"]["allowed_action"], allowed)

    def test_deterministic_recommendations(self):
        first = run_sim_suite(quiet=True)
        second = run_sim_suite(quiet=True)
        for name in first:
            with self.subTest(name=name):
                self.assertEqual(first[name]["recommendation"], second[name]["recommendation"])

    def test_pendulum_energy_drift_signal_is_measured(self):
        diagnostics = run_sim_suite(quiet=True)["pendulum"]["diagnostics"]
        measured = diagnostics["_measured"]
        self.assertIs(diagnostics["symplectic_improves_vs_controls"], True)
        self.assertLess(measured["energy_drift_symplectic"], measured["energy_drift_control"])

    def test_object_persistence_loss_is_measured(self):
        diagnostics = run_sim_suite(quiet=True)["object_persistence"]["diagnostics"]
        self.assertIs(diagnostics["object_tracking_valid"], False)
        self.assertLess(diagnostics["_measured"]["id_present_fraction"], 1.0)

    def test_contact_is_labeled_proxy(self):
        diagnostics = run_sim_suite(quiet=True)["contact"]["diagnostics"]
        self.assertIs(diagnostics["_measured"]["is_proxy"], True)


if __name__ == "__main__":
    unittest.main()
