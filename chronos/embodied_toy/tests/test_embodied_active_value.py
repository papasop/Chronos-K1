import unittest

from chronos.embodied_toy.active_value import (
    ZONE,
    ActiveRailExplorer,
    RailWorld,
    probe_energy_drifts,
    random_rail_run,
)
from chronos.embodied_toy.run_active_value_suite import run_value_suite
from chronos.s0.diagnostics_schema import ACT_ARCHIVE, ACT_CONTINUE, ACT_DO_NOT_PROMOTE, K2_SYMPLECTIC


class EmbodiedActiveValueTests(unittest.TestCase):
    def test_active_reaches_structure_zone(self):
        active_reached = max(ActiveRailExplorer(RailWorld()).run(200))
        self.assertGreaterEqual(active_reached, ZONE)

    def test_random_rarely_reaches_structure_zone(self):
        random_far = sum(max(random_rail_run(RailWorld(), 200, seed=seed)) >= ZONE for seed in range(20))
        self.assertLessEqual(random_far, 2)

    def test_active_gets_k2_random_does_not(self):
        result = run_value_suite(quiet=True)
        self.assertEqual(result["active"]["recommendation"]["candidate_family"], K2_SYMPLECTIC)
        self.assertEqual(result["active"]["recommendation"]["allowed_action"], ACT_CONTINUE)
        self.assertIs(result["active"]["in_zone"], True)
        self.assertNotEqual(result["random"]["recommendation"]["candidate_family"], K2_SYMPLECTIC)
        self.assertIs(result["random"]["in_zone"], False)

    def test_partitioned_probe_has_real_value_difference(self):
        drift_symplectic_zone, drift_control_zone = probe_energy_drifts(True, n_steps=200)
        drift_symplectic_near, drift_control_near = probe_energy_drifts(False, n_steps=200)
        self.assertLess(drift_symplectic_zone, 0.5 * drift_control_zone)
        self.assertFalse(drift_symplectic_near < 0.5 * drift_control_near)

    def test_value_suite_is_deterministic(self):
        first = run_value_suite(quiet=True)
        second = run_value_suite(quiet=True)
        self.assertEqual(first["active"]["recommendation"], second["active"]["recommendation"])
        self.assertEqual(first["random"]["recommendation"], second["random"]["recommendation"])

    def test_s0_never_certifies_on_value_suite(self):
        result = run_value_suite(quiet=True)
        allowed = {ACT_CONTINUE, ACT_ARCHIVE, ACT_DO_NOT_PROMOTE}
        self.assertIn(result["active"]["recommendation"]["allowed_action"], allowed)
        self.assertIn(result["random"]["recommendation"]["allowed_action"], allowed)


if __name__ == "__main__":
    unittest.main()
