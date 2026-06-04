import unittest

from chronos.embodied_toy.active import (
    ActiveToyExplorer,
    PendulumWorld,
    diagnostics_from_active_pendulum,
    most_novel_state_from_transitions,
    state_novelty,
)
from chronos.embodied_toy.run_active_suite import run_active_suite
from chronos.s0.diagnostics_schema import ACT_ARCHIVE, ACT_CONTINUE, ACT_DO_NOT_PROMOTE, K2_SYMPLECTIC


class EmbodiedActiveTests(unittest.TestCase):
    def test_active_suite_routes_to_k2(self):
        result = run_active_suite(quiet=True)
        self.assertEqual(result["recommendation"]["candidate_family"], K2_SYMPLECTIC)
        self.assertEqual(result["recommendation"]["allowed_action"], ACT_CONTINUE)

    def test_symplectic_diagnostic_is_measured(self):
        result = run_active_suite(quiet=True)
        measured = result["diagnostics"]["_measured"]
        self.assertLess(measured["energy_drift_symplectic"], measured["energy_drift_control"])
        self.assertIs(result["diagnostics"]["symplectic_improves_vs_controls"], True)

    def test_active_coverage_beats_random_control(self):
        result = run_active_suite(quiet=True)
        self.assertGreater(result["coverage_active"], result["coverage_random"])

    def test_novelty_sanity(self):
        self.assertEqual(state_novelty((0.0, 0.0), []), 1.0)
        self.assertEqual(state_novelty((0.0, 0.0), [(0.0, 0.0)]), 0.0)
        self.assertGreater(
            state_novelty((1.0, 0.0), [(0.0, 0.0)]),
            state_novelty((0.1, 0.0), [(0.0, 0.0)]),
        )

    def test_active_suite_is_deterministic(self):
        first = run_active_suite(quiet=True)
        second = run_active_suite(quiet=True)
        self.assertEqual(first["recommendation"], second["recommendation"])
        self.assertEqual(first["coverage_active"], second["coverage_active"])

    def test_s0_never_certifies_on_active_suite(self):
        result = run_active_suite(quiet=True)
        self.assertIn(result["recommendation"]["allowed_action"], {ACT_CONTINUE, ACT_ARCHIVE, ACT_DO_NOT_PROMOTE})

    def test_probe_uses_active_most_novel_state(self):
        world = PendulumWorld()
        explorer = ActiveToyExplorer(world)
        transitions = explorer.run(120)
        diagnostics = diagnostics_from_active_pendulum(transitions)
        expected = most_novel_state_from_transitions(transitions)
        self.assertEqual(diagnostics["_measured"]["probe_source"], "active_most_novel_state")
        self.assertEqual(diagnostics["_measured"]["probe_start"], [round(expected[0], 4), round(expected[1], 4)])
        self.assertNotEqual(diagnostics["_measured"]["probe_start"], [1.0, 0.0])

    def test_probe_can_use_active_terminal_state(self):
        world = PendulumWorld()
        explorer = ActiveToyExplorer(world)
        transitions = explorer.run(120)
        diagnostics = diagnostics_from_active_pendulum(transitions, probe_from="terminal")
        terminal = transitions[-1]["next"]
        self.assertEqual(diagnostics["_measured"]["probe_source"], "active_terminal_state")
        self.assertEqual(diagnostics["_measured"]["probe_start"], [round(terminal[0], 4), round(terminal[1], 4)])


if __name__ == "__main__":
    unittest.main()
