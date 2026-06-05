import os
import tempfile
import unittest

import numpy as np

from chronos.k3.active_gp_search import (
    BOX_L,
    DX,
    SCORE_MARGIN,
    START,
    TRANSPORT_OK_THRESHOLD,
    VortexRegime,
    apply_action,
    diagnostics_from_vortex_search,
    evaluate_gp_vortex_regime,
    find_vortices,
    passed_admission,
    seed_vortex_pair,
    write_memory_event,
)
from chronos.k3.run_active_gp_search import run_search_suite
from chronos.memory import load_events
from chronos.s0.structure_selector import recommend


class ActiveGpSearchTests(unittest.TestCase):
    def test_evaluator_keys_and_score_range(self):
        metrics = evaluate_gp_vortex_regime(START)
        for key in ("pair_intact", "topology_score", "transport_score", "mean_pos_err", "transport_ok"):
            self.assertIn(key, metrics)
        self.assertGreaterEqual(metrics["transport_score"], 0.0)
        self.assertLessEqual(metrics["transport_score"], 1.0)
        self.assertEqual(metrics["mean_pos_err_units"], "physical coordinate units")

    def test_plaquette_vortex_coordinates_use_centers(self):
        cores = find_vortices(seed_vortex_pair(pair_sep=6.0, core_size=2.0))
        self.assertTrue(cores)
        for x, y, _charge in cores:
            self.assertAlmostEqual(((x + BOX_L / 2) / DX) % 1.0, 0.5)
            self.assertAlmostEqual(((y + BOX_L / 2) / DX) % 1.0, 0.5)

    def test_push_creates_continuous_failure_region(self):
        high_push = VortexRegime(pair_sep=6.0, core_size=2.0, push=2.6)
        low_push = VortexRegime(pair_sep=6.0, core_size=2.0, push=0.0)
        self.assertLess(
            evaluate_gp_vortex_regime(high_push)["transport_score"],
            evaluate_gp_vortex_regime(low_push)["transport_score"],
        )

    def test_mean_position_error_matches_score(self):
        metrics = evaluate_gp_vortex_regime(START)
        self.assertAlmostEqual(metrics["mean_pos_err"], (BOX_L / 2) * (1.0 - metrics["transport_score"]))

    def test_active_passes_all_admission_criteria(self):
        result = run_search_suite(quiet=True)
        self.assertLessEqual(result["random_success_count_20"], 5)
        self.assertIs(result["active"]["found_transport_ok"], True)
        self.assertGreater(result["active"]["best_score"], result["random_median_best_score"] + SCORE_MARGIN)
        self.assertTrue(
            passed_admission(
                result["active"],
                result["random_median_best_score"],
                result["random_success_count_20"],
            )
        )
        self.assertEqual(result["verdict"], "GP_ACTIVE_DIAGNOSTIC_VALUE_PASSED")

    def test_start_is_failure_region(self):
        self.assertLess(evaluate_gp_vortex_regime(START)["transport_score"], TRANSPORT_OK_THRESHOLD)

    def test_s0_routing_and_never_certify(self):
        result = run_search_suite(quiet=True)
        self.assertEqual(result["active_rec"]["candidate_family"], "K3_TOPOLOGICAL")
        self.assertEqual(result["active_rec"]["allowed_action"], "continue")
        for rec in (result["active_rec"], result["random_rec"]):
            self.assertIn(rec["allowed_action"], {"continue", "archive", "do_not_promote"})

        bad_rec = recommend(diagnostics_from_vortex_search(evaluate_gp_vortex_regime(START))).to_dict()
        self.assertEqual(bad_rec["candidate_family"], "K3_TOPOLOGICAL")
        self.assertEqual(bad_rec["allowed_action"], "do_not_promote")

    def test_memory_write_readback(self):
        result = run_search_suite(quiet=True)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "events.jsonl")
            event = write_memory_event(result, path, run_id="test-k3e2d")
            self.assertTrue(os.path.exists(path))
            loaded = load_events(path)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].module, "k3.active.gp_d")
            self.assertEqual(loaded[0].run_id, event.run_id)
            self.assertIn("not k3 prior validation", loaded[0].claim_boundary.lower())

    def test_apply_action_reduces_push(self):
        moved = apply_action(START, "decrease_push")
        self.assertLess(moved.push, START.push)

    def test_seed_has_complex_field(self):
        field = seed_vortex_pair(6.0, 2.0)
        self.assertEqual(field.shape, (64, 64))
        self.assertTrue(np.iscomplexobj(field))


if __name__ == "__main__":
    unittest.main()
