import unittest

from chronos.k3.active_topology_search import (
    ACTIONS,
    SCORE_MARGIN,
    START,
    VortexRegime,
    active_search,
    apply_action,
    diagnostics_from_vortex_search,
    evaluate_vortex_regime,
    has_active_advantage,
    novelty,
    pure_novelty_search,
    random_search,
)
from chronos.k3.run_active_topology_search import run_search_suite
from chronos.s0.structure_selector import recommend


class ActiveTopologySearchTests(unittest.TestCase):
    def test_evaluate_returns_expected_keys(self):
        metrics = evaluate_vortex_regime(START)
        for key in (
            "pair_intact",
            "vortex_lifetime_fraction",
            "net_charge_error",
            "pos_err",
            "topology_score",
            "transport_ok",
        ):
            self.assertIn(key, metrics)
        self.assertGreaterEqual(metrics["topology_score"], 0.0)
        self.assertLessEqual(metrics["topology_score"], 1.0)

    def test_stable_regime_scores_above_bad_regime(self):
        stable = VortexRegime(pair_sep=13.0, core_size=3.0, dt_phys=0.01, horizon=40)
        bad = VortexRegime(pair_sep=4.0, core_size=0.8, dt_phys=0.08, horizon=160)
        self.assertLess(evaluate_vortex_regime(bad)["topology_score"], evaluate_vortex_regime(stable)["topology_score"])
        self.assertIs(evaluate_vortex_regime(stable)["transport_ok"], True)

    def test_active_improves_over_start(self):
        active = active_search(START)
        self.assertGreater(active["best_score"], evaluate_vortex_regime(START)["topology_score"])

    def test_active_beats_random_control(self):
        active = active_search(START)
        random_result = random_search(START, seed=0)
        self.assertTrue(has_active_advantage(active, random_result))
        self.assertTrue(
            active["found_transport_ok"]
            and not random_result["found_transport_ok"]
            or active["best_score"] > random_result["best_score"] + SCORE_MARGIN
            or (
                active["trials_to_success"] is not None
                and (
                    random_result["trials_to_success"] is None
                    or active["trials_to_success"] < random_result["trials_to_success"]
                )
            )
        )

    def test_active_beats_median_random_across_seeds(self):
        active = active_search(START)
        random_median = sorted(random_search(START, seed=seed)["best_score"] for seed in range(20))[10]
        self.assertGreater(active["best_score"], random_median)

    def test_active_best_routes_to_k3_continue(self):
        active = active_search(START)
        rec = recommend(diagnostics_from_vortex_search(active["best_metrics"])).to_dict()
        self.assertEqual(rec["candidate_family"], "K3_TOPOLOGICAL")
        self.assertEqual(rec["allowed_action"], "continue")

    def test_bad_regime_routes_to_k3_do_not_promote(self):
        bad = VortexRegime(pair_sep=4.0, core_size=0.8, dt_phys=0.08, horizon=160)
        rec = recommend(diagnostics_from_vortex_search(evaluate_vortex_regime(bad))).to_dict()
        self.assertEqual(rec["candidate_family"], "K3_TOPOLOGICAL")
        self.assertEqual(rec["allowed_action"], "do_not_promote")

    def test_s0_never_certifies(self):
        active = active_search(START)
        bad = VortexRegime(pair_sep=4.0, core_size=0.8, dt_phys=0.08, horizon=160)
        recs = [
            recommend(diagnostics_from_vortex_search(active["best_metrics"])).to_dict(),
            recommend(diagnostics_from_vortex_search(evaluate_vortex_regime(bad))).to_dict(),
        ]
        for rec in recs:
            self.assertIn(rec["allowed_action"], {"continue", "archive", "do_not_promote"})

    def test_active_search_is_deterministic(self):
        first = active_search(START)
        second = active_search(START)
        self.assertEqual(first["best_score"], second["best_score"])
        self.assertEqual(first["trials_to_success"], second["trials_to_success"])

    def test_pure_novelty_does_not_solve_landscape(self):
        self.assertIs(pure_novelty_search(START), False)

    def test_action_and_novelty_helpers(self):
        moved = apply_action(START, "increase_sep")
        self.assertGreater(moved.pair_sep, START.pair_sep)
        self.assertGreater(novelty(moved, [START]), 0.0)
        self.assertIn("rotate_angle", ACTIONS)

    def test_suite_verdict(self):
        result = run_search_suite(quiet=True)
        self.assertEqual(result["verdict"], "ACTIVE_DIAGNOSTIC_VALUE_PASSED")
        self.assertEqual(result["active_rec"]["candidate_family"], "K3_TOPOLOGICAL")
        self.assertEqual(result["active_rec"]["allowed_action"], "continue")
        self.assertEqual(result["random_rec"]["candidate_family"], "K3_TOPOLOGICAL")
        self.assertEqual(result["random_rec"]["allowed_action"], "do_not_promote")
        self.assertEqual(result["random_success_count_20"], 0)


if __name__ == "__main__":
    unittest.main()
