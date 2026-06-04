import unittest
import csv
import os
import tempfile

from chronos.s0.adapters import diagnostics_from_k2_summary, diagnostics_from_k32d_summary, diagnostics_from_summary
from chronos.k3 import verdicts as k3_verdicts
from chronos.k3.verdicts import k32d_verdict
from chronos.s0.diagnostics_schema import (
    ACT_CONTINUE,
    ACT_DO_NOT_PROMOTE,
    CTX_SYMPLECTIC,
    CTX_TOPOLOGY,
    GATE_REGIME,
    K1_LORENTZ,
    K2_SYMPLECTIC,
    K3_TOPOLOGICAL,
    UNRESOLVED,
)
from chronos.s0.emitter import emit_recommendation
from chronos.s0.run_selector import run_cli
from chronos.s0.structure_selector import Recommendation, recommend


class StructureSelectorTests(unittest.TestCase):
    def test_missing_diagnostics(self):
        rec = recommend({})
        self.assertEqual(rec.candidate_family, UNRESOLVED)
        self.assertEqual(rec.allowed_action, ACT_DO_NOT_PROMOTE)

    def test_topology_transport_fail_guard(self):
        rec = recommend(
            {
                "field_learnable": True,
                "object_tracking_valid": False,
                "topological_transport_score": 0.0,
                "baseline_divergence": 0.01,
                "diagnostic_context": CTX_TOPOLOGY,
            }
        )
        self.assertEqual(rec.candidate_family, K3_TOPOLOGICAL)
        self.assertEqual(rec.allowed_action, ACT_DO_NOT_PROMOTE)
        self.assertEqual(rec.next_vpsl_gate, GATE_REGIME)

    def test_bounded_only_is_not_field_learned(self):
        rec = recommend(
            {
                "object_tracking_valid": False,
                "topological_transport_score": 0.0,
                "baseline_divergence": 0.01,
            }
        )
        self.assertEqual(rec.candidate_family, UNRESOLVED)
        self.assertEqual(rec.allowed_action, ACT_DO_NOT_PROMOTE)

    def test_transport_fail_without_context_falls_back_to_k3(self):
        rec = recommend({"field_learnable": True, "object_tracking_valid": False})
        self.assertEqual(rec.candidate_family, K3_TOPOLOGICAL)
        self.assertEqual(rec.allowed_action, ACT_DO_NOT_PROMOTE)

    def test_transport_fail_does_not_shadow_k2_without_context(self):
        rec = recommend(
            {
                "field_learnable": True,
                "object_tracking_valid": False,
                "symplectic_improves_vs_controls": True,
            }
        )
        self.assertEqual(rec.candidate_family, K2_SYMPLECTIC)

    def test_topology_context_transport_fail_preempts_k2(self):
        rec = recommend(
            {
                "field_learnable": True,
                "object_tracking_valid": False,
                "baseline_divergence": 0.01,
                "symplectic_improves_vs_controls": True,
                "diagnostic_context": CTX_TOPOLOGY,
            }
        )
        self.assertEqual(rec.candidate_family, K3_TOPOLOGICAL)

    def test_transport_score_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            recommend({"topological_transport_score": 3.2})

    def test_topology_context_does_not_require_hard_frac(self):
        rec = recommend(
            {
                "field_learnable": True,
                "object_tracking_valid": False,
                "diagnostic_context": CTX_TOPOLOGY,
            }
        )
        self.assertEqual(rec.candidate_family, K3_TOPOLOGICAL)
        self.assertEqual(rec.allowed_action, ACT_DO_NOT_PROMOTE)

    def test_strong_symplectic(self):
        rec = recommend({"symplectic_improves_vs_controls": True, "symplectic_jacobian_error": 0.1})
        self.assertEqual(rec.candidate_family, K2_SYMPLECTIC)
        self.assertEqual(rec.allowed_action, ACT_CONTINUE)

    def test_causal_signal(self):
        self.assertEqual(recommend({"causal_violation_rate": 0.3}).candidate_family, K1_LORENTZ)

    def test_topology_ok(self):
        rec = recommend({"topological_transport_score": 0.85, "object_tracking_valid": True})
        self.assertEqual(rec.candidate_family, K3_TOPOLOGICAL)
        self.assertEqual(rec.allowed_action, ACT_CONTINUE)

    def test_k32d_smoke_transport_fail(self):
        self.assertEqual(
            k32d_verdict("SMOKE", ref_med=0.0374, hard_frac=0.0, pair_frac=0.0, pos_med=8.0),
            "SMOKE_PIPELINE_OK_TRANSPORT_FAIL",
        )

    def test_k32d_verdicts(self):
        self.assertEqual(k32d_verdict("smoke", 0.02, 0.0, 0.8, 2.0), "SMOKE_TRANSPORT_OK")
        self.assertEqual(k32d_verdict("SMOKE", 0.02, 0.0, 0.8, 2.0), "SMOKE_TRANSPORT_OK")
        self.assertEqual(k32d_verdict("SMOKE", 0.2, 0.0, 0.0, 8.0), "SMOKE_PIPELINE_FAIL")
        self.assertEqual(k32d_verdict("FULL", 0.02, 0.0, 0.8, 2.0), "FULL_REGIME_VALIDATED")
        self.assertEqual(k32d_verdict("FULL", 0.02, 0.0, 0.0, 8.0), "REGIME_UNRESOLVED")

    def test_k32d_invalid_mode(self):
        with self.assertRaises(ValueError):
            k32d_verdict("scout", 0.02, 0.0, 0.8, 2.0)

    def test_k32d_canonical_truth_table(self):
        self.assertEqual(k3_verdicts.REF_CEIL, 0.05)
        self.assertEqual(k3_verdicts.HARD_DIV_MAX, 0.5)
        self.assertEqual(k3_verdicts.PAIR_INTACT_MIN, 0.6)
        self.assertEqual(k3_verdicts.POS_ERR_CEIL, 8.0)
        cases = [
            (("SMOKE", 0.0374, 0.0, 0.0, 8.0), "SMOKE_PIPELINE_OK_TRANSPORT_FAIL"),
            (("SMOKE", 0.02, 0.0, 0.8, 2.0), "SMOKE_TRANSPORT_OK"),
            (("SMOKE", 0.2, 0.0, 0.0, 8.0), "SMOKE_PIPELINE_FAIL"),
            (("FULL", 0.02, 0.0, 0.8, 2.0), "FULL_REGIME_VALIDATED"),
            (("FULL", 0.02, 0.0, 0.0, 8.0), "REGIME_UNRESOLVED"),
            (("smoke", 0.02, 0.0, 0.8, 2.0), "SMOKE_TRANSPORT_OK"),
        ]
        for args, expected in cases:
            self.assertEqual(k3_verdicts.k32d_verdict(*args), expected)

    def test_k32d_adapter_transport_fail(self):
        diagnostics = diagnostics_from_k32d_summary(
            {"pipeline_ok": True, "transport_ok": False, "hard_frac": 0.0, "pair_frac": 0.0}
        )
        rec = recommend(diagnostics)
        self.assertEqual(rec.candidate_family, K3_TOPOLOGICAL)
        self.assertEqual(rec.allowed_action, ACT_DO_NOT_PROMOTE)

    def test_k32d_adapter_transport_ok(self):
        diagnostics = diagnostics_from_k32d_summary(
            {"pipeline_ok": True, "transport_ok": True, "hard_frac": 0.0, "pair_frac": 0.9}
        )
        rec = recommend(diagnostics)
        self.assertEqual(rec.candidate_family, K3_TOPOLOGICAL)
        self.assertEqual(rec.allowed_action, ACT_CONTINUE)

    def test_k32d_adapter_clamps_pair_frac(self):
        diagnostics = diagnostics_from_k32d_summary({"pipeline_ok": True, "transport_ok": True, "pair_frac": 1.7})
        self.assertLessEqual(diagnostics["topological_transport_score"], 1.0)
        self.assertEqual(diagnostics["diagnostic_context"], CTX_TOPOLOGY)

    def test_k32d_adapter_missing_pair_frac_raises_when_transport_ok(self):
        with self.assertRaises(ValueError):
            diagnostics_from_k32d_summary({"pipeline_ok": True, "transport_ok": True})

    def test_k32d_adapter_transport_fail_can_omit_pair_frac(self):
        diagnostics = diagnostics_from_k32d_summary({"pipeline_ok": True, "transport_ok": False})
        rec = recommend(diagnostics)
        self.assertEqual(rec.candidate_family, K3_TOPOLOGICAL)
        self.assertEqual(rec.allowed_action, ACT_DO_NOT_PROMOTE)

    def test_k2_adapter(self):
        diagnostics = diagnostics_from_k2_summary(
            {"beats_baseline_and_controls": True, "symp_err": 0.1, "hard_frac": 0.0}
        )
        self.assertEqual(diagnostics["diagnostic_context"], CTX_SYMPLECTIC)
        rec = recommend(diagnostics)
        self.assertEqual(rec.candidate_family, K2_SYMPLECTIC)
        self.assertEqual(rec.allowed_action, ACT_CONTINUE)

    def test_unknown_adapter_kind_raises(self):
        with self.assertRaises(ValueError):
            diagnostics_from_summary("nope", {})

    def test_run_selector_cli_from_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = os.path.join(tmpdir, "summary.csv")
            with open(summary_path, "w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["pipeline_ok", "transport_ok", "hard_frac", "pair_frac"])
                writer.writeheader()
                writer.writerow(
                    {"pipeline_ok": "True", "transport_ok": "False", "hard_frac": "0.0", "pair_frac": "0.0"}
                )
            rec = run_cli(["--kind", "k3_2d", "--summary", summary_path], quiet=True)
        self.assertEqual(rec["candidate_family"], K3_TOPOLOGICAL)
        self.assertEqual(rec["allowed_action"], ACT_DO_NOT_PROMOTE)

    def test_emit_recommendation_writes_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rec = emit_recommendation(
                "k3_2d",
                {"pipeline_ok": True, "transport_ok": False, "hard_frac": 0.0, "pair_frac": 0.0},
                tmpdir,
                verbose=False,
            )
            out_path = os.path.join(tmpdir, "s0_recommendation.csv")
            self.assertTrue(os.path.exists(out_path))
            self.assertEqual(rec["candidate_family"], K3_TOPOLOGICAL)
            self.assertEqual(rec["allowed_action"], ACT_DO_NOT_PROMOTE)

    def test_emit_recommendation_degrades_without_crashing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rec = emit_recommendation("bogus", {}, tmpdir, verbose=False)
            self.assertIsNone(rec)
            self.assertFalse(os.path.exists(os.path.join(tmpdir, "s0_recommendation.csv")))

    def test_emit_recommendation_strict_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                emit_recommendation("bogus", {}, tmpdir, verbose=False, strict=True)
            self.assertFalse(os.path.exists(os.path.join(tmpdir, "s0_recommendation.csv")))

    def test_recommendation_rejects_certifying_actions(self):
        for bad_action in ("certified", "certify", "promote"):
            with self.subTest(bad_action=bad_action):
                with self.assertRaises(ValueError):
                    Recommendation(K2_SYMPLECTIC, "high", "x", "mechanism", bad_action)


if __name__ == "__main__":
    unittest.main()
