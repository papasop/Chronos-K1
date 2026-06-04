import unittest

from chronos.s0.diagnostics_schema import (
    ACT_CONTINUE,
    ACT_DO_NOT_PROMOTE,
    GATE_REGIME,
    K1_LORENTZ,
    K2_SYMPLECTIC,
    K3_TOPOLOGICAL,
    UNRESOLVED,
)
from chronos.s0.structure_selector import k32d_verdict, recommend


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
            }
        )
        self.assertEqual(rec.candidate_family, K3_TOPOLOGICAL)
        self.assertEqual(rec.allowed_action, ACT_DO_NOT_PROMOTE)
        self.assertEqual(rec.next_vpsl_gate, GATE_REGIME)

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
        self.assertEqual(k32d_verdict("SMOKE", 0.02, 0.0, 0.8, 2.0), "SMOKE_TRANSPORT_OK")
        self.assertEqual(k32d_verdict("SMOKE", 0.2, 0.0, 0.0, 8.0), "SMOKE_PIPELINE_FAIL")
        self.assertEqual(k32d_verdict("FULL", 0.02, 0.0, 0.8, 2.0), "FULL_REGIME_VALIDATED")
        self.assertEqual(k32d_verdict("FULL", 0.02, 0.0, 0.0, 8.0), "REGIME_UNRESOLVED")


if __name__ == "__main__":
    unittest.main()

