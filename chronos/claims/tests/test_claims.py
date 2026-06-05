import os
import tempfile
import unittest

from chronos.claims import (
    ClaimRecord,
    append_claim,
    claim_from_k2_summary,
    claim_from_k3_2d_0_summary,
    claim_from_k3_e2b,
    claim_from_k3_e2c,
    claim_from_k3_e2d,
    claims_requiring_next_gate,
    is_known_failure_mode,
    load_claims,
    summarize_claims,
    supersede_claim,
    write_claims,
)
from chronos.claims.failure_taxonomy import (
    ACTIVE_NO_ADVANTAGE,
    PIPELINE_OK_TRANSPORT_FAIL,
    RANDOM_ALSO_SUCCEEDS,
    REGIME_INVALID,
)
from chronos.s0.diagnostics_schema import ACT_ARCHIVE, ACT_CONTINUE, ACT_DO_NOT_PROMOTE, K2_SYMPLECTIC, K3_TOPOLOGICAL


def _claim(**overrides):
    base = {
        "claim_id": "c1",
        "structure_family": K3_TOPOLOGICAL,
        "verdict": "GP_ACTIVE_DIAGNOSTIC_VALUE_PASSED",
        "evidence_level": "gp_truth_active_search",
        "gate": "regime",
        "diagnostics": {"score": 1.0},
        "controls": {"random_action_search": True},
        "supports": ["active search finds a GP vortex regime better than random"],
        "does_not_support": ["K3 prior validation"],
        "failure_mode": None,
        "next_gate": "K3.2D.0 CNN baseline regime validation",
        "claim_boundary": "truth-level GP only; not prior validation",
        "allowed_action": ACT_CONTINUE,
        "timestamp": "2026-06-05T00:00:00+00:00",
        "source_module": "chronos.test",
        "code_version": "claims_test",
    }
    base.update(overrides)
    return ClaimRecord(**base)


class ClaimRecordTests(unittest.TestCase):
    def test_required_fields_and_roundtrip(self):
        claim = _claim()
        self.assertEqual(ClaimRecord.from_dict(claim.to_dict()), claim)
        for field in ("claim_id", "structure_family", "verdict", "claim_boundary", "timestamp"):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    _claim(**{field: ""})
        with self.assertRaises(ValueError):
            _claim(source_module="")

    def test_allowed_action_bans_certification_words_only_as_actions(self):
        for action in ("certified", "promote", "proved", "validated"):
            with self.subTest(action=action):
                with self.assertRaises(ValueError):
                    _claim(allowed_action=action)

        claim = _claim(verdict="FULL_REGIME_VALIDATED")
        self.assertEqual(claim.verdict, "FULL_REGIME_VALIDATED")

    def test_supports_and_does_not_support_are_required(self):
        with self.assertRaises(ValueError):
            _claim(supports=[])
        with self.assertRaises(ValueError):
            _claim(does_not_support=[])

    def test_certified_rule_only_checks_verdict(self):
        ok = _claim(
            verdict="K2_VPSL_CERTIFIED",
            structure_family=K2_SYMPLECTIC,
            evidence_level="vpsl_certified_structure",
            gate="transfer",
        )
        self.assertEqual(ok.verdict, "K2_VPSL_CERTIFIED")

        with self.assertRaises(ValueError):
            _claim(verdict="K2_CERTIFIED", evidence_level="gp_truth_active_search", gate="regime")

        boundary_only = _claim(claim_boundary="not certified; not prior validation")
        self.assertIn("not certified", boundary_only.claim_boundary)

    def test_claim_status_validation(self):
        self.assertEqual(_claim().claim_status, "active")
        with self.assertRaises(ValueError):
            _claim(claim_status="validated")

    def test_unknown_failure_mode_raises(self):
        with self.assertRaises(ValueError):
            _claim(failure_mode="MADE_UP")

    def test_controls_and_diagnostics_must_be_dicts(self):
        with self.assertRaises(TypeError):
            _claim(controls=["random_action_search"])
        with self.assertRaises(TypeError):
            _claim(diagnostics=["score"])


class BuilderTests(unittest.TestCase):
    def test_k3_e2b_builder(self):
        claim = claim_from_k3_e2b({"run_id": "e2b", "timestamp": "2026-01-01T00:00:00+00:00"})
        self.assertEqual(claim.structure_family, K3_TOPOLOGICAL)
        self.assertEqual(claim.allowed_action, ACT_CONTINUE)
        self.assertIn("toy", claim.claim_boundary)
        self.assertEqual(claim.source_module, "chronos.k3.run_active_topology_search")
        self.assertEqual(claim.code_version, "claims_v1")
        self.assertIsInstance(claim.controls, dict)

    def test_k3_e2c_builder_records_no_advantage(self):
        claim = claim_from_k3_e2c(
            {
                "verdict": "GP_ACTIVE_NO_ADVANTAGE",
                "active_rec": {"allowed_action": "continue"},
                "random_success_count_20": 18,
            }
        )
        self.assertEqual(claim.failure_mode, RANDOM_ALSO_SUCCEEDS)
        self.assertEqual(claim.allowed_action, ACT_ARCHIVE)
        self.assertEqual(claim.next_gate, "K3-E2d discriminating GP truth active search")
        self.assertIn("active/random comparison is operational", claim.supports)
        self.assertIn("active exploration has diagnostic advantage", claim.does_not_support[0])

    def test_k3_e2c_no_advantage_never_continue(self):
        claim = claim_from_k3_e2c({"verdict": "GP_ACTIVE_NO_ADVANTAGE", "active_rec": {"allowed_action": "continue"}})
        self.assertEqual(claim.failure_mode, ACTIVE_NO_ADVANTAGE)
        self.assertEqual(claim.allowed_action, ACT_ARCHIVE)

    def test_k3_e2d_builder_next_gate(self):
        claim = claim_from_k3_e2d(
            {
                "verdict": "GP_ACTIVE_DIAGNOSTIC_VALUE_PASSED",
                "active": {"best_score": 0.96, "best_metrics": {"mean_pos_err": 0.31, "pair_intact": True}},
                "random_median_best_score": 0.82,
            }
        )
        self.assertEqual(claim.next_gate, "K3.2D.0 CNN baseline regime validation")
        self.assertIn("CNN baseline regime validation", claim.does_not_support)
        self.assertEqual(claim.diagnostics["active_mean_pos_err"], 0.31)

    def test_k3_2d_0_transport_fail_builder(self):
        claim = claim_from_k3_2d_0_summary(
            {
                "verdict": "SMOKE_PIPELINE_OK_TRANSPORT_FAIL",
                "pipeline_ok": True,
                "transport_ok": False,
                "ref_med": 0.01,
                "pos_med": 8.0,
                "pair_frac": 0.0,
                "hard_frac": 0.0,
            }
        )
        self.assertEqual(claim.failure_mode, PIPELINE_OK_TRANSPORT_FAIL)
        self.assertEqual(claim.allowed_action, ACT_DO_NOT_PROMOTE)
        self.assertEqual(claim.next_gate, "K3.2D.0-B neural baseline learnability rescue or archive as regime unresolved")

    def test_k3_2d_0_invalid_builder(self):
        with self.assertRaises(ValueError):
            claim_from_k3_2d_0_summary({"verdict": "REGIME_UNRESOLVED"})
        with self.assertRaises(ValueError):
            claim_from_k3_2d_0_summary({"pipeline_ok": True, "transport_ok": True})

    def test_k2_builder_allows_certified_when_transfer_evidence(self):
        claim = claim_from_k2_summary({"verdict": "K2_VPSL_CERTIFIED"})
        self.assertEqual(claim.structure_family, K2_SYMPLECTIC)
        self.assertEqual(claim.evidence_level, "vpsl_certified_structure")
        self.assertEqual(claim.gate, "transfer")
        self.assertEqual(claim.source_module, "chronos.k2")


class ReplayTests(unittest.TestCase):
    def test_summarize_and_next_gate(self):
        claims = [
            _claim(claim_id="old", timestamp="2026-01-01T00:00:00+00:00"),
            _claim(claim_id="new", timestamp="2026-01-02T00:00:00+00:00", verdict="FULL_REGIME_VALIDATED"),
        ]
        summary = summarize_claims(claims)
        self.assertEqual(summary["count_total"], 2)
        self.assertEqual(summary["count_by_structure_family"], {K3_TOPOLOGICAL: 2})
        self.assertEqual(summary["latest_claim_by_structure_family"][K3_TOPOLOGICAL]["claim_id"], "new")
        self.assertEqual(len(claims_requiring_next_gate(claims)), 2)

    def test_supersede_claim(self):
        updated = supersede_claim([_claim()], "c1", "narrowed by later evidence")
        self.assertEqual(updated[0].claim_status, "superseded")
        self.assertEqual(updated[0].superseded_reason, "narrowed by later evidence")
        with self.assertRaises(ValueError):
            supersede_claim(updated, "missing")

    def test_jsonl_roundtrip_and_errors(self):
        claims = [_claim(claim_id="a"), _claim(claim_id="b", verdict="FULL_REGIME_VALIDATED")]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "claims.jsonl")
            write_claims(claims, path)
            self.assertEqual(load_claims(path), claims)
            append_claim(_claim(claim_id="c"), path)
            self.assertEqual(len(load_claims(path)), 3)
            self.assertEqual(load_claims(os.path.join(temp_dir, "missing.jsonl")), [])

            bad_path = os.path.join(temp_dir, "bad.jsonl")
            with open(bad_path, "w", encoding="utf-8") as handle:
                handle.write("{broken json\n")
            with self.assertRaises(ValueError):
                load_claims(bad_path)

    def test_failure_mode_taxonomy(self):
        self.assertTrue(is_known_failure_mode(None))
        self.assertTrue(is_known_failure_mode(PIPELINE_OK_TRANSPORT_FAIL))
        self.assertFalse(is_known_failure_mode("MADE_UP"))


if __name__ == "__main__":
    unittest.main()
