import os
import tempfile
import unittest

from chronos.memory.logging import MemoryEvent, append_event, load_events, new_timestamp, summarize_events
from chronos.s0.diagnostics_schema import ALLOWED_ACTIONS, ACT_DO_NOT_PROMOTE


def make_event(**kwargs):
    base = {
        "timestamp": new_timestamp(),
        "run_id": "r1",
        "module": "k3.active",
        "experiment_kind": "k3_e2b",
        "verdict": "ACTIVE_DIAGNOSTIC_VALUE_PASSED",
        "candidate_family": "K3_TOPOLOGICAL",
        "allowed_action": "continue",
        "score": 1.0,
        "payload": {"a": 1},
        "claim_boundary": "toy only",
        "code_version": "v1",
    }
    base.update(kwargs)
    return MemoryEvent(**base)


class MemoryLoggingTests(unittest.TestCase):
    def test_jsonl_roundtrip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "events.jsonl")
            first = make_event(run_id="a")
            second = make_event(run_id="b", allowed_action=ACT_DO_NOT_PROMOTE, verdict="X")
            append_event(first, path)
            append_event(second, path)

            loaded = load_events(path)
            self.assertEqual(len(loaded), 2)
            self.assertEqual(loaded[0].run_id, "a")
            self.assertEqual(loaded[1].run_id, "b")
            self.assertEqual(loaded[0].to_dict(), first.to_dict())
            self.assertEqual(loaded[1].allowed_action, ACT_DO_NOT_PROMOTE)

    def test_claim_boundary_is_required(self):
        for bad in ("", "   ", None):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    make_event(claim_boundary=bad)

    def test_allowed_actions_mirror_s0_never_certify(self):
        for bad in ("certified", "certify", "promote", "PROMOTE", "done"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    make_event(allowed_action=bad)
        for action in ALLOWED_ACTIONS:
            with self.subTest(action=action):
                self.assertEqual(make_event(allowed_action=action).allowed_action, action)

    def test_score_must_be_number_or_none(self):
        with self.assertRaises(ValueError):
            make_event(score="not-a-number")
        self.assertIsNone(make_event(score=None).score)
        self.assertEqual(make_event(score="1.25").score, 1.25)

    def test_summarize_counts(self):
        events = [
            make_event(module="k3.active", verdict="PASS", candidate_family="K3_TOPOLOGICAL", timestamp="2026-01-01T00:00:00+00:00"),
            make_event(module="k3.active", verdict="PASS", candidate_family="K3_TOPOLOGICAL", timestamp="2026-01-02T00:00:00+00:00"),
            make_event(module="s0.e1", verdict="FAIL", candidate_family="K2_SYMPLECTIC", timestamp="2026-01-03T00:00:00+00:00"),
        ]
        summary = summarize_events(events)
        self.assertEqual(summary["count_total"], 3)
        self.assertEqual(summary["count_by_module"], {"k3.active": 2, "s0.e1": 1})
        self.assertEqual(summary["count_by_verdict"], {"PASS": 2, "FAIL": 1})
        self.assertEqual(summary["count_by_candidate_family"], {"K3_TOPOLOGICAL": 2, "K2_SYMPLECTIC": 1})
        self.assertEqual(summary["latest_event_by_module"]["k3.active"]["timestamp"], "2026-01-02T00:00:00+00:00")
        self.assertEqual(summary["latest_event_by_module"]["s0.e1"]["timestamp"], "2026-01-03T00:00:00+00:00")

    def test_missing_file_returns_empty_list(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.assertEqual(load_events(os.path.join(temp_dir, "missing.jsonl")), [])

    def test_k3_event_carries_toy_only_boundary(self):
        event = make_event(
            module="k3.active",
            claim_boundary="toy search landscape only; NOT GP truth, NOT K3 prior validation.",
        )
        boundary = event.claim_boundary.lower()
        self.assertIn("toy", boundary)
        self.assertIn("not gp truth", boundary)

    def test_malformed_jsonl_surfaces_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "bad.jsonl")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write('{"not":"valid-event but valid-json"}\n{ broken json\n')
            with self.assertRaises(ValueError):
                load_events(path)

    def test_append_event_requires_memory_event(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(TypeError):
                append_event({"not": "an event"}, os.path.join(temp_dir, "events.jsonl"))  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
