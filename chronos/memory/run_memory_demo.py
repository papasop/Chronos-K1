"""Demo runner for the S0-M0 memory logging layer."""

from __future__ import annotations

import argparse
import json
import os
import tempfile

from chronos.memory.logging import MemoryEvent, append_event, load_events, new_timestamp, summarize_events


def demo_memory_loop(path: str | None = None) -> dict:
    """Write two demo events and return their read-only summary."""

    if path is None:
        temp_dir = tempfile.mkdtemp(prefix="chronos-memory-")
        path = os.path.join(temp_dir, "events.jsonl")
    append_event(
        MemoryEvent(
            timestamp=new_timestamp(),
            run_id="demo-1",
            module="k3.active",
            experiment_kind="k3_e2b",
            verdict="ACTIVE_DIAGNOSTIC_VALUE_PASSED",
            candidate_family="K3_TOPOLOGICAL",
            allowed_action="continue",
            score=1.0,
            payload={"source": "demo"},
            claim_boundary="toy landscape only; not GP truth; not K3 prior validation",
            code_version="demo",
        ),
        path,
    )
    append_event(
        MemoryEvent(
            timestamp=new_timestamp(),
            run_id="demo-2",
            module="s0.e1",
            experiment_kind="s0_e1",
            verdict="K2_MEASURED",
            candidate_family="K2_SYMPLECTIC",
            allowed_action="continue",
            score=None,
            payload={"source": "demo"},
            claim_boundary="toy pendulum sim; symplectic measured, not certified",
            code_version="demo",
        ),
        path,
    )
    return {"path": path, "summary": summarize_events(load_events(path))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Chronos S0-M0 memory logging demo")
    parser.add_argument("--path", default=None, help="optional JSONL path to append demo events")
    parser.add_argument("--json", action="store_true", help="print full summary JSON")
    args = parser.parse_args()

    result = demo_memory_loop(args.path)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    summary = result["summary"]
    print("S0-M0 Memory Logging Layer")
    print("logging only; not a learning layer; no feedback into S0\n")
    print(f"wrote demo events to {result['path']}")
    print(
        json.dumps(
            {
                "count_total": summary["count_total"],
                "count_by_module": summary["count_by_module"],
                "count_by_verdict": summary["count_by_verdict"],
                "count_by_candidate_family": summary["count_by_candidate_family"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
