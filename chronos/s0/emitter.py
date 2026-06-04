"""Experiment-side helpers for emitting Chronos-S0 recommendations."""

from __future__ import annotations

import csv
import os
from typing import Any

from .adapters import diagnostics_from_summary
from .structure_selector import recommend


def emit_recommendation(
    kind: str,
    summary: dict[str, Any],
    results_dir: str | os.PathLike,
    filename: str = "s0_recommendation.csv",
    verbose: bool = True,
) -> dict[str, Any] | None:
    """Write an S0 recommendation CSV beside an experiment summary.

    This helper is intentionally non-throwing so experiment scripts can call it
    at the end of a run without risking loss of their primary result.
    """

    try:
        recommendation = recommend(diagnostics_from_summary(kind, summary)).to_dict()
        os.makedirs(results_dir, exist_ok=True)
        out = os.path.join(str(results_dir), filename)
        with open(out, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(recommendation.keys()))
            writer.writeheader()
            writer.writerow(recommendation)
        if verbose:
            print(
                f"[S0] {recommendation['candidate_family']} / {recommendation['confidence']} / "
                f"next_gate={recommendation['next_vpsl_gate']} / action={recommendation['allowed_action']}"
            )
            print(f"[S0] {recommendation['reason']}")
            print(f"[S0] wrote {out}")
        return recommendation
    except Exception as exc:
        if verbose:
            print(
                f"[S0] recommendation skipped ({type(exc).__name__}: {exc}); "
                "no s0_recommendation.csv emitted."
            )
        return None
