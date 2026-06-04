"""CLI for running Chronos-S0 on experiment summary files."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Sequence

from .adapters import ADAPTERS, diagnostics_from_summary, load_summary
from .structure_selector import recommend


def emit_recommendation(
    kind: str,
    summary: dict,
    results_dir: str | os.PathLike,
    filename: str = "s0_recommendation.csv",
    verbose: bool = True,
) -> dict | None:
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


def run_cli(argv: Sequence[str] | None = None, *, quiet: bool = False) -> dict:
    parser = argparse.ArgumentParser(description="Chronos-S0: experiment summary -> structure recommendation")
    parser.add_argument("--kind", required=True, choices=sorted(ADAPTERS), help="experiment kind / adapter")
    parser.add_argument("--summary", required=True, help="path to summary .csv or .json")
    parser.add_argument("--out", default=None, help="optional output path (.csv or .json)")
    args = parser.parse_args(argv)

    diagnostics = diagnostics_from_summary(args.kind, load_summary(args.summary))
    recommendation = recommend(diagnostics).to_dict()

    if not quiet:
        print(json.dumps(recommendation, ensure_ascii=False, indent=2))

    if args.out:
        ext = os.path.splitext(args.out)[1].lower()
        if ext == ".json":
            with open(args.out, "w") as handle:
                json.dump(recommendation, handle, ensure_ascii=False, indent=2)
        else:
            with open(args.out, "w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(recommendation.keys()))
                writer.writeheader()
                writer.writerow(recommendation)
        if not quiet:
            print(f"(saved to {args.out})", file=sys.stderr)

    return recommendation


def main() -> None:
    run_cli()


if __name__ == "__main__":
    main()
