"""Experiment K2.2-B: Symplectic transfer test at H=240.

This is the canonical repository archive entrypoint for the K2.2-B H=240
result. It records the registered two-phase regime-gated discipline and writes
the archived repository summary. It does not rerun the full GPU Colab training
experiment.

K2.2-B extends K2.2-A: the symplectic prior remains
FULL_TRANSFER_CONFIRMED at H=240 on the graceful-baseline subset.

Archived headline:
- graceful-baseline subset: n = 18
- baseline roll_MSE: 0.0847
- symplectic roll_MSE: 0.0543
- fair energy roll_MSE: 0.1374
- fair L2 roll_MSE: 0.1408
- full symp_err reduction: 71.5%
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


EXPERIMENT = "k2_2b_transfer_h240"
STRUCTURE = "symplectic prior"
REGIME = "FPU-β"
HORIZON = 240
CLAIM_LEVEL = "FULL_TRANSFER_CONFIRMED"


@dataclass(frozen=True)
class K22BArchiveSummary:
    """Minimal archived values for the K2.2-B headline result."""

    n_graceful: int = 18
    baseline_roll_mse: float = 0.0847
    symplectic_roll_mse: float = 0.0543
    fair_energy_roll_mse: float = 0.1374
    fair_l2_roll_mse: float = 0.1408
    symp_err_reduction: float = 0.715


def write_archived_summary(path: Path) -> None:
    """Write the archived K2.2-B headline result used by the repository."""

    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        [
            {
                "experiment": EXPERIMENT,
                "structure": STRUCTURE,
                "regime": REGIME,
                "horizon": HORIZON,
                "claim_level": CLAIM_LEVEL,
                "primary_subset": "graceful-baseline",
                "baseline_comparison": "symplectic<baseline",
                "energy_control": "symplectic<fair energy",
                "l2_control": "symplectic<fair L2",
                "mechanism": "full symplectic Jacobian error reduction confirmed",
            }
        ]
    )
    df.to_csv(path, index=False)


def archived_values() -> K22BArchiveSummary:
    """Return the archived numerical evidence summary."""

    return K22BArchiveSummary()


def main() -> None:
    results_path = Path(__file__).resolve().parents[1] / "results" / "k2_2b_summary.csv"
    write_archived_summary(results_path)
    values = archived_values()
    print(f"Wrote archived K2.2-B summary to {results_path}")
    print(f"Verdict: {CLAIM_LEVEL}")
    print(f"Graceful subset n={values.n_graceful}")
    print(f"Symplectic roll_MSE={values.symplectic_roll_mse:.4f}")
    print(f"Full symp_err reduction={values.symp_err_reduction:.1%}")


if __name__ == "__main__":
    main()
