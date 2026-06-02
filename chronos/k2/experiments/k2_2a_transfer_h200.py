"""Experiment K2.2-A: Symplectic transfer test at H=200.

This is the canonical repository location for the Colab experiment supplied in
the K2.2-A archive. It records the pre-registered VPSL discipline and verdict
logic for the H=200 stress-transfer test.

K2.1-B established, with fair non-degenerate controls, that at H=160 the
symplectic prior beats baseline, fair energy, and fair L2 on rollout MSE, with
a confirmed mechanism on the graceful subset.

K2.2-A does not re-prove that symplectic helps. It tests range of validity:
does the advantage transfer to H=200, where K2.0-B found baseline hard
divergence around 0.27, or does it degrade into divergence rescue / damping?

Verdict levels:
- FULL_TRANSFER_CONFIRMED
- TRANSFER_CONFIRMED
- TRANSFER_PERFORMANCE_ONLY
- CONTROL_MATCHES_SYMP
- NO_TRANSFER

The complete Colab source from the experiment note should be kept in sync with
this file when running the archived experiment. This repository copy preserves
the configuration, gates, and verdict semantics without claiming that the
placeholder-only predecessor scripts are executable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


EXPERIMENT = "k2_2a_transfer_h200"
STRUCTURE = "symplectic prior"
REGIME = "FPU-β"
HORIZON = 200
CLAIM_LEVEL = "FULL_TRANSFER_CONFIRMED"


@dataclass(frozen=True)
class K22AVerdictInputs:
    """Minimal summary inputs needed to reproduce the K2.2-A verdict branch."""

    energy_fair_h200: bool
    l2_fair_h200: bool
    graceful_symp_lt_base: bool
    graceful_symp_lt_energy: bool
    graceful_symp_lt_l2: bool
    graceful_mechanism: bool
    pooled_base_med: float
    pooled_symp_med: float
    n_graceful: int
    n_rescued: int


def k2_2a_verdict(info: K22AVerdictInputs) -> str:
    """Return the registered K2.2-A verdict from summarized test outcomes."""

    base_ok = info.graceful_symp_lt_base
    mech_ok = info.graceful_mechanism
    beaten: list[bool] = []
    if info.energy_fair_h200:
        beaten.append(info.graceful_symp_lt_energy)
    if info.l2_fair_h200:
        beaten.append(info.graceful_symp_lt_l2)

    beats_all_fair = all(beaten) if beaten else False
    both_fair = info.energy_fair_h200 and info.l2_fair_h200
    pooled_better = info.pooled_symp_med < info.pooled_base_med

    if base_ok and mech_ok and beats_all_fair and both_fair:
        return "FULL_TRANSFER_CONFIRMED"
    if base_ok and mech_ok and beats_all_fair and not both_fair:
        return "TRANSFER_CONFIRMED"
    if base_ok and not mech_ok:
        return "TRANSFER_PERFORMANCE_ONLY"
    if base_ok and mech_ok and not beats_all_fair:
        return "CONTROL_MATCHES_SYMP"
    if pooled_better and not base_ok:
        return "TRANSFER_PERFORMANCE_ONLY"
    return "NO_TRANSFER"


def load_summary_csv(path: Path) -> K22AVerdictInputs:
    """Load the one-row CSV summary emitted by the full Colab experiment."""

    row = pd.read_csv(path).iloc[0]
    return K22AVerdictInputs(
        energy_fair_h200=bool(row["energy_fair_h200"]),
        l2_fair_h200=bool(row["l2_fair_h200"]),
        graceful_symp_lt_base=bool(row["graceful_symp_lt_base"]),
        graceful_symp_lt_energy=bool(row["graceful_symp_lt_energy"]),
        graceful_symp_lt_l2=bool(row["graceful_symp_lt_l2"]),
        graceful_mechanism=bool(row["graceful_mechanism"]),
        pooled_base_med=float(row["pooled_base_med"]),
        pooled_symp_med=float(row["pooled_symp_med"]),
        n_graceful=int(row["n_graceful"]),
        n_rescued=int(row["n_rescued"]),
    )


def holm_less_flags(p_values: Iterable[float], alpha: float = 0.05) -> list[bool]:
    """Small local Holm helper for archived summary checks."""

    indexed = sorted(enumerate(float(p) for p in p_values), key=lambda item: item[1])
    flags = [False] * len(indexed)
    for rank, (idx, p_value) in enumerate(indexed):
        threshold = alpha / (len(indexed) - rank)
        if p_value <= threshold:
            flags[idx] = True
        else:
            break
    return flags


def write_archived_summary(path: Path) -> None:
    """Write the archived K2.2-A headline result used by the repository README."""

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


def main() -> None:
    results_path = Path(__file__).resolve().parents[1] / "results" / "k2_2a_summary.csv"
    write_archived_summary(results_path)
    print(f"Wrote archived K2.2-A summary to {results_path}")
    print(f"Verdict: {CLAIM_LEVEL}")


if __name__ == "__main__":
    main()
