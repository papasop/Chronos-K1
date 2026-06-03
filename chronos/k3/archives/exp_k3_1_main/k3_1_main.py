"""Archive entrypoint for K3.1 MAIN.

This file preserves the archived K3.1 result package. It regenerates the
headline result CSV and summary CSV for the repository archive. It does not
rerun the full GPU Colab training sweep.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


EXPERIMENT = "k3_1_main"
VERDICT = "NO_EFFECT"
HORIZON = 160
N_SEEDS = 30
NON_CLAIM = "not integer topological-charge certification"


@dataclass(frozen=True)
class GroupResult:
    group: str
    prior: str
    lambda_value: float
    ratio: float | None
    roll_med: float
    wdens_med: float
    wind_err: float
    inc_norm: float
    gate_pass: bool


GROUPS: tuple[GroupResult, ...] = (
    GroupResult("baseline", "none", 0.0, None, 0.0068, 0.0379, 0.20, 0.0107, True),
    GroupResult("density_l0.1", "winding_density_continuity", 0.1, 0.51, 0.0059, 0.0377, 0.10, 0.0103, True),
    GroupResult("density_l1.0", "winding_density_continuity", 1.0, 1.08, 0.0087, 0.0419, 0.07, 0.0106, True),
    GroupResult("offtarget_l0.01", "off_target_continuity", 0.01, 0.83, 0.0067, 0.0383, 0.20, 0.0103, True),
    GroupResult("offtarget_l0.1", "off_target_continuity", 0.1, 5.7, 0.0039, 0.0251, 0.07, 0.0077, True),
    GroupResult("smoothness_l0.1", "gradient_smoothness", 0.1, 0.50, 0.0059, 0.0343, 0.17, 0.0104, True),
    GroupResult("increment_l2_l0.1", "increment_l2", 0.1, 0.36, 0.0065, 0.0364, 0.20, 0.0106, True),
)


def group_rows() -> list[dict[str, object]]:
    return [
        {
            "group": group.group,
            "prior": group.prior,
            "lambda": group.lambda_value,
            "ratio": group.ratio,
            "roll_med": group.roll_med,
            "wdens_med": group.wdens_med,
            "wind_err": group.wind_err,
            "inc_norm": group.inc_norm,
            "gate_pass": group.gate_pass,
        }
        for group in GROUPS
    ]


def summary_row() -> dict[str, object]:
    return {
        "experiment": EXPERIMENT,
        "verdict": VERDICT,
        "horizon": HORIZON,
        "n_seeds": N_SEEDS,
        "density_l1_vs_baseline": False,
        "density_l1_vs_offtarget_matched": False,
        "density_l01_vs_smoothness": False,
        "density_l01_vs_increment_l2": False,
        "mechanism": False,
        "wdens_reduction": -0.107,
        "wind_intact": 0.93,
        "non_claim": NON_CLAIM,
    }


def main() -> None:
    root = Path(__file__).resolve().parent
    results = root / "results"
    results.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(group_rows()).to_csv(results / "k3_1_main_results.csv", index=False)
    pd.DataFrame([summary_row()]).to_csv(results / "k3_1_main_summary.csv", index=False)

    print("K3.1 MAIN archived package")
    print(f"verdict: {VERDICT}")
    print("wdens_err reduction: -10.7%")
    print(f"non-claim: {NON_CLAIM}")


if __name__ == "__main__":
    main()
