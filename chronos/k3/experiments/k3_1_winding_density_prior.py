"""Experiment K3.1 MAIN: winding-density continuity prior test.

System:
    Periodic Sine-Gordon, angle representation ``[sin(u), cos(u), u_t]``.

Stage:
    K3 Stage 2, prior test after K3.0-D regime validation.

Archived verdict:
    NO_EFFECT

Scientific question:
    Does a winding-density continuity prior improve neural rollout in the
    K3.0-D validated Sine-Gordon regime, and does it do so because of
    topological alignment rather than generic continuity / smoothness?

Matched-ratio primary comparison:
    density_l1.0, prior/roll ratio approximately 1.08
    versus
    offtarget_l0.01, prior/roll ratio approximately 0.83

Honest claim boundary:
    The target is winding-density / local topological structure, not integer
    topological-charge certification. K3.1 is archived as a negative result:
    the density prior did not beat baseline, did not beat the matched
    off-target control, and did not reduce the winding-density diagnostic.

This repository file preserves the K3.1 archived result and verdict logic.
The full GPU Colab training run is not executed by CI.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


EXPERIMENT = "k3_1_main"
SYSTEM = "periodic Sine-Gordon"
STRUCTURE = "winding-density continuity prior"
REGIME = "FPU-style VPSL regime discipline; Sine-Gordon angle representation"
HORIZON = 160
CLAIM_LEVEL = "NO_EFFECT"


@dataclass(frozen=True)
class GroupSummary:
    group: str
    roll_med: float
    wdens_med: float
    wind_err: float
    inc_norm: float
    gate_pass: bool


@dataclass(frozen=True)
class K31ArchivedResult:
    verdict: str = CLAIM_LEVEL
    density_l1_vs_baseline_pass: bool = False
    density_l1_vs_offtarget_matched_pass: bool = False
    density_l01_vs_smoothness_pass: bool = False
    density_l01_vs_increment_l2_pass: bool = False
    mechanism_pass: bool = False
    wdens_reduction: float = -0.107
    wind_intact: float = 0.93
    matched_density_ratio: float = 1.08
    matched_offtarget_ratio: float = 0.83


GROUPS: tuple[GroupSummary, ...] = (
    GroupSummary("baseline", 0.0068, 0.0379, 0.20, 0.0107, True),
    GroupSummary("density_l0.1", 0.0059, 0.0377, 0.10, 0.0103, True),
    GroupSummary("density_l1.0", 0.0087, 0.0419, 0.07, 0.0106, True),
    GroupSummary("offtarget_l0.01", 0.0067, 0.0383, 0.20, 0.0103, True),
    GroupSummary("offtarget_l0.1", 0.0039, 0.0251, 0.07, 0.0077, True),
    GroupSummary("smoothness_l0.1", 0.0059, 0.0343, 0.17, 0.0104, True),
    GroupSummary("increment_l2_l0.1", 0.0065, 0.0364, 0.20, 0.0106, True),
)


def archived_result() -> K31ArchivedResult:
    """Return the archived K3.1 verdict summary."""

    return K31ArchivedResult()


def write_group_summary(path: Path) -> None:
    """Write the archived per-group headline medians."""

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([group.__dict__ for group in GROUPS]).to_csv(path, index=False)


def write_verdict_summary(path: Path) -> None:
    """Write the archived K3.1 verdict summary."""

    path.parent.mkdir(parents=True, exist_ok=True)
    result = archived_result()
    pd.DataFrame(
        [
            {
                "experiment": EXPERIMENT,
                "system": SYSTEM,
                "structure": STRUCTURE,
                "horizon": HORIZON,
                "claim_level": result.verdict,
                "primary_comparison": "density_l1.0<offtarget_l0.01 matched-ratio",
                "density_l1_vs_baseline": result.density_l1_vs_baseline_pass,
                "density_l1_vs_offtarget_matched": result.density_l1_vs_offtarget_matched_pass,
                "density_l01_vs_smoothness": result.density_l01_vs_smoothness_pass,
                "density_l01_vs_increment_l2": result.density_l01_vs_increment_l2_pass,
                "mechanism": result.mechanism_pass,
                "wdens_reduction": result.wdens_reduction,
                "wind_intact": result.wind_intact,
                "non_claim": "not integer topological-charge certification",
            }
        ]
    ).to_csv(path, index=False)


def verdict_from_summary(result: K31ArchivedResult) -> str:
    """Reproduce the archived verdict branch from boolean summary flags."""

    if not result.density_l1_vs_baseline_pass:
        return "NO_EFFECT"
    if (
        result.density_l1_vs_baseline_pass
        and result.density_l1_vs_offtarget_matched_pass
        and result.density_l01_vs_smoothness_pass
        and result.density_l01_vs_increment_l2_pass
        and result.mechanism_pass
    ):
        return "WINDING_DENSITY_CONFIRMED"
    if result.density_l1_vs_baseline_pass and not result.density_l1_vs_offtarget_matched_pass:
        return "DENSITY_HELP_BUT_NOT_OFFTARGET"
    return "SMOOTHNESS_ONLY"


def main() -> None:
    results_dir = Path(__file__).resolve().parents[1] / "results"
    write_group_summary(results_dir / "k3_1_group_summary.csv")
    write_verdict_summary(results_dir / "k3_1_summary.csv")
    result = archived_result()
    verdict = verdict_from_summary(result)
    print(f"K3.1 archived verdict: {verdict}")
    print(f"winding-density reduction: {result.wdens_reduction:.1%}")
    print("Honest boundary: winding-density prior only; not integer-charge certification.")


if __name__ == "__main__":
    main()
