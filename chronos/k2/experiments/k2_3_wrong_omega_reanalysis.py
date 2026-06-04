"""K2.3 wrong-Ω specificity reanalysis.

Reads an existing `k2_3_main_results.csv` and re-derives the non-degeneracy-
aware wrong-Ω specificity verdict. This script performs no retraining.

Default Colab result path:

```text
/content/exp_k2_3_main/k2_3_main_results.csv
```

Repository archive summary:

```text
chronos/k2/results/k2_3_reanalysis_summary.csv
```
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ALPHA = 0.05
MECH_REL_THRESHOLD = 0.20
COLLAPSE_FRAC = 0.50
REF_CEIL = 0.01
GROUPS = ["baseline", "symplectic_true", "symp_shuffled", "symp_randantisym"]
OFFTARGETS = ["symp_shuffled", "symp_randantisym"]


def paired_less(df: pd.DataFrame, a: str, b: str, metric: str = "roll_mse") -> tuple[float, bool, float]:
    x = df[df["group"] == a].sort_values("seed")[metric].values
    y = df[df["group"] == b].sort_values("seed")[metric].values
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 5:
        return np.nan, False, np.nan
    if np.allclose(x - y, 0):
        return 1.0, False, 0.0
    try:
        _, p_value = wilcoxon(x, y, alternative="less")
    except Exception:
        p_value = np.nan
    median_x, median_y = np.median(x), np.median(y)
    reduction = (median_y - median_x) / median_y if median_y > 0 else np.nan
    return (float(p_value) if p_value == p_value else np.nan), bool(median_x < median_y), float(reduction)


def holm(p_values: dict[str, float]) -> dict[str, float]:
    items = [(key, value) for key, value in p_values.items() if value == value]
    out: dict[str, float] = {}
    if not items:
        return {key: np.nan for key in p_values}
    keys = [key for key, _ in items]
    values = [value for _, value in items]
    try:
        from statsmodels.stats.multitest import multipletests

        _, corrected, _, _ = multipletests(values, alpha=ALPHA, method="holm")
        for key, value in zip(keys, corrected):
            out[key] = float(value)
    except Exception:
        order = np.argsort(values)
        count = len(values)
        adjusted = np.empty(count)
        running = 0.0
        for rank, idx in enumerate(order):
            value = min(1.0, values[idx] * (count - rank))
            running = max(running, value)
            adjusted[idx] = running
        for key, value in zip(keys, adjusted):
            out[key] = float(value)
    for key, value in p_values.items():
        if value != value:
            out[key] = np.nan
    return out


def reanalyze(input_csv: Path) -> dict[str, object]:
    df = pd.read_csv(input_csv)
    med = {group: df[df["group"] == group]["roll_mse"].median() for group in GROUPS}
    symp_err = {group: df[df["group"] == group]["symp_err_true"].median() for group in GROUPS}
    inc = {group: df[df["group"] == group]["inc_norm"].median() for group in GROUPS}
    ref = {group: df[df["group"] == group]["ref_mse"].median() for group in GROUPS}
    base_inc = inc["baseline"]

    degenerate = {
        group: group != "baseline" and ((inc[group] / base_inc < COLLAPSE_FRAC) or (ref[group] > REF_CEIL))
        for group in GROUPS
    }

    raw: dict[str, float] = {}
    lt: dict[str, bool] = {}
    rel: dict[str, float] = {}
    comparisons = {
        "Q1_true_vs_baseline": ("symplectic_true", "baseline"),
        "Q2_true_vs_shuffled": ("symplectic_true", "symp_shuffled"),
        "Q3_true_vs_randantisym": ("symplectic_true", "symp_randantisym"),
    }
    for key, (a, b) in comparisons.items():
        p_value, is_less, reduction = paired_less(df, a, b)
        raw[key] = p_value
        lt[key] = is_less
        rel[key] = reduction
    corrected = holm(raw)
    q1 = lt["Q1_true_vs_baseline"] and corrected["Q1_true_vs_baseline"] < ALPHA
    q2 = lt["Q2_true_vs_shuffled"] and corrected["Q2_true_vs_shuffled"] < ALPHA
    q3 = lt["Q3_true_vs_randantisym"] and corrected["Q3_true_vs_randantisym"] < ALPHA

    red_base = (symp_err["baseline"] - symp_err["symplectic_true"]) / symp_err["baseline"]
    mech_present = np.isfinite(red_base) and red_base >= MECH_REL_THRESHOLD
    interpretable = [group for group in OFFTARGETS if not degenerate[group]]

    norm_symp_err = {group: symp_err[group] / (inc[group] + 1e-9) for group in GROUPS}
    symp_true_lowest_normalized = (
        all(norm_symp_err["symplectic_true"] < norm_symp_err[group] for group in OFFTARGETS)
        and norm_symp_err["symplectic_true"] < norm_symp_err["baseline"]
    )

    if not interpretable:
        mech_specific = mech_present and symp_true_lowest_normalized
    else:
        direct_ok = all(
            (symp_err[group] - symp_err["symplectic_true"]) / symp_err[group] >= MECH_REL_THRESHOLD
            for group in interpretable
            if symp_err[group] > 0
        )
        mech_specific = mech_present and direct_ok and symp_true_lowest_normalized

    if q1 and q2 and q3 and mech_specific:
        verdict = "OMEGA_SPECIFICITY_CONFIRMED_NONDEGEN_AWARE"
    elif q2 and q3 and not q1:
        verdict = "OFFTARGET_HARMS_ONLY"
    else:
        verdict = "NO_SPECIFICITY_OR_OTHER"

    return {
        "experiment": "k2_3_reanalysis",
        "verdict": verdict,
        "q1": bool(q1),
        "q2": bool(q2),
        "q3": bool(q3),
        "mech_present": bool(mech_present),
        "mech_specific": bool(mech_specific),
        "red_base": float(red_base),
        "norm_se_true": float(norm_symp_err["symplectic_true"]),
        "norm_se_baseline": float(norm_symp_err["baseline"]),
        "norm_se_shuffled": float(norm_symp_err["symp_shuffled"]),
        "norm_se_randantisym": float(norm_symp_err["symp_randantisym"]),
        "symp_true_lowest_normalized": bool(symp_true_lowest_normalized),
        "shuffled_degenerate": bool(degenerate["symp_shuffled"]),
        "randantisym_degenerate": bool(degenerate["symp_randantisym"]),
        "p_Q1": corrected["Q1_true_vs_baseline"],
        "p_Q2": corrected["Q2_true_vs_shuffled"],
        "p_Q3": corrected["Q3_true_vs_randantisym"],
        "baseline_roll": float(med["baseline"]),
        "true_roll": float(med["symplectic_true"]),
        "shuffled_roll": float(med["symp_shuffled"]),
        "randantisym_roll": float(med["symp_randantisym"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/content/exp_k2_3_main/k2_3_main_results.csv"),
        help="Existing K2.3 main results CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results" / "k2_3_reanalysis_summary.csv",
        help="Output summary CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = reanalyze(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([summary]).to_csv(args.output, index=False)
    print(f"K2.3 reanalysis verdict: {summary['verdict']}")
    print(f"Saved summary to {args.output}")


if __name__ == "__main__":
    main()
