"""Experiment 5 postmortem analysis.

Investigates the gap between Lorentzian point dynamics and neural learning
objectives without over-claiming against the underlying theory.

Preferred use:

    python exp5-diagnostic/experiment_5_postmortem_analysis.py \
        --csv exp5-diagnostic/results/exp5_extended_fixed_v2_results.csv \
        --out-dir exp5-diagnostic/results

When a CSV contains multiple `box` values, the script defaults to the primary
task `box == 2.0`. Pass `--box 32.0` for an OOD slice or `--box nan` to skip
box filtering.

If the raw CSV is unavailable, an explicitly marked demo dataset can be used:

    python exp5-diagnostic/experiment_5_postmortem_analysis.py \
        --use-embedded-demo-data

The embedded data are manually entered from supplied experiment notes. They are
useful for reproducing the narrative analysis, but they are not a substitute for
loading the actual per-seed CSV.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import wilcoxon


EMBEDDED_DEMO_DATA = {
    "lambda": [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2],
    "euclid_mean": [0.7652] * 8,
    "chronos_mean": [0.8154, 0.8214, 0.8277, 0.8135, 0.8149, 0.8183, 0.8143, 0.8095],
    "p_value": [0.1876, 0.0371, 0.0039, 0.0840, 0.1055, 0.0098, 0.0371, 0.0840],
    "n_seeds": [10] * 8,
}


@dataclass
class AnalysisSummary:
    data_source: str
    data_source_id: str
    box_filter: float | None
    n_lambda: int
    negative_count: int
    positive_count: int
    significant_count: int
    marginal_count: int
    mean_improvement: float
    min_improvement: float
    max_improvement: float
    trend_slope: float
    trend_r2: float
    trend_p_value: float
    has_per_seed_data: bool


LAMBDA_COL = "lambda"


def safe_wilcoxon(a: Iterable[float], b: Iterable[float]) -> float:
    a_arr = np.asarray(list(a), dtype=float)
    b_arr = np.asarray(list(b), dtype=float)
    mask = np.isfinite(a_arr) & np.isfinite(b_arr)
    if mask.sum() < 2:
        return float("nan")
    diff = a_arr[mask] - b_arr[mask]
    if np.allclose(diff, 0):
        return 1.0
    return float(wilcoxon(a_arr[mask], b_arr[mask]).pvalue)


def canonical_model_name(name: str) -> str:
    lower = str(name).strip().lower()
    if lower in {"euclid", "euclidean", "euclid_jepa", "euclidean latent predictor"}:
        return "euclid"
    if lower in {"chronos", "chronos_jepa", "chronos latent predictor"}:
        return "chronos"
    return lower


def filter_box(df: pd.DataFrame, box_filter: float | None) -> pd.DataFrame:
    if box_filter is None or "box" not in df.columns:
        return df
    filtered = df[np.isclose(df["box"].astype(float), float(box_filter))].copy()
    if filtered.empty:
        available = sorted(df["box"].dropna().unique().tolist())
        raise ValueError(f"No rows found for box={box_filter}. Available boxes: {available}")
    return filtered


def load_summary_from_raw_csv(
    path: Path,
    box_filter: float | None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    raw = pd.read_csv(path)
    if "model" not in raw.columns:
        raise ValueError("Raw CSV must contain a 'model' column.")
    if LAMBDA_COL not in raw.columns:
        raise ValueError("Raw CSV must contain a 'lambda' column.")

    metric = "violation_rate" if "violation_rate" in raw.columns else None
    if metric is None and "causal_violation_rate" in raw.columns:
        metric = "causal_violation_rate"
    if metric is None:
        raise ValueError("Raw CSV must contain 'violation_rate' or 'causal_violation_rate'.")

    raw = raw.copy()
    raw = filter_box(raw, box_filter)
    raw["model"] = raw["model"].map(canonical_model_name)
    raw = raw[raw["model"].isin(["euclid", "chronos"])]

    means = (
        raw.groupby([LAMBDA_COL, "model"], as_index=False)[metric]
        .mean()
        .pivot(index=LAMBDA_COL, columns="model", values=metric)
        .reset_index()
    )
    if not {"euclid", "chronos"}.issubset(means.columns):
        raise ValueError("Raw CSV must include both Euclid and Chronos rows.")

    summary = means.rename(columns={"euclid": "euclid_mean", "chronos": "chronos_mean"})

    n_by_lambda = raw.groupby(LAMBDA_COL).size().rename("n_records").reset_index()
    summary = summary.merge(n_by_lambda, on=LAMBDA_COL, how="left")
    summary["n_seeds"] = np.nan
    summary["p_value"] = np.nan

    per_seed = None
    if "seed" in raw.columns:
        pivot = raw.pivot_table(
            index=[LAMBDA_COL, "seed"],
            columns="model",
            values=metric,
            aggfunc="mean",
        ).reset_index()
        if {"euclid", "chronos"}.issubset(pivot.columns):
            pivot["improvement"] = (
                (pivot["euclid"] - pivot["chronos"]) / pivot["euclid"] * 100.0
            )
            p_values = []
            seed_counts = []
            for lambda_value, group in pivot.groupby(LAMBDA_COL):
                p_values.append((lambda_value, safe_wilcoxon(group["euclid"], group["chronos"])))
                seed_counts.append(
                    (lambda_value, int(group[["euclid", "chronos"]].dropna().shape[0]))
                )
            p_df = pd.DataFrame(p_values, columns=[LAMBDA_COL, "p_value"])
            n_df = pd.DataFrame(seed_counts, columns=[LAMBDA_COL, "n_seeds"])
            summary = summary.drop(columns=["p_value", "n_seeds"]).merge(
                p_df, on=LAMBDA_COL, how="left"
            )
            summary = summary.merge(n_df, on=LAMBDA_COL, how="left")
            per_seed = pivot

    return summary, per_seed


def load_summary_table(
    path: Path | None,
    use_embedded_demo_data: bool,
    box_filter: float | None,
) -> tuple[pd.DataFrame, pd.DataFrame | None, str, str]:
    if path is not None:
        df = pd.read_csv(path)
        if {LAMBDA_COL, "model"}.issubset(df.columns):
            summary, per_seed = load_summary_from_raw_csv(path, box_filter)
            source = f"raw CSV: {path}"
            source_id = "raw_csv"
        elif {LAMBDA_COL, "euclid_mean", "chronos_mean"}.issubset(df.columns):
            summary = filter_box(df.copy(), box_filter)
            per_seed = None
            source = f"summary CSV: {path}"
            source_id = "summary_csv"
        else:
            raise ValueError(
                "CSV must be either raw rows with lambda/model/violation_rate "
                "or summary rows with lambda/euclid_mean/chronos_mean."
            )
    elif use_embedded_demo_data:
        summary = pd.DataFrame(EMBEDDED_DEMO_DATA)
        per_seed = None
        source = "embedded demo data manually entered from supplied experiment notes"
        source_id = "embedded_demo_data"
    else:
        raise ValueError("Provide --csv or explicitly pass --use-embedded-demo-data.")

    summary = summary.copy()
    summary["improvement"] = (
        (summary["euclid_mean"] - summary["chronos_mean"]) / summary["euclid_mean"] * 100.0
    )
    if "p_value" not in summary.columns:
        summary["p_value"] = np.nan
    if "n_seeds" not in summary.columns:
        summary["n_seeds"] = np.nan
    summary = summary.sort_values(LAMBDA_COL).reset_index(drop=True)
    return summary, per_seed, source, source_id


def build_summary(
    summary: pd.DataFrame,
    per_seed: pd.DataFrame | None,
    source: str,
    source_id: str,
    box_filter: float | None,
) -> AnalysisSummary:
    slope, intercept, r_value, p_trend, _ = stats.linregress(
        summary[LAMBDA_COL].to_numpy(dtype=float),
        summary["improvement"].to_numpy(dtype=float),
    )
    p_values = summary["p_value"].to_numpy(dtype=float)
    significant = np.isfinite(p_values) & (p_values < 0.05)
    marginal = np.isfinite(p_values) & (p_values >= 0.05) & (p_values < 0.10)
    return AnalysisSummary(
        data_source=source,
        data_source_id=source_id,
        box_filter=box_filter,
        n_lambda=int(summary.shape[0]),
        negative_count=int((summary["improvement"] < 0).sum()),
        positive_count=int((summary["improvement"] > 0).sum()),
        significant_count=int(significant.sum()),
        marginal_count=int(marginal.sum()),
        mean_improvement=float(summary["improvement"].mean()),
        min_improvement=float(summary["improvement"].min()),
        max_improvement=float(summary["improvement"].max()),
        trend_slope=float(slope),
        trend_r2=float(r_value**2),
        trend_p_value=float(p_trend),
        has_per_seed_data=per_seed is not None,
    )


def significance_label(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "not available"
    if p_value < 0.05:
        return "p<0.05"
    if p_value < 0.10:
        return "p<0.10"
    return "not significant"


def write_report(
    summary: pd.DataFrame,
    analysis: AnalysisSummary,
    per_seed: pd.DataFrame | None,
    out_dir: Path,
) -> Path:
    report_path = out_dir / "experiment_5_postmortem_analysis.md"
    lines: list[str] = []
    lines.append("# Experiment 5 Postmortem")
    lines.append("")
    lines.append("Investigating the gap between Lorentzian point dynamics and neural learning objectives.")
    lines.append("")
    lines.append("## Data Source")
    lines.append("")
    lines.append(analysis.data_source)
    if analysis.box_filter is not None:
        lines.append("")
        lines.append(f"Primary-task filter: `box == {analysis.box_filter}`.")
    if analysis.data_source.startswith("embedded demo"):
        lines.append("")
        lines.append(
            "Important: these values were manually entered from supplied notes. "
            "Use a raw CSV for production analysis."
        )
    lines.append("")
    lines.append("## Summary Table")
    lines.append("")
    lines.append("| lambda | Euclid | Chronos | improvement | p-value | label |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | --- |")
    for _, row in summary.iterrows():
        p_value = float(row["p_value"]) if "p_value" in row else float("nan")
        p_text = f"{p_value:.4f}" if np.isfinite(p_value) else "n/a"
        lines.append(
            f"| {row[LAMBDA_COL]:.4g} | {row['euclid_mean']:.4f} | "
            f"{row['chronos_mean']:.4f} | {row['improvement']:.2f}% | "
            f"{p_text} | {significance_label(p_value)} |"
        )

    lines.append("")
    lines.append("## Objective Observations")
    lines.append("")
    lines.append(
        f"- Negative improvement appears in {analysis.negative_count}/{analysis.n_lambda} tested lambda values."
    )
    lines.append(
        f"- Positive improvement appears in {analysis.positive_count}/{analysis.n_lambda} tested lambda values."
    )
    lines.append(
        f"- Several settings show p<0.05: {analysis.significant_count}/{analysis.n_lambda}."
    )
    lines.append(
        f"- Marginal settings with 0.05<=p<0.10: {analysis.marginal_count}/{analysis.n_lambda}."
    )
    lines.append(
        f"- Linear trend R^2={analysis.trend_r2:.3f}, p={analysis.trend_p_value:.4f}."
    )
    lines.append("")
    if per_seed is None:
        lines.append("Per-seed distribution analysis is not shown because raw per-seed rows were not available.")
    else:
        lines.append("Per-seed rows were available; paired Wilcoxon p-values were computed from seed-paired rows.")

    lines.append("")
    lines.append("## Interpretation Boundary")
    lines.append("")
    lines.append(
        "Experiment 5 can support a claim about the current ChronosJEPA implementation. "
        "It does not test or falsify the point-level GCD theory itself."
    )
    lines.append("")
    lines.append("Can say:")
    lines.append("")
    lines.append("- The current ChronosJEPA implementation under this benchmark does not improve causal violation.")
    lines.append("- In the analyzed data, Chronos has higher violation than Euclid across the tested lambda values.")
    lines.append("- Several lambda settings show p<0.05 degradation, but not all settings do.")
    lines.append("")
    lines.append("Cannot say:")
    lines.append("")
    lines.append("- K=1 can never work in neural networks.")
    lines.append("- GCD theory is wrong.")
    lines.append("- All Lorentzian or geometric constraints are harmful.")
    lines.append("")
    lines.append("Recommended framing:")
    lines.append("")
    lines.append(
        "Experiment 5 is a postmortem on the gap between Lorentzian point dynamics "
        "and the current neural learning objective."
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def plot_results(
    summary: pd.DataFrame,
    analysis: AnalysisSummary,
    per_seed: pd.DataFrame | None,
    out_dir: Path,
) -> Path:
    plot_path = out_dir / "experiment_5_postmortem_analysis.png"
    if per_seed is None:
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.28)
    else:
        fig = plt.figure(figsize=(16, 13))
        gs = fig.add_gridspec(3, 2, hspace=0.42, wspace=0.28)

    colors = [
        "tab:red" if np.isfinite(p) and p < 0.05 else "tab:orange" if np.isfinite(p) and p < 0.10 else "tab:gray"
        for p in summary["p_value"]
    ]

    ax1 = fig.add_subplot(gs[0, :])
    ax1.bar(range(len(summary)), summary["improvement"], color=colors, alpha=0.75, edgecolor="black")
    ax1.axhline(0, color="black", linewidth=1)
    ax1.set_xticks(range(len(summary)))
    ax1.set_xticklabels([f"{x:.4g}" for x in summary[LAMBDA_COL]], rotation=45, ha="right")
    ax1.set_ylabel("Improvement (%)")
    ax1.set_xlabel("lambda")
    ax1.set_title("Experiment 5 Postmortem: Chronos vs Euclid")
    ax1.grid(True, alpha=0.25, axis="y")

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.scatter(summary[LAMBDA_COL], summary["improvement"], s=90, edgecolor="black", alpha=0.75)
    x_line = np.linspace(summary[LAMBDA_COL].min(), summary[LAMBDA_COL].max(), 100)
    y_line = analysis.mean_improvement + analysis.trend_slope * (x_line - summary[LAMBDA_COL].mean())
    ax2.plot(x_line, y_line, "r--", label=f"R^2={analysis.trend_r2:.3f}, p={analysis.trend_p_value:.3f}")
    ax2.axhline(0, color="black", linewidth=1)
    ax2.set_xlabel("lambda")
    ax2.set_ylabel("Improvement (%)")
    ax2.set_title("Trend Check")
    ax2.legend()
    ax2.grid(True, alpha=0.25)

    ax3 = fig.add_subplot(gs[1, 1])
    width = 0.35
    x = np.arange(len(summary))
    ax3.bar(x - width / 2, summary["euclid_mean"], width, label="Euclid", alpha=0.75)
    ax3.bar(x + width / 2, summary["chronos_mean"], width, label="Chronos", alpha=0.75)
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"{val:.4g}" for val in summary[LAMBDA_COL]], rotation=45, ha="right")
    ax3.set_ylabel("Violation rate")
    ax3.set_xlabel("lambda")
    ax3.set_title("Absolute Violation Rates")
    ax3.legend()
    ax3.grid(True, alpha=0.25, axis="y")

    if per_seed is not None:
        ax4 = fig.add_subplot(gs[2, :])
        for lambda_value, group in per_seed.groupby(LAMBDA_COL):
            ax4.scatter(
                np.full(group.shape[0], float(lambda_value)),
                group["improvement"],
                alpha=0.65,
                edgecolor="black",
                linewidth=0.5,
            )
        ax4.axhline(0, color="black", linewidth=1)
        ax4.set_xlabel("lambda")
        ax4.set_ylabel("Per-seed improvement (%)")
        ax4.set_title("Per-Seed Paired Improvement (raw CSV only)")
        ax4.grid(True, alpha=0.25)

    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, help="Raw or summary CSV to analyze.")
    parser.add_argument("--out-dir", type=Path, default=Path("exp5-diagnostic/results"))
    parser.add_argument(
        "--box",
        type=float,
        default=2.0,
        help="Primary-task box value to analyze when CSV contains a box column. Use --box nan to skip filtering.",
    )
    parser.add_argument(
        "--use-embedded-demo-data",
        action="store_true",
        help="Use manually entered demo data when a CSV is unavailable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    box_filter = None if np.isnan(args.box) else args.box

    summary, per_seed, source, source_id = load_summary_table(
        args.csv,
        args.use_embedded_demo_data,
        box_filter,
    )
    analysis = build_summary(summary, per_seed, source, source_id, box_filter)

    summary_path = args.out_dir / "experiment_5_postmortem_summary.csv"
    json_path = args.out_dir / "experiment_5_postmortem_summary.json"
    summary.to_csv(summary_path, index=False)
    json_path.write_text(json.dumps(asdict(analysis), indent=2), encoding="utf-8")
    report_path = write_report(summary, analysis, per_seed, args.out_dir)
    plot_path = plot_results(summary, analysis, per_seed, args.out_dir)

    print("Experiment 5 Postmortem")
    print("=" * 80)
    print(f"Data source: {analysis.data_source}")
    print(f"Data source id: {analysis.data_source_id}")
    print(f"Box filter: {analysis.box_filter}")
    print(f"Negative improvement: {analysis.negative_count}/{analysis.n_lambda}")
    print(f"p<0.05 settings: {analysis.significant_count}/{analysis.n_lambda}")
    print(f"Mean improvement: {analysis.mean_improvement:.2f}%")
    print(f"Trend R^2: {analysis.trend_r2:.3f}, p={analysis.trend_p_value:.4f}")
    print()
    print("Interpretation: current ChronosJEPA implementation result, not a falsification of GCD theory.")
    print()
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {json_path}")
    print(f"Wrote: {report_path}")
    print(f"Wrote: {plot_path}")


if __name__ == "__main__":
    main()
