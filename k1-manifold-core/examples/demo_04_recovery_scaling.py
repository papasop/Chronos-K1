"""Demo: rank-one null flow and inverse-square recovery scaling."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from k1_manifold_core.experiments.null_flow import (
    ideal_leaf_probability,
    integrate_null_flow,
    leaf_box_probability,
    leaf_coordinate,
    null_flow_matrix,
    recovery_scaling_fit,
    recovery_time,
)


def main() -> None:
    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_png = output_dir / "demo_04_recovery_scaling.png"
    output_svg = output_dir / "demo_04_recovery_scaling.svg"

    A_c = null_flow_matrix()
    x0 = np.array([0.75, -0.25])
    trajectory = integrate_null_flow(x0, dt=1e-3, steps=5000)
    c_values = np.array([leaf_coordinate(x) for x in trajectory])
    c_drift = float(np.max(np.abs(c_values - c_values[0])))

    radius = 0.1
    estimated_probability = leaf_box_probability(radius)
    ideal_probability = ideal_leaf_probability(radius)

    epsilons = np.array([0.05, 0.075, 0.1, 0.15, 0.2])
    times = np.array([recovery_time(epsilon) for epsilon in epsilons])
    slope, r_squared = recovery_scaling_fit(epsilons)
    probability_ratio = ideal_probability / (4.0 * radius * radius)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(epsilons, times, "o-", label=f"slope ≈ {slope:.3f}")
    ax.set_xlabel("epsilon")
    ax.set_ylabel("recovery time")
    ax.set_title("Null-flow recovery scaling")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_png, dpi=300)
    fig.savefig(output_svg)
    plt.close(fig)

    print("===== Null Flow Recovery Scaling Demo =====")
    print(f"rank(A_c) = {np.linalg.matrix_rank(A_c)}")
    print(f"max |c(t)-c(0)| = {c_drift:.3e}")
    print(f"leaf probability estimate = {estimated_probability:.6f}")
    print(f"ideal 4c^2 probability = {ideal_probability:.6f}")
    print(f"ideal probability ratio = {probability_ratio:.4f}")
    print(f"log-log recovery slope = {slope:.6f}")
    print(f"log-log fit R^2 = {r_squared:.6f}")
    print(f"saved png = {output_png}")
    print(f"saved svg = {output_svg}")


if __name__ == "__main__":
    main()
