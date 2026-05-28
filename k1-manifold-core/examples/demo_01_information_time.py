"""Plot dt_info = dPhi / H as H varies."""

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

from k1_manifold_core.axioms.information_time import information_time


def main() -> None:
    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "demo_01_information_time.png"

    d_phi = 1.0
    H = np.linspace(0.1, 5.0, 400)
    dt_info = information_time(d_phi, H)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(H, dt_info, color="#1f77b4", linewidth=2)
    ax.set_title("Information time")
    ax.set_xlabel("H")
    ax.set_ylabel("dt_info = dPhi / H")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    print(f"dPhi = {d_phi}")
    print(f"H range = [{H[0]:.3f}, {H[-1]:.3f}]")
    print(f"dt_info range = [{dt_info[-1]:.6f}, {dt_info[0]:.6f}]")
    print(f"saved figure = {output_path}")


if __name__ == "__main__":
    main()
