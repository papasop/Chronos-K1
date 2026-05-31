"""Plot timelike, lightlike, and spacelike regions for G = diag(1, -1)."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    output_png = output_dir / "demo_02_causal_cone.png"
    output_svg = output_dir / "demo_02_causal_cone.svg"

    t = np.linspace(-2.0, 2.0, 401)
    x = np.linspace(-2.0, 2.0, 401)
    T, X = np.meshgrid(t, x)
    K = T**2 - X**2

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.contourf(T, X, K, levels=[-10, -1e-9, 1e-9, 10], colors=["#d9ecff", "#ffffff", "#ffe0cc"], alpha=0.9)
    ax.plot(t, t, color="black", linewidth=1.5, label="lightlike")
    ax.plot(t, -t, color="black", linewidth=1.5)
    ax.text(0.15, 1.35, "spacelike", color="#1f77b4")
    ax.text(1.05, 0.2, "timelike", color="#d55e00")
    ax.set_title("Causal cone: K=t^2-x^2")
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_png, dpi=300)
    fig.savefig(output_svg)
    plt.close(fig)

    timelike_count = int(np.sum(K > 1e-9))
    lightlike_count = int(np.sum(np.abs(K) <= 1e-9))
    spacelike_count = int(np.sum(K < -1e-9))

    print("G = diag(1, -1)")
    print("timelike: K > 0")
    print("lightlike: K = 0")
    print("spacelike: K < 0")
    print(f"Timelike : {timelike_count}")
    print(f"Lightlike: {lightlike_count}")
    print(f"Spacelike: {spacelike_count}")
    print(f"saved png = {output_png}")
    print(f"saved svg = {output_svg}")


if __name__ == "__main__":
    main()
