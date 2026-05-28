"""Demo: null cone for the 2D signed cost form."""

from __future__ import annotations

import numpy as np

from k1_manifold_core.geometry.lorentzian import QuadraticForm2D


def main() -> None:
    G = np.diag([1.0, -1.0])
    form = QuadraticForm2D(G)
    samples = {
        "future_timelike": np.array([2.0, 0.5]),
        "right_null": np.array([1.0, 1.0]),
        "left_null": np.array([1.0, -1.0]),
        "spacelike": np.array([0.5, 2.0]),
    }

    print("Light-cone classification for G = diag(1, -1)")
    for name, x in samples.items():
        K = form.K(x)
        if abs(K) < 1e-12:
            kind = "null"
        elif K > 0:
            kind = "positive-cost/timelike convention"
        else:
            kind = "spacelike convention"
        print(f"{name:16s} x={x} K={K: .6f} -> {kind}")


if __name__ == "__main__":
    main()
