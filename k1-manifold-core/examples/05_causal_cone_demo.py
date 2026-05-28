"""Demo: timelike/lightlike/spacelike classification by x.T @ G @ x."""

from __future__ import annotations

import numpy as np

from k1_manifold_core.geometry.lorentzian import QuadraticForm2D, causal_classification


def main() -> None:
    G = np.diag([1.0, -1.0])
    form = QuadraticForm2D(G)
    samples = {
        "timelike": np.array([2.0, 0.5]),
        "right_lightlike": np.array([1.0, 1.0]),
        "left_lightlike": np.array([1.0, -1.0]),
        "spacelike": np.array([0.5, 2.0]),
    }

    print("Causal cone classification for G = diag(1, -1)")
    for name, x in samples.items():
        print(f"{name:16s} K={form.K(x): .6f} class={causal_classification(G, x)}")


if __name__ == "__main__":
    main()
