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

    G_tilted = np.array([[2.0, 0.4], [0.4, -1.0]])
    root = np.sqrt(0.8**2 + 8.0)
    tilted_samples = {
        "timelike": np.array([1.0, 0.0]),
        "right_lightlike": np.array([1.0, (0.8 + root) / 2.0]),
        "left_lightlike": np.array([1.0, (0.8 - root) / 2.0]),
        "spacelike": np.array([0.0, 1.0]),
    }
    tilted_form = QuadraticForm2D(G_tilted)

    print("\nCausal cone classification for non-diagonal G")
    print(G_tilted)
    for name, x in tilted_samples.items():
        print(f"{name:16s} K={tilted_form.K(x): .6f} class={causal_classification(G_tilted, x)}")


if __name__ == "__main__":
    main()
