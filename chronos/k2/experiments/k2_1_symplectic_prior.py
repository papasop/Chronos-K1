"""Experiment K2.1: Symplectic prior on FPU-β.

Canonical archive path for the K2.1 Colab experiment.

K2.1 tested whether a symplectic structural prior helps the neural baseline in
the validated graceful-fail regime:

```text
FPU-β, S=1, H=160
```

The experiment used a three-way design:

- baseline
- symplectic prior, lambda in {0.01, 0.1, 1.0}
- energy prior, lambda = 0.1
- L2 prior, lambda = 0.1

Primary metric:

```text
roll_MSE at H=160
```

Mechanism diagnostic:

```text
full ||J^T Omega J - Omega|| on the graceful-baseline subset
```

K2.1 verdict:

```text
SYMPLECTIC_CONFIRMED
```

Archive boundary:

K2.1 established the initial symplectic structure candidate result, but its
energy and L2 controls were later found degenerate. The preserved K2 claim is
therefore not K2.1 alone. K2.1-B repaired the controls, and K2.2-A confirmed
transfer at H=200.

Use the full Colab source from the K2.1 experiment note for executable
reproduction. This file anchors the canonical repository path and the
pre-registered claim semantics.
"""

from __future__ import annotations

from dataclasses import dataclass


EXPERIMENT = "k2_1_symplectic_prior"
REGIME = "FPU-β"
S_FIXED = 1
HORIZON = 160
PRIMARY_LAMBDA = 0.1
VERDICT = "SYMPLECTIC_CONFIRMED"


@dataclass(frozen=True)
class K21Summary:
    """Summary of the K2.1 pre-registered verdict conditions."""

    q1_symp_lt_base: bool
    q2_symp_lt_energy: bool
    q3_symp_lt_l2: bool
    q4_symp_mechanism: bool


def k2_1_verdict(summary: K21Summary) -> str:
    """Return the K2.1 pre-registered verdict from the four query outcomes."""

    if (
        summary.q1_symp_lt_base
        and summary.q2_symp_lt_energy
        and summary.q3_symp_lt_l2
        and summary.q4_symp_mechanism
    ):
        return "SYMPLECTIC_CONFIRMED"
    if summary.q1_symp_lt_base and not summary.q2_symp_lt_energy:
        return "ENERGY_ONLY"
    if summary.q1_symp_lt_base and not summary.q3_symp_lt_l2:
        return "GENERIC_REGULARIZATION"
    if summary.q1_symp_lt_base and not summary.q4_symp_mechanism:
        return "PERFORMANCE_ONLY"
    return "NO_PRIOR_HELPS"


def main() -> None:
    print(f"{EXPERIMENT}: {VERDICT}")
    print("Canonical archive path for K2.1. Use the full Colab source for reruns.")


if __name__ == "__main__":
    main()
