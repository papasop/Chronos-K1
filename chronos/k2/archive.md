# K2 Archive

K2 is the first VPSL-certified physical-structure stage.

## Structure

Symplectic prior on FPU-beta.

## Status

```text
FULL_TRANSFER_CONFIRMED
```

## What K2 Established

K2 tells a three-step VPSL story.

| Stage | Result | Claim Role |
| --- | --- | --- |
| K2.0-B | Valid FPU-beta window refined; H=200 identified as stress transfer horizon | Regime / window validation |
| K2.1-B | Fair non-degenerate controls repaired at H=160 | `CONTROLS_REPAIRED` |
| K2.2-A | Symplectic transfer confirmed at H=200 | `FULL_TRANSFER_CONFIRMED` |

K2.2-A confirms transfer at H=200 on the graceful-baseline subset:

- `symplectic < baseline`
- `symplectic < fair energy`
- `symplectic < fair L2`
- mechanism transfer confirmed

The mechanism diagnostic is full symplectic Jacobian error:

```text
||J^T Omega J - Omega||
```

## Boundary

K2 currently supports a narrow claim: the symplectic prior transfers on the
validated FPU-beta regime under the tested controls. It does not yet claim
generalization beyond FPU-beta.

Not claimed:

- all Hamiltonian systems benefit from the tested symplectic prior
- all systems benefit from symplectic constraints
- H=240 or beyond is already certified
- pooled rescue is sufficient for a structure claim

## Numbering Discipline

K2 uses K-stage numbering rather than the historical ExpXX numbering:

```text
K2.0   regime validation
K2.0-B window refinement
K2.1   first symplectic comparison
K2.1-B fair-control repair
K2.2-A transfer test at H=200
```

Historical ExpXX labels remain in K1 archive material only.

## Preserved Materials

- `chronos/k2/experiments/k2_0_fpu_regime.py`
- `chronos/k2/experiments/k2_0b_refine_window.py`
- `chronos/k2/experiments/k2_1b_repair_controls.py`
- `chronos/k2/experiments/k2_2a_transfer_h200.py`
- `chronos/k2/results/k2_2a_summary.csv`
