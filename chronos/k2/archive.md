# K2 Archive

K2 is the first VPSL-certified physical-structure stage.

## Structure

Symplectic prior on FPU-beta.

## Status

```text
FULL_TRANSFER_CONFIRMED
```

## What K2 Established

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

## Preserved Materials

- `chronos/k2/experiments/k2_0_fpu_regime.py`
- `chronos/k2/experiments/k2_0b_refine_window.py`
- `chronos/k2/experiments/k2_1b_repair_controls.py`
- `chronos/k2/experiments/k2_2a_transfer_h200.py`
- `chronos/k2/results/k2_2a_summary.csv`
