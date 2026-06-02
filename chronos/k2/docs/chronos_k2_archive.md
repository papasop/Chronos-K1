# Chronos K2 Archive

K2 is the first VPSL-certified physical-structure milestone.

## Structure

Symplectic prior on FPU-beta.

## Current Verdict

```text
FULL_TRANSFER_CONFIRMED
```

## Evidence Summary

K2.2-A tests transfer at H=200 after K2.1-B repaired the controls at H=160.

The confirmed H=200 result requires the primary claim to hold on the
graceful-baseline subset, not only in pooled results.

Confirmed comparisons:

- `symplectic < baseline`
- `symplectic < fair energy`
- `symplectic < fair L2`
- full symplectic mechanism transfer confirmed

Mechanism:

```text
||J^T Omega J - Omega||
```

The full symplectic Jacobian error reduction exceeds the K2.2-A mechanism
threshold on the graceful-baseline subset.

## Boundary

K2 currently supports a narrow structure claim:

- validated regime: FPU-beta
- validated structure: symplectic prior
- validated transfer horizon: H=200
- validated controls: fair energy and fair L2

K2 does not yet claim generalization beyond FPU-beta.

## Experiment Files

- `chronos/k2/experiments/k2_0_fpu_regime.py`
- `chronos/k2/experiments/k2_0b_refine_window.py`
- `chronos/k2/experiments/k2_1b_repair_controls.py`
- `chronos/k2/experiments/k2_2a_transfer_h200.py`
