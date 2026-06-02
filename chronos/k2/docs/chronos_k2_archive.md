# Chronos K2 Archive

K2 is the first VPSL-certified physical-structure milestone.

## Structure

Symplectic prior on FPU-β.

## Current Verdict

```text
FULL_TRANSFER_CONFIRMED
```

## Evidence Summary

K2 is now sealed as the first VPSL-certified structure milestone.

| Stage | Archive Role |
| --- | --- |
| K2.0 | FPU-β regime validation |
| K2.0-B | Valid window and H=200 stress horizon |
| K2.1 | Initial symplectic candidate comparison |
| K2.1-B | Fair controls repaired |
| K2.2-A | Full transfer confirmed |

K2.2-A tests transfer at H=200 after K2.1-B repaired the controls at H=160.

Full reproducibility currently starts from K2.1. K2.0 and K2.0-B established
the FPU-β graceful-fail regime; their logs are archived, but source scripts
are not currently available.

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

- validated regime: FPU-β
- validated structure: symplectic prior
- validated transfer horizon: H=200
- validated controls: fair energy and fair L2

K2 does not yet claim generalization beyond FPU-β.

## Future Stress-Horizon Note

For H=240+ tests, use a dual baseline-failure label:

```text
hard_diverged or functional_diverged
```

This prevents the graceful-baseline subset from admitting seeds that are not
hard-diverged by the strict tail rule but are already functionally failed.

This note is for future K2.3+ design only. K2.2-A remains archived as:

```text
FULL_TRANSFER_CONFIRMED
```

## Experiment Files

- `chronos/k2/historical_logs/k2_0_regime_validation_log.txt`
- `chronos/k2/historical_logs/k2_0b_refine_window_log.txt`
- `chronos/k2/reconstruction_notes.md`
- `chronos/k2/experiments/k2_1_symplectic_prior.py`
- `chronos/k2/experiments/k2_1b_repair_controls.py`
- `chronos/k2/experiments/k2_2a_transfer_h200.py`
