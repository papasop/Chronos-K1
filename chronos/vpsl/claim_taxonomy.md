# VPSL Claim Taxonomy

## Claim Levels

### `BOUNDED_POSITIVE`

The framework or prior shows a positive result under bounded conditions, but it
is not promoted to a full transferable structure claim.

Chronos example:
- K1 spectral / Lorentz-sensitive prior evidence.

### `CONTROLS_REPAIRED`

The experiment has established fair, non-degenerate controls for the relevant
comparison.

Chronos example:
- K2.1-B repaired the energy and L2 control comparisons for the symplectic
  prior.

### `TRANSFER_PERFORMANCE_ONLY`

The prior improves performance at the transfer horizon, but the mechanism does
not transfer or the improvement is explained mainly by pooled rescue.

This is not a physical-structure transfer claim.

### `TRANSFER_CONFIRMED`

The prior beats baseline and every control still fair at the transfer horizon
on the graceful-baseline subset, with mechanism transfer confirmed.

This claim is used when one or more previously selected controls degenerated at
the transfer horizon, reducing the completeness of the three-way comparison.

### `FULL_TRANSFER_CONFIRMED`

The strongest current VPSL claim level.

Requirements:
- prior beats baseline on the graceful-baseline subset
- prior beats fair energy control
- prior beats fair L2 control
- all controls remain fair at the tested horizon
- mechanism transfer is confirmed

Chronos example:
- K2.2-A symplectic transfer at H=200 on FPU-beta.

### `NO_TRANSFER`

The prior does not robustly beat baseline at the transfer horizon.

## Non-Claim Defaults

Unless explicitly validated, Chronos does not claim:

- transfer beyond the tested regime
- transfer beyond the tested horizon
- universality across architectures
- that every physical prior helps
- that pooled rescue is enough for mechanism transfer
