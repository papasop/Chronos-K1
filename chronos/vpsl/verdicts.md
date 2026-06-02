# VPSL Verdicts

VPSL verdicts are used across K1, K2, and future K-stage experiments.

## Verdict Table

| Verdict | Meaning |
| --- | --- |
| `NO_EFFECT` | The prior does not improve the target metric. |
| `PERFORMANCE_ONLY` | The prior improves performance, but mechanism or fair controls are missing. |
| `MECHANISM_CONFIRMED` | The mechanism diagnostic improves, but transfer is not yet established. |
| `BOUNDED_POSITIVE` | Positive bounded evidence; not a full certified structure. |
| `TRANSFER_CONFIRMED` | Transfer and mechanism hold against every control still fair at the transfer horizon. |
| `FULL_TRANSFER_CONFIRMED` | Transfer, mechanism, baseline separation, and all fair controls hold. |
| `REJECTED` | The candidate fails the gates or is not predictive enough to retain as a structure claim. |

## How To Compare K1 And K2

K1:

```text
Spectral / Lorentz-sensitive prior -> BOUNDED_POSITIVE
```

K2:

```text
Symplectic prior on FPU-beta -> FULL_TRANSFER_CONFIRMED
```

This makes K2 a stronger evidence claim than K1, while preserving K1 as the
framework-proven stage.

## Use In Future K-Stages

K3 and later stages should report one verdict from this file. If a new verdict
is needed, it should be added here before the experiment is promoted to archive
status.
