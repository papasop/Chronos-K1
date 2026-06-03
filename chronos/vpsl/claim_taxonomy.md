# VPSL Claim Taxonomy

VPSL claims are graded by what survives controls, mechanism checks, and
transfer tests. The taxonomy prevents a local performance gain from being
overstated as a certified physical structure.

## Claim Levels

| Level | Label | Meaning |
| --- | --- | --- |
| 0 | `NO_EFFECT` | The prior does not improve the target metric. |
| 1 | `PERFORMANCE_ONLY` | The prior improves performance, but mechanism or controls are not confirmed. |
| 2 | `BOUNDED_POSITIVE` | The prior/framework has a bounded positive result under a validated setting. |
| 3 | `TRANSFER_CONFIRMED` | The advantage transfers with a confirmed mechanism against every fair control still valid at the transfer horizon. |
| 4 | `FULL_TRANSFER_CONFIRMED` | Transfer holds with mechanism, baseline separation, and all pre-registered fair controls intact. |

## Level 0 - `NO_EFFECT`

The tested prior does not robustly beat the baseline.

Use this when:

- primary performance does not improve
- pooled and graceful-subset results both fail
- the result does not justify mechanism interpretation

## Level 1 - `PERFORMANCE_ONLY`

The prior improves the target metric, but the evidence does not establish a
physical-structure mechanism.

Use this when:

- improvement is pooled-only rescue
- graceful-subset performance fails
- mechanism fails
- controls are missing, degenerate, or unfair

K2 synonym:

```text
TRANSFER_PERFORMANCE_ONLY
```

## Level 2 - `BOUNDED_POSITIVE`

The result is positive and meaningful, but bounded.

Use this when:

- the framework works in a controlled setting
- the mechanism is suggestive but not fully transfer-certified
- the claim should remain local and historical

Chronos example:

```text
K1 spectral / Lorentz-sensitive prior
```

## Level 3 - `TRANSFER_CONFIRMED`

The prior transfers to a harder setting with mechanism confirmed on the
graceful-baseline subset, and it beats every control that remains fair at that
transfer horizon.

Use this when:

- a pre-registered control degenerates at the transfer horizon
- the transfer claim is real, but the full three-way comparison is incomplete

## Level 4 - `FULL_TRANSFER_CONFIRMED`

The strongest current VPSL claim level.

Requirements:

- prior beats baseline on the graceful-baseline subset
- prior beats fair energy control
- prior beats fair L2 control
- all controls remain fair at the tested horizon
- mechanism transfer is confirmed

Chronos example:

```text
K2.2-B symplectic transfer at H=240 on FPU-β
```

## Auxiliary Labels

`CONTROLS_REPAIRED`

The experiment has established fair, non-degenerate controls for the relevant
comparison.

Chronos example:

```text
K2.1-B repaired fair energy and fair L2 controls.
```

`CONTROL_MATCHES_SYMP`

The symplectic mechanism persists and beats baseline, but a fair control
matches or beats the target prior on performance.

## Non-Claim Defaults

Unless explicitly validated, Chronos does not claim:

- transfer beyond the tested regime
- transfer beyond the tested horizon
- universality across architectures
- that every physical prior helps
- that all systems benefit from symplectic priors
- that pooled rescue is enough for mechanism transfer
