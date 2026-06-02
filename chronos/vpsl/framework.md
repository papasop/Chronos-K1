# What Is VPSL?

VPSL means Validated Physics Structure Learning.

VPSL is the Chronos framework for deciding when a physical structure should be
treated as a learning constraint, whether it improves prediction for the right
reason, and whether that advantage transfers beyond the setting where it was
first observed.

Chronos is not a single model family. It is a validation program for physical
structure priors.

## Core Question

When a candidate physical structure is added to a learner, does it become a
validated structure or only a local performance trick?

VPSL answers this through four gates.

## Gate 1 - Regime Validation

Before testing a prior, validate that the system is in the intended physical
regime.

This gate asks:

- Is the dataset actually expressing the structure being tested?
- Is the rollout window meaningful?
- Is the baseline neither trivial nor fully broken?
- Are stress horizons identified separately from clean horizons?

K2 example:

```text
FPU-beta, S=1, H=160 clean comparison, H=200 stress transfer.
```

## Gate 2 - Constraint Validation

A structure prior must be compared against fair, non-degenerate controls.

This gate asks:

- Does the prior improve the primary metric?
- Do controls avoid collapse, damping, or identity shortcuts?
- Are energy, L2, or other controls matched fairly?
- Are controls re-checked at transfer horizons?

K2 example:

```text
K2.1-B repaired fair energy and fair L2 controls before transfer testing.
```

## Gate 3 - Mechanism Validation

A performance gain is not enough. The claimed physical mechanism must be
measured directly.

This gate asks:

- Does the diagnostic tied to the structure improve?
- Does the improvement survive on the graceful-baseline subset?
- Is the gain more than pooled rescue?

K2 example:

```text
Full ||J^T Omega J - Omega|| reduction confirms symplectic mechanism transfer.
```

## Gate 4 - Transfer Validation

A certified structure must transfer to a harder setting without reducing to
divergence rescue.

This gate asks:

- Does the prior beat baseline at the transfer horizon?
- Does it beat every control still fair at that horizon?
- Does it hold on the graceful-baseline subset?
- Does the mechanism still transfer?

K2 example:

```text
K2.2-A at H=200: FULL_TRANSFER_CONFIRMED.
```

## Pipeline

```text
Physical Structure
-> Regime Validation
-> Constraint Validation
-> Mechanism Validation
-> Transfer Validation
-> Certified Structure
```

The rendered repository diagram is stored at:

```text
chronos/vpsl/vpsl_pipeline.svg
```

## Chronos Mapping

| Milestone | Role | Current Claim |
| --- | --- | --- |
| K1 | Framework validation / spectral prior | `BOUNDED_POSITIVE` |
| K2 | First certified structure: symplectic prior | `FULL_TRANSFER_CONFIRMED` |
| K3 | Future structure discovery program | `PENDING` |

## Structure Claim Template

A VPSL structure claim should name:

- the physical structure
- the system / regime
- the horizon or stress setting
- the baseline
- the fair controls
- the primary metric
- the mechanism diagnostic
- the accepted verdict
- the explicit non-claims
