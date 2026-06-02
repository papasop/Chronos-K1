# VPSL Framework

VPSL means Validated Physics Structure Learning.

The framework asks a narrow question: when a physical structure is added as an
inductive bias, does it improve a model for the right reason, in a validated
regime, against fair controls?

## Core Principles

1. Validate the regime first.
2. Compare the structure prior against baseline and non-degenerate controls.
3. Separate pooled performance from graceful-subset performance.
4. Require mechanism transfer for structure claims.
5. State claim boundaries at the same level of precision as the evidence.

## Chronos Mapping

| Milestone | Role | Current Claim |
| --- | --- | --- |
| K1 | Framework and bounded metric/spectral result | `BOUNDED_POSITIVE` |
| K2 | First validated physical structure | `FULL_TRANSFER_CONFIRMED` |

K1 remains important because it established the discipline: a prior can appear
to help for reasons that do not justify a broad structure claim. K2 applies that
discipline to a symplectic prior and validates transfer on the graceful subset.

## Pipeline

```text
Physical Structure
-> Regime Validation
-> Constraint Validation
-> Intervention / Mechanism
-> Transfer
-> Certified Structure
```

The rendered repository diagram is stored at:

```text
chronos/vpsl/vpsl_pipeline.svg
```

## Structure Claim Template

A VPSL structure claim should name:

- the physical structure
- the regime
- the horizon or stress setting
- the baseline
- the fair controls
- the primary metric
- the mechanism diagnostic
- the accepted claim level
- the explicit non-claims
