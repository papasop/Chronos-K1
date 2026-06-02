# VPSL Verdict Taxonomy

## REJECTED

A proposed physical structure fails validation.

Examples:

- Not predictive.
- Mechanism unsupported.
- Control outperforms structure.

## NO_EFFECT

No measurable improvement over baseline.

Criteria:

- No significant reduction in primary metric.

## PERFORMANCE_ONLY

Performance improves but mechanism is unsupported.

Criteria:

- Primary metric improves.
- Mechanism test fails.

Interpretation:

- Likely rescue, damping, or regularization effect.

## MECHANISM_CONFIRMED

Mechanism operates as intended.

Criteria:

- Mechanism diagnostic passes.
- Transfer not yet tested.

## BOUNDED_POSITIVE

Positive result with known limitations.

Criteria:

- Predictive improvement.
- Mechanism partially supported.
- Transfer incomplete or bounded.

Example:

- K1 spectral prior.

## TRANSFER_CONFIRMED

Mechanism and performance transfer.

Criteria:

- Transfer regime passes.
- Mechanism persists.
- Beats all remaining fair controls.

## FULL_TRANSFER_CONFIRMED

Highest VPSL evidence level.

Criteria:

- Predictive improvement.
- Mechanism confirmed.
- Transfer confirmed.
- Fair controls remain valid.
- Beats baseline and all fair controls.

Example:

- K2 symplectic prior on FPU-beta.
