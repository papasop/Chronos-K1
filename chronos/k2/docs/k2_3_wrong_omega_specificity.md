# K2.3 — Wrong-Ω Specificity Control

**Status:** `OMEGA_SPECIFICITY_CONFIRMED_NONDEGEN_AWARE`

**Role:** K2 hardening / specificity control.
**Does not replace:** K2.2-B, which remains the headline certified result:
`FULL_TRANSFER_CONFIRMED through H=240`.

## Question

K2.2-B showed that the canonical symplectic prior transfers through H=240 on
FPU-β.

K2.3 asks a sharper control question:

> Is the gain caused by preserving the correct canonical symplectic form Ω,
> or would any antisymmetric Jacobian-structure penalty work?

## Tested Controls

All priors use the same Jacobian penalty form:

```text
||J^T Ω' J - Ω'||
```

Only `Ω'` changes.

| Group | Ω used |
| --- | --- |
| `symplectic_true` | canonical `Ω = [[0,I],[-I,0]]` |
| `symp_shuffled` | shuffled / permuted Ω, breaking q-p pairing |
| `symp_randantisym` | random antisymmetric 2-form, Frobenius-norm matched |

## Main H=160 Result

| Group | Rollout MSE | true-Ω error | increment | status |
| --- | ---: | ---: | ---: | --- |
| baseline | 0.0512 | 0.203 | 0.2141 | healthy |
| symplectic_true | 0.0269 | 0.108 | 0.2108 | healthy |
| symp_shuffled | 0.1690 | 0.069 | 0.0373 | degenerate / under-dynamic |
| symp_randantisym | 0.1786 | 0.080 | 0.0401 | degenerate / under-dynamic |

Paired rollout tests:

- `symplectic_true < baseline`: Holm p = 0.0110
- `symplectic_true < symp_shuffled`: Holm p = 0.0030
- `symplectic_true < symp_randantisym`: Holm p = 0.0030

Mechanism vs baseline:

- true-Ω error reduction: 46.7%

## Non-Degeneracy-Aware Interpretation

The wrong-Ω controls reduce the model's motion amplitude:

- shuffled inc/base ≈ 0.17
- random antisymmetric inc/base ≈ 0.19

They also degrade short-horizon fit:

- shuffled ref@40 ≈ 0.0426
- random antisymmetric ref@40 ≈ 0.0423

Thus their low raw true-Ω Jacobian error is not interpreted as valid geometry
learning. A near-identity or under-driven map can make raw Jacobian errors look
small without producing useful dynamics.

The reanalysis therefore treats wrong-Ω Jacobian errors as uninterpretable when
the controls are degenerate.

## Verdict

```text
OMEGA_SPECIFICITY_CONFIRMED_NONDEGEN_AWARE
```

Meaning:

- canonical Ω improves rollout relative to baseline
- canonical Ω improves true-Ω mechanism relative to baseline
- wrong-Ω controls are harmful on rollout
- wrong-Ω controls collapse / under-drive the dynamics, so their low raw
  Jacobian error is an artifact
- K2 is not merely an arbitrary antisymmetric Jacobian-regularization result

## Caveat

This is a non-degeneracy-aware confirmation.

Direct raw Jacobian-error comparison against collapsed wrong-Ω controls is
excluded. A future hardening step may recompute Jacobians on true rollout
states, but this is not required for the current K2.3 bounded conclusion.

## Relationship To K2.2-B

K2.2-B remains the main certified result:

```text
Symplectic Prior — FULL_TRANSFER_CONFIRMED through H=240
```

K2.3 strengthens the mechanism boundary by ruling out healthy wrong-Ω
alternatives in the tested setting.
