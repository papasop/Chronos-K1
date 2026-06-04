# K2 Archive

K2 is the first VPSL-certified physical-structure stage.

## Structure

Symplectic prior on FPU-β.

## Status

```text
FULL_TRANSFER_CONFIRMED through H=240
```

## Archive Verdict

K2.2-B extends the K2 transfer result to H=240. The symplectic prior remains
`FULL_TRANSFER_CONFIRMED` through H=240.

```text
K2 = first VPSL-certified physical structure
K2.2-A = first FULL_TRANSFER_CONFIRMED structure result at H=200
K2.2-B = FULL_TRANSFER_CONFIRMED extended to H=240
```

K2.2-A remains the first full transfer result. K2.2-B extends the transfer
ceiling. Future work should start from K3 rather than weakening the K2 archive
boundary through open-ended horizon hunting.

## Source Availability

Available entrypoints:

- K2.1
- K2.1-B
- K2.2-A
- K2.2-B full audit-fixed Colab source

Missing code:

- K2.0
- K2.0-B

Available evidence:

- K2.0 / K2.0-B logs
- K2.1 / K2.1-B archive entrypoints and archived summaries
- K2.2-A archive entrypoint, archived summary, and verdict wiring
- K2.2-B full audit-fixed Colab source, archived summary, and verdict wiring

K2.1, K2.1-B, and K2.2-A currently have repository archive entrypoints and
verdict logic. K2.2-B has the full audit-fixed Colab training source restored.
The regime-discovery stages K2.0 and K2.0-B are preserved as logs and are being
reconstructed.

## What K2 Established

K2 tells a six-step VPSL story.

| Stage | Result | Claim Role |
| --- | --- | --- |
| K2.0 | FPU-β regime validation | Regime validation |
| K2.0-B | Valid FPU-β window refined; H=200 identified as stress transfer horizon | Regime / window validation |
| K2.1 | Initial symplectic prior comparison | Structure candidate test |
| K2.1-B | Fair non-degenerate controls repaired at H=160 | `CONTROLS_REPAIRED` |
| K2.2-A | Symplectic transfer confirmed at H=200 | `FULL_TRANSFER_CONFIRMED` |
| K2.2-B | Symplectic transfer extended to H=240 | `FULL_TRANSFER_CONFIRMED` |

K2.2-A confirms transfer at H=200 on the graceful-baseline subset:

- `symplectic < baseline`
- `symplectic < fair energy`
- `symplectic < fair L2`
- mechanism transfer confirmed

K2.2-B extends the transfer result to H=240 on the graceful-baseline subset:

- `symplectic < baseline`
- `symplectic < fair energy`
- `symplectic < fair L2`
- mechanism transfer confirmed

## K2.2-A Evidence Summary

Primary subset:

```text
graceful-baseline subset, n = 22
```

Rollout MSE at H=200:

| Variant | roll_MSE |
| --- | ---: |
| baseline | 0.0722 |
| symplectic | 0.0398 |
| fair energy | 0.1038 |
| fair L2 | 0.0975 |

Mechanism:

```text
full symp_err reduction = 70.7%
```

Verdict:

```text
FULL_TRANSFER_CONFIRMED
```

## K2.2-B Evidence Summary

Primary subset:

```text
graceful-baseline subset, n = 18
```

Rollout MSE at H=240:

| Variant | roll_MSE |
| --- | ---: |
| baseline | 0.0847 |
| symplectic | 0.0543 |
| fair energy | 0.1374 |
| fair L2 | 0.1408 |

Mechanism:

```text
full symp_err reduction = 71.5%
```

Verdict:

```text
FULL_TRANSFER_CONFIRMED
```

The mechanism diagnostic is full symplectic Jacobian error:

```text
||J^T Omega J - Omega||
```

## K2.3 — Wrong-Ω Specificity Control

Status:

```text
OMEGA_SPECIFICITY_CONFIRMED_NONDEGEN_AWARE
```

K2.3 tests whether the K2 result is specific to the canonical symplectic form Ω
or merely an arbitrary antisymmetric Jacobian penalty.

Result:

- canonical Ω beats baseline on rollout
- canonical Ω beats shuffled Ω and random antisymmetric Ω on rollout
- canonical Ω reduces true-Ω Jacobian error vs baseline
- wrong-Ω controls collapse / under-drive the dynamics, making their raw
  Jacobian error uninterpretable

This strengthens K2 by showing that the certified symplectic effect is not
explained by generic antisymmetric Jacobian regularization.

Detailed note:

```text
chronos/k2/docs/k2_3_wrong_omega_specificity.md
```

Summary:

```text
chronos/k2/results/k2_3_reanalysis_summary.csv
```

## Boundary

K2 currently supports a narrow claim: the symplectic prior transfers through
H=240 on the validated FPU-β regime under the tested controls. It does not yet
claim generalization beyond FPU-β.

Not claimed:

- all Hamiltonian systems benefit from the tested symplectic prior
- all systems benefit from symplectic constraints
- H=300 or beyond is already certified
- pooled rescue is sufficient for a structure claim

## Numbering Discipline

K2 uses K-stage numbering rather than the historical ExpXX numbering:

```text
K2.0   regime validation
K2.0-B window refinement
K2.1   first symplectic comparison
K2.1-B fair-control repair
K2.2-A transfer test at H=200
K2.2-B transfer test at H=240
```

Historical ExpXX labels remain in K1 archive material only.

## H=300+ Note

K2.2-B preserves the K2 discipline: transfer claims require the
graceful-baseline subset, not pooled rescue. Future H=300+ stress tests should
again be regime-gated before training priors.

K2.2-A / K2.2-B use `hard_diverged` as the primary baseline-stratification
label. The definition is intentionally strong:

```text
FUNC_DIV_THR = 10
tail failure >= 80%
```

For H=300 or later stress tests, future experiments should track both labels:

```text
hard_diverged
functional_diverged
```

Reason: at longer horizons, a seed may become functionally failed without
meeting the stronger hard-divergence tail criterion. Future graceful subsets
should avoid mixing such half-failed samples into the primary transfer test.

This is a future stress-test design note only. It does not change the K2.2-B
archive verdict.

## Preserved Materials

- `chronos/k2/README.md`
- `chronos/k2/reconstruction_notes.md`
- `chronos/k2/historical_logs/k2_0_regime_validation_log.txt`
- `chronos/k2/historical_logs/k2_0b_refine_window_log.txt`
- `chronos/k2/experiments/k2_1_symplectic_prior.py`
- `chronos/k2/experiments/k2_1b_repair_controls.py`
- `chronos/k2/experiments/k2_2a_transfer_h200.py`
- `chronos/k2/experiments/k2_2b_transfer_h240.py`
- `chronos/k2/experiments/k2_3_wrong_omega_reanalysis.py`
- `chronos/k2/results/k2_2a_summary.csv`
- `chronos/k2/results/k2_2b_summary.csv`
- `chronos/k2/results/k2_3_reanalysis_summary.csv`
