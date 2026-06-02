# K2 Archive

K2 is the first VPSL-certified physical-structure stage.

## Structure

Symplectic prior on FPU-beta.

## Status

```text
FULL_TRANSFER_CONFIRMED
```

## Archive Verdict

K2.2-A is archived as a milestone result, not as another positive experiment.

```text
K2 = first VPSL-certified physical structure
K2.2-A = first FULL_TRANSFER_CONFIRMED structure result
```

No further changes are required to K2.2-A before archival. Future work should
start from K3 or from a clearly labeled K2.3 extension, not by weakening the
K2.2-A claim boundary.

## What K2 Established

K2 tells a five-step VPSL story.

| Stage | Result | Claim Role |
| --- | --- | --- |
| K2.0 | FPU-beta regime validation | Regime validation |
| K2.0-B | Valid FPU-beta window refined; H=200 identified as stress transfer horizon | Regime / window validation |
| K2.1 | Initial symplectic prior comparison | Structure candidate test |
| K2.1-B | Fair non-degenerate controls repaired at H=160 | `CONTROLS_REPAIRED` |
| K2.2-A | Symplectic transfer confirmed at H=200 | `FULL_TRANSFER_CONFIRMED` |

K2.2-A confirms transfer at H=200 on the graceful-baseline subset:

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

The mechanism diagnostic is full symplectic Jacobian error:

```text
||J^T Omega J - Omega||
```

## Boundary

K2 currently supports a narrow claim: the symplectic prior transfers on the
validated FPU-beta regime under the tested controls. It does not yet claim
generalization beyond FPU-beta.

Not claimed:

- all Hamiltonian systems benefit from the tested symplectic prior
- all systems benefit from symplectic constraints
- H=240 or beyond is already certified
- pooled rescue is sufficient for a structure claim

## Numbering Discipline

K2 uses K-stage numbering rather than the historical ExpXX numbering:

```text
K2.0   regime validation
K2.0-B window refinement
K2.1   first symplectic comparison
K2.1-B fair-control repair
K2.2-A transfer test at H=200
```

Historical ExpXX labels remain in K1 archive material only.

## H=240+ Note

K2.2-A uses `hard_diverged` as the primary baseline-stratification label. The
definition is intentionally strong:

```text
FUNC_DIV_THR = 10
tail failure >= 80%
```

For H=240 or later stress tests, future experiments should track both labels:

```text
hard_diverged
functional_diverged
```

Reason: at longer horizons, a seed may become functionally failed without
meeting the stronger hard-divergence tail criterion. Future graceful subsets
should avoid mixing such half-failed samples into the primary transfer test.

This is a K2.3+ design note only. It does not change the K2.2-A archive
verdict.

## Preserved Materials

- `chronos/k2/experiments/k2_0_fpu_regime.py`
- `chronos/k2/experiments/k2_0b_refine_window.py`
- `chronos/k2/experiments/k2_1_symplectic_prior.py`
- `chronos/k2/experiments/k2_1b_repair_controls.py`
- `chronos/k2/experiments/k2_2a_transfer_h200.py`
- `chronos/k2/results/k2_2a_summary.csv`
