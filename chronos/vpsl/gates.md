# VPSL Gates

VPSL gates prevent a performance improvement from being over-read as a
validated physical-structure result.

## 1. Regime Gate

The tested system must be in the intended physical regime before priors are
compared.

For K2 this means the FPU-β setting is validated before claiming symplectic
transfer.

## 2. Control Fairness Gate

Controls must remain non-degenerate at the tested horizon.

Examples of degeneracy:

- near-identity collapse
- excessive damping
- pathological amplitude growth or shrinkage
- prior loss overwhelming data loss
- control divergence that makes comparison meaningless

Controls that were fair at a shorter horizon must be re-checked at a longer
stress horizon.

## 3. Graceful-Subset Gate

At stress horizons, pooled improvement can be inflated by rescuing seeds where
the baseline hard-diverged.

The primary transfer claim must therefore hold on the graceful-baseline subset:
the seeds where the baseline did not hard-diverge.

## 4. Mechanism Gate

A structure prior must transfer its mechanism, not only lower error.

For K2 the mechanism diagnostic is full symplectic Jacobian error:

```text
||J^T Omega J - Omega||
```

K2.2-B requires more than 20% reduction on the graceful-baseline subset.

## 5. Claim Boundary Gate

The final claim must match the passed gates.

Passing a narrow regime test supports a narrow structure claim. It does not
support universal claims about all physical priors, all systems, or all
architectures.
