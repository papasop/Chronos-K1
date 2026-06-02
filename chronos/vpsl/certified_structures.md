# Certified Structure Registry

This registry records candidate physical structures and their current VPSL
status.

## Registry

| Structure | System / Regime | Status | Evidence |
| --- | --- | --- | --- |
| Symplectic prior | FPU-beta, S=1, H=200 transfer | `FULL_TRANSFER_CONFIRMED` | K2.2-A |
| Spectral / Lorentz-sensitive prior | K1 metric-sensitive setting | `BOUNDED_POSITIVE` | K1 Exp6 / Exp7 |
| Energy prior | FPU-beta control family | `REJECTED` as certified structure; useful as fair control | K2.1-B / K2.2-A |
| Causal leakage prior | K1 candidate | `REJECTED` | Negative archive |

## Certified Structure Criteria

A structure can be marked certified only if it passes:

- regime validation
- constraint validation
- mechanism validation
- transfer validation
- claim boundary review

## Current Certified Structure

### Symplectic Prior

Status:

```text
FULL_TRANSFER_CONFIRMED
```

System:

```text
FPU-beta, S=1, H=200 stress transfer
```

Evidence:

- beats baseline on graceful-baseline subset
- beats fair energy control
- beats fair L2 control
- full symplectic Jacobian mechanism transfers

Boundary:

- not yet certified beyond FPU-beta
- not yet certified beyond tested horizons
- not a claim that all systems benefit from symplectic priors
