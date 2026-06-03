# Negative Results Archive

Chronos keeps negative and bounded results as first-class evidence. VPSL is not
validated by showing only successes; it is validated by rejecting structures
that do not survive the gates.

## Summary

| Stage | Candidate | Verdict | Reason |
| --- | --- | --- | --- |
| Exp10 | Causal leakage prior | `REJECTED` | Leakage was not predictive enough to certify a structure. |
| Exp10.4 | Energy conservation prior | `REJECTED` | Energy conservation alone was not sufficient as a fair structure claim. |
| K1 | Spectral / Lorentz-sensitive prior | `BOUNDED_POSITIVE` | Positive framework result, but mechanism transfer was bounded. |
| K1 / Exp11.2-E | Spectral transfer boundary | `PERFORMANCE_ONLY` / partial | Pooled gains can reflect rescue rather than mechanism transfer. |
| K2 controls | Energy and L2 controls | `CONTROLS_REPAIRED` | Initial controls required repair before fair K2 comparison. |
| K3.0-A | phi^4 single kink | `NO_GRACEFUL_BAND` | Field blowup before a graceful topological regime emerged. |
| K3.0-B | phi^4 kink-antikink | `NO_GRACEFUL_BAND` | Discrete wall-count failure was too abrupt for a clean graceful window. |
| K3.0-C | periodic Sine-Gordon lifted field | `NO_GRACEFUL_BAND` | Lifted-field seam representation caused the baseline failure. |
| K3.1-A | winding-density continuity prior | `NO_EFFECT` | Form-C prior did not beat baseline or matched-ratio off-target; mechanism worsened. |

## Why This Archive Exists

Negative results protect the claim boundary.

They show that Chronos does not assume:

- every physical prior helps
- conserved quantities are automatically sufficient
- pooled performance is a mechanism
- a prior is certified without transfer

## Current Lessons

### Causal Leakage

Causal leakage was not promoted to a certified structure. The result is treated
as rejected unless a future experiment establishes regime, mechanism, and
transfer gates.

### Energy Conservation

Energy conservation is physically meaningful, but K2 showed that an energy
control is not enough by itself. It must be fair, non-degenerate, and
performance-competitive against the structure under test.

### Spectral / Lorentz K1 Boundary

K1 remains important because it established the framework and a bounded
positive result. It is not a full transfer-certified structure.

### Pooled Rescue

At stress horizons, pooled improvement can be caused by rescuing seeds where
the baseline diverged. VPSL therefore requires graceful-baseline transfer and a
mechanism diagnostic.

### K3 Winding-Density Boundary

K3.0-D found a valid winding-density regime for periodic Sine-Gordon, but the
first tested Form-C winding-density continuity prior failed at K3.1-A. This
rejects that prior design only; it does not reject all winding-density or
topological structures.
