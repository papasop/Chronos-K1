# K3 Negative Results: Pre-K3.0-D Regime Attempts

These notes preserve the failed Stage-1 regime searches that motivated K3.0-D.

## K3.0-A - phi^4 Single Kink

Verdict:

```text
NO_GRACEFUL_BAND
```

Reason:

The unbounded `u^3` nonlinearity caused rollout blowup before a useful
graceful-failure regime emerged.

## K3.0-B - phi^4 Kink-Antikink

Verdict:

```text
NO_GRACEFUL_BAND
```

Reason:

The integer wall-count / sector diagnostic changed too abruptly. Small
continuous timing errors could produce large discrete sector errors, violating
the VPSL graceful-failure premise.

## K3.0-C - Periodic Sine-Gordon, Lifted Field

Verdict:

```text
NO_GRACEFUL_BAND
```

Reason:

The lifted angle representation was not translation-equivariant on the ring.
The fixed seam and changing plateau width made the circular-padding CNN fail
the one-step map. This is treated as a representation failure, not a negative
result about periodic Sine-Gordon physics.

## K3.0-D Response

K3.0-D keeps periodic Sine-Gordon but changes the state representation to:

```text
[sin(u), cos(u), u_t]
```

The target structure is downgraded honestly to winding-density / local
topological structure. Integer winding remains a secondary hard check, not the
primary graceful metric.

## K3.1 - Winding-Density Continuity Prior

Detailed archive:

- `negative_results/k3_1a_winding_density_continuity_rejected.md`

Verdict:

```text
NO_EFFECT
```

Reason:

The density prior failed the primary paired tests:

- `density_l1.0` did not beat baseline.
- `density_l1.0` did not beat the matched-ratio off-target continuity control.
- `density_l0.1` did not beat generic smoothness.
- `density_l0.1` did not beat increment L2.
- Winding-density diagnostic did not improve; median reduction was `-10.7%`.

Boundary:

This is a negative result for the tested winding-density continuity prior, not
a rejection of all topological or winding-density structures. Integer
topological-charge certification was never claimed.
