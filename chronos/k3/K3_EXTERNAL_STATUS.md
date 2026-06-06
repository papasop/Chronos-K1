# K3 External Status

K3 is not certified.

## Current Status

- K3.1 winding-density prior returned `NO_EFFECT`.
- K3.2D is still regime-first.
- K3-E2b and K3-E2d provide bounded active-search evidence.
- No topology prior should be tested until baseline transport gates pass.

## Key Boundary

Field prediction OK does not imply object transport OK.

Low field MSE may mean the field is easy to predict while the vortex pair,
defect, winding, kink, or other topological object is not transported with
identity continuity.

## Current Next Gate

K3.2D.0d:

```text
baseline-only strict pipeline + object transport
```

Only if the baseline passes transport in a reference-stable regime should K3
move to FULL. Only after FULL passes should a K3 prior test be considered.

## What K3 Does Not Claim

- K3 is VPSL-certified.
- A topology prior has been validated.
- Low field error proves topological object transport.
- A failed baseline transport result means a topology prior failed.
