# K2 Reconstruction Notes

K2.0 and K2.0-B established the FPU-beta graceful-fail regime, but their source
scripts are not currently available in this repository.

## Current Source Availability

Available code:

- K2.1
- K2.1-B
- K2.2-A

Missing code:

- K2.0
- K2.0-B

Available evidence:

- K2.0 / K2.0-B logs
- K2.1 / K2.1-B code and results
- K2.2-A code and results

## Reproducibility Boundary

Full reproducibility currently starts from K2.1.

The regime-discovery stages K2.0 and K2.0-B are preserved as logs in:

```text
chronos/k2/historical_logs/
```

They should be reconstructed before claiming full source-code reproducibility
for the entire K2 chain.

## Reconstruction Targets

Future files:

```text
chronos/k2/experiments/k2_0_reconstructed.py
chronos/k2/experiments/k2_0b_reconstructed.py
```

Acceptance criteria:

- recover FPU-beta S=1 as the validated K2 regime
- recover H=160 as the clean graceful-fail comparison window
- recover H=200 as the K2.2-A stress transfer horizon
- match the archived hard-divergence conclusions closely enough to preserve
  the K2.1 / K2.2-A claim boundary

## Archive Statement

K2 can be archived as:

```text
complete result chain, partial source missing
```

The final K2 verdict remains:

```text
FULL_TRANSFER_CONFIRMED
```
