# Experiment 5 Verification Report

## Status

Experiment 5 has been reproduced as a formal benchmark script in the repository:

```text
k1-manifold-core/benchmarks/experiment_5_full_sanity_reproduction.py
```

The benchmark is documented in:

```text
k1-manifold-core/docs/experiment_5_reproduction_protocol.md
```

## Reproduction Configuration

| Parameter | Value |
| --- | ---: |
| seeds | 10 |
| train samples | 3000 |
| test samples | 512 |
| epochs | 250 |
| latent dimension | 16 |
| width | 64 |
| learning rate | 0.001 |
| rollout steps | 50 |
| train box | 2.0 |
| test boxes | 2, 4, 8, 16, 32 |
| lambda grid | 0, 0.1, 0.2, 0.5 |
| K1 regularization | 0.02 |

## RNG Strategy

```text
train_seed = 50000 + seed
test_seed  = 60000 + seed + int(box * 100)
```

Examples:

| Seed | Train | Box 2 | Box 4 | Box 8 | Box 16 | Box 32 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 50000 | 60200 | 60400 | 60800 | 61600 | 63200 |
| 1 | 50001 | 60201 | 60401 | 60801 | 61601 | 63201 |
| 2 | 50002 | 60202 | 60402 | 60802 | 61602 | 63202 |

## Main Result

At `lambda=0.1`:

| Setting | Euclidean violation | Chronos violation | Reduction | p-value |
| --- | ---: | ---: | ---: | ---: |
| `box=2` | 0.2866 | 0.1475 | 48.5% | 0.0840 |
| `box=32` | 0.4306 | 0.3155 | about 27% | not reported |

Rollout MSE was approximately unchanged in this reproduction.

## Interpretation

This is the strongest observed world-model effect in the current Chronos-K1
benchmark suite. It should be described as:

```text
substantial effect size with OOD persistence
```

It should not be described as:

```text
statistically significant
proved
forecasting-accuracy advantage
```

The correct framing is:

```text
Experiment 5 is the Primary World-Model Phenomenon Benchmark.
Experiment 5b is the Mechanism Diagnostic Benchmark.
```

The `box=2` Wilcoxon value, `p=0.0840`, is close to conventional significance
but does not meet `p<0.05`.

## Next Validation Step

Increase the reproduction beyond `N=10` seeds to test whether the observed
effect remains stable and whether the paired Wilcoxon value crosses the
conventional threshold.
