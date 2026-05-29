# Experiment 5 Extended Colab Reproduction Result

Date recorded: 2026-05-29

Source: external Google Colab console output supplied after a completed
Experiment 5 extended fixed-v2 run.

This note records the numeric result from that run. It does not fabricate the
raw Colab artifacts (`exp5_reproduction_results.csv`,
`exp5_reproduction_results.png`, or `exp5_reproduction_config.json`). Copy those
files into this directory separately if preserving the exact external session
artifacts is required.

## Environment

```text
PyTorch: 2.11.0+cu128
Device: cuda
CUDA available: True
Project root: /content
Results dir: /content/results
```

## Configuration

```text
n_seeds: 10
n_train: 3000
n_test: 512
epochs: 250
dim: 4
latent_dim: 16
width: 64
lr: 0.001
t_total: 80
t_obs: 10
roll_steps: 50
train_box: 2.0
test_boxes: [2.0, 4.0, 8.0, 16.0, 32.0]
lambda_grid: [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
lambda_k1: 0.02
device: cuda
```

## RNG Strategy

The run used offset-based seeding with a box-dependent test offset:

```text
train_seed = 50000 + seed
test_seed = 60000 + seed + int(box * 100)
```

Examples:

| Seed | Train | Box 2 | Box 4 | Box 8 | Box 16 | Box 32 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 50000 | 60200 | 60400 | 60800 | 61600 | 63200 |
| 1 | 50001 | 60201 | 60401 | 60801 | 61601 | 63201 |
| 2 | 50002 | 60202 | 60402 | 60802 | 61602 | 63202 |

## Extended Run

The run collected 800 results:

```text
8 lambda values * 10 seeds * 2 models * 5 test boxes = 800
```

For `box=2.0`, `lambda=0.1`:

| Model | N | Mean causal violation | Std | Range |
| --- | ---: | ---: | ---: | --- |
| Euclidean latent predictor | 10 | 0.2866 | 0.2321 | 0.0372 - 0.5974 |
| Chronos latent predictor | 10 | 0.1475 | 0.1452 | 0.0280 - 0.5429 |

Relative reduction:

```text
48.5%
```

Paired Wilcoxon p-value:

```text
p = 0.0840
```

The extended lambda grid reproduced the same headline `lambda=0.1` result as
the earlier full sanity reproduction:

```text
Euclidean violation: 0.2866 +/- 0.2321
Chronos violation: 0.1475 +/- 0.1452
Relative reduction: 48.5%
Wilcoxon p-value: 0.083984375
```

## Mechanism Analysis Status

This v2 run attempted interval-preservation and latent-geometry mechanism
analysis for seed 0 at `box=2.0`, but that diagnostic failed for every lambda
with the same CUDA tensor conversion error:

```text
can't convert cuda:0 device type tensor to numpy
```

Therefore, the rollout/causal-violation sweep is usable, but the mechanism
analysis output from this run should be treated as incomplete. Do not cite this
run as successful mechanism evidence.

## Interpretation Boundary

This is the current primary world-model phenomenon benchmark result for
Chronos-K1. It records a substantial causal-violation reduction and OOD
persistence, while statistical significance has not yet been established:
`p=0.0840` does not meet the conventional `p<0.05` threshold.

The result should not be described as proved or statistically significant.

Recommended wording:

```text
Experiment 5 records a substantial causal-violation reduction in Chronos latent
world-model rollouts, with OOD persistence, while preserving comparable
prediction error. The current N=10 reproduction is close to conventional
significance but does not meet p<0.05.
```

## Output Files Reported By Colab

The external Colab session reported the following saved files:

```text
exp5_extended_fixed_v2_results.png
exp5_extended_fixed_v2_results.csv
exp5_mechanism_analysis_v2.json
```

Only this summary note is currently committed here unless those generated files
are copied from the Colab session.
