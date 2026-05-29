# Experiment 5 Reproduction Protocol

This note records the reproducible configuration for the Experiment 5 full
sanity reproduction. It is a research benchmark, not a unit test and not a
theorem-level claim.

## Command

From `k1-manifold-core`:

```bash
python -m pip install -r requirements-benchmarks.txt
CHRONOS_DEVICE=cuda python benchmarks/experiment_5_full_sanity_reproduction.py --full
```

For a tiny CPU-friendly smoke check:

```bash
python benchmarks/experiment_5_full_sanity_reproduction.py --smoke
```

## Configuration

| Parameter | Value |
| --- | ---: |
| `n_seeds` | 10 |
| `n_train` | 3000 |
| `n_test` | 512 |
| `epochs` | 250 |
| `dim` | 4 |
| `latent_dim` | 16 |
| `width` | 64 |
| `lr` | 0.001 |
| `t_total` | 80 |
| `t_obs` | 10 |
| `roll_steps` | 50 |
| `train_box` | 2.0 |
| `test_boxes` | 2, 4, 8, 16, 32 |
| `lambda_grid` | 0, 0.1, 0.2, 0.5 |
| `lambda_k1` | 0.02 |

## RNG Seeding

The benchmark uses offset-based seeding:

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

## Headline Result

The `N=10` CUDA reproduction gave:

| Setting | Euclidean violation | Chronos violation | Reduction | p-value |
| --- | ---: | ---: | ---: | ---: |
| `lambda=0.1`, `box=2` | 0.2866 | 0.1475 | 48.5% | 0.0840 |
| `lambda=0.1`, `box=32` | 0.4306 | 0.3155 | about 27% | not reported |

Rollout MSE was approximately unchanged in this run.

## Interpretation Boundary

This result is evidence that the Chronos latent predictor can reduce decoded
causal-violation rates on this synthetic Lorentzian oscillator benchmark while
preserving comparable rollout error. The `box=2` Wilcoxon value is close to,
but does not meet, the conventional `p<0.05` threshold. It should therefore be
reported as reproducible benchmark evidence, not as a final statistical
significance claim and not as a broad forecasting-accuracy claim.

## Artifacts

The script writes:

```text
results/experiment_5_full_sanity_summary.csv
results/experiment_5_full_sanity_payload.json
results/rng_seeding_documentation.json
results/experiment_5_full_sanity_violation_vs_box.png
results/experiment_5_full_sanity_mse_vs_box.png
results/experiment_5_full_sanity_violation_by_step.png
```
