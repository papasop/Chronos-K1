# Experiment 5 Diagnostic Package

This directory is the public entry point for the Chronos-K1 Experiment 5
diagnostic.

Experiment 5 is the current **primary world-model phenomenon benchmark** for
Chronos-K1. It tests whether a Chronos latent predictor can reduce decoded
causal-violation rates on synthetic Lorentzian oscillator rollouts while
preserving comparable rollout error.

The maintained implementation lives in:

```text
k1-manifold-core/benchmarks/experiment_5_full_sanity_reproduction.py
```

This diagnostic directory provides:

- `chronos_k1_complete_colab.py`: a Colab-friendly launcher for the maintained
  benchmark script.
- `VERIFICATION_REPORT.md`: concise milestone verification report.
- `里程碑验证报告.md`: Chinese milestone verification report.
- `COLAB_INSTRUCTIONS.md`: step-by-step Colab usage guide.
- `guides/COLAB_COPY_PASTE_VERSION.md`: direct copy-paste Colab cell version
  of the Experiment 5 reproduction, with cautious interpretation wording.
- `guides/COLAB_EXTENDED_FIXED_V2.md`: extended 8-lambda Colab guide matching
  the latest supplied run, with the mechanism-analysis CUDA/NumPy issue fixed.
- `results/`: target directory for full reproduction artifacts.
- `results/EXP5_COLAB_REPRODUCTION_2026-05-29.md`: committed summary of the
  supplied extended Colab `N=10` reproduction console output.

## Headline Result

Extended fixed-v2 Colab reproduction (`N=10`, CUDA, `lambda=0.1`):

| Setting | Euclidean violation | Chronos violation | Reduction | p-value |
| --- | ---: | ---: | ---: | ---: |
| `box=2` | 0.2866 | 0.1475 | 48.5% | 0.0840 |
| `box=32` | 0.4306 | 0.3155 | about 27% | not reported |

The `box=32` result is important because the reduction persists under OOD
extrapolation rather than appearing only at the training-scale distribution.

The extended fixed-v2 run used the lambda grid
`[0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]` and collected 800 rollout
results. Its attempted mechanism analysis did not complete because of a CUDA
tensor-to-NumPy conversion error, so that part should be treated as incomplete.

## Interpretation Boundary

This is a substantial effect-size benchmark and is close to conventional
significance at `box=2`, but it is not a final statistical significance claim:
`p=0.0840` does not meet `p<0.05`.

Experiment 5 should be cited as:

```text
Primary World-Model Phenomenon Benchmark
```

Experiment 5b should be cited as:

```text
Mechanism Diagnostic Benchmark
```

Do not describe the result as proved or statistically significant.

## Run Locally

From repository root:

```bash
cd k1-manifold-core
python -m pip install -r requirements-benchmarks.txt
CHRONOS_DEVICE=cuda python benchmarks/experiment_5_full_sanity_reproduction.py --full
```

For a smoke check:

```bash
cd k1-manifold-core
python benchmarks/experiment_5_full_sanity_reproduction.py --smoke
```

## Output Artifacts

The maintained benchmark writes:

```text
experiment_5_full_sanity_summary.csv
experiment_5_full_sanity_payload.json
rng_seeding_documentation.json
experiment_5_full_sanity_violation_vs_box.png
experiment_5_full_sanity_mse_vs_box.png
experiment_5_full_sanity_violation_by_step.png
```

To write those artifacts into this package, run:

```bash
CHRONOS_DEVICE=cuda python exp5-diagnostic/chronos_k1_complete_colab.py --full --output-dir exp5-diagnostic/results
```

The historical Colab raw files `exp5_reproduction_results.csv`,
`exp5_reproduction_results.png`, `exp5_reproduction_config.json`,
`exp5_extended_fixed_v2_results.csv`, `exp5_extended_fixed_v2_results.png`,
and `exp5_mechanism_analysis_v2.json` should be copied into
`exp5-diagnostic/results/` if you want to preserve the exact external session
output. They are not fabricated from summary statistics.

The supplied Colab console result is recorded in:

```text
exp5-diagnostic/results/EXP5_COLAB_REPRODUCTION_2026-05-29.md
```

See also:

- `exp5-diagnostic/guides/COLAB_COPY_PASTE_VERSION.md`
- `exp5-diagnostic/guides/COLAB_EXTENDED_FIXED_V2.md`
- `k1-manifold-core/docs/experiment_5_reproduction_protocol.md`
- `k1-manifold-core/docs/benchmark_report.md`
