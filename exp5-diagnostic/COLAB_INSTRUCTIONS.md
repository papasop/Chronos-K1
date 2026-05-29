# Colab Instructions

Use this guide when running the Experiment 5 full sanity reproduction in
Google Colab.

## 1. Open Colab

Go to:

```text
https://colab.research.google.com
```

Set runtime:

```text
Runtime -> Change runtime type -> GPU
```

## 2. Clone The Repository

```python
!git clone https://github.com/papasop/Chronos-K1.git
%cd Chronos-K1/k1-manifold-core
```

## 3. Install Benchmark Dependencies

```python
!python -m pip install -r requirements-benchmarks.txt
```

## 4. Run A Smoke Check

```python
!python benchmarks/experiment_5_full_sanity_reproduction.py --smoke
```

## 5. Run The Full Reproduction

```python
!CHRONOS_DEVICE=cuda python benchmarks/experiment_5_full_sanity_reproduction.py --full
```

Expected runtime depends on GPU availability. On a hosted GPU runtime, plan for
a long benchmark run.

## 6. Inspect Results

The full run writes:

```text
results/experiment_5_full_sanity_summary.csv
results/experiment_5_full_sanity_payload.json
results/rng_seeding_documentation.json
results/experiment_5_full_sanity_violation_vs_box.png
results/experiment_5_full_sanity_mse_vs_box.png
results/experiment_5_full_sanity_violation_by_step.png
```

## Interpretation Reminder

The target result is a substantial causal-violation reduction with OOD
persistence. The currently recorded `N=10` reproduction gives `p=0.0840` at
`box=2`, so do not describe the result as statistically significant.
