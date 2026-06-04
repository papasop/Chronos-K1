# K2 Experiment Entrypoints

This directory contains repository entrypoints for the K2 symplectic-prior
milestone.

The K2.1, K2.1-B, and K2.2-A scripts preserve:

- experiment configuration metadata
- registered verdict semantics
- archived summary generation
- repository navigation for the K2 result chain

They do not rerun the full GPU Colab training experiments.

The K2.2-B script is different: it is the restored audit-fixed full Colab
training source for the two-phase H=240 transfer experiment. Running it directly
starts the expensive full experiment.

## Current Entrypoints

- `k2_1_symplectic_prior.py`
- `k2_1b_repair_controls.py`
- `k2_2a_transfer_h200.py`
- `k2_2b_transfer_h240.py`
- `k2_3_wrong_omega_reanalysis.py`

## Boundary

K2.1, K2.1-B, and K2.2-A can currently be inspected through repository archive
entrypoints and archived summaries. K2.2-B can be inspected syntactically with
`py_compile`, or run intentionally on a GPU runtime for the full H=240
two-phase experiment.

K2.3 is a no-retraining reanalysis script. It reads an existing
`k2_3_main_results.csv` and derives the non-degeneracy-aware wrong-Ω
specificity verdict.
