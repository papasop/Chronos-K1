# K2 Experiment Archive Entrypoints

This directory contains archive entrypoints for the K2 symplectic-prior
milestone.

These scripts preserve:

- experiment configuration metadata
- registered verdict semantics
- archived summary generation
- repository navigation for the K2 result chain

They do not rerun the full GPU Colab training experiments unless the original
training sources are explicitly restored.

## Current Entrypoints

- `k2_1_symplectic_prior.py`
- `k2_1b_repair_controls.py`
- `k2_2a_transfer_h200.py`

## Boundary

K2.1 through K2.2-A can currently be inspected through repository archive
entrypoints and archived summaries. Full training reproduction remains pending
restoration of the original Colab experiment sources.
