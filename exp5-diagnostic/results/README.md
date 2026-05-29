# Experiment 5 Result Artifacts

This directory is reserved for full Experiment 5 reproduction artifacts.

The maintained script generates the result files:

```text
experiment_5_full_sanity_summary.csv
experiment_5_full_sanity_payload.json
rng_seeding_documentation.json
experiment_5_full_sanity_violation_vs_box.png
experiment_5_full_sanity_mse_vs_box.png
experiment_5_full_sanity_violation_by_step.png
```

Run from repository root:

```bash
CHRONOS_DEVICE=cuda python exp5-diagnostic/chronos_k1_complete_colab.py --full --output-dir exp5-diagnostic/results
```

Historical Colab filenames such as:

```text
exp5_reproduction_results.csv
exp5_reproduction_results.png
exp5_reproduction_config.json
```

refer to raw files generated in the external Colab session. They are not
included here unless copied from that completed session. The repository does
not fabricate those raw artifacts from summary statistics.
