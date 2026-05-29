# Causal Projection Experiment

This experiment tests a simple causal-projection constraint in a 2D Lorentzian state space.

It compares:

- Euclidean baseline: accepts raw Gaussian displacements.
- Chronos causal projection: projects each displacement into the future causal cone.

The causal validity condition is:

```text
dx.T @ G @ dx >= 0
G = diag(1, -1)
```

## Run

```bash
cd k1-manifold-core
python experiments/causal_projection_demo.py
```

The script writes:

```text
results/causal_projection_experiment.json
examples/outputs/causal_projection_trajectories.png
examples/outputs/causal_projection_intervals.png
```

## Metrics

- `violations`: number of spacelike or past-invalid displacement steps.
- `violation_rate`: violations divided by total steps.
- `mean_interval`: mean signed interval over all proposed steps.
- `min_interval`: minimum signed interval.

## Interpretation Boundary

This experiment tests a projected reachability constraint. It does not prove a general physical law or world-modeling advantage.

Its value is narrower: under the implemented Lorentzian constraint, causal projection eliminates step-level causal violations that raw Euclidean random displacements can produce.
