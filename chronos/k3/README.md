# K3 Structure Discovery Program

K3 is the active VPSL structure-discovery program after K2.

No K3 structure is certified yet.

Candidate structures should enter this directory only after the claim plan is
defined:

- regime validation
- constraint validation
- intervention / mechanism test
- transfer test
- claim boundary

K3 status:

```text
STAGE_1_REGIME_VALIDATION
```

## Current Candidate

```text
Topological / winding-density local structure
```

Current entrypoint:

- `experiments/k3_0d_sine_gordon_winding_density.py`

Prior negative regime attempts are archived in:

- `K3_NEGATIVE_RESULTS_phi4_regime.md`

K3.0-D is baseline-only periodic Sine-Gordon regime validation using the angle
representation `[sin(u), cos(u), u_t]`.

Boundary:

- target structure: winding-density / local topological structure
- not claimed: integer topological-charge certification
- no prior tested yet
- no K3 certified structure yet
