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
ACTIVE / BOUNDED_NEGATIVE
```

## Current Candidate

```text
Topological / winding-density local structure
```

Current entrypoint:

- `experiments/k3_0d_sine_gordon_winding_density.py`
- `experiments/k3_1_winding_density_prior.py`
- `experiments/k3_2d_0_vortex_regime.py`

Archived package:

- `archives/exp_k3_1_main/`

Prior negative regime attempts are archived in:

- `K3_NEGATIVE_RESULTS_phi4_regime.md`

Detailed K3.1-A negative prior result:

- `negative_results/k3_1a_winding_density_continuity_rejected.md`

K3.0-D is baseline-only periodic Sine-Gordon regime validation using the angle
representation `[sin(u), cos(u), u_t]`.

K3.1 tests a winding-density continuity prior at H=160. It returns:

```text
NO_EFFECT
```

The density prior does not beat baseline, does not beat the matched-ratio
off-target continuity control, and does not pass the winding-density mechanism
test. This is archived as a negative Stage-2 prior result.

The `archives/exp_k3_1_main/` package preserves the experiment design, config,
headline results, summary verdict, and an archive entrypoint for regenerating
the stored CSV summaries. It does not rerun the full GPU Colab training sweep.

Boundary:

- target structure: winding-density / local topological structure
- not claimed: integer topological-charge certification
- no K3 prior confirmed yet
- no K3 certified structure yet

## K3.2D Restart

K3.2D.0 starts the 2D topological-structure restart using Gross-Pitaevskii
vortex-antivortex pairs.

Status:

```text
SMOKE-FIRST REGIME VALIDATION
```

Role:

- baseline-only
- no prior tested
- validates whether a continuous vortex-position error can form a graceful 2D
  regime before K3.2D.1 spends compute on priors
- separates field-learning pipeline health from topological transport health

Run mode is intentionally `SMOKE` by default. A scientific regime decision
requires switching the script to `FULL`.

Developmental lesson:

```text
Field prediction OK does not imply topological-object transport OK.
```

The smoke verdicts are therefore split:

- `SMOKE_PIPELINE_FAIL`: the baseline did not learn the field map.
- `SMOKE_PIPELINE_OK_TRANSPORT_FAIL`: the baseline learned field prediction but
  failed vortex-antivortex transport; do not promote.
- `SMOKE_TRANSPORT_OK`: both field prediction and vortex transport passed the
  smoke sanity check; promotable to `FULL`.

If K3.2D.0 returns transport failure, it is recorded as
`REGIME_UNRESOLVED` / `TRANSPORT_FAIL`, not as a rejected topological prior.
