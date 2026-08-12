# vaccine_tcell_model

A modular, scientifically-traceable Python package modeling:

```
vaccine dose + schedule + adjuvant + T-cell functional affinity
    -> antigen/adjuvant kinetics -> innate cells/DC
    -> antigen-loaded activated DC -> pMHC
    -> antigen-specific T cells -> Tfh cells
```

It independently reproduces two published models and then combines them
into a clearly-labeled integrated extension:

- **Science model** — Bhagchandani et al., *Science Immunology* 9,
  eadl3755 (2024): innate/DC/T-cell priming kinetics. Reproduced exactly
  (supplement Eq. 1-7), including the finding that the published model
  has **no pMHC state and no affinity parameter K**.
- **Mayer model** — Mayer et al., *PNAS* 116, 5914-5919 (2019):
  pMHC-driven T-cell expansion and functional avidity (K). Reproduced
  exactly (main text Eq. 1-2), with three independent literature presets
  (Fig.1/2/4) that are never merged.
- **Integrated model** — this project's own extension, combining
  Science's upstream/Tfh dynamics (reused unchanged) with a new pMHC
  layer and a hybrid T-cell equation. Every equation and parameter that
  isn't published verbatim is explicitly labeled `model_extension` in
  its provenance metadata — never presented as literature.

Full documentation:
- [`docs/model_specification.md`](docs/model_specification.md) — what's modeled and why
- [`docs/equations.md`](docs/equations.md) — equation-by-equation provenance and every discrepancy found against the original MATLAB reference implementation (Zenodo 12595650 from the Science model)
- [`docs/parameter_table.md`](docs/parameter_table.md) — every parameter, value, and source
- [`docs/architecture.md`](docs/architecture.md) — software design and how to extend it

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

```python
import numpy as np
from vaccine_tcell_model.dosing import DoseSchedule
from vaccine_tcell_model.models.integrated import simulate_integrated
from vaccine_tcell_model.parameters import (
    default_integrated_parameters,
    integrated_default_initial_conditions,
)

schedule = DoseSchedule(
    times=[0, 7],
    antigen_fractions=[0.2, 0.8],
    adjuvant_fractions=[0.2, 0.8],
)
params = default_integrated_parameters(K=10.0)
ic = integrated_default_initial_conditions(params)

result = simulate_integrated(
    params, ic, schedule, t_end=21.0, t_eval=np.linspace(0, 21, 211)
)
print(result.data.tail())          # tidy DataFrame: time, Ag, Adj, TC, DC, aDC_Ag, pMHC, T, TFH
print(result.trajectory("TFH")[-1])
```

Parameter sweeps, sensitivity analysis, fitting, and dose-schedule
optimization all return tidy `pandas.DataFrame`s or typed result objects:

```python
from vaccine_tcell_model.analysis import sweep_parameter, compare_dose_schedules_including_optimized

df = sweep_parameter(model="integrated", parameter="K", values=np.logspace(-2, 3, 50))

comparison_df, opt_result = compare_dose_schedules_including_optimized(
    "integrated", time_window=12.0, num_doses=7, metric="TFH_max",
)
```

## Examples

Each is runnable directly (`python examples/NN_*.py`) and writes its
plot(s) alongside itself:

| File | Demonstrates |
|---|---|
| `00_dosing_and_events.py` | Exact dose jumps (bolus/2-dose/7-dose), no numerical smearing |
| `01_science_reproduction.py` | Science model end-to-end, DC/aDC_Ag/T/TFH trajectories |
| `02_mayer_reproduction.py` | Mayer model, affinity direction, the paper's inverse power law |
| `03_integrated_model.py` | Full Ag→DC→aDC_Ag→pMHC→T→TFH pipeline, affinity/schedule response |
| `04_dose_schedule_comparison.py` | `sweep_dose_schedules` comparing bolus/2-ED/7-ED |
| `05_affinity_sweep.py` | `sweep_parameter` over K, the 4-panel comparison the spec asks for |

Start with `00_dosing_and_events.py` if you're new to the package, or
`03_integrated_model.py` if you want the full pipeline right away.

## Exploratory plots (`runs/`)

`runs/` holds ad-hoc scripts (and their output plots) written while
exploring the model interactively — dosing-schedule illustrations,
adjuvant dose-response comparisons, and a full dose-schedule
optimization deep dive (multi-start search, number-of-doses sweep,
non-zero-dose-constrained sweep with a global optimizer). Unlike
`examples/`, these aren't a curated tutorial sequence — see
[`runs/README.md`](runs/README.md) for what's in each subfolder.

Every script in both `examples/` and `runs/` is self-contained and saves
its `.png` output(s) next to itself, so you can just open the `.png`
files directly (in this repo or after cloning) without running anything.
To regenerate a plot or try your own parameters:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python runs/dosing_and_adjuvant/dosing_schedule_comparison_run.py   # or any other script
```

## Package layout

```
src/vaccine_tcell_model/
    core/        state containers (StateSpec), pluggable RHSComponent interfaces,
                 ComposedModel, SimulationResult
    parameters/  Parameter/ParameterSet + provenance (SourceType); per-model builders
    dosing/      DoseSchedule, DoseEvent, validation
    models/      science/, mayer/, integrated/
    solvers/     dose-event-aware ODE integration
    analysis/    metrics, sweeps, sensitivity, fitting, optimization
    validation/  per-model validation checks (used by the test suite)
```

See [`docs/architecture.md`](docs/architecture.md) for how these fit
together and how to extend the package (new components, new models, new
dosing patterns).
