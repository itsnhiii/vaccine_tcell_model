# Architecture

How this package is put together, and how to extend it. For *what* is
modeled, see [`model_specification.md`](model_specification.md). For
equation provenance, see [`equations.md`](equations.md).

## Package layout

```
src/vaccine_tcell_model/
    core/          state containers, pluggable-component interfaces, ComposedModel, SimulationResult
    parameters/    Parameter/ParameterSet + provenance (SourceType); per-model parameter builders
    dosing/        DoseSchedule, DoseEvent, validation
    models/        science/, mayer/, integrated/ -- the three models (see model_specification.md)
    solvers/       dose-event-aware ODE integration (integrate_with_events, simulate)
    analysis/      metrics, sweeps, sensitivity, fitting, optimization
    validation/    per-model validation checks used by the test suite (Phases 4, 8)
```

Fifteen phases of the master specification map roughly onto this
directory structure in the order they were built: `core`/`parameters`
(Phase 1) → `dosing`/`solvers` (Phase 2) → `models/science` (Phase 3-4)
→ `models/mayer` (Phase 5) → `core.composition` (Phase 6) →
`models/integrated` (Phase 7-8) → `analysis` (Phase 9-11).

## The state/solver separation (core/)

`StateSpec` (`core/state.py`) is the single source of truth mapping a
named biological quantity (`"aDC_Ag"`) to its index in the numpy array
the solver actually integrates. No model or solver code hard-codes an
index. Three module-level `StateSpec` instances (`SCIENCE_STATES`,
`MAYER_STATES`, `INTEGRATED_STATES`) exist as constants, built once from
the equation audit in `equations.md`, not redefined ad hoc per model file.

`SimulationResult` (`core/result.py`) wraps a tidy pandas DataFrame
(`time` + one column per state) plus the `StateSpec`, the `ParameterSet`
used, the `DoseSchedule` used (or `None`), and a `model_name` string —
every `simulate_*` function returns one of these, so downstream code
(`analysis/`) never needs to know which model produced a result to
compute metrics on it.

## Pluggable components (core/interfaces.py, core/composition.py)

Every model's right-hand side is built from `RHSComponent` objects, not
one monolithic function:

```python
class RHSComponent(ABC):
    owned_states: tuple[str, ...]        # which states this component computes derivatives for
    def rhs(self, t, state) -> dict: ...  # may READ any state, only WRITES its own
```

Four semantic subclasses (`InnateDCModel`, `PresentationModel`,
`TCellActivationModel`, `TfhModel`) name the four conceptual roles the
specification calls for, with no additional abstract methods — they
exist for `isinstance` checks and documentation, not new mechanics.

`ComposedModel` (`core/composition.py`) assembles a list of components
into one solver-facing RHS, and validates at construction time that
every state in a `StateSpec` is owned by exactly one component — no
silent gaps (a state nobody updates) and no silent overlaps (two
components both writing the same state).

**This is what makes the integrated model (Phase 7) cheap to build.**
`ScienceUpstreamModel` and `ScienceTfhModel` are composed into the
integrated model completely unchanged — building the integrated model
required writing exactly two new files (`models/integrated/pmhc.py`,
`tcell.py`), not touching the DC or Tfh code at all. This was verified,
not just claimed by design: `tests/test_integrated_model.py` runs the
standalone Science model and the integrated model side by side under
identical inputs and checks the shared states match to solver tolerance.

**Swapping a component requires no solver changes.** Every
`simulate_*` function takes an optional `model=` argument accepting a
custom `ComposedModel`. `tests/test_modular_interfaces.py` demonstrates
this concretely: a toy `MockConstantTCellModel` (defined only in the
test file) is substituted for `ScienceTCellModel`, and the untouched
upstream/Tfh components, and `solvers/`, need no changes to accommodate it.

## Dosing and the solver (dosing/, solvers/)

`DoseSchedule` tracks antigen and adjuvant as two independent channels
(possibly different times, possibly different dose counts), merging them
into a sorted list of `DoseEvent`s only at the point of consumption.
`DoseEvent.to_state_jump_event()` translates a dose into the solver's
generic, model-agnostic `StateJumpEvent` (`core/events.py`) — the solver
never knows "antigen" or "adjuvant" as concepts, only "add this amount
to this named state at this time."

`solvers/events.py`'s `integrate_with_events` implements dosing as an
exact discontinuity: it splits the integration span at every event time,
integrates each segment independently with `scipy.integrate.solve_ivp`,
and applies each jump before integrating the segment that starts at that
time. The value recorded at a jump time is right-continuous (the
post-jump value) — a deliberate choice documented in that module, since
it differs slightly from how the reference MATLAB implementation records
its output arrays at dose boundaries (see `equations.md` Section 1.3).

`solvers/ode.py`'s `simulate()` is the model-agnostic wrapper: given a
dict-based RHS function, a `StateSpec`, initial conditions, and an
optional `DoseSchedule`, it handles the array↔dict conversion and
delegates to `integrate_with_events`, returning a `SimulationResult`.
Every model's `simulate_science`/`simulate_mayer`/`simulate_integrated`
is a thin wrapper around this one function.

## Parameters and provenance (parameters/)

`Parameter` bundles a value with mandatory provenance: `source` (a
citation or fit-target string) and `source_type` (one of `literature`,
`fitted`, `assumed`, `user_defined`, `derived`, `model_extension` — no
other value is accepted). `ParameterSet` is an immutable, named
collection of `Parameter`s with no merge/update-in-place API — combining
two different papers' parameter sets is structurally impossible without
an explicit, visible new `ParameterSet` construction (e.g.
`parameters/integrated.py`), which is how master-spec rule 4 ("never
silently combine parameter values from different papers") is enforced
in code, not just by convention.

The one sanctioned mutation-like operation is `ParameterSet.with_value()`
— returns a *new* `ParameterSet` with one named value replaced and its
own provenance updated to record that it's now an override. This exists
specifically for `analysis/sweeps.py`, `sensitivity.py`, `fitting.py`,
and `optimization.py`, none of which ever touch a `ParameterSet`'s
private state directly.

Each model has its own parameter-builder module
(`science.py`/`mayer.py`/`integrated.py`); Mayer additionally exposes
three named presets (`mayer_parameters_fig1_cd4`, `_fig2_cd8`,
`_fig4_dosing`) since the paper itself reports three independent fits
that must never be averaged together.

## Analysis layer (analysis/)

`_common.py` holds the shared "given a model name, build its default
params/IC, run it, summarize its metrics" dispatch logic — factored out
once `sweeps.py` and `sensitivity.py` needed the identical thing, rather
than duplicated a third time in `fitting.py`/`optimization.py`. It is
not part of the public API (not re-exported from `analysis/__init__.py`).

- `metrics.py`: single-trajectory summary functions (`peak_value`,
  `auc`, ...) plus per-state-group summaries (`summarize_tcell`, etc.)
  matching the exact metric names the specification lists.
- `sweeps.py`: `sweep_parameter` (vary one kinetic parameter via
  `with_value`) and `sweep_dose_schedules` (compare named
  `DoseSchedule`s) — both return tidy `pandas.DataFrame`s, one row per
  swept condition.
- `sensitivity.py`: `local_sensitivity` computes `d log(metric)/d
  log(parameter)` by central finite difference in log-parameter space.
- `fitting.py`: `fit_parameters` wraps `scipy.optimize.least_squares`
  (bounded, optionally log-transformed, multi-observable). Always
  returns a *new* `ParameterSet` inside its `FitResult` — never mutates
  the base set or a model's module-level default.
- `optimization.py`: `optimize_dose_schedule` wraps
  `scipy.optimize.minimize(method="SLSQP")` to search the dose-fraction
  simplex for a fixed, evenly-spaced time grid. Documented, verified
  finding: this objective is non-convex, so results depend on the
  starting point (`initial_fractions`) — "reproducible" means
  deterministic given a fixed starting point, not globally optimal from
  any starting point. See that module's docstring for the numbers.

## Extending the package

**Add a new component implementation** (e.g. an
`ExplicitPeptideProcessingModel` to replace `SimplePmhcProductionModel`):
implement `RHSComponent` (or the relevant semantic subclass), owning the
same state name(s) the component it replaces owned, and pass a custom
`ComposedModel` via any `simulate_*`'s `model=` argument. No other file
needs to change.

**Add a new model entirely** (e.g. the Science B-cell/GC model, master
spec Section 25): create `models/<name>/` with its own components and
`full_model.py` following the existing three models' pattern; create
`parameters/<name>.py` with its own `default_<name>_parameters()`;
extend `core/state.py` with a new `StateSpec` constant if new states are
needed. `analysis/_common.py`'s `MODELS` tuple and dispatch functions
are the one place that would need a new branch to make the new model
usable from the generic sweep/sensitivity/fitting/optimization APIs.

**Add a new dosing pattern**: add a classmethod to `DoseSchedule`
(following `bolus`/`equal_doses`/`from_exponential_escalation`) rather
than a new top-level function — keeps all schedule-construction logic in
one place.
