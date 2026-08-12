"""Parameter sweeps and dose-schedule comparisons (master spec Section 19).

    results = sweep_parameter(
        model="integrated", parameter="K", values=np.logspace(-2, 3, 50),
    )

Both sweep functions return a tidy pandas DataFrame: one row per swept
value/schedule, plus one column per output metric for whichever of
T/TFH/pMHC (or Mayer's C) the chosen model has.

Sweeping a dosing dimension (total dose, dose interval, number of doses,
dose fractions -- master spec Section 19's dosing entries) is done by
constructing the corresponding DoseSchedule objects for each value and
passing them to sweep_dose_schedules, rather than by a separate
bespoke sweep function per dosing dimension -- DoseSchedule already
covers all of those construction patterns (Phase 2), so a second,
overlapping API for the same thing would just be duplication.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from vaccine_tcell_model.dosing import DoseSchedule
from vaccine_tcell_model.parameters import ParameterSet

from ._common import MODELS, compute_metrics, default_dose_schedule, default_params_and_ic, run_model


def sweep_parameter(
    model: str,
    parameter: str,
    values: Sequence[float],
    *,
    base_params: ParameterSet | None = None,
    initial_conditions: dict[str, float] | None = None,
    dose_schedule: DoseSchedule | None = None,
    t_end: float = 21.0,
    t_eval: np.ndarray | None = None,
) -> pd.DataFrame:
    """Sweep one named kinetic parameter (e.g. K, T0, d_Ag, k_p, mu_pMHC),
    holding everything else fixed. Uses ParameterSet.with_value, so the
    swept values are individually provenance-tagged as overrides, never a
    silent cross-model merge (master spec rule 4).

    `model` is one of "science", "mayer", "integrated". `dose_schedule`
    defaults to a single bolus for science/integrated (ignored for mayer,
    which has no dosing concept -- antigen enters via the initial
    condition's C(0), see docs/equations.md Section 2).

    Returns a tidy DataFrame with a `parameter` column (containing the
    swept values) plus one column per output metric.
    """
    if model not in MODELS:
        raise ValueError(f"Unknown model {model!r}; expected one of {MODELS}")

    default_params, default_ic = default_params_and_ic(model)
    params_base = base_params if base_params is not None else default_params
    ic = initial_conditions if initial_conditions is not None else default_ic
    schedule = default_dose_schedule(model, dose_schedule)

    rows = []
    for value in values:
        params = params_base.with_value(parameter, float(value))
        result = run_model(model, params, ic, schedule, t_end, t_eval)
        row = {"parameter": parameter, "value": float(value)}
        row.update(compute_metrics(result))
        rows.append(row)
    return pd.DataFrame(rows)


def sweep_dose_schedules(
    model: str,
    schedules: dict[str, DoseSchedule],
    *,
    params: ParameterSet | None = None,
    initial_conditions: dict[str, float] | None = None,
    t_end: float = 21.0,
    t_eval: np.ndarray | None = None,
) -> pd.DataFrame:
    """Compare named dose schedules (master spec Section 19's dose-schedule
    sweeps: total dose, interval, number of doses, dose fractions -- vary
    whichever dimension by constructing the corresponding DoseSchedule
    objects yourself and passing them here as a {label: DoseSchedule} dict).

    Returns a tidy DataFrame with a `schedule` column (the dict keys) plus
    one column per output metric.
    """
    if model == "mayer":
        raise ValueError(
            "Mayer has no dosing concept (docs/equations.md Section 2); "
            "use sweep_parameter on 'T' or 'C' via initial_conditions instead"
        )
    if model not in MODELS:
        raise ValueError(f"Unknown model {model!r}; expected one of {MODELS}")

    default_params, default_ic = default_params_and_ic(model)
    params = params if params is not None else default_params
    ic = initial_conditions if initial_conditions is not None else default_ic

    rows = []
    for name, schedule in schedules.items():
        result = run_model(model, params, ic, schedule, t_end, t_eval)
        row = {"schedule": name}
        row.update(compute_metrics(result))
        rows.append(row)
    return pd.DataFrame(rows)
