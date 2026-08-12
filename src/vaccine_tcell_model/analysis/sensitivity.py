"""Local log sensitivity analysis (master spec Section 20).

    S_i = d log(output) / d log(parameter_i)

estimated by central finite difference in log-parameter space around a
base ParameterSet: perturb the parameter by +/-`relative_step`, run the
model at each perturbation, and take the log-log slope between the two.

Only meaningful once the base model is already validated (Phases 4/8) --
this module assumes the model is trustworthy and asks how its outputs
respond to small parameter perturbations, not whether the model itself
is correct.

Global sensitivity (SALib) is explicitly out of scope for now (master
spec Section 20: "Later, optionally support SALib").
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from vaccine_tcell_model.dosing import DoseSchedule
from vaccine_tcell_model.parameters import ParameterSet

from ._common import compute_metrics, default_dose_schedule, default_params_and_ic, run_model

DEFAULT_METRICS: tuple[str, ...] = ("T_max", "TFH_max", "AUC_T", "AUC_TFH")
"""The four outputs master spec Section 20 names explicitly."""


def local_sensitivity(
    model: str,
    parameter: str,
    metric: str,
    *,
    base_params: ParameterSet | None = None,
    initial_conditions: dict[str, float] | None = None,
    dose_schedule: DoseSchedule | None = None,
    t_end: float = 21.0,
    t_eval: np.ndarray | None = None,
    relative_step: float = 0.01,
) -> float:
    """S_i = d log(metric) / d log(parameter) via central finite
    difference: perturb `parameter` to p*(1+relative_step) and
    p*(1-relative_step), and take the log-log slope between the two
    resulting metric values.
    """
    default_params, default_ic = default_params_and_ic(model)
    params = base_params if base_params is not None else default_params
    ic = initial_conditions if initial_conditions is not None else default_ic
    schedule = default_dose_schedule(model, dose_schedule)

    p0 = params.value(parameter)
    if p0 <= 0:
        raise ValueError(
            f"local_sensitivity requires a strictly positive base value for "
            f"{parameter!r} (log-sensitivity is undefined at/below 0); got {p0}"
        )

    p_hi = p0 * (1.0 + relative_step)
    p_lo = p0 * (1.0 - relative_step)

    def metric_at(p_value: float) -> float:
        swept = params.with_value(parameter, p_value)
        result = run_model(model, swept, ic, schedule, t_end, t_eval)
        row = compute_metrics(result)
        if metric not in row:
            raise KeyError(f"Metric {metric!r} not available for model {model!r}; got {sorted(row)}")
        return row[metric]

    f_hi = metric_at(p_hi)
    f_lo = metric_at(p_lo)
    if f_hi <= 0 or f_lo <= 0:
        raise ValueError(
            f"local_sensitivity requires strictly positive metric values to take "
            f"a log; got {metric}({p_lo})={f_lo}, {metric}({p_hi})={f_hi}"
        )

    return (np.log(f_hi) - np.log(f_lo)) / (np.log(p_hi) - np.log(p_lo))


def sensitivity_analysis(
    model: str,
    parameters: Sequence[str],
    *,
    metrics: Sequence[str] = DEFAULT_METRICS,
    base_params: ParameterSet | None = None,
    initial_conditions: dict[str, float] | None = None,
    dose_schedule: DoseSchedule | None = None,
    t_end: float = 21.0,
    t_eval: np.ndarray | None = None,
    relative_step: float = 0.01,
) -> pd.DataFrame:
    """Local log sensitivity S_i for every (parameter, metric) pair.

    Returns a tidy DataFrame with a `parameter` column and one column per
    metric in `metrics`.
    """
    rows = []
    for parameter in parameters:
        row: dict[str, float | str] = {"parameter": parameter}
        for metric in metrics:
            row[metric] = local_sensitivity(
                model,
                parameter,
                metric,
                base_params=base_params,
                initial_conditions=initial_conditions,
                dose_schedule=dose_schedule,
                t_end=t_end,
                t_eval=t_eval,
                relative_step=relative_step,
            )
        rows.append(row)
    return pd.DataFrame(rows)
