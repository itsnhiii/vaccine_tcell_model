"""Shared model-dispatch helpers for sweeps, sensitivity, and fitting.

Internal plumbing only -- not part of the public `analysis` API (not
re-exported from analysis/__init__.py). Factored out once a second module
(sensitivity.py) needed the exact same "given a model name, get its
default params/IC, run it, and summarize its metrics" logic that
sweeps.py already had, rather than duplicating it.
"""

from __future__ import annotations

import numpy as np

from vaccine_tcell_model.dosing import DoseSchedule
from vaccine_tcell_model.models.integrated import simulate_integrated
from vaccine_tcell_model.models.mayer import simulate_mayer
from vaccine_tcell_model.models.science import simulate_science
from vaccine_tcell_model.parameters import (
    ParameterSet,
    default_integrated_parameters,
    default_mayer_parameters,
    default_science_parameters,
    integrated_default_initial_conditions,
    mayer_demo_initial_conditions,
    science_default_initial_conditions,
)

from .metrics import summarize_pmhc, summarize_tcell, summarize_tfh

MODELS = ("science", "mayer", "integrated")


def default_params_and_ic(model: str) -> tuple[ParameterSet, dict[str, float]]:
    if model == "science":
        params = default_science_parameters()
        return params, science_default_initial_conditions(params)
    if model == "mayer":
        params = default_mayer_parameters()
        return params, mayer_demo_initial_conditions()
    if model == "integrated":
        params = default_integrated_parameters()
        return params, integrated_default_initial_conditions(params)
    raise ValueError(f"Unknown model {model!r}; expected one of {MODELS}")


def default_dose_schedule(model: str, dose_schedule: DoseSchedule | None) -> DoseSchedule | None:
    """Fill in a sensible default dose schedule (a single bolus) for
    science/integrated when none is given; always None for mayer, which
    has no dosing concept."""
    if dose_schedule is not None:
        return dose_schedule
    return None if model == "mayer" else DoseSchedule.bolus()


def run_model(
    model: str,
    params: ParameterSet,
    ic: dict[str, float],
    dose_schedule: DoseSchedule | None,
    t_end: float,
    t_eval: np.ndarray | None,
):
    if model == "science":
        return simulate_science(params, ic, dose_schedule, t_end=t_end, t_eval=t_eval)
    if model == "mayer":
        return simulate_mayer(params, ic, t_end=t_end, t_eval=t_eval)
    if model == "integrated":
        return simulate_integrated(params, ic, dose_schedule, t_end=t_end, t_eval=t_eval)
    raise ValueError(f"Unknown model {model!r}; expected one of {MODELS}")


def compute_metrics(result) -> dict[str, float]:
    row: dict[str, float] = {}
    names = result.states.names
    if "T" in names:
        row.update(summarize_tcell(result))
    if "TFH" in names:
        row.update(summarize_tfh(result))
    if "pMHC" in names:
        row.update(summarize_pmhc(result, state="pMHC"))
    if "C" in names:  # Mayer's cognate-pMHC state
        row.update(summarize_pmhc(result, state="C"))
    return row
