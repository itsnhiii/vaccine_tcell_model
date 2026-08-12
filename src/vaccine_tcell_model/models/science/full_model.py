"""Assembled Science model (Eq. 1-7), built by composing three
independently pluggable components (Phase 6): ScienceUpstreamModel,
ScienceTCellModel, ScienceTfhModel. No equation here differs from the
individual component modules -- this file only wires them together via
core.composition.ComposedModel.

A future caller can replace any one component (e.g. swap ScienceTCellModel
for a Mayer-based or mock T-cell module) by passing a custom `model=` to
simulate_science, without touching this file, the other two components,
or the solver in solvers/ (Phase 6 success criterion).
"""

from __future__ import annotations

import numpy as np

from vaccine_tcell_model.core.composition import ComposedModel
from vaccine_tcell_model.core.state import SCIENCE_STATES
from vaccine_tcell_model.dosing import DoseSchedule
from vaccine_tcell_model.parameters import ParameterSet
from vaccine_tcell_model.solvers import simulate as _simulate

from .tcell import ScienceTCellModel, ScienceTfhModel
from .upstream import ScienceUpstreamModel


def build_science_model(params: ParameterSet) -> ComposedModel:
    """The default Science model, composed from its three pluggable components."""
    return ComposedModel(
        SCIENCE_STATES,
        [
            ScienceUpstreamModel(params),
            ScienceTCellModel(params),
            ScienceTfhModel(params),
        ],
    )


def science_rhs(t: float, state: dict[str, float], params: ParameterSet) -> dict[str, float]:
    """Function-style RHS for direct use/testing -- equivalent to
    build_science_model(params).rhs(t, state). `t` is unused (the RHS has
    no explicit time dependence once dosing is externalized as jumps),
    kept in the signature for solve_ivp/simulate compatibility.
    """
    return build_science_model(params).rhs(t, state)


def simulate_science(
    params: ParameterSet,
    initial_conditions: dict[str, float],
    dose_schedule: DoseSchedule | None,
    t_end: float,
    *,
    model: ComposedModel | None = None,
    t0: float = 0.0,
    t_eval: np.ndarray | None = None,
    method: str = "RK45",
    rtol: float = 1e-8,
    atol: float = 1e-10,
    max_step: float = np.inf,
):
    """Integrate the Science model from t0 to t_end and return a SimulationResult.

    Pass `model=` with a custom ComposedModel (e.g. swapping in a mock or
    alternative T-cell component -- see tests/test_modular_interfaces.py)
    to override the default three-component composition. This is the
    Phase 6 module-swap entry point; nothing below this parameter, nor
    anything in solvers/, needs to change to support it.
    """
    composed = model if model is not None else build_science_model(params)

    def rhs(t: float, state: dict[str, float]) -> dict[str, float]:
        return composed.rhs(t, state)

    return _simulate(
        rhs,
        SCIENCE_STATES,
        initial_conditions,
        t_end,
        t0=t0,
        dose_schedule=dose_schedule,
        t_eval=t_eval,
        method=method,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
        parameters=params,
        model_name="science",
    )
