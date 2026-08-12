"""Assembled Mayer model (Eq. 1-2), built by composing two independently
pluggable components (Phase 6): MayerTCellModel, MayerPresentationModel.

No dosing/event system: antigen "administration" enters entirely through
the initial condition C(0) (docs/equations.md Section 2), so
`simulate_mayer` never takes a DoseSchedule.
"""

from __future__ import annotations

import numpy as np

from vaccine_tcell_model.core.composition import ComposedModel
from vaccine_tcell_model.core.state import MAYER_STATES
from vaccine_tcell_model.parameters import ParameterSet
from vaccine_tcell_model.solvers import simulate as _simulate

from .pmhc import MayerPresentationModel
from .tcell import MayerTCellModel


def build_mayer_model(params: ParameterSet) -> ComposedModel:
    """The default Mayer model, composed from its two pluggable components."""
    return ComposedModel(MAYER_STATES, [MayerTCellModel(params), MayerPresentationModel(params)])


def mayer_rhs(t: float, state: dict[str, float], params: ParameterSet) -> dict[str, float]:
    """Function-style RHS for direct use/testing -- equivalent to
    build_mayer_model(params).rhs(t, state)."""
    return build_mayer_model(params).rhs(t, state)


def simulate_mayer(
    params: ParameterSet,
    initial_conditions: dict[str, float],
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
    """Integrate the Mayer model from t0 to t_end and return a SimulationResult.

    Pass `model=` with a custom ComposedModel to override the default
    two-component composition (Phase 6 module-swap entry point).
    """
    composed = model if model is not None else build_mayer_model(params)

    def rhs(t: float, state: dict[str, float]) -> dict[str, float]:
        return composed.rhs(t, state)

    return _simulate(
        rhs,
        MAYER_STATES,
        initial_conditions,
        t_end,
        t0=t0,
        dose_schedule=None,
        t_eval=t_eval,
        method=method,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
        parameters=params,
        model_name="mayer",
    )
