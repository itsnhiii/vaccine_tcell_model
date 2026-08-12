"""Assembled integrated model: Ag -> DC -> aDC_Ag -> pMHC -> T -> TFH.

Composes FOUR pluggable components (master spec Section 8-9):

    upstream     = ScienceUpstreamModel       Bhagchandani et al. 2024 Eq.1-5, UNCHANGED
    presentation = SimplePmhcProductionModel  THIS PROJECT'S EXTENSION
    tcell        = IntegratedTCellModel       THIS PROJECT'S EXTENSION
    tfh          = ScienceTfhModel            Bhagchandani et al. 2024 Eq.7, UNCHANGED

Two of the four components (upstream, tfh) are the exact, unmodified
Science classes from Phases 3/6 -- concrete proof that Phase 6's modular
architecture does its job: adding pMHC required writing only the two
genuinely new pieces below, without touching the DC or Tfh code at all.
"""

from __future__ import annotations

import numpy as np

from vaccine_tcell_model.core.composition import ComposedModel
from vaccine_tcell_model.core.state import INTEGRATED_STATES
from vaccine_tcell_model.dosing import DoseSchedule
from vaccine_tcell_model.models.science import ScienceUpstreamModel
from vaccine_tcell_model.parameters import ParameterSet
from vaccine_tcell_model.solvers import simulate as _simulate

from .pmhc import SimplePmhcProductionModel
from .tcell import IntegratedTCellModel
from .tfh import ScienceTfhModel


def build_integrated_model(params: ParameterSet) -> ComposedModel:
    """The default integrated model, composed from its four pluggable components."""
    return ComposedModel(
        INTEGRATED_STATES,
        [
            ScienceUpstreamModel(params),
            SimplePmhcProductionModel(params),
            IntegratedTCellModel(params),
            ScienceTfhModel(params),
        ],
    )


def integrated_rhs(t: float, state: dict[str, float], params: ParameterSet) -> dict[str, float]:
    """Function-style RHS for direct use/testing -- equivalent to
    build_integrated_model(params).rhs(t, state)."""
    return build_integrated_model(params).rhs(t, state)


def simulate_integrated(
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
    """Integrate the integrated model from t0 to t_end and return a SimulationResult.

    Pass `model=` with a custom ComposedModel to override the default
    four-component composition (Phase 6 module-swap entry point, e.g. to
    try an ExplicitPeptideProcessingModel in place of
    SimplePmhcProductionModel without editing this file).
    """
    composed = model if model is not None else build_integrated_model(params)

    def rhs(t: float, state: dict[str, float]) -> dict[str, float]:
        return composed.rhs(t, state)

    return _simulate(
        rhs,
        INTEGRATED_STATES,
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
        model_name="integrated",
    )
