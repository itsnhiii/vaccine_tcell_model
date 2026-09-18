"""Assembled cancer/TCR-signaling model: Ag -> DC -> aDC_Ag -> pMHC_density -> T.

Composes THREE pluggable components (docs/cancer_tcr_model.md):

    upstream     = ScienceUpstreamModel   Bhagchandani et al. 2024 Eq.1-5, UNCHANGED
    presentation = PMHCDensityModel       THIS PROJECT'S CANCER-CONTEXT EXTENSION
    tcell        = CancerTCellModel       THIS PROJECT'S CANCER-CONTEXT EXTENSION

No TfhModel component -- there is no humoral/germinal-center output in
this cancer-context model (see core.state.CANCER_TCR_STATES). Same
proof-of-modularity point as models/integrated/full_model.py: reusing
ScienceUpstreamModel unchanged here required writing only the two
genuinely new pieces, without touching the DC code at all.

The TCR signaling strength itself (S_contact, S_pop) is NOT part of this
ODE -- it is a closed-form function of the simulated trajectories,
computed separately by models.cancer_tcr.signal.compute_tcr_signal on
this function's output.
"""

from __future__ import annotations

import numpy as np

from vaccine_tcell_model.core.composition import ComposedModel
from vaccine_tcell_model.core.state import CANCER_TCR_STATES
from vaccine_tcell_model.dosing import DoseSchedule
from vaccine_tcell_model.models.science import ScienceUpstreamModel
from vaccine_tcell_model.parameters import ParameterSet
from vaccine_tcell_model.solvers import simulate as _simulate

from .pmhc_density import PMHCDensityModel
from .tcell import CancerTCellModel


def build_cancer_tcr_model(params: ParameterSet) -> ComposedModel:
    """The default cancer_tcr model, composed from its three pluggable components."""
    return ComposedModel(
        CANCER_TCR_STATES,
        [
            ScienceUpstreamModel(params),
            PMHCDensityModel(params),
            CancerTCellModel(params),
        ],
    )


def cancer_tcr_rhs(t: float, state: dict[str, float], params: ParameterSet) -> dict[str, float]:
    """Function-style RHS for direct use/testing -- equivalent to
    build_cancer_tcr_model(params).rhs(t, state)."""
    return build_cancer_tcr_model(params).rhs(t, state)


def simulate_cancer_tcr(
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
    """Integrate the cancer_tcr model from t0 to t_end and return a SimulationResult.

    Pass `model=` with a custom ComposedModel to override the default
    three-component composition (same module-swap entry point as
    simulate_integrated).

    This returns population-dynamics trajectories only (Ag, Adj, TC, DC,
    aDC_Ag, pMHC_density, T). To get the TCR-signal columns (S_contact,
    S_pop), pass the result to
    models.cancer_tcr.signal.compute_tcr_signal(result, tcr_signal_params)
    -- kept as a separate step since the TCR-signal parameters (K_D,
    k_on, k_p, N) live in an independent ParameterSet, never merged with
    this one (see parameters/cancer_tcr.py).
    """
    composed = model if model is not None else build_cancer_tcr_model(params)

    def rhs(t: float, state: dict[str, float]) -> dict[str, float]:
        return composed.rhs(t, state)

    return _simulate(
        rhs,
        CANCER_TCR_STATES,
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
        model_name="cancer_tcr",
    )
