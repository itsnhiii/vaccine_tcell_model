"""Phase 4: Science-model validation against the reference implementation.

This module never changes a published equation. It exists to:

1. Reproduce the exact dosing schemes used to generate Bhagchandani et
   al. 2024 Fig. 1B/3F (transcribed from the reference implementation's
   `tCellModel.m` `defineSchemes()`), so DoseSchedule construction can be
   checked against known reference values rather than only our own
   hand-derived numbers (docs/equations.md Flag S5).
2. Provide a reference-parity T-cell RHS variant reproducing the
   undocumented floor clamp found in the reference code (`odeInnate.m`
   lines 180-184, docs/equations.md Flag S-code-1) -- NOT used by
   `models.science` -- purely so tests can quantify whether it matters
   for the paper's own dosing regimens, instead of assuming either way.
"""

from __future__ import annotations

import math

import numpy as np

from vaccine_tcell_model.core.state import SCIENCE_STATES
from vaccine_tcell_model.dosing import DoseSchedule
from vaccine_tcell_model.models.science.antigen_adjuvant import antigen_adjuvant_rhs
from vaccine_tcell_model.models.science.innate_dc import innate_dc_rhs
from vaccine_tcell_model.parameters import ParameterSet
from vaccine_tcell_model.solvers import simulate as _simulate

# -- Reference dosing schemes, transcribed verbatim from tCellModel.m -------
#
#   scheme.T       = {[12,12],[11,11],[12,12],[12,12],[7,7],[0,0]}
#   scheme.numshot = {[7,7],[6,6],[4,4],[3,3],[2,2],[1,1]}
#   scheme.k       = {[1,1],[1,1],[1,1],[1,1],[log(4),log(4)],[0,0]}
#   scheme.names   = {'7-ED','6-ED','4-ED','3-ED','2-ED','Bolus'}
#
# Note (docs/equations.md Flag S7): 6-ED uses an 11-day window, not 12 --
# transcribed as-is, not "corrected" to 12 to match the main text's prose
# ("the total time interval (12 days) ... constant").


def _escalation(numshot: int, k: float, duration: float, name: str) -> DoseSchedule:
    return DoseSchedule.from_exponential_escalation(
        numshot=numshot, k=k, duration=duration, name=name
    )


REFERENCE_DOSE_SCHEMES: dict[str, DoseSchedule] = {
    "7-ED": _escalation(7, 1.0, 12, "7-ED"),
    "6-ED": _escalation(6, 1.0, 11, "6-ED"),
    "4-ED": _escalation(4, 1.0, 12, "4-ED"),
    "3-ED": _escalation(3, 1.0, 12, "3-ED"),
    "2-ED": _escalation(2, math.log(4), 7, "2-ED"),
    "Bolus": _escalation(1, 0.0, 0, "Bolus"),
}
"""Ordered from most to fewest doses, matching the paper's canonical
Fig. 1B/3F comparison order."""

REFERENCE_DOSE_SCHEMES_ADJUVANT_BOLUS: dict[str, DoseSchedule] = {
    "7-ED (adjuvant bolus)": DoseSchedule.from_exponential_escalation(
        numshot=7, k=1.0, duration=12,
        adjuvant_numshot=1, adjuvant_k=0.0, adjuvant_duration=0.0,
        name="7-ED (adjuvant bolus)",
    ),
}
"""Science supplement fig. S2B: antigen escalates over 7 shots while
adjuvant is given as a single bolus at t=0."""


def tfh_at_day(result, day: float = 21.0) -> float:
    """TFH value at the trajectory point closest to `day`.

    Mirrors the reference code's `conc(2100, end)` -- "take day 21 T cell
    number" -- used to fit D0/T0 against Fig. 3F (docs/equations.md Flag S4).
    """
    idx = int(np.argmin(np.abs(result.time - day)))
    return float(result.trajectory("TFH")[idx])


def tcell_rhs_reference_with_floor(state: dict[str, float], params: ParameterSet) -> dict[str, float]:
    """Reference-parity T-cell RHS reproducing the undocumented floor
    clamp in `odeInnate.m` (lines 180-184): when T <= T0, dT/dt is
    replaced by the constant T0 instead of the published formula.

    NOT part of the Science reproduction in models.science -- exists
    only so tests can quantify whether this code-level safeguard changes
    anything under the paper's own dosing regimens.
    """
    T = state["T"]
    aDC_Ag = state["aDC_Ag"]
    alpha = params.value("alpha")
    eta = params.value("eta")
    T0 = params.value("T0")

    if T >= T0:
        denom = T + aDC_Ag
        proliferation = alpha * aDC_Ag * T / denom if denom > 0 else 0.0
        d_T = proliferation - eta * (T - T0)
    else:
        d_T = T0

    d_TFH = eta * (T - T0)
    return {"T": d_T, "TFH": d_TFH}


def _science_rhs_with_reference_floor(t, state, params):
    derivatives: dict[str, float] = {}
    derivatives.update(antigen_adjuvant_rhs(state, params))
    derivatives.update(innate_dc_rhs(state, params))
    derivatives.update(tcell_rhs_reference_with_floor(state, params))
    return derivatives


def simulate_science_reference_floor(
    params: ParameterSet,
    initial_conditions: dict[str, float],
    dose_schedule,
    t_end: float,
    **solver_kwargs,
):
    """Same as models.science.simulate_science, but with the reference
    code's undocumented T<=T0 floor clamp reproduced -- for validation
    comparison only."""

    def rhs(t, state):
        return _science_rhs_with_reference_floor(t, state, params)

    return _simulate(
        rhs,
        SCIENCE_STATES,
        initial_conditions,
        t_end,
        dose_schedule=dose_schedule,
        parameters=params,
        model_name="science_reference_floor",
        **solver_kwargs,
    )
