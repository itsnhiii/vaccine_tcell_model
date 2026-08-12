"""pMHC (cognate presented antigen) dynamics -- Mayer et al. 2019 main text Eq.2.

    dC/dt = -mu*C

No production term: in the specific form implemented here (used in every
main-text figure), antigen "administration" enters through the initial
condition C(0), not through a dosing event. The general form with a
time-varying input nu(t) (Methods Eq.8) is documented but not
implemented -- see docs/equations.md Section 2.1.
"""

from __future__ import annotations

from vaccine_tcell_model.core.interfaces import PresentationModel
from vaccine_tcell_model.parameters import ParameterSet


def pmhc_rhs(state: dict[str, float], params: ParameterSet) -> dict[str, float]:
    mu = params.value("mu")
    return {"C": -mu * state["C"]}


class MayerPresentationModel(PresentationModel):
    """Owns C, the cognate presented pMHC (Mayer et al. 2019 main text Eq.2)."""

    def __init__(self, params: ParameterSet):
        self.params = params

    @property
    def owned_states(self) -> tuple[str, ...]:
        return ("C",)

    def rhs(self, t: float, state: dict[str, float]) -> dict[str, float]:
        return pmhc_rhs(state, self.params)
