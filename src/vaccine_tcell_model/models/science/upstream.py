"""ScienceUpstreamModel: the pluggable "upstream innate/DC dynamics"
component (master spec Section 3/9, core.interfaces.InnateDCModel) --
antigen/adjuvant kinetics through activated antigen-loaded DCs (aDC_Ag).

Wraps the independently-tested antigen_adjuvant_rhs and innate_dc_rhs
functions (Eq. 1-5) verbatim. This file adds no new equations -- only the
RHSComponent shape needed for composition (Phase 6) and future swapping
(Phase 7, e.g. an integrated model that reuses this component unchanged
while swapping out the T-cell/Tfh components downstream of it).
"""

from __future__ import annotations

from vaccine_tcell_model.core.interfaces import InnateDCModel
from vaccine_tcell_model.parameters import ParameterSet

from .antigen_adjuvant import antigen_adjuvant_rhs
from .innate_dc import innate_dc_rhs


class ScienceUpstreamModel(InnateDCModel):
    """Owns Ag, Adj, TC, DC, aDC_Ag (Science supplement Eq. 1-5)."""

    def __init__(self, params: ParameterSet):
        self.params = params

    @property
    def owned_states(self) -> tuple[str, ...]:
        return ("Ag", "Adj", "TC", "DC", "aDC_Ag")

    def rhs(self, t: float, state: dict[str, float]) -> dict[str, float]:
        derivatives: dict[str, float] = {}
        derivatives.update(antigen_adjuvant_rhs(state, self.params))
        derivatives.update(innate_dc_rhs(state, self.params))
        return derivatives
