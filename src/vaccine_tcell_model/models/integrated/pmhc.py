"""SimplePmhcProductionModel: pMHC production/decay.

*** THIS PROJECT'S INTEGRATED-MODEL EXTENSION (docs/equations.md Section
3) -- NOT a published equation from either source paper. ***

    dpMHC/dt = k_p*aDC_Ag - mu_pMHC*pMHC

Neither Bhagchandani et al. 2024 (no pMHC state at all -- docs/equations.md
Flag S1) nor Mayer et al. 2019 (whose pMHC/C has no production term tied
to a DC state; the general nu(t) input in Methods Eq.8 is never
instantiated as k_p*aDC_Ag) contains this equation. k_p and mu_pMHC are
source_type=model_extension parameters (master spec Section 12) unless a
specific literature source is supplied later.

A future ExplicitPeptideProcessingModel can replace this component
without changing the upstream DC code or the T-cell code downstream
(master spec Section 9) -- it only needs to own "pMHC" and read whatever
state the upstream component produces (here, "aDC_Ag").
"""

from __future__ import annotations

from vaccine_tcell_model.core.interfaces import PresentationModel
from vaccine_tcell_model.parameters import ParameterSet


def pmhc_production_rhs(state: dict[str, float], params: ParameterSet) -> dict[str, float]:
    """dpMHC/dt = k_p*aDC_Ag - mu_pMHC*pMHC.

    Limiting case (master spec Section 16): if aDC_Ag=0, this reduces to
    dpMHC/dt = -mu_pMHC*pMHC, matching the analytic exponential decay
    solution exactly -- verified in tests/test_integrated_model.py.
    """
    k_p = params.value("k_p")
    mu_pMHC = params.value("mu_pMHC")
    return {"pMHC": k_p * state["aDC_Ag"] - mu_pMHC * state["pMHC"]}


class SimplePmhcProductionModel(PresentationModel):
    """Owns pMHC. Reads aDC_Ag from whatever InnateDCModel it is composed with."""

    def __init__(self, params: ParameterSet):
        self.params = params

    @property
    def owned_states(self) -> tuple[str, ...]:
        return ("pMHC",)

    def rhs(self, t: float, state: dict[str, float]) -> dict[str, float]:
        return pmhc_production_rhs(state, self.params)
