"""Tissue-resident innate cell and dendritic cell kinetics --
Science supplement Eq. 3-5.

Published equations:

    d[TC]/dt     = [Adj]/(S_Adj+[Adj]) - mu*[TC]
    d[DC]/dt     = D0*[TC] - (1 + k*[Adj]/(S_Adj+[Adj]))*[DC]*[Ag] - mu*[DC]
    d[aDC_Ag]/dt = (1 + k*[Adj]/(S_Adj+[Adj]))*[DC]*[Ag] - mu*[aDC_Ag]
"""

from __future__ import annotations

from vaccine_tcell_model.parameters import ParameterSet


def innate_dc_rhs(state: dict[str, float], params: ParameterSet) -> dict[str, float]:
    Ag = state["Ag"]
    Adj = state["Adj"]
    TC = state["TC"]
    DC = state["DC"]
    aDC_Ag = state["aDC_Ag"]

    S_Adj = params.value("S_Adj")
    mu = params.value("mu")
    k = params.value("k")
    D0 = params.value("D0")

    adjuvant_activity = Adj / (S_Adj + Adj)
    uptake_enhancement = 1.0 + k * adjuvant_activity

    d_TC = adjuvant_activity - mu * TC
    d_DC = D0 * TC - uptake_enhancement * DC * Ag - mu * DC
    d_aDC_Ag = uptake_enhancement * DC * Ag - mu * aDC_Ag

    return {"TC": d_TC, "DC": d_DC, "aDC_Ag": d_aDC_Ag}
