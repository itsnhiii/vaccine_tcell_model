"""IntegratedTCellModel: T-cell activation for the integrated model.

*** THIS PROJECT'S INTEGRATED-MODEL EXTENSION (docs/equations.md Section
3) -- NOT a published equation from either source paper. ***

    dT/dt = alpha*T*pMHC/(K + T + pMHC) - eta*(T - T0)

This is a hybrid: the proliferation term's saturating-competition form
(T*pMHC/(K+T+pMHC)) is structurally Mayer et al. 2019's Eq.1
(docs/equations.md Flag M1) with pMHC substituted for Mayer's C -- but
the loss term is Science's differentiation-sink -eta*(T-T0)
(Bhagchandani et al. 2024 Eq.6), NOT Mayer's own -delta*T death term.
Neither paper combines these. Despite the structural resemblance to
Mayer's equation, this is therefore implemented fresh here rather than
by reusing models.mayer.MayerTCellModel, which hard-codes Mayer's own
-delta*T sink and the state name "C".

K is a first-class, user-configurable affinity parameter (master spec
Section 23): lower K -> stronger functional avidity -> faster/greater
expansion, same interpretation as in the Mayer model. Do not equate K to
an experimentally measured biochemical KD.
"""

from __future__ import annotations

from vaccine_tcell_model.core.interfaces import TCellActivationModel
from vaccine_tcell_model.parameters import ParameterSet


def integrated_tcell_rhs(state: dict[str, float], params: ParameterSet) -> dict[str, float]:
    T = state["T"]
    pMHC = state["pMHC"]
    alpha = params.value("alpha")
    eta = params.value("eta")
    T0 = params.value("T0")
    K = params.value("K")

    denom = K + T + pMHC
    proliferation = alpha * T * pMHC / denom if denom > 0 else 0.0
    return {"T": proliferation - eta * (T - T0)}


class IntegratedTCellModel(TCellActivationModel):
    """Owns T. Reads pMHC from whatever PresentationModel it is composed with."""

    def __init__(self, params: ParameterSet):
        self.params = params

    @property
    def owned_states(self) -> tuple[str, ...]:
        return ("T",)

    def rhs(self, t: float, state: dict[str, float]) -> dict[str, float]:
        return integrated_tcell_rhs(state, self.params)
