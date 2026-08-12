"""Antigen-specific T-cell dynamics -- Mayer et al. 2019 main text Eq.1.

    dT/dt = alpha*T*C/(K+T+C) - delta*T

This is the first-order-expansion approximation of the exact competitive
quasi-steady-state binding solution (SI Eq. S12-S13), which is what every
figure in the paper actually uses -- see docs/equations.md Flag M1 for
the exact form and its documented worst-case ~2x deviation at K->0, T=C.
Not implemented here: the exact form, the multi-clone competitive
extension (SI section 1C), or the T-cell-grazing alternative (SI
section 3).

Interpretation: lower K -> stronger functional avidity -> faster/greater
T-cell expansion; higher K -> weaker functional avidity -> slower/lesser
expansion. Do not equate K to an experimentally measured biochemical KD
(main text: a future mapping from biochemical affinity to functional K
must be explicitly modeled, not assumed).
"""

from __future__ import annotations

from vaccine_tcell_model.core.interfaces import TCellActivationModel
from vaccine_tcell_model.parameters import ParameterSet


def tcell_rhs(state: dict[str, float], params: ParameterSet) -> dict[str, float]:
    T = state["T"]
    C = state["C"]
    alpha = params.value("alpha")
    delta = params.value("delta")
    K = params.value("K")

    denom = K + T + C
    proliferation = alpha * T * C / denom if denom > 0 else 0.0
    d_T = proliferation - delta * T
    return {"T": d_T}


class MayerTCellModel(TCellActivationModel):
    """Owns T (Mayer et al. 2019 main text Eq.1), reads C."""

    def __init__(self, params: ParameterSet):
        self.params = params

    @property
    def owned_states(self) -> tuple[str, ...]:
        return ("T",)

    def rhs(self, t: float, state: dict[str, float]) -> dict[str, float]:
        return tcell_rhs(state, self.params)
