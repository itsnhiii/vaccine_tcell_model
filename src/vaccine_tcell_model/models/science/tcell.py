"""T-cell priming and Tfh differentiation -- Science supplement Eq. 6-7.

    d[T]/dt   = alpha*[aDC_Ag]*[T]/([T]+[aDC_Ag]) - eta*([T]-T0)   (Eq.6)
    d[TFH]/dt = eta*([T]-T0)                                       (Eq.7)

Split into two independent functions/components (Phase 6), rather than
one combined function returning both derivatives, because the integrated
model (Phase 7) swaps Eq.6 out for a different T-cell equation while
keeping Eq.7's Tfh differentiation unchanged, reading whatever "T" the
swapped-in T-cell component produces. This is a pure architectural
decomposition -- neither formula changes.

No K (affinity) term anywhere in Eq.6 -- confirmed absent from the
published Science equations (docs/equations.md Flag S1). K is introduced
only by the Mayer model (Phase 5) and this project's integrated
extension (Phase 7).
"""

from __future__ import annotations

from vaccine_tcell_model.core.interfaces import TCellActivationModel, TfhModel
from vaccine_tcell_model.parameters import ParameterSet


def tcell_activation_rhs(state: dict[str, float], params: ParameterSet) -> dict[str, float]:
    """Eq.6: d[T]/dt = alpha*aDC_Ag*T/(T+aDC_Ag) - eta*(T-T0)."""
    T = state["T"]
    aDC_Ag = state["aDC_Ag"]
    alpha = params.value("alpha")
    eta = params.value("eta")
    T0 = params.value("T0")

    denom = T + aDC_Ag
    # denom == 0 only if T == aDC_Ag == 0; the published formula's limit
    # there is 0 (no T cells, nothing to proliferate), not undefined.
    proliferation = alpha * aDC_Ag * T / denom if denom > 0 else 0.0
    return {"T": proliferation - eta * (T - T0)}


def tfh_rhs(state: dict[str, float], params: ParameterSet) -> dict[str, float]:
    """Eq.7: d[TFH]/dt = eta*(T-T0)."""
    T = state["T"]
    eta = params.value("eta")
    T0 = params.value("T0")
    return {"TFH": eta * (T - T0)}


class ScienceTCellModel(TCellActivationModel):
    """Owns T (Science supplement Eq.6)."""

    def __init__(self, params: ParameterSet):
        self.params = params

    @property
    def owned_states(self) -> tuple[str, ...]:
        return ("T",)

    def rhs(self, t: float, state: dict[str, float]) -> dict[str, float]:
        return tcell_activation_rhs(state, self.params)


class ScienceTfhModel(TfhModel):
    """Owns TFH (Science supplement Eq.7)."""

    def __init__(self, params: ParameterSet):
        self.params = params

    @property
    def owned_states(self) -> tuple[str, ...]:
        return ("TFH",)

    def rhs(self, t: float, state: dict[str, float]) -> dict[str, float]:
        return tfh_rhs(state, self.params)
