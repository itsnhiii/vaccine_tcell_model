"""CancerTCellModel: antigen-specific T-cell dynamics for the cancer/
TCR-signaling model.

*** THIS PROJECT'S EXTENSION (docs/cancer_tcr_model.md) -- NOT a
published equation from either source paper. ***

    dT/dt = alpha*T*pMHC_density/(K_T + T + pMHC_density) - delta*(T - T0)

Structurally the same saturating-competition proliferation term as
models/integrated/tcell.py (itself structurally Mayer et al. 2019's
Eq.1, docs/equations.md Flag M1), with pMHC_density substituted for
pMHC/C.

*** CORRECTED FORM -- caught by tests/test_cancer_tcr_model.py, recorded
here so it isn't silently "fixed" back. *** An earlier draft of this
equation used Mayer's own unconditional death term `-delta*T` (reasoning:
"no Tfh compartment, so nothing to differentiate into"). That is wrong,
not just differently-labeled: it makes the antigen-free baseline
UNSTABLE -- T decays exponentially to zero over the simulation window
even with Ag(t)=0 for all time, which contradicts basic T-cell biology
(circulating baseline T cells are maintained by homeostatic
IL-7/IL-15-driven turnover, not left to die off absent antigen).
`test_cancer_tcr_model_null_case_no_dose_stays_at_baseline` and
`test_cancer_tcr_model_pipeline_causal_ordering` (T never exceeding T0
under a real dose) both caught this. The loss term is therefore
`-delta*(T-T0)`, a homeostatic RELAXATION toward baseline T0 -- zero
exactly at T=T0, pulling T back toward T0 if perturbed either direction.

This is the SAME functional form as Science's `-eta*(T-T0)`, but NOT the
same biological claim: Science's term specifically means "T cells
differentiate into Tfh" (there is a TFH state receiving exactly this
flux). Here there is no Tfh state and no differentiation claim -- this
term means only "net T-cell turnover relaxes the population toward a
homeostatic set point in the absence of continued antigen," a standard
and distinct population-biology mechanism that happens to share the same
math. `delta`'s value (0.22/day) is still borrowed from Mayer et al.
2019's Fig.1 fit (see parameters/cancer_tcr.py) as an order-of-magnitude
turnover rate -- not because Mayer's OWN interpretation of that number
(plain effector death) applies unmodified here.

K_T is a first-class, user-configurable growth-saturation parameter,
kept as a SEPARATE symbol from the TCR-signal module's K_D
(models/cancer_tcr/signal.py, docs/cancer_tcr_model.md) -- same warning
as models/integrated/tcell.py's K: do not equate K_T to an
experimentally measured biochemical K_D. K_T shapes how fast the T-cell
POPULATION grows; K_D shapes how strongly a single engaged TCR SIGNALS.
Conflating them would hide two independently-tunable, physically
different knobs behind one number.
"""

from __future__ import annotations

from vaccine_tcell_model.core.interfaces import TCellActivationModel
from vaccine_tcell_model.parameters import ParameterSet


def cancer_tcell_rhs(state: dict[str, float], params: ParameterSet) -> dict[str, float]:
    T = state["T"]
    pMHC_density = state["pMHC_density"]
    alpha = params.value("alpha")
    delta = params.value("delta")
    K_T = params.value("K_T")
    T0 = params.value("T0")

    denom = K_T + T + pMHC_density
    proliferation = alpha * T * pMHC_density / denom if denom > 0 else 0.0
    return {"T": proliferation - delta * (T - T0)}


class CancerTCellModel(TCellActivationModel):
    """Owns T. Reads pMHC_density from whatever PresentationModel it is composed with."""

    def __init__(self, params: ParameterSet):
        self.params = params

    @property
    def owned_states(self) -> tuple[str, ...]:
        return ("T",)

    def rhs(self, t: float, state: dict[str, float]) -> dict[str, float]:
        return cancer_tcell_rhs(state, self.params)
