"""PMHCDensityModel: pMHC surface density (per representative DC) for the
cancer/TCR-signaling model.

*** THIS PROJECT'S EXTENSION (docs/cancer_tcr_model.md) -- NOT a published
equation from either source paper. ***

    dpMHC_density/dt = k_load*aDC_Ag - mu_pMHC*pMHC_density

*** CORRECTED FORM -- caught by tests/test_cancer_tcr_model.py, recorded
here so it isn't silently "fixed" back. *** An earlier draft drove
production directly from ambient antigen (`k_load*Ag`), reasoning that
pMHC density is an intensive (per-DC) quantity and so shouldn't scale
with the *number* of DCs (aDC_Ag), unlike the integrated model's bulk
`dpMHC/dt = k_p*aDC_Ag - mu_pMHC*pMHC`. That is wrong, not just
differently-labeled: `Ag` decays fast (d_Ag=3/day) and independently of
DC recruitment, so pMHC_density could rise and peak BEFORE aDC_Ag itself
had built up -- i.e. antigen could apparently be "displayed" before any
DC had actually captured it, which is physically backwards.
`test_cancer_tcr_model_pipeline_causal_ordering` caught this (aDC_Ag's
peak must come no later than pMHC_density's peak).

The fix ties production to `aDC_Ag` -- the same causal dependency the
integrated model's pMHC layer uses, and for the same documented reason
(docs/equations.md Section 3): this makes pMHC_density a delayed,
low-pass-filtered readout of aDC_Ag, guaranteeing pMHC_density=0 whenever
aDC_Ag=0 and that its rise cannot precede aDC_Ag's own rise. The honest
cost of this fix, recorded rather than glossed over: pMHC_density is no
longer a strictly *intensive* per-single-DC quantity independent of DC
count -- it scales with aDC_Ag like an extensive pool would, just as the
integrated model's bulk pMHC does. This package does not separately
model per-DC heterogeneity in antigen loading; "per representative DC"
here means "the ensemble-average antigen-presentation signal driving TCR
engagement," not a literally-tracked single cell's surface density. If
real per-DC pMHC copy-number data becomes available, this is the
equation to revisit first.

k_load and mu_pMHC have no literature source for this specific
cancer-context pairing -- both are source_type=model_extension (or
user_defined when overridden) parameters, exactly like the integrated
model's k_p/mu_pMHC.
"""

from __future__ import annotations

from vaccine_tcell_model.core.interfaces import PresentationModel
from vaccine_tcell_model.parameters import ParameterSet


def pmhc_density_rhs(state: dict[str, float], params: ParameterSet) -> dict[str, float]:
    """dpMHC_density/dt = k_load*aDC_Ag - mu_pMHC*pMHC_density.

    Limiting case: if aDC_Ag=0 (no activated antigen-loaded DCs, e.g.
    before the first dose or long after full clearance), this reduces to
    dpMHC_density/dt = -mu_pMHC*pMHC_density, the analytic exponential
    decay -- same structural limiting-case guarantee as
    models/integrated/pmhc.py's aDC_Ag=0 case.
    """
    k_load = params.value("k_load")
    mu_pMHC = params.value("mu_pMHC")
    return {"pMHC_density": k_load * state["aDC_Ag"] - mu_pMHC * state["pMHC_density"]}


class PMHCDensityModel(PresentationModel):
    """Owns pMHC_density. Reads aDC_Ag from whatever InnateDCModel it is composed with."""

    def __init__(self, params: ParameterSet):
        self.params = params

    @property
    def owned_states(self) -> tuple[str, ...]:
        return ("pMHC_density",)

    def rhs(self, t: float, state: dict[str, float]) -> dict[str, float]:
        return pmhc_density_rhs(state, self.params)
