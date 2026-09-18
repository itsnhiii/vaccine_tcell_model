"""TCR signaling strength: closed-form kinetic-proofreading calculation.

*** THIS PROJECT'S EXTENSION, built on the mechanism reviewed in
Chakraborty & Weiss 2014 (Nat. Immunol. 15:797-807, "Insights into the
initiation of TCR signaling") and McKeithan 1995 (PNAS 92:5042-5046,
the original kinetic-proofreading model) -- NOT a published equation
verbatim from either paper. The mechanism (a chain of first-order steps,
each fully reset by ligand dissociation) is McKeithan's; the specific
combination with a pMHC-density occupancy term and a population
contact-rate term below is this project's own construction, built to
answer the question this package exists to answer: how does TCR
signaling strength compare across different dosing schedules. See
docs/cancer_tcr_model.md for the full derivation and every caveat below.

Deliberately NOT an RHSComponent / ODE state. Both pieces below are
algebraic, closed-form functions of already-simulated trajectories
(pMHC_density, T, aDC_Ag) from models.cancer_tcr.full_model, not new
differential equations:

1. Occupancy (pseudo-first-order, Michaelis-Menten-shaped):
       Theta = pMHC_density / (pMHC_density + K_D)
   The true TCR+pMHC binding step is bimolecular (second-order), but
   because pMHC sits at a site with a roughly constant local TCR density,
   both source documents treat it as pseudo-first-order saturation in
   pMHC availability -- this is exactly that reduction.

2. Kinetic-proofreading amplification (first-order chain, reduced to its
   quasi-steady-state closed form):
       k_off = K_D * k_on
       A     = (k_p / (k_p + k_off)) ** N
   McKeithan's model: after binding, N first-order forward steps (rate
   k_p each) must complete before signal; ANY intermediate state fully
   resets to unbound the instant the ligand dissociates (rate k_off, not
   a separate parameter). A is the steady-state probability a bound
   complex survives all N steps. Using the closed form (rather than
   integrating the N-state linear ODE chain explicitly) is justified by a
   timescale-separation argument: the chain completes in seconds
   (Tischer & Weiner 2019; Yousefi et al. 2019 report per-step delays of
   seconds for a ~2-4 step chain), while this package's population
   dynamics (Ag/DC/T) evolve over hours-days -- a >1000-fold separation.
   See docs/cancer_tcr_model.md "Quasi-steady-state justification" for
   the caveats this glosses over (notably: momentarily invalid at a
   discontinuous dose-jump instant, invisible at this model's day-scale
   output resolution).

K_D-UNITS CAVEAT (flagged, not silently resolved -- read before trusting
absolute S_contact/S_pop magnitudes): `pMHC_density` is an arbitrary,
uncalibrated model quantity (docs/cancer_tcr_model.md), not a measured
molecule count. `K_D` (supplied in uM) is used DIRECTLY as the occupancy
saturation constant against `pMHC_density` -- i.e. this implicitly
assumes K_D's numeric value is expressed on the same normalized scale as
pMHC_density, not literal molar concentration. This is the single
biggest unvalidated assumption in this module. It does NOT affect
`k_off` (used only in the proofreading term A), which is computed from
K_D and k_on in real, physically consistent units (uM and uM^-1 s^-1) --
only Theta (and therefore S_contact/S_pop) inherits this caveat. Until
real pMHC copy-number data lets us calibrate pMHC_density's scale, trust
RELATIVE comparisons (across affinities/dosing schedules) from this
module, not absolute magnitudes.

SERIAL-TRIGGERING: Chakraborty & Weiss 2014 report "very little evidence"
for an optimal-dwell-time/bell-curve response (Holler & Kranz 2003's
TCR-affinity-variant data is monotonic, no observed optimum) -- so no
such correction is applied here. `tcr_signal_params.value("include_serial_
triggering")` is reserved for this and raises NotImplementedError if set,
rather than silently doing nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vaccine_tcell_model.core.result import SimulationResult
from vaccine_tcell_model.parameters import ParameterSet


def tcr_signal_from_arrays(
    pmhc_density: np.ndarray,
    T: np.ndarray,
    DC: np.ndarray,
    tcr_signal_params: ParameterSet,
) -> dict[str, np.ndarray]:
    """Closed-form TCR signaling strength from already-simulated trajectories.

    Args:
        pmhc_density: pMHC_density(t) trajectory (per representative DC).
        T: T(t) trajectory (antigen-specific T-cell count).
        DC: the DC count relevant to T-cell engagement -- pass aDC_Ag(t)
            (activated, antigen-loaded DCs), not the naive DC(t) pool.
        tcr_signal_params: from parameters.default_tcr_signal_parameters
            (K_D, k_on, k_p, N, include_serial_triggering).

    Returns a dict of same-length arrays:
        k_off, Theta, A, S_contact (per-cell/per-engagement signal),
        contacts (saturating T-DC meeting rate), S_pop (population
        aggregate signal = S_contact * contacts).
    """
    if tcr_signal_params.value("include_serial_triggering"):
        raise NotImplementedError(
            "include_serial_triggering is reserved but not yet implemented. "
            "Chakraborty & Weiss 2014 find no clear experimental support for "
            "an optimal-dwell-time/bell-curve downturn (Holler & Kranz 2003 "
            "data is monotonic), so this correction is deliberately left "
            "unimplemented pending validation against experimental data. "
            "See models/cancer_tcr/signal.py and docs/cancer_tcr_model.md."
        )

    K_D = tcr_signal_params.value("K_D")
    k_on = tcr_signal_params.value("k_on")
    k_p = tcr_signal_params.value("k_p")
    N = tcr_signal_params.value("N")

    k_off = K_D * k_on  # 1/s: K_D [uM] * k_on [1/uM/s] -- no separate unit conversion needed, see module docstring

    pmhc_density = np.asarray(pmhc_density, dtype=float)
    T = np.asarray(T, dtype=float)
    DC = np.asarray(DC, dtype=float)

    Theta = pmhc_density / (pmhc_density + K_D)
    A = (k_p / (k_p + k_off)) ** N
    S_contact = Theta * A

    contact_denom = T + DC
    contacts = np.divide(
        T * DC, contact_denom, out=np.zeros_like(contact_denom, dtype=float), where=contact_denom > 0
    )
    S_pop = S_contact * contacts

    return {
        "k_off": np.full_like(pmhc_density, k_off),
        "Theta": Theta,
        "A": np.full_like(pmhc_density, A),
        "S_contact": S_contact,
        "contacts": contacts,
        "S_pop": S_pop,
    }


def compute_tcr_signal(result: SimulationResult, tcr_signal_params: ParameterSet) -> pd.DataFrame:
    """Augment a cancer_tcr SimulationResult with TCR-signal columns.

    Reads pMHC_density, T, and aDC_Ag from `result` (must be a
    models.cancer_tcr.full_model.simulate_cancer_tcr output, or anything
    with those three state columns). Returns a NEW DataFrame
    (result.data plus k_off/Theta/A/S_contact/contacts/S_pop columns) --
    does not mutate `result` in place.
    """
    signal = tcr_signal_from_arrays(
        result.trajectory("pMHC_density"),
        result.trajectory("T"),
        result.trajectory("aDC_Ag"),
        tcr_signal_params,
    )
    out = result.data.copy()
    for name, values in signal.items():
        out[name] = values
    return out
