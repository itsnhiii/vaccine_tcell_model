"""Phase 8: integrated model validation.

Covers master spec Phase 8's required checks: dose response, schedule
response, affinity response, pMHC response, T-cell response, Tfh
response, and a conservation/flow-logic check. Where a check overlaps
with something Phase 7 already tested (e.g. affinity direction), this
file goes further (finer sweeps, cross-condition consistency) rather
than duplicating the same assertion.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import spearmanr

from vaccine_tcell_model.dosing import DoseSchedule
from vaccine_tcell_model.models.integrated import simulate_integrated
from vaccine_tcell_model.parameters import default_integrated_parameters, integrated_default_initial_conditions
from vaccine_tcell_model.validation.science import REFERENCE_DOSE_SCHEMES, tfh_at_day

T_EVAL_21D = np.linspace(0, 21, 211)


def _run(params=None, ic=None, schedule=None, t_end=21.0, t_eval=T_EVAL_21D):
    params = params if params is not None else default_integrated_parameters(K=10.0)
    ic = ic if ic is not None else integrated_default_initial_conditions(params)
    schedule = schedule if schedule is not None else DoseSchedule.bolus()
    return simulate_integrated(params, ic, schedule, t_end=t_end, t_eval=t_eval)


# -- dose response --------------------------------------------------------------


def test_dose_response_increasing_total_dose_increases_pmhc_and_t():
    params = default_integrated_parameters(K=10.0)
    ic = integrated_default_initial_conditions(params)

    pmhc_peaks, t_peaks = {}, {}
    for total_dose in [0.5, 1.0, 2.0, 5.0, 10.0]:
        schedule = DoseSchedule.bolus(total_antigen_dose=total_dose, total_adjuvant_dose=total_dose)
        result = _run(params, ic, schedule)
        pmhc_peaks[total_dose] = result.trajectory("pMHC").max()
        t_peaks[total_dose] = result.trajectory("T").max()

    doses = sorted(pmhc_peaks)
    assert all(pmhc_peaks[a] <= pmhc_peaks[b] for a, b in zip(doses, doses[1:])), pmhc_peaks
    assert all(t_peaks[a] <= t_peaks[b] for a, b in zip(doses, doses[1:])), t_peaks


# -- schedule response ------------------------------------------------------------


def test_schedule_response_broadly_matches_validated_science_ordering():
    # Phase 4 validated a STRICT 7-ED > 6-ED > 4-ED > 3-ED > 2-ED > Bolus
    # ordering for the Science model's own TFH(day21) output. Re-running
    # the same six regimens here shows that ordering is NOT strictly
    # preserved once the integrated model's pMHC layer is added: 7-ED and
    # 6-ED, already separated by only ~0.6% in the Science-only
    # validation (docs/equations.md Section 6), swap (6-ED slightly
    # exceeds 7-ED here). This was confirmed reproducible at tight solver
    # tolerance on a 20x finer grid (rtol=1e-10, atol=1e-12, 4200 points)
    # -- not numerical noise. It's a genuine, documented behavior change:
    # the pMHC accumulation layer (its own k_p/mu_pMHC timescale) acts as
    # a low-pass filter on the upstream aDC_Ag signal and can reorder
    # near-ties (docs/equations.md Section 7). This test therefore checks
    # the ROBUST part of the trend -- Bolus is unambiguously weakest, and
    # dose count positively correlates with response overall -- rather
    # than asserting strict pairwise monotonicity the data doesn't support.
    order = ["7-ED", "6-ED", "4-ED", "3-ED", "2-ED", "Bolus"]
    dose_counts = [7, 6, 4, 3, 2, 1]
    params = default_integrated_parameters(K=10.0)
    ic = integrated_default_initial_conditions(params)

    tfh_day21 = {}
    for name in order:
        result = _run(params, ic, REFERENCE_DOSE_SCHEMES[name])
        tfh_day21[name] = tfh_at_day(result, 21.0)

    values = [tfh_day21[name] for name in order]

    assert tfh_day21["Bolus"] == min(values), tfh_day21
    assert tfh_day21["Bolus"] < 0.5 * min(values[:-1]), tfh_day21

    rho, _ = spearmanr(dose_counts, values)
    assert rho > 0.8, (rho, tfh_day21)


# -- affinity response (finer than Phase 7's spot-check) --------------------------


def test_affinity_response_fold_expansion_strictly_decreases_across_fine_k_grid():
    params_ref = default_integrated_parameters(K=10.0)
    ic = integrated_default_initial_conditions(params_ref)
    schedule = DoseSchedule.from_exponential_escalation(numshot=7, k=1.0, duration=12)

    fold = {}
    for K in np.logspace(0, 4, 9):
        params = default_integrated_parameters(K=float(K))
        result = _run(params, ic, schedule)
        fold[K] = result.trajectory("T").max() / ic["T"]

    ks = sorted(fold)
    values = [fold[k] for k in ks]
    assert all(a >= b for a, b in zip(values, values[1:])), fold


# -- pMHC response: sensitivity to its own new parameters -------------------------


def test_pmhc_response_increases_with_kp():
    ic = integrated_default_initial_conditions(default_integrated_parameters())
    schedule = DoseSchedule.bolus()

    peaks = {}
    for k_p in [0.1, 1.0, 10.0, 100.0]:
        params = default_integrated_parameters(K=10.0, k_p=k_p)
        result = _run(params, ic, schedule)
        peaks[k_p] = result.trajectory("pMHC").max()

    values = [peaks[k] for k in sorted(peaks)]
    assert all(a <= b for a, b in zip(values, values[1:])), peaks


def test_pmhc_response_decreases_with_mu_pmhc():
    ic = integrated_default_initial_conditions(default_integrated_parameters())
    schedule = DoseSchedule.bolus()

    peaks = {}
    for mu_pMHC in [0.1, 1.0, 10.0, 100.0]:
        params = default_integrated_parameters(K=10.0, mu_pMHC=mu_pMHC)
        result = _run(params, ic, schedule)
        peaks[mu_pMHC] = result.trajectory("pMHC").max()

    values = [peaks[m] for m in sorted(peaks)]
    assert all(a >= b for a, b in zip(values, values[1:])), peaks


# -- T-cell response to pMHC exposure ----------------------------------------------


def test_tcell_response_scales_with_pmhc_exposure_via_kp():
    # Indirect analogue of Mayer's "response changes with pMHC exposure"
    # check (test_mayer_model.py::test_response_changes_with_pmhc_exposure):
    # more pMHC production (higher k_p) must drive greater T expansion.
    ic = integrated_default_initial_conditions(default_integrated_parameters())
    schedule = DoseSchedule.bolus()

    t_peaks = {}
    for k_p in [0.1, 1.0, 10.0, 100.0]:
        params = default_integrated_parameters(K=10.0, k_p=k_p)
        result = _run(params, ic, schedule)
        t_peaks[k_p] = result.trajectory("T").max()

    values = [t_peaks[k] for k in sorted(t_peaks)]
    assert all(a <= b for a, b in zip(values, values[1:])), t_peaks


# -- Tfh response tracks T expansion across many varied conditions ----------------


def test_tfh_response_is_rank_consistent_with_t_expansion_across_varied_conditions():
    # Across a broad, heterogeneous set of conditions (schedule x K),
    # whichever run has a larger peak T should never end with a strictly
    # smaller final TFH than a run with smaller peak T -- TFH is fed
    # entirely by T's excess above baseline, so this should hold as a
    # systemic (not single-sweep) consistency check.
    params_template = default_integrated_parameters(K=10.0)
    ic = integrated_default_initial_conditions(params_template)

    conditions = []
    for K in [1.0, 10.0, 100.0]:
        for name, schedule in {
            "bolus": DoseSchedule.bolus(),
            "2-ED": DoseSchedule(times=[0, 7], antigen_fractions=[0.2, 0.8], adjuvant_fractions=[0.2, 0.8]),
            "7-ED": DoseSchedule.from_exponential_escalation(numshot=7, k=1.0, duration=12),
        }.items():
            params = default_integrated_parameters(K=K)
            result = _run(params, ic, schedule)
            conditions.append((result.trajectory("T").max(), result.trajectory("TFH")[-1]))

    conditions.sort(key=lambda pair: pair[0])
    tfh_values = [tfh for _, tfh in conditions]
    # Allow ties, but never a decrease as T_max increases across the sorted set.
    assert all(a <= b + 1e-6 for a, b in zip(tfh_values, tfh_values[1:])), conditions


# -- conservation / flow-logic check -----------------------------------------------


def test_flow_logic_t_plus_tfh_is_nondecreasing():
    # d(T+TFH)/dt = [proliferation - eta*(T-T0)] + [eta*(T-T0)]
    #             = alpha*T*pMHC/(K+T+pMHC) >= 0 always
    # (given alpha,T,pMHC,K >= 0), regardless of the eta-mediated
    # redistribution between T and TFH. This is a real invariant of the
    # equations, not just an empirical trend -- verify it holds under
    # every schedule x K combination Phase 8 exercises above.
    for K in [1.0, 10.0, 100.0, 1000.0]:
        params = default_integrated_parameters(K=K)
        ic = integrated_default_initial_conditions(params)
        for schedule in [
            DoseSchedule.bolus(),
            DoseSchedule(times=[0, 7], antigen_fractions=[0.2, 0.8], adjuvant_fractions=[0.2, 0.8]),
            DoseSchedule.from_exponential_escalation(numshot=7, k=1.0, duration=12),
        ]:
            result = _run(params, ic, schedule)
            t_plus_tfh = result.trajectory("T") + result.trajectory("TFH")
            assert np.all(np.diff(t_plus_tfh) >= -1e-6), (K, schedule.name)
