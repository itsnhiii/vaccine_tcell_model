"""Tests for the integrated model extension (docs/equations.md Section 3).

Covers master spec Section 16's required checks (test_integrated_pmhc_
production, test_integrated_tcell_affinity_direction,
test_integrated_tfh_generation), the explicit aDC_Ag=0 analytic limiting
case from Section 16, and Phase 7's success criterion: the full
Ag -> DC -> aDC_Ag -> pMHC -> T -> TFH pipeline runs correctly end-to-end.
"""

from __future__ import annotations

import numpy as np
import pytest

from vaccine_tcell_model.dosing import DoseSchedule
from vaccine_tcell_model.models.integrated import (
    integrated_tcell_rhs,
    pmhc_production_rhs,
    simulate_integrated,
    tfh_rhs,
)
from vaccine_tcell_model.models.science import simulate_science
from vaccine_tcell_model.parameters import (
    default_integrated_parameters,
    default_science_parameters,
    integrated_default_initial_conditions,
    science_default_initial_conditions,
)

PARAMS = default_integrated_parameters(K=10.0)
IC = integrated_default_initial_conditions(PARAMS)


# -- component-level equation checks -----------------------------------------


def test_integrated_pmhc_production():
    # dpMHC/dt = k_p*aDC_Ag - mu_pMHC*pMHC
    state = {"aDC_Ag": 50.0, "pMHC": 10.0}
    out = pmhc_production_rhs(state, PARAMS)
    k_p, mu_pMHC = PARAMS.value("k_p"), PARAMS.value("mu_pMHC")
    assert out["pMHC"] == pytest.approx(k_p * 50.0 - mu_pMHC * 10.0)


def test_integrated_pmhc_production_zero_adc_ag_matches_analytic_decay():
    # Master spec Section 16's explicit limiting case: if aDC_Ag=0,
    # dpMHC/dt = -mu_pMHC*pMHC, matching the analytic exponential exactly.
    schedule = None  # no dosing -> Ag/Adj/aDC_Ag all stay 0 for all time
    ic = dict(IC)
    ic["pMHC"] = 100.0  # start with some pMHC, aDC_Ag=0 throughout
    t_eval = np.linspace(0, 10, 101)
    result = simulate_integrated(PARAMS, ic, schedule, t_end=10.0, t_eval=t_eval, rtol=1e-10, atol=1e-12)

    assert np.allclose(result.trajectory("aDC_Ag"), 0.0)
    analytic = 100.0 * np.exp(-PARAMS.value("mu_pMHC") * result.time)
    assert np.allclose(result.trajectory("pMHC"), analytic, rtol=1e-6)


def test_integrated_tcell_equation():
    # dT/dt = alpha*T*pMHC/(K+T+pMHC) - eta*(T-T0)
    state = {"T": 100.0, "pMHC": 50.0}
    out = integrated_tcell_rhs(state, PARAMS)
    alpha, eta, T0, K = (PARAMS.value(p) for p in ("alpha", "eta", "T0", "K"))
    expected = alpha * 100.0 * 50.0 / (K + 100.0 + 50.0) - eta * (100.0 - T0)
    assert out["T"] == pytest.approx(expected)


def test_integrated_tfh_generation():
    # dTFH/dt = eta*(T-T0) -- literally Science's Eq.7, reused.
    state = {"T": 200.0}
    out = tfh_rhs(state, PARAMS)
    eta, T0 = PARAMS.value("eta"), PARAMS.value("T0")
    assert out["TFH"] == pytest.approx(eta * (200.0 - T0))


# -- affinity direction --------------------------------------------------------


def test_integrated_tcell_affinity_direction():
    schedule = DoseSchedule.from_exponential_escalation(numshot=7, k=1.0, duration=12)
    t_eval = np.linspace(0, 21, 211)

    fold_expansions = {}
    for K in [1.0, 10.0, 100.0, 1000.0]:
        params = default_integrated_parameters(K=K)
        ic = integrated_default_initial_conditions(params)
        result = simulate_integrated(params, ic, schedule, t_end=21.0, t_eval=t_eval)
        fold_expansions[K] = result.trajectory("T").max() / ic["T"]

    ks = sorted(fold_expansions)
    values = [fold_expansions[k] for k in ks]
    assert all(a >= b for a, b in zip(values, values[1:])), (
        f"expected fold expansion to decrease with increasing K: {fold_expansions}"
    )


# -- end-to-end pipeline --------------------------------------------------------


def test_integrated_model_runs_end_to_end():
    schedule = DoseSchedule.bolus()
    result = simulate_integrated(PARAMS, IC, schedule, t_end=21.0, t_eval=np.linspace(0, 21, 211))
    assert result.model_name == "integrated"
    assert set(result.states.names) == {"Ag", "Adj", "TC", "DC", "aDC_Ag", "pMHC", "T", "TFH"}


def test_integrated_model_null_case_no_dose_stays_at_baseline():
    result = simulate_integrated(PARAMS, IC, dose_schedule=None, t_end=21.0, t_eval=np.linspace(0, 21, 22))
    for name in ["Ag", "Adj", "TC", "DC", "aDC_Ag", "pMHC"]:
        assert np.allclose(result.trajectory(name), 0.0), name
    assert np.allclose(result.trajectory("T"), PARAMS.value("T0"))
    assert np.allclose(result.trajectory("TFH"), 0.0)


def test_integrated_model_pipeline_ag_drives_adc_ag_drives_pmhc_drives_t_drives_tfh():
    schedule = DoseSchedule.bolus()
    result = simulate_integrated(PARAMS, IC, schedule, t_end=21.0, t_eval=np.linspace(0, 21, 211))

    assert result.trajectory("Ag").max() > 0.0
    assert result.trajectory("aDC_Ag").max() > 0.0
    assert result.trajectory("pMHC").max() > 0.0
    assert result.trajectory("T").max() > IC["T"]
    assert result.trajectory("TFH")[-1] > 0.0

    # Causal ordering: each downstream quantity should peak no earlier
    # than the quantity that drives it (aDC_Ag feeds pMHC feeds T).
    t = result.time
    t_peak_adc_ag = t[np.argmax(result.trajectory("aDC_Ag"))]
    t_peak_pmhc = t[np.argmax(result.trajectory("pMHC"))]
    t_peak_t = t[np.argmax(result.trajectory("T"))]
    assert t_peak_adc_ag <= t_peak_pmhc <= t_peak_t


def test_integrated_model_dose_schedule_changes_pmhc_and_t_trajectories():
    t_eval = np.linspace(0, 21, 211)
    bolus_result = simulate_integrated(PARAMS, IC, DoseSchedule.bolus(), t_end=21.0, t_eval=t_eval)
    seven_ed = DoseSchedule.from_exponential_escalation(numshot=7, k=1.0, duration=12)
    seven_ed_result = simulate_integrated(PARAMS, IC, seven_ed, t_end=21.0, t_eval=t_eval)

    assert not np.allclose(bolus_result.trajectory("pMHC"), seven_ed_result.trajectory("pMHC"))
    assert not np.allclose(bolus_result.trajectory("T"), seven_ed_result.trajectory("T"))


def test_integrated_upstream_and_tfh_components_are_literally_reused_from_science():
    # ScienceUpstreamModel and ScienceTfhModel are composed into the
    # integrated model unchanged (Phase 6/7) -- so, given the SAME
    # Science-shared parameters, dose schedule, and initial conditions,
    # the states they own must match a standalone Science-model run to
    # solver tolerance (not bit-for-bit, per the Phase 6 coupled-solver
    # floating-point note in test_modular_interfaces.py).
    science_params = default_science_parameters()
    science_ic = science_default_initial_conditions(science_params)
    schedule = DoseSchedule.from_exponential_escalation(numshot=7, k=1.0, duration=12)
    t_eval = np.linspace(0, 21, 211)

    science_result = simulate_science(science_params, science_ic, schedule, t_end=21.0, t_eval=t_eval)

    integrated_params = default_integrated_parameters(K=10.0)
    integrated_ic = integrated_default_initial_conditions(integrated_params)
    integrated_result = simulate_integrated(integrated_params, integrated_ic, schedule, t_end=21.0, t_eval=t_eval)

    for name in ["Ag", "Adj", "TC", "DC", "aDC_Ag"]:
        assert np.allclose(
            science_result.trajectory(name), integrated_result.trajectory(name), rtol=1e-4, atol=1e-9
        ), name
