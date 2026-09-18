"""Tests for the cancer/TCR-signaling model (docs/cancer_tcr_model.md).

Mirrors tests/test_integrated_model.py's structure: component-level
equation checks, a direction-of-effect check, end-to-end pipeline checks,
plus a new section for the closed-form TCR-signal calculation
(models/cancer_tcr/signal.py) that the integrated model has no
equivalent of.
"""

from __future__ import annotations

import numpy as np
import pytest

from vaccine_tcell_model.dosing import DoseSchedule
from vaccine_tcell_model.models.cancer_tcr import (
    cancer_tcell_rhs,
    compute_tcr_signal,
    pmhc_density_rhs,
    simulate_cancer_tcr,
    tcr_signal_from_arrays,
)
from vaccine_tcell_model.models.science import simulate_science
from vaccine_tcell_model.parameters import (
    cancer_tcr_default_initial_conditions,
    default_cancer_tcr_parameters,
    default_science_parameters,
    default_tcr_signal_parameters,
    science_default_initial_conditions,
)

PARAMS = default_cancer_tcr_parameters()
IC = cancer_tcr_default_initial_conditions(PARAMS)
SIGNAL_PARAMS = default_tcr_signal_parameters()


# -- component-level equation checks -----------------------------------------


def test_cancer_tcr_pmhc_density_rhs():
    # dpMHC_density/dt = k_load*aDC_Ag - mu_pMHC*pMHC_density
    state = {"aDC_Ag": 5.0, "pMHC_density": 10.0}
    out = pmhc_density_rhs(state, PARAMS)
    k_load, mu_pMHC = PARAMS.value("k_load"), PARAMS.value("mu_pMHC")
    assert out["pMHC_density"] == pytest.approx(k_load * 5.0 - mu_pMHC * 10.0)


def test_cancer_tcr_pmhc_density_zero_adc_ag_matches_analytic_decay():
    ic = dict(IC)
    ic["pMHC_density"] = 100.0  # start with some density, aDC_Ag=0 throughout (no dosing)
    t_eval = np.linspace(0, 10, 101)
    result = simulate_cancer_tcr(PARAMS, ic, dose_schedule=None, t_end=10.0, t_eval=t_eval, rtol=1e-10, atol=1e-12)

    assert np.allclose(result.trajectory("aDC_Ag"), 0.0)
    analytic = 100.0 * np.exp(-PARAMS.value("mu_pMHC") * result.time)
    assert np.allclose(result.trajectory("pMHC_density"), analytic, rtol=1e-6)


def test_cancer_tcr_tcell_equation():
    # dT/dt = alpha*T*pMHC_density/(K_T+T+pMHC_density) - delta*(T-T0)
    state = {"T": 100.0, "pMHC_density": 50.0}
    out = cancer_tcell_rhs(state, PARAMS)
    alpha, delta, K_T, T0 = (PARAMS.value(p) for p in ("alpha", "delta", "K_T", "T0"))
    expected = alpha * 100.0 * 50.0 / (K_T + 100.0 + 50.0) - delta * (100.0 - T0)
    assert out["T"] == pytest.approx(expected)


def test_cancer_tcr_tcell_homeostatic_baseline_is_stable_without_antigen():
    # No Tfh sink, but the loss term is -delta*(T-T0) (a homeostatic
    # relaxation, not Mayer's own unconditional -delta*T death) so that
    # the antigen-free baseline is a stable fixed point, not a slow decay
    # to zero -- see models/cancer_tcr/tcell.py's docstring for why the
    # unconditional-death version was wrong (caught by
    # test_cancer_tcr_model_null_case_no_dose_stays_at_baseline below).
    T0 = PARAMS.value("T0")
    delta = PARAMS.value("delta")

    at_baseline = cancer_tcell_rhs({"T": T0, "pMHC_density": 0.0}, PARAMS)
    assert at_baseline["T"] == pytest.approx(0.0)

    above_baseline = cancer_tcell_rhs({"T": T0 + 10.0, "pMHC_density": 0.0}, PARAMS)
    assert above_baseline["T"] == pytest.approx(-delta * 10.0)
    assert above_baseline["T"] < 0.0  # relaxes back down toward T0

    below_baseline = cancer_tcell_rhs({"T": T0 - 10.0, "pMHC_density": 0.0}, PARAMS)
    assert below_baseline["T"] == pytest.approx(delta * 10.0)
    assert below_baseline["T"] > 0.0  # relaxes back up toward T0


# -- affinity/growth direction --------------------------------------------------


def test_cancer_tcr_tcell_growth_saturation_direction():
    schedule = DoseSchedule.from_exponential_escalation(numshot=7, k=1.0, duration=12)
    t_eval = np.linspace(0, 21, 211)

    fold_expansions = {}
    for K_T in [1.0, 10.0, 100.0, 1000.0]:
        params = default_cancer_tcr_parameters(K_T=K_T)
        ic = cancer_tcr_default_initial_conditions(params)
        result = simulate_cancer_tcr(params, ic, schedule, t_end=21.0, t_eval=t_eval)
        fold_expansions[K_T] = result.trajectory("T").max() / ic["T"]

    ks = sorted(fold_expansions)
    values = [fold_expansions[k] for k in ks]
    assert all(a >= b for a, b in zip(values, values[1:])), (
        f"expected fold expansion to decrease with increasing K_T: {fold_expansions}"
    )


# -- end-to-end pipeline --------------------------------------------------------


def test_cancer_tcr_model_runs_end_to_end():
    schedule = DoseSchedule.bolus()
    result = simulate_cancer_tcr(PARAMS, IC, schedule, t_end=21.0, t_eval=np.linspace(0, 21, 211))
    assert result.model_name == "cancer_tcr"
    assert set(result.states.names) == {"Ag", "Adj", "TC", "DC", "aDC_Ag", "pMHC_density", "T"}
    assert "TFH" not in result.data.columns


def test_cancer_tcr_model_null_case_no_dose_stays_at_baseline():
    result = simulate_cancer_tcr(PARAMS, IC, dose_schedule=None, t_end=21.0, t_eval=np.linspace(0, 21, 22))
    for name in ["Ag", "Adj", "TC", "DC", "aDC_Ag", "pMHC_density"]:
        assert np.allclose(result.trajectory(name), 0.0), name
    assert np.allclose(result.trajectory("T"), PARAMS.value("T0"))


def test_cancer_tcr_model_pipeline_causal_ordering():
    schedule = DoseSchedule.bolus()
    result = simulate_cancer_tcr(PARAMS, IC, schedule, t_end=21.0, t_eval=np.linspace(0, 21, 211))

    assert result.trajectory("Ag").max() > 0.0
    assert result.trajectory("aDC_Ag").max() > 0.0
    assert result.trajectory("pMHC_density").max() > 0.0
    assert result.trajectory("T").max() > IC["T"]

    t = result.time
    t_peak_adc_ag = t[np.argmax(result.trajectory("aDC_Ag"))]
    t_peak_pmhc = t[np.argmax(result.trajectory("pMHC_density"))]
    t_peak_t = t[np.argmax(result.trajectory("T"))]
    assert t_peak_adc_ag <= t_peak_pmhc <= t_peak_t


def test_cancer_tcr_upstream_component_is_literally_reused_from_science():
    science_params = default_science_parameters()
    science_ic = science_default_initial_conditions(science_params)
    schedule = DoseSchedule.from_exponential_escalation(numshot=7, k=1.0, duration=12)
    t_eval = np.linspace(0, 21, 211)

    science_result = simulate_science(science_params, science_ic, schedule, t_end=21.0, t_eval=t_eval)
    cancer_result = simulate_cancer_tcr(PARAMS, IC, schedule, t_end=21.0, t_eval=t_eval)

    for name in ["Ag", "Adj", "TC", "DC", "aDC_Ag"]:
        assert np.allclose(
            science_result.trajectory(name), cancer_result.trajectory(name), rtol=1e-4, atol=1e-9
        ), name


# -- TCR signal: closed-form kinetic proofreading --------------------------------


def test_tcr_signal_zero_pmhc_density_gives_zero_signal():
    out = tcr_signal_from_arrays(
        pmhc_density=np.array([0.0]), T=np.array([100.0]), DC=np.array([50.0]), tcr_signal_params=SIGNAL_PARAMS
    )
    assert out["Theta"][0] == pytest.approx(0.0)
    assert out["S_contact"][0] == pytest.approx(0.0)
    assert out["S_pop"][0] == pytest.approx(0.0)


def test_tcr_signal_occupancy_saturates_towards_one():
    K_D = SIGNAL_PARAMS.value("K_D")
    out = tcr_signal_from_arrays(
        pmhc_density=np.array([1e6 * K_D]), T=np.array([1.0]), DC=np.array([1.0]), tcr_signal_params=SIGNAL_PARAMS
    )
    assert out["Theta"][0] == pytest.approx(1.0, abs=1e-4)


def test_tcr_signal_higher_affinity_lower_kd_increases_amplification():
    # Lower K_D -> lower k_off -> proofreading amplification A closer to 1.
    pmhc = np.array([50.0])
    T = np.array([100.0])
    DC = np.array([50.0])

    strong = default_tcr_signal_parameters(K_D=1.0)
    weak = default_tcr_signal_parameters(K_D=200.0)

    out_strong = tcr_signal_from_arrays(pmhc, T, DC, strong)
    out_weak = tcr_signal_from_arrays(pmhc, T, DC, weak)

    assert out_strong["A"][0] > out_weak["A"][0]
    assert out_strong["k_off"][0] < out_weak["k_off"][0]


def test_tcr_signal_population_zero_when_no_t_or_dc():
    out = tcr_signal_from_arrays(
        pmhc_density=np.array([100.0, 100.0]),
        T=np.array([0.0, 100.0]),
        DC=np.array([50.0, 0.0]),
        tcr_signal_params=SIGNAL_PARAMS,
    )
    assert out["contacts"][0] == pytest.approx(0.0)
    assert out["contacts"][1] == pytest.approx(0.0)
    assert out["S_pop"][0] == pytest.approx(0.0)
    assert out["S_pop"][1] == pytest.approx(0.0)


def test_tcr_signal_serial_triggering_flag_raises_not_implemented():
    params = default_tcr_signal_parameters(include_serial_triggering=1.0)
    with pytest.raises(NotImplementedError):
        tcr_signal_from_arrays(np.array([1.0]), np.array([1.0]), np.array([1.0]), params)


def test_compute_tcr_signal_end_to_end_adds_expected_columns():
    schedule = DoseSchedule.bolus()
    result = simulate_cancer_tcr(PARAMS, IC, schedule, t_end=21.0, t_eval=np.linspace(0, 21, 211))
    signal_df = compute_tcr_signal(result, SIGNAL_PARAMS)

    for col in ["k_off", "Theta", "A", "S_contact", "contacts", "S_pop"]:
        assert col in signal_df.columns
    assert (signal_df["S_pop"] >= 0.0).all()
    assert (signal_df["Theta"] >= 0.0).all() and (signal_df["Theta"] <= 1.0).all()
    # compute_tcr_signal must not mutate the original result's DataFrame
    assert "S_pop" not in result.data.columns


def test_compute_tcr_signal_affinity_direction_matches_dosing_schedule_comparison_use_case():
    # The model's whole purpose: comparing TCR signal across dosing
    # schedules/affinities. A stronger antigen (lower K_D) should not
    # produce a WEAKER peak population signal than a much weaker one,
    # holding the dosing schedule fixed.
    schedule = DoseSchedule.from_exponential_escalation(numshot=7, k=1.0, duration=12)
    t_eval = np.linspace(0, 21, 211)
    result = simulate_cancer_tcr(PARAMS, IC, schedule, t_end=21.0, t_eval=t_eval)

    strong = compute_tcr_signal(result, default_tcr_signal_parameters(K_D=1.0))
    weak = compute_tcr_signal(result, default_tcr_signal_parameters(K_D=200.0))

    assert strong["S_pop"].max() > weak["S_pop"].max()
