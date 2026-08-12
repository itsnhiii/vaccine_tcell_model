"""End-to-end integration tests for the assembled Science model (Eq. 1-7).

Complements test_science_equations.py's component-level checks with
whole-trajectory sanity checks: dosed vs. undosed behavior, exact
decoupled decay of Ag/Adj, and the monotonicity properties the equations
imply (docs/equations.md).
"""

from __future__ import annotations

import numpy as np
import pytest

from vaccine_tcell_model.dosing import DoseSchedule
from vaccine_tcell_model.models.science import simulate_science
from vaccine_tcell_model.parameters import default_science_parameters, science_default_initial_conditions

PARAMS = default_science_parameters()
IC = science_default_initial_conditions(PARAMS)


def test_science_model_runs_end_to_end():
    schedule = DoseSchedule.bolus()
    result = simulate_science(PARAMS, IC, schedule, t_end=21.0, t_eval=np.linspace(0, 21, 211))
    assert result.model_name == "science"
    assert set(result.states.names) == {"Ag", "Adj", "TC", "DC", "aDC_Ag", "T", "TFH"}
    assert len(result.data) == 211


def test_science_model_null_case_no_dose_stays_at_baseline():
    result = simulate_science(PARAMS, IC, dose_schedule=None, t_end=21.0, t_eval=np.linspace(0, 21, 22))
    assert np.allclose(result.trajectory("Ag"), 0.0)
    assert np.allclose(result.trajectory("Adj"), 0.0)
    assert np.allclose(result.trajectory("TC"), 0.0)
    assert np.allclose(result.trajectory("DC"), 0.0)
    assert np.allclose(result.trajectory("aDC_Ag"), 0.0)
    assert np.allclose(result.trajectory("T"), PARAMS.value("T0"))
    assert np.allclose(result.trajectory("TFH"), 0.0)


def test_science_model_antigen_matches_analytic_decay():
    # Ag depends only on itself (dosing aside) -- Eq.1's decay term should
    # match an exact exponential regardless of what TC/DC/T are doing.
    schedule = DoseSchedule.bolus(total_antigen_dose=1.0, total_adjuvant_dose=0.0)
    t_eval = np.linspace(0, 10, 101)
    result = simulate_science(PARAMS, IC, schedule, t_end=10.0, t_eval=t_eval, rtol=1e-10, atol=1e-12)
    analytic = 1.0 * np.exp(-PARAMS.value("d_Ag") * result.time)
    assert np.allclose(result.trajectory("Ag"), analytic, rtol=1e-6)


def test_science_model_adjuvant_matches_analytic_decay():
    schedule = DoseSchedule.bolus(total_antigen_dose=0.0, total_adjuvant_dose=1.0)
    t_eval = np.linspace(0, 10, 101)
    result = simulate_science(PARAMS, IC, schedule, t_end=10.0, t_eval=t_eval, rtol=1e-10, atol=1e-12)
    analytic = 1.0 * np.exp(-PARAMS.value("d_Adj") * result.time)
    assert np.allclose(result.trajectory("Adj"), analytic, rtol=1e-6)


def test_science_model_tfh_is_monotonically_nondecreasing():
    # d[TFH]/dt = eta*(T-T0) and T >= T0 for all t (T'=0 at T=T0 given
    # aDC_Ag>=0, so T never dips below T0) -- TFH must never decrease.
    schedule = DoseSchedule.bolus()
    result = simulate_science(PARAMS, IC, schedule, t_end=21.0, t_eval=np.linspace(0, 21, 211))
    tfh = result.trajectory("TFH")
    assert np.all(np.diff(tfh) >= -1e-8)


def test_science_model_t_never_dips_below_t0():
    schedule = DoseSchedule.bolus()
    result = simulate_science(PARAMS, IC, schedule, t_end=21.0, t_eval=np.linspace(0, 21, 211))
    assert np.all(result.trajectory("T") >= PARAMS.value("T0") - 1e-6)


def test_science_model_states_remain_nonnegative():
    schedule = DoseSchedule.from_exponential_escalation(numshot=7, k=1.0, duration=12)
    result = simulate_science(PARAMS, IC, schedule, t_end=21.0, t_eval=np.linspace(0, 21, 211))
    for name in result.states.names:
        assert np.all(result.trajectory(name) >= -1e-8), f"{name} went negative"


def test_science_model_dosing_produces_more_dc_and_tfh_than_no_dose():
    dosed = simulate_science(
        PARAMS, IC, DoseSchedule.bolus(), t_end=14.0, t_eval=np.linspace(0, 14, 141)
    )
    undosed = simulate_science(
        PARAMS, IC, dose_schedule=None, t_end=14.0, t_eval=np.linspace(0, 14, 141)
    )
    assert dosed.trajectory("DC").max() > undosed.trajectory("DC").max()
    assert dosed.trajectory("aDC_Ag").max() > undosed.trajectory("aDC_Ag").max()
    assert dosed.trajectory("TFH")[-1] > undosed.trajectory("TFH")[-1]


def test_science_model_initial_t_equals_t0():
    schedule = DoseSchedule.bolus()
    result = simulate_science(PARAMS, IC, schedule, t_end=1.0, t_eval=np.array([0.0, 1.0]))
    assert result.trajectory("T")[0] == pytest.approx(PARAMS.value("T0"))
