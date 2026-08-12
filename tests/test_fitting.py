"""Tests for analysis.fitting (master spec Phase 10 / Section 21).

The load-bearing test is synthetic-data recovery: generate data from a
KNOWN parameter value, fit starting from a deliberately wrong guess, and
check the fit recovers the known value -- master spec's explicit Phase 10
success criterion ("known parameters can be approximately recovered").
"""

from __future__ import annotations

import numpy as np
import pytest

from vaccine_tcell_model.analysis import FitResult, Observable, fit_parameters
from vaccine_tcell_model.models.mayer import simulate_mayer
from vaccine_tcell_model.parameters import SourceType, default_mayer_parameters


def test_fit_recovers_known_alpha_from_synthetic_data():
    true_params = default_mayer_parameters(K=10.0)
    ic = {"T": 300.0, "C": 1e6}
    t_eval = np.linspace(0, 10, 21)

    true_result = simulate_mayer(true_params, ic, t_end=10.0, t_eval=t_eval)
    observed_T = true_result.trajectory("T")

    # Deliberately wrong starting point for alpha (3x the true value).
    wrong_guess_params = true_params.with_value("alpha", true_params.value("alpha") * 3.0)

    fit_result = fit_parameters(
        model="mayer",
        param_names=["alpha"],
        observables=[Observable(state="T", time=t_eval, value=observed_T)],
        base_params=wrong_guess_params,
        initial_conditions=ic,
        t_end=10.0,
        bounds={"alpha": (0.1, 10.0)},
    )

    assert isinstance(fit_result, FitResult)
    assert fit_result.success
    assert fit_result.fitted_values["alpha"] == pytest.approx(true_params.value("alpha"), rel=0.05)
    assert fit_result.fitted_params["alpha"].source_type == SourceType.FITTED


def test_fit_recovers_multiple_parameters_from_multiple_observables():
    true_params = default_mayer_parameters(K=10.0)
    ic = {"T": 300.0, "C": 1e6}
    t_eval = np.linspace(0, 10, 21)

    true_result = simulate_mayer(true_params, ic, t_end=10.0, t_eval=t_eval)

    wrong_params = true_params.with_value("alpha", true_params.value("alpha") * 1.5).with_value(
        "mu", true_params.value("mu") * 0.6
    )

    fit_result = fit_parameters(
        model="mayer",
        param_names=["alpha", "mu"],
        observables=[
            Observable(state="T", time=t_eval, value=true_result.trajectory("T")),
            Observable(state="C", time=t_eval, value=true_result.trajectory("C")),
        ],
        base_params=wrong_params,
        initial_conditions=ic,
        t_end=10.0,
        bounds={"alpha": (0.1, 10.0), "mu": (0.1, 10.0)},
    )

    assert fit_result.success
    assert fit_result.fitted_values["alpha"] == pytest.approx(true_params.value("alpha"), rel=0.05)
    assert fit_result.fitted_values["mu"] == pytest.approx(true_params.value("mu"), rel=0.05)


def test_fit_does_not_mutate_base_params_or_module_default():
    base = default_mayer_parameters(K=10.0)
    original_alpha = base.value("alpha")
    ic = {"T": 300.0, "C": 1e6}
    t_eval = np.linspace(0, 10, 11)
    result = simulate_mayer(base, ic, t_end=10.0, t_eval=t_eval)

    fit_parameters(
        model="mayer",
        param_names=["alpha"],
        observables=[Observable(state="T", time=t_eval, value=result.trajectory("T"))],
        base_params=base,
        initial_conditions=ic,
        t_end=10.0,
        bounds={"alpha": (0.1, 10.0)},
    )

    assert base.value("alpha") == original_alpha
    assert default_mayer_parameters(K=10.0).value("alpha") == original_alpha


def test_fit_result_reports_bounds_and_initial_guess():
    base = default_mayer_parameters(K=10.0)
    ic = {"T": 300.0, "C": 1e6}
    t_eval = np.linspace(0, 10, 11)
    result = simulate_mayer(base, ic, t_end=10.0, t_eval=t_eval)

    fit_result = fit_parameters(
        model="mayer",
        param_names=["alpha"],
        observables=[Observable(state="T", time=t_eval, value=result.trajectory("T"))],
        base_params=base,
        initial_conditions=ic,
        t_end=10.0,
        bounds={"alpha": (0.1, 10.0)},
        initial_guess={"alpha": 2.0},
    )
    assert fit_result.initial_guess["alpha"] == 2.0
    assert fit_result.bounds["alpha"] == pytest.approx((0.1, 10.0))


def test_fit_parameters_requires_at_least_one_observable():
    with pytest.raises(ValueError, match="at least one Observable"):
        fit_parameters(model="mayer", param_names=["alpha"], observables=[])


def test_observable_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        Observable(state="T", time=np.array([0, 1, 2]), value=np.array([1, 2]))


def test_fit_recovers_integrated_model_kp_from_synthetic_data():
    from vaccine_tcell_model.dosing import DoseSchedule
    from vaccine_tcell_model.models.integrated import simulate_integrated
    from vaccine_tcell_model.parameters import default_integrated_parameters, integrated_default_initial_conditions

    true_params = default_integrated_parameters(K=10.0, k_p=5.0)
    ic = integrated_default_initial_conditions(true_params)
    schedule = DoseSchedule.bolus()
    t_eval = np.linspace(0, 21, 43)

    true_result = simulate_integrated(true_params, ic, schedule, t_end=21.0, t_eval=t_eval)

    wrong_params = true_params.with_value("k_p", 1.0)
    fit_result = fit_parameters(
        model="integrated",
        param_names=["k_p"],
        observables=[Observable(state="pMHC", time=t_eval, value=true_result.trajectory("pMHC"))],
        base_params=wrong_params,
        initial_conditions=ic,
        dose_schedule=schedule,
        t_end=21.0,
        bounds={"k_p": (0.1, 50.0)},
    )

    assert fit_result.success
    assert fit_result.fitted_values["k_p"] == pytest.approx(5.0, rel=0.05)
