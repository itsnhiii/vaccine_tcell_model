"""Tests for analysis.sensitivity (master spec Section 20).

Where possible, cross-checks the finite-difference S_i against a
closed-form analytic derivative (Mayer's pure-decay C(t)), rather than
just checking "looks about the right sign."
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vaccine_tcell_model.analysis import local_sensitivity, sensitivity_analysis
from vaccine_tcell_model.analysis.sensitivity import DEFAULT_METRICS
from vaccine_tcell_model.parameters import default_mayer_parameters


def test_local_sensitivity_of_pmhc_peak_to_mu_is_exactly_zero():
    # C(t) = C0*exp(-mu*t) is monotonically non-increasing from C0, so
    # C_max = C0 for ANY mu > 0 -- an exact, provable zero-sensitivity case.
    params = default_mayer_parameters(K=10.0)
    s = local_sensitivity(
        "mayer", "mu", "C_max",
        base_params=params, initial_conditions={"T": 300.0, "C": 1e6},
        t_end=10.0, t_eval=np.linspace(0, 10, 201),
    )
    assert s == pytest.approx(0.0, abs=1e-6)


def test_local_sensitivity_of_pmhc_auc_to_mu_matches_analytic_derivative():
    # AUC_C(mu) = C0/mu * (1 - exp(-mu*T)) for pure exponential decay.
    # d ln(AUC)/d ln(mu) = mu*T*exp(-mu*T)/(1-exp(-mu*T)) - 1, analytically.
    params = default_mayer_parameters(K=10.0)
    mu = params.value("mu")
    t_end = 10.0

    numeric = local_sensitivity(
        "mayer", "mu", "C_AUC",
        base_params=params, initial_conditions={"T": 300.0, "C": 1e6},
        t_end=t_end, t_eval=np.linspace(0, t_end, 4001), relative_step=1e-3,
    )
    analytic = mu * t_end * np.exp(-mu * t_end) / (1 - np.exp(-mu * t_end)) - 1
    assert numeric == pytest.approx(analytic, rel=1e-2)


def test_local_sensitivity_requires_positive_base_value():
    params = default_mayer_parameters(K=0.0)
    with pytest.raises(ValueError, match="strictly positive"):
        local_sensitivity("mayer", "K", "T_max", base_params=params)


def test_local_sensitivity_unknown_metric_raises():
    with pytest.raises(KeyError):
        local_sensitivity("mayer", "mu", "not_a_real_metric")


def test_sensitivity_analysis_returns_tidy_dataframe_with_expected_shape():
    df = sensitivity_analysis(
        "integrated",
        parameters=["K", "alpha", "eta"],
        t_eval=np.linspace(0, 21, 106),
    )
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert list(df["parameter"]) == ["K", "alpha", "eta"]
    for metric in DEFAULT_METRICS:
        assert metric in df.columns


def test_sensitivity_analysis_k_is_negative_consistent_with_affinity_direction():
    # Already-established finding (Phases 5/7/8): higher K -> weaker
    # response. Local sensitivity of T_max/TFH_max/AUC_T/AUC_TFH to K
    # should therefore be negative.
    df = sensitivity_analysis(
        "integrated", parameters=["K"], t_eval=np.linspace(0, 21, 106),
    )
    row = df.iloc[0]
    for metric in DEFAULT_METRICS:
        assert row[metric] < 0, (metric, row[metric])
