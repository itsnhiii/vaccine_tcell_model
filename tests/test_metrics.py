"""Tests for analysis.metrics (master spec Section 18).

Uses a known analytic trajectory (exponential decay) so peak/AUC/final
values can be checked against closed-form expressions, not just
"looks about right."
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vaccine_tcell_model.analysis import (
    auc,
    duration_above_threshold,
    final_value,
    fold_expansion,
    peak_value,
    summarize_pmhc,
    summarize_tcell,
    summarize_tfh,
    time_to_peak,
)
from vaccine_tcell_model.core import SimulationResult, StateSpec

STATES = StateSpec(names=("X",))
DECAY_RATE = 0.5
X0 = 10.0


def _decay_result(t_end: float = 10.0, n: int = 1001) -> SimulationResult:
    t = np.linspace(0, t_end, n)
    x = X0 * np.exp(-DECAY_RATE * t)
    data = pd.DataFrame({"time": t, "X": x})
    return SimulationResult(data=data, states=STATES, parameters=None, dose_schedule=None, model_name="decay")


def test_peak_value_and_time_to_peak_for_monotonic_decay():
    result = _decay_result()
    assert peak_value(result, "X") == pytest.approx(X0)
    assert time_to_peak(result, "X") == pytest.approx(0.0)


def test_auc_matches_analytic_integral():
    result = _decay_result(t_end=10.0, n=100001)  # fine grid for tight trapezoidal accuracy
    analytic = X0 / DECAY_RATE * (1 - np.exp(-DECAY_RATE * 10.0))
    assert auc(result, "X") == pytest.approx(analytic, rel=1e-4)


def test_final_value():
    result = _decay_result(t_end=10.0)
    assert final_value(result, "X") == pytest.approx(X0 * np.exp(-DECAY_RATE * 10.0), rel=1e-6)


def test_fold_expansion():
    result = _decay_result()
    assert fold_expansion(result, "X", baseline=5.0) == pytest.approx(X0 / 5.0)


def test_duration_above_threshold_matches_analytic_crossing_time():
    # X(t) = X0*exp(-rate*t) > threshold  for  t < ln(X0/threshold)/rate
    result = _decay_result(t_end=10.0, n=100001)
    threshold = 2.0
    analytic_duration = np.log(X0 / threshold) / DECAY_RATE
    assert duration_above_threshold(result, "X", threshold) == pytest.approx(analytic_duration, rel=1e-3)


def test_duration_above_threshold_returns_zero_if_never_exceeded():
    result = _decay_result()
    assert duration_above_threshold(result, "X", threshold=1e9) == 0.0


def test_summarize_tcell_keys_and_values():
    t = np.linspace(0, 10, 101)
    x = 28.0 + 100.0 * np.sin(np.pi * t / 10) ** 2  # rises from baseline then returns
    data = pd.DataFrame({"time": t, "T": x})
    result = SimulationResult(
        data=data, states=StateSpec(names=("T",)), parameters=None, dose_schedule=None, model_name="t"
    )
    summary = summarize_tcell(result, baseline=28.0)
    assert set(summary) == {"T_max", "t_T_max", "T_final", "AUC_T", "fold_expansion"}
    assert summary["T_max"] == pytest.approx(x.max())
    assert summary["fold_expansion"] == pytest.approx(x.max() / 28.0)


def test_summarize_tfh_keys():
    t = np.linspace(0, 10, 101)
    x = np.cumsum(np.ones_like(t))  # monotonic increasing
    data = pd.DataFrame({"time": t, "TFH": x})
    result = SimulationResult(
        data=data, states=StateSpec(names=("TFH",)), parameters=None, dose_schedule=None, model_name="tfh"
    )
    summary = summarize_tfh(result)
    assert set(summary) == {"TFH_max", "t_TFH_max", "TFH_final", "AUC_TFH"}


def test_summarize_pmhc_keys_with_and_without_threshold():
    t = np.linspace(0, 10, 1001)
    x = X0 * np.exp(-DECAY_RATE * t)
    data = pd.DataFrame({"time": t, "pMHC": x})
    result = SimulationResult(
        data=data, states=StateSpec(names=("pMHC",)), parameters=None, dose_schedule=None, model_name="pmhc"
    )

    summary = summarize_pmhc(result)
    assert set(summary) == {"pMHC_max", "pMHC_AUC"}

    summary_with_threshold = summarize_pmhc(result, threshold=1.0)
    assert set(summary_with_threshold) == {"pMHC_max", "pMHC_AUC", "duration_above_threshold"}
