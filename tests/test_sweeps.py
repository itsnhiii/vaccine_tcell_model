"""Tests for analysis.sweeps (master spec Section 19).

Checks the sweeps return tidy DataFrames of the right shape, and
cross-validates against already-established behavior (K-affinity
direction, Phases 5/7/8) as a sanity check on the new API itself, not a
re-derivation of the underlying science.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vaccine_tcell_model.analysis import sweep_dose_schedules, sweep_parameter
from vaccine_tcell_model.dosing import DoseSchedule
from vaccine_tcell_model.parameters import ParameterSet, SourceType, default_integrated_parameters


def test_parameter_set_with_value_overrides_and_preserves_others():
    base = default_integrated_parameters(K=10.0)
    swept = base.with_value("K", 42.0)

    assert swept.value("K") == 42.0
    assert base.value("K") == 10.0  # original untouched
    assert swept["K"].source_type == SourceType.USER_DEFINED
    # everything else carried over unchanged
    assert swept.value("alpha") == base.value("alpha")
    assert swept.value("T0") == base.value("T0")


def test_parameter_set_with_value_rejects_unknown_name():
    base = default_integrated_parameters()
    with pytest.raises(KeyError):
        base.with_value("not_a_real_parameter", 1.0)


def test_sweep_parameter_returns_tidy_dataframe_integrated_k():
    values = np.logspace(0, 3, 5)
    df = sweep_parameter(model="integrated", parameter="K", values=values, t_eval=np.linspace(0, 21, 106))

    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(values)
    assert list(df["value"]) == pytest.approx(list(values))
    for col in ["T_max", "TFH_max", "AUC_T", "AUC_TFH", "fold_expansion", "pMHC_max", "pMHC_AUC"]:
        assert col in df.columns

    # Cross-check against the already-established K-affinity direction
    # (Phase 7/8): fold expansion should be non-increasing as K increases.
    assert all(a >= b for a, b in zip(df["fold_expansion"], df["fold_expansion"][1:]))


def test_sweep_parameter_science_model():
    df = sweep_parameter(
        model="science", parameter="eta", values=[0.1, 0.22, 0.5], t_eval=np.linspace(0, 21, 106)
    )
    assert len(df) == 3
    assert "T_max" in df.columns
    assert "TFH_max" in df.columns
    assert "pMHC_max" not in df.columns  # Science has no pMHC state


def test_sweep_parameter_mayer_model_ignores_dose_schedule():
    df = sweep_parameter(model="mayer", parameter="K", values=[1.0, 10.0, 100.0])
    assert len(df) == 3
    assert "C_max" in df.columns  # Mayer's pMHC-equivalent state is "C"


def test_sweep_parameter_rejects_unknown_model():
    with pytest.raises(ValueError, match="Unknown model"):
        sweep_parameter(model="not_a_model", parameter="K", values=[1.0])


def test_sweep_dose_schedules_returns_tidy_dataframe():
    schedules = {
        "Bolus": DoseSchedule.bolus(),
        "2-ED": DoseSchedule(times=[0, 7], antigen_fractions=[0.2, 0.8], adjuvant_fractions=[0.2, 0.8]),
        "7-ED": DoseSchedule.from_exponential_escalation(numshot=7, k=1.0, duration=12),
    }
    df = sweep_dose_schedules(model="integrated", schedules=schedules, t_eval=np.linspace(0, 21, 211))

    assert isinstance(df, pd.DataFrame)
    assert list(df["schedule"]) == ["Bolus", "2-ED", "7-ED"]
    assert "TFH_max" in df.columns

    # Cross-check: 7-ED should give a larger response than Bolus (already
    # established, Phase 4/8).
    bolus_tfh = df.loc[df["schedule"] == "Bolus", "TFH_max"].iloc[0]
    seven_ed_tfh = df.loc[df["schedule"] == "7-ED", "TFH_max"].iloc[0]
    assert seven_ed_tfh > bolus_tfh


def test_sweep_dose_schedules_rejects_mayer():
    with pytest.raises(ValueError, match="no dosing concept"):
        sweep_dose_schedules(model="mayer", schedules={"x": DoseSchedule.bolus()})


def test_sweep_parameter_does_not_mutate_base_params():
    base = default_integrated_parameters(K=10.0)
    sweep_parameter(model="integrated", parameter="K", values=[1.0, 100.0], base_params=base)
    assert base.value("K") == 10.0
