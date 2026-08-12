"""Tests for analysis.optimization (master spec Phase 11).

Covers the phase's explicit success criterion -- "optimization results
are reproducible; constraints are verified" -- plus a quality check that
the optimizer actually finds something at least as good as the
hand-picked bolus/2-dose/7-dose heuristics it's compared against.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vaccine_tcell_model.analysis import (
    OptimizationResult,
    compare_dose_schedules_including_optimized,
    optimize_dose_schedule,
)

T_EVAL = np.linspace(0, 21, 106)


def test_optimize_dose_schedule_satisfies_constraints():
    result = optimize_dose_schedule(
        "integrated", num_doses=4, time_window=12.0, metric="TFH_max", t_end=21.0, t_eval=T_EVAL
    )
    assert isinstance(result, OptimizationResult)
    assert all(result.constraints_satisfied.values()), result.constraints_satisfied
    assert len(result.fractions) == 4
    assert result.fractions.sum() == pytest.approx(1.0, abs=1e-6)
    assert np.all(result.fractions >= 0.0)
    assert len(result.schedule.antigen_times) == 4
    assert max(result.schedule.antigen_times) <= 12.0


def test_optimize_dose_schedule_is_reproducible():
    kwargs = dict(model="integrated", num_doses=4, time_window=12.0, metric="TFH_max", t_end=21.0, t_eval=T_EVAL)
    result1 = optimize_dose_schedule(**kwargs)
    result2 = optimize_dose_schedule(**kwargs)

    assert result1.fractions == pytest.approx(result2.fractions, abs=1e-9)
    assert result1.objective_value == pytest.approx(result2.objective_value, abs=1e-9)
    assert result1.success == result2.success


def test_optimize_dose_schedule_rejects_mayer():
    with pytest.raises(ValueError, match="no dosing concept"):
        optimize_dose_schedule("mayer", num_doses=4, time_window=12.0)


def test_optimize_dose_schedule_rejects_invalid_num_doses():
    with pytest.raises(ValueError, match="num_doses"):
        optimize_dose_schedule("integrated", num_doses=0, time_window=12.0)


def test_optimize_dose_schedule_single_dose_is_trivial_bolus():
    result = optimize_dose_schedule(
        "integrated", num_doses=1, time_window=12.0, metric="TFH_max", t_end=21.0, t_eval=T_EVAL
    )
    assert result.fractions == pytest.approx([1.0])
    assert result.schedule.antigen_times == (0.0,)


def test_optimize_dose_schedule_matches_or_beats_evenly_spaced_baseline():
    # The optimizer's search space (fixed evenly-spaced times, free
    # fractions) strictly contains the "N equal fractions" starting point
    # it's initialized from -- so it must never do worse than that
    # specific point after optimizing.
    from vaccine_tcell_model.dosing import DoseSchedule
    from vaccine_tcell_model.models.integrated import simulate_integrated
    from vaccine_tcell_model.parameters import default_integrated_parameters, integrated_default_initial_conditions

    params = default_integrated_parameters(K=10.0)
    ic = integrated_default_initial_conditions(params)
    baseline_schedule = DoseSchedule.equal_doses(times=list(np.round(np.linspace(0, 12, 4))))
    baseline_result = simulate_integrated(params, ic, baseline_schedule, t_end=21.0, t_eval=T_EVAL)
    baseline_tfh_max = baseline_result.trajectory("TFH").max()

    opt_result = optimize_dose_schedule(
        "integrated", num_doses=4, time_window=12.0, metric="TFH_max", t_end=21.0, t_eval=T_EVAL
    )
    assert opt_result.objective_value >= baseline_tfh_max * (1 - 1e-6)


def test_optimize_dose_schedule_is_a_local_not_global_optimizer():
    # Documented, verified finding (see analysis/optimization.py's module
    # docstring caveat): SLSQP converges to DIFFERENT local optima
    # depending on initial_fractions, for this non-convex objective.
    # Starting from uniform fractions gives a markedly weaker result than
    # the default (exponential-escalation) start -- this is real
    # non-convexity, not a bug, and callers should not assume
    # optimize_dose_schedule finds a global optimum from an arbitrary
    # starting point.
    kwargs = dict(model="integrated", num_doses=7, time_window=12.0, metric="TFH_max", t_end=21.0, t_eval=T_EVAL)

    default_start = optimize_dose_schedule(**kwargs)
    uniform_start = optimize_dose_schedule(**kwargs, initial_fractions=np.full(7, 1.0 / 7))

    # Both succeed and satisfy constraints...
    assert default_start.success and uniform_start.success
    assert all(default_start.constraints_satisfied.values())
    assert all(uniform_start.constraints_satisfied.values())
    # ...but land at genuinely different objective values, proving the
    # landscape is non-convex rather than both finding the same optimum.
    assert default_start.objective_value > uniform_start.objective_value * 1.05
    assert not np.allclose(default_start.fractions, uniform_start.fractions, atol=1e-3)


def test_compare_dose_schedules_including_optimized_returns_tidy_dataframe():
    df, opt_result = compare_dose_schedules_including_optimized(
        "integrated", time_window=12.0, num_doses=7, metric="TFH_max", t_end=21.0, t_eval=T_EVAL
    )
    assert isinstance(df, pd.DataFrame)
    assert isinstance(opt_result, OptimizationResult)
    assert set(df["schedule"]) == {"Bolus", "2-dose (20/80)", "7-dose (exp. escalation)", "Optimized (7-dose)"}
    assert "TFH_max" in df.columns

    optimized_row = df.loc[df["schedule"] == "Optimized (7-dose)", "TFH_max"].iloc[0]
    best_baseline = df.loc[df["schedule"] != "Optimized (7-dose)", "TFH_max"].max()
    assert optimized_row >= best_baseline * (1 - 1e-6)
