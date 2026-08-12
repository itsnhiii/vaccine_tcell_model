"""Tests for event-aware ODE integration: exact dose jumps, no smearing.

Uses trivial first-order decay systems (not the Science/Mayer equations,
which arrive in Phases 3/5) purely to isolate and verify the solver's
event-handling machinery itself, per master spec Phase 2's success
criterion: "dose jumps occur exactly at requested times; no numerical
smearing of the jump."
"""

from __future__ import annotations

import numpy as np
import pytest

from vaccine_tcell_model.core import SCIENCE_STATES, StateJumpEvent, StateSpec
from vaccine_tcell_model.dosing import DoseSchedule
from vaccine_tcell_model.solvers import integrate_with_events, simulate

DECAY_RATE = 0.3
ONE_STATE = StateSpec(names=("Ag",))
TWO_STATE = StateSpec(names=("Ag", "Adj"))


def decay_rhs_1d(t, y):
    return np.array([-DECAY_RATE * y[0]])


def decay_rhs_2d(t, state):
    return {"Ag": -DECAY_RATE * state["Ag"], "Adj": -DECAY_RATE * state["Adj"]}


def test_dose_jump_occurs_at_exact_time_bolus_at_t0():
    event = StateJumpEvent(time=0.0, jumps={"Ag": 1.0})
    t_eval = np.linspace(0, 5, 51)
    t, y = integrate_with_events(
        decay_rhs_1d, ONE_STATE, y0=[0.0], t_span=(0.0, 5.0),
        events=[event], t_eval=t_eval,
    )
    # First reported value is the post-jump (dosed) value exactly -- not
    # a smeared ramp-up over the first few solver steps.
    assert y[0, 0] == pytest.approx(1.0, abs=1e-12)
    # And it decays smoothly from there.
    analytic = 1.0 * np.exp(-DECAY_RATE * t)
    assert np.allclose(y[0], analytic, rtol=1e-6)


def test_dose_jump_occurs_at_exact_time_interior_dose():
    dose_time = 5.0
    dose_amount = 2.0
    event = StateJumpEvent(time=dose_time, jumps={"Ag": dose_amount})
    y0 = 10.0
    t_eval = np.linspace(0, 10, 101)

    t, y = integrate_with_events(
        decay_rhs_1d, ONE_STATE, y0=[y0], t_span=(0.0, 10.0),
        events=[event], t_eval=t_eval, rtol=1e-10, atol=1e-12,
    )

    pre_jump_analytic = y0 * np.exp(-DECAY_RATE * dose_time)
    idx = np.where(np.isclose(t, dose_time))[0]
    assert len(idx) == 1, "dose time must appear exactly once in the trajectory"
    reported_at_dose_time = y[0, idx[0]]

    # The single reported value at the dose time equals the pre-jump
    # (decayed) baseline plus the exact dose -- proving the jump is a
    # clean discontinuity, not spread across neighboring solver steps.
    assert reported_at_dose_time == pytest.approx(
        pre_jump_analytic + dose_amount, rel=1e-6
    )

    # The point just before the dose reflects undosed decay only.
    idx_before = np.where(t < dose_time)[0][-1]
    assert y[0, idx_before] == pytest.approx(
        y0 * np.exp(-DECAY_RATE * t[idx_before]), rel=1e-6
    )

    # The point just after decays from the POST-jump value, not the pre-jump one.
    idx_after = np.where(t > dose_time)[0][0]
    dt = t[idx_after] - dose_time
    expected_after = (pre_jump_analytic + dose_amount) * np.exp(-DECAY_RATE * dt)
    assert y[0, idx_after] == pytest.approx(expected_after, rel=1e-6)


def test_no_dose_does_not_change_ag_instantaneously():
    """With zero events, the trajectory must be pure smooth decay --
    no spurious jump is ever introduced by the event machinery itself."""
    t_eval = np.linspace(0, 10, 101)
    t, y = integrate_with_events(
        decay_rhs_1d, ONE_STATE, y0=[5.0], t_span=(0.0, 10.0), events=[], t_eval=t_eval,
    )
    analytic = 5.0 * np.exp(-DECAY_RATE * t)
    assert np.allclose(y[0], analytic, rtol=1e-6)
    # No single step should show a jump larger than the natural decay step.
    steps = np.diff(y[0])
    max_natural_step = np.max(np.abs(np.diff(analytic)))
    assert np.all(np.abs(steps) <= max_natural_step * 1.01)


def test_event_outside_span_raises():
    event = StateJumpEvent(time=99.0, jumps={"Ag": 1.0})
    with pytest.raises(ValueError, match="outside the integration span"):
        integrate_with_events(decay_rhs_1d, ONE_STATE, y0=[1.0], t_span=(0.0, 5.0), events=[event])


def test_multiple_states_jump_simultaneously_via_dose_schedule():
    schedule = DoseSchedule(times=[0, 5], antigen_fractions=[0.5, 0.5], adjuvant_fractions=[0.3, 0.7])
    result = simulate(
        decay_rhs_2d, TWO_STATE, initial_conditions={"Ag": 0.0, "Adj": 0.0},
        t_end=10.0, dose_schedule=schedule, t_eval=np.linspace(0, 10, 101),
        model_name="two_state_decay_smoke_test",
    )
    ag = result.trajectory("Ag")
    adj = result.trajectory("Adj")

    # At t=0, both channels show their exact first-dose value.
    assert ag[0] == pytest.approx(0.5, abs=1e-9)
    assert adj[0] == pytest.approx(0.3, abs=1e-9)

    # At t=5, both jump by their respective second-dose amounts.
    idx = np.where(np.isclose(result.time, 5.0))[0][0]
    ag_pre = 0.5 * np.exp(-DECAY_RATE * 5.0)
    adj_pre = 0.3 * np.exp(-DECAY_RATE * 5.0)
    assert ag[idx] == pytest.approx(ag_pre + 0.5, rel=1e-6)
    assert adj[idx] == pytest.approx(adj_pre + 0.7, rel=1e-6)


def test_seven_dose_schedule_no_smearing_at_any_dose_time():
    schedule = DoseSchedule.from_exponential_escalation(numshot=7, k=1.0, duration=12)
    # Dense grid so we can find points immediately around every dose time.
    t_eval = np.linspace(0, 21, 2101)

    result = simulate(
        lambda t, s: {"Ag": -DECAY_RATE * s["Ag"], "Adj": -DECAY_RATE * s["Adj"]},
        TWO_STATE,
        initial_conditions={"Ag": 0.0, "Adj": 0.0},
        t_end=21.0,
        dose_schedule=schedule,
        t_eval=t_eval,
        rtol=1e-10,
        atol=1e-12,
        model_name="seven_dose_smoke_test",
    )
    ag = result.trajectory("Ag")
    t = result.time

    for dose_time, dose_amount in zip(schedule.antigen_times, schedule.antigen_doses):
        idx = np.where(np.isclose(t, dose_time))[0]
        assert len(idx) == 1
        i = idx[0]
        if i > 0:
            step_at_dose = ag[i] - ag[i - 1]
            # The jump at the dose time must be >= the dose amount minus a
            # tiny decay correction over the adjacent (tiny) sampling step
            # -- i.e. it shows up as one big step, not smeared across many.
            assert step_at_dose >= dose_amount * 0.99, (
                f"dose at t={dose_time} appears smeared: step={step_at_dose}, "
                f"expected close to {dose_amount}"
            )
