"""Tests for DoseSchedule / DoseEvent / validation (master spec Section 10)."""

from __future__ import annotations

import math

import pytest

from vaccine_tcell_model.dosing import DoseEvent, DoseSchedule


def test_bolus_single_dose_at_time_zero():
    s = DoseSchedule.bolus()
    assert s.antigen_times == (0.0,)
    assert s.antigen_doses == (1.0,)
    assert s.adjuvant_times == (0.0,)
    assert s.adjuvant_doses == (1.0,)
    events = s.to_dose_events()
    assert events == [DoseEvent(time=0.0, antigen_jump=1.0, adjuvant_jump=1.0)]


def test_two_dose_20_80():
    s = DoseSchedule(
        times=[0, 7],
        antigen_fractions=[0.2, 0.8],
        adjuvant_fractions=[0.2, 0.8],
    )
    assert s.antigen_doses == (0.2, 0.8)
    assert s.total_antigen_dose == pytest.approx(1.0)
    events = s.to_dose_events()
    assert [e.time for e in events] == [0.0, 7.0]
    assert events[0].antigen_jump == pytest.approx(0.2)
    assert events[1].antigen_jump == pytest.approx(0.8)


def test_two_dose_50_50():
    s = DoseSchedule(times=[0, 7], antigen_fractions=[0.5, 0.5], adjuvant_fractions=[0.5, 0.5])
    assert s.antigen_doses == (0.5, 0.5)


def test_seven_dose_equal():
    s = DoseSchedule.equal_doses(times=list(range(0, 14, 2)))
    assert len(s.antigen_doses) == 7
    assert sum(s.antigen_doses) == pytest.approx(1.0)
    assert all(d == pytest.approx(1 / 7) for d in s.antigen_doses)


def test_absolute_doses_do_not_need_to_sum_to_one():
    s = DoseSchedule(times=[0, 7], antigen_doses=[2.0, 8.0], adjuvant_doses=[1.0, 4.0])
    assert s.total_antigen_dose == pytest.approx(10.0)
    assert s.total_adjuvant_dose == pytest.approx(5.0)


def test_fractions_must_sum_to_one():
    with pytest.raises(ValueError, match="must sum to 1"):
        DoseSchedule(times=[0, 7], antigen_fractions=[0.2, 0.5], adjuvant_fractions=[0.5, 0.5])


def test_fractions_and_doses_are_mutually_exclusive():
    with pytest.raises(ValueError, match="not both"):
        DoseSchedule(
            times=[0], antigen_fractions=[1.0], antigen_doses=[1.0], adjuvant_fractions=[1.0]
        )


def test_duplicate_times_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        DoseSchedule(times=[0, 0], antigen_fractions=[0.5, 0.5], adjuvant_fractions=[0.5, 0.5])


def test_negative_times_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        DoseSchedule(times=[-1, 7], antigen_fractions=[0.5, 0.5], adjuvant_fractions=[0.5, 0.5])


def test_mismatched_lengths_rejected():
    with pytest.raises(ValueError, match="must have the same length"):
        DoseSchedule(times=[0, 7], antigen_fractions=[1.0], adjuvant_fractions=[0.5, 0.5])


def test_independently_scheduled_antigen_and_adjuvant_merge_correctly():
    # e.g. antigen escalates over 2 doses, adjuvant given once up front.
    s = DoseSchedule(
        antigen_times=[0, 7],
        antigen_fractions=[0.2, 0.8],
        adjuvant_times=[0],
        adjuvant_fractions=[1.0],
    )
    events = s.to_dose_events()
    assert [e.time for e in events] == [0.0, 7.0]
    assert events[0].antigen_jump == pytest.approx(0.2)
    assert events[0].adjuvant_jump == pytest.approx(1.0)
    assert events[1].antigen_jump == pytest.approx(0.8)
    assert events[1].adjuvant_jump == 0.0  # no adjuvant dose at t=7


def test_exponential_escalation_reproduces_reference_2ed_split():
    # docs/equations.md Flag S5: reference code's 2-ED uses numshot=2,
    # k=log(4), duration=7 -> fractions [1,4]/5 = [0.2, 0.8].
    s = DoseSchedule.from_exponential_escalation(numshot=2, k=math.log(4), duration=7)
    assert s.antigen_times == (0.0, 7.0)
    assert s.antigen_doses[0] == pytest.approx(0.2, abs=1e-9)
    assert s.antigen_doses[1] == pytest.approx(0.8, abs=1e-9)


def test_exponential_escalation_bolus_is_k_zero_single_shot():
    s = DoseSchedule.from_exponential_escalation(numshot=1, k=0.0, duration=0.0)
    assert s.antigen_times == (0.0,)
    assert s.antigen_doses == (1.0,)


def test_exponential_escalation_seven_dose_sums_to_total_and_is_increasing():
    s = DoseSchedule.from_exponential_escalation(numshot=7, k=1.0, duration=12)
    assert len(s.antigen_times) == 7
    assert sum(s.antigen_doses) == pytest.approx(1.0)
    # Escalating pattern: each dose fraction larger than the previous.
    assert all(
        s.antigen_doses[i] < s.antigen_doses[i + 1] for i in range(len(s.antigen_doses) - 1)
    )


def test_exponential_escalation_adjuvant_bolus_override():
    # Science supplement fig. S2B "7-ED (adjuvant bolus)": antigen
    # escalates over 7 shots, adjuvant given as a single bolus at t=0.
    s = DoseSchedule.from_exponential_escalation(
        numshot=7, k=1.0, duration=12,
        adjuvant_numshot=1, adjuvant_k=0.0, adjuvant_duration=0.0,
    )
    assert len(s.antigen_times) == 7
    assert s.adjuvant_times == (0.0,)
    assert s.adjuvant_doses == (1.0,)


def test_dose_event_rejects_negative_jump():
    with pytest.raises(ValueError, match="cannot be negative"):
        DoseEvent(time=0.0, antigen_jump=-1.0)
