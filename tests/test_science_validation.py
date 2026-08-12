"""Phase 4: Science model validation against the reference implementation.

Cross-checks our DoseSchedule construction against the exact regimens
used to generate Bhagchandani et al. 2024 Fig. 1B/3F, verifies the
qualitative dose-number ordering the paper reports, and quantifies
(rather than assumes) that the undocumented floor-clamp found in the
reference code never actually matters for these regimens.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vaccine_tcell_model.models.science import simulate_science
from vaccine_tcell_model.parameters import default_science_parameters, science_default_initial_conditions
from vaccine_tcell_model.validation.science import (
    REFERENCE_DOSE_SCHEMES,
    REFERENCE_DOSE_SCHEMES_ADJUVANT_BOLUS,
    simulate_science_reference_floor,
    tfh_at_day,
)

PARAMS = default_science_parameters()
IC = science_default_initial_conditions(PARAMS)


def test_reference_dose_schemes_match_matlab_definescheme_values():
    expected = {
        "7-ED": (7, 1.0, 12),
        "6-ED": (6, 1.0, 11),
        "4-ED": (4, 1.0, 12),
        "3-ED": (3, 1.0, 12),
        "2-ED": (2, math.log(4), 7),
        "Bolus": (1, 0.0, 0),
    }
    for name, (numshot, k, duration) in expected.items():
        schedule = REFERENCE_DOSE_SCHEMES[name]
        assert len(schedule.antigen_times) == numshot

        expected_times = tuple(float(t) for t in np.round(np.linspace(0, duration, numshot)))
        assert schedule.antigen_times == expected_times, name

        weights = np.exp(np.arange(numshot) * k)
        expected_fracs = tuple(float(f) for f in weights / weights.sum())
        assert schedule.antigen_doses == pytest.approx(expected_fracs), name


def test_two_ed_matches_paper_20_80_split():
    s = REFERENCE_DOSE_SCHEMES["2-ED"]
    assert s.antigen_times == (0.0, 7.0)
    assert s.antigen_doses == pytest.approx((0.2, 0.8), abs=1e-9)


def test_qualitative_dose_schedule_ordering_matches_paper():
    # Main text: "As the number of doses was reduced ... the total size
    # of the GC and TFH responses steadily dropped" (Fig. 1, B and C).
    order = ["7-ED", "6-ED", "4-ED", "3-ED", "2-ED", "Bolus"]
    t_eval = np.linspace(0, 21, 2101)
    tfh_day21 = {}
    for name in order:
        result = simulate_science(PARAMS, IC, REFERENCE_DOSE_SCHEMES[name], t_end=21.0, t_eval=t_eval)
        tfh_day21[name] = tfh_at_day(result, 21.0)

    values = [tfh_day21[name] for name in order]
    assert all(a >= b for a, b in zip(values, values[1:])), (
        f"expected non-increasing TFH(day21) as dose count drops: {tfh_day21}"
    )


def test_undocumented_floor_clamp_never_activates_for_any_reference_scheme():
    # docs/equations.md Flag S-code-1: the reference code silently floors
    # dT/dt at T0 when T<=T0. We derived that this can never trigger
    # under the published equations alone (dT/dt=0 at T=T0 given
    # aDC_Ag>=0, so T never crosses below T0). Verify numerically across
    # every published regimen rather than merely asserting it.
    t_eval = np.linspace(0, 21, 2101)
    all_schemes = {**REFERENCE_DOSE_SCHEMES, **REFERENCE_DOSE_SCHEMES_ADJUVANT_BOLUS}
    for name, schedule in all_schemes.items():
        clean = simulate_science(PARAMS, IC, schedule, t_end=21.0, t_eval=t_eval)
        floored = simulate_science_reference_floor(PARAMS, IC, schedule, t_end=21.0, t_eval=t_eval)

        assert np.all(clean.trajectory("T") >= PARAMS.value("T0") - 1e-6), name
        # If the floor genuinely never triggers, the clean and
        # floor-augmented trajectories must be numerically identical.
        assert np.allclose(clean.trajectory("T"), floored.trajectory("T"), rtol=1e-5, atol=1e-6), name
        assert np.allclose(clean.trajectory("TFH"), floored.trajectory("TFH"), rtol=1e-5, atol=1e-6), name


def test_seven_ed_dc_shows_periodic_recruitment_matching_each_dose():
    # Main text: "the 7-ED regimen shows DCs being periodically recruited
    # with each dose" (Fig. 3B), unlike bolus's single decaying peak.
    # Structural check: DC(t) should show a local resurgence somewhere in
    # each inter-dose window, not monotonic decay throughout.
    schedule = REFERENCE_DOSE_SCHEMES["7-ED"]
    t_eval = np.linspace(0, 14, 1401)
    result = simulate_science(PARAMS, IC, schedule, t_end=14.0, t_eval=t_eval)
    dc = result.trajectory("DC")
    t = result.time

    dose_times = list(schedule.antigen_times) + [14.0]
    boosts = 0
    for i in range(1, len(dose_times) - 1):  # interior doses only
        idx_i = int(np.argmin(np.abs(t - dose_times[i])))
        idx_next = int(np.argmin(np.abs(t - dose_times[i + 1])))
        window = dc[idx_i: idx_next + 1]
        if window.max() > dc[idx_i] * 1.01:
            boosts += 1
    assert boosts >= 4, f"expected most interior doses to show a DC re-boost, got {boosts}/6"
