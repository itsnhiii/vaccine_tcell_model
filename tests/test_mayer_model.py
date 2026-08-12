"""End-to-end integration tests for the assembled Mayer model (Eq. 1-2).

Covers master spec Phase 5's required checks (Section 15.B): pMHC
decays, T-cell expansion occurs, lower K gives a stronger response,
response changes with pMHC exposure, and initial T-cell number affects
expansion (the paper's headline inverse-power-law finding).
"""

from __future__ import annotations

import numpy as np
import pytest

from vaccine_tcell_model.models.mayer import simulate_mayer
from vaccine_tcell_model.parameters import default_mayer_parameters, mayer_demo_initial_conditions


def test_mayer_model_runs_end_to_end():
    params = default_mayer_parameters()
    ic = mayer_demo_initial_conditions()
    result = simulate_mayer(params, ic, t_end=10.0, t_eval=np.linspace(0, 10, 101))
    assert result.model_name == "mayer"
    assert set(result.states.names) == {"T", "C"}
    assert len(result.data) == 101


def test_mayer_model_pmhc_matches_analytic_decay():
    # C depends only on itself -- must match an exact exponential
    # regardless of T's dynamics.
    params = default_mayer_parameters()
    ic = {"T": 300.0, "C": 1e6}
    t_eval = np.linspace(0, 10, 101)
    result = simulate_mayer(params, ic, t_end=10.0, t_eval=t_eval, rtol=1e-10, atol=1e-12)
    analytic = 1e6 * np.exp(-params.value("mu") * result.time)
    assert np.allclose(result.trajectory("C"), analytic, rtol=1e-6)


def test_mayer_model_tcell_expansion_occurs():
    params = default_mayer_parameters()
    ic = mayer_demo_initial_conditions(T0=300.0, C0=1e6)
    result = simulate_mayer(params, ic, t_end=10.0, t_eval=np.linspace(0, 10, 101))
    assert result.trajectory("T").max() > ic["T"]


def test_affinity_direction_lower_k_gives_stronger_expansion():
    # Master spec Phase 5 requirement: K_low -> stronger T expansion,
    # K_high -> weaker T expansion, all else equal.
    ic = {"T": 300.0, "C": 1e6}
    t_eval = np.linspace(0, 10, 201)

    fold_expansions = {}
    for K in [0.1, 1.0, 10.0, 100.0, 1000.0]:
        params = default_mayer_parameters(K=K)
        result = simulate_mayer(params, ic, t_end=10.0, t_eval=t_eval)
        fold_expansions[K] = result.trajectory("T").max() / ic["T"]

    ks = sorted(fold_expansions)
    values = [fold_expansions[k] for k in ks]
    assert all(a >= b for a, b in zip(values, values[1:])), (
        f"expected fold expansion to strictly decrease with increasing K: {fold_expansions}"
    )
    # And a coarse low-vs-high sanity bound.
    assert fold_expansions[0.1] > fold_expansions[1000.0]


def test_response_changes_with_pmhc_exposure():
    # Higher initial pMHC (C0) -> greater T-cell expansion, all else equal.
    params = default_mayer_parameters(K=10.0)
    t_eval = np.linspace(0, 10, 201)

    fold_expansions = {}
    for C0 in [1e4, 1e5, 1e6, 1e7]:
        ic = {"T": 300.0, "C": C0}
        result = simulate_mayer(params, ic, t_end=10.0, t_eval=t_eval)
        fold_expansions[C0] = result.trajectory("T").max() / ic["T"]

    values = [fold_expansions[c] for c in sorted(fold_expansions)]
    assert all(a <= b for a, b in zip(values, values[1:])), (
        f"expected fold expansion to increase with C0: {fold_expansions}"
    )


def test_initial_tcell_number_affects_expansion_inverse_power_law():
    # Mayer et al.'s headline finding: fold expansion T(t*)/T(0) declines
    # (as an inverse power law) with increasing precursor number T(0),
    # all else (including C0) held fixed.
    params = default_mayer_parameters(K=10.0)
    C0 = 1e6
    t_eval = np.linspace(0, 10, 201)

    fold_expansions = {}
    for T0 in [10.0, 100.0, 1000.0, 10000.0]:
        ic = {"T": T0, "C": C0}
        result = simulate_mayer(params, ic, t_end=10.0, t_eval=t_eval)
        fold_expansions[T0] = result.trajectory("T").max() / T0

    values = [fold_expansions[t0] for t0 in sorted(fold_expansions)]
    assert all(a >= b for a, b in zip(values, values[1:])), (
        f"expected fold expansion to decrease as precursor number T0 increases: {fold_expansions}"
    )


def test_mayer_parameter_presets_are_independent_and_not_merged():
    from vaccine_tcell_model.parameters import (
        mayer_parameters_fig1_cd4,
        mayer_parameters_fig2_cd8,
        mayer_parameters_fig4_dosing,
    )

    fig1 = mayer_parameters_fig1_cd4()
    fig2 = mayer_parameters_fig2_cd8()
    fig4 = mayer_parameters_fig4_dosing()

    assert fig1.name == "mayer_fig1_cd4"
    assert fig2.name == "mayer_fig2_cd8"
    assert fig4.name == "mayer_fig4_dosing"
    # Distinct fits -- values must differ, not be silently reconciled.
    assert fig1.value("alpha") != fig2.value("alpha")
    assert fig1.value("K") == 0.0
    assert fig2.value("K") == pytest.approx(1.0)  # SIINFEKL default
