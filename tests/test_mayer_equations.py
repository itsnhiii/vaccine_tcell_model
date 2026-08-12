"""Direct, component-level tests of the Mayer equations (main text Eq. 1-2)."""

from __future__ import annotations

import pytest

from vaccine_tcell_model.models.mayer import pmhc_rhs, tcell_rhs
from vaccine_tcell_model.parameters import default_mayer_parameters

PARAMS = default_mayer_parameters(K=10.0)


def test_mayer_pmhc_decay_term():
    # dC/dt = -mu*C   (Eq.2)
    state = {"T": 0.0, "C": 5.0}
    out = pmhc_rhs(state, PARAMS)
    assert out["C"] == pytest.approx(-PARAMS.value("mu") * 5.0)


def test_mayer_tcell_equation():
    # dT/dt = alpha*T*C/(K+T+C) - delta*T   (Eq.1)
    state = {"T": 100.0, "C": 50.0}
    out = tcell_rhs(state, PARAMS)
    alpha, delta, K = (PARAMS.value(p) for p in ("alpha", "delta", "K"))
    expected = alpha * 100.0 * 50.0 / (K + 100.0 + 50.0) - delta * 100.0
    assert out["T"] == pytest.approx(expected)


def test_mayer_tcell_zero_pmhc_gives_pure_death():
    # C=0 -> proliferation term vanishes, dT/dt = -delta*T
    state = {"T": 40.0, "C": 0.0}
    out = tcell_rhs(state, PARAMS)
    assert out["T"] == pytest.approx(-PARAMS.value("delta") * 40.0)


def test_mayer_tcell_saturates_at_alpha_when_pmhc_dominates():
    # C >> K+T -> proliferation term -> alpha*T (saturating limit)
    state = {"T": 1.0, "C": 1e9}
    out = tcell_rhs(state, PARAMS)
    alpha, delta = PARAMS.value("alpha"), PARAMS.value("delta")
    expected = alpha * 1.0 - delta * 1.0
    assert out["T"] == pytest.approx(expected, rel=1e-6)


def test_mayer_tcell_zero_state_gives_zero_derivative():
    state = {"T": 0.0, "C": 0.0}
    out = tcell_rhs(state, PARAMS)
    assert out["T"] == 0.0
