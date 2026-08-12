"""Direct, component-level tests of the Science equations (supplement Eq. 1-7).

Each test calls the RHS function with hand-picked state values and checks
the output against the formula computed independently in the test itself
-- a token-for-token regression check against Table S2 / docs/equations.md
Section 1.1, so a silent equation edit fails immediately.
"""

from __future__ import annotations

import pytest

from vaccine_tcell_model.models.science import (
    antigen_adjuvant_rhs,
    innate_dc_rhs,
    tcell_activation_rhs,
    tfh_rhs,
)
from vaccine_tcell_model.parameters import default_science_parameters

PARAMS = default_science_parameters()


def test_science_antigen_first_order_decay_term():
    state = {"Ag": 4.0, "Adj": 0.0}
    out = antigen_adjuvant_rhs(state, PARAMS)
    assert out["Ag"] == pytest.approx(-PARAMS.value("d_Ag") * 4.0)


def test_science_adjuvant_first_order_decay_term():
    state = {"Ag": 0.0, "Adj": 2.5}
    out = antigen_adjuvant_rhs(state, PARAMS)
    assert out["Adj"] == pytest.approx(-PARAMS.value("d_Adj") * 2.5)


def test_science_tc_equation():
    # d[TC]/dt = Adj/(S_Adj+Adj) - mu*TC   (Eq.3)
    state = {"Ag": 0.3, "Adj": 0.4, "TC": 1.0, "DC": 0.0, "aDC_Ag": 0.0}
    out = innate_dc_rhs(state, PARAMS)
    S_Adj = PARAMS.value("S_Adj")
    mu = PARAMS.value("mu")
    expected_dTC = 0.4 / (S_Adj + 0.4) - mu * 1.0
    assert out["TC"] == pytest.approx(expected_dTC)


def test_science_dc_equation():
    # d[DC]/dt = D0*TC - (1+k*Adj/(S_Adj+Adj))*DC*Ag - mu*DC   (Eq.4)
    state = {"Ag": 0.3, "Adj": 0.4, "TC": 2.0, "DC": 5.0, "aDC_Ag": 0.0}
    out = innate_dc_rhs(state, PARAMS)
    S_Adj, mu, k, D0 = (PARAMS.value(p) for p in ("S_Adj", "mu", "k", "D0"))
    uptake = 1.0 + k * 0.4 / (S_Adj + 0.4)
    expected_dDC = D0 * 2.0 - uptake * 5.0 * 0.3 - mu * 5.0
    assert out["DC"] == pytest.approx(expected_dDC)


def test_science_adc_equation():
    # d[aDC_Ag]/dt = (1+k*Adj/(S_Adj+Adj))*DC*Ag - mu*aDC_Ag   (Eq.5)
    state = {"Ag": 0.3, "Adj": 0.4, "TC": 2.0, "DC": 5.0, "aDC_Ag": 7.0}
    out = innate_dc_rhs(state, PARAMS)
    S_Adj, mu, k = (PARAMS.value(p) for p in ("S_Adj", "mu", "k"))
    uptake = 1.0 + k * 0.4 / (S_Adj + 0.4)
    expected_daDC = uptake * 5.0 * 0.3 - mu * 7.0
    assert out["aDC_Ag"] == pytest.approx(expected_daDC)


def test_science_tcell_equation():
    # d[T]/dt = alpha*aDC_Ag*T/(T+aDC_Ag) - eta*(T-T0)   (Eq.6)
    state = {"T": 100.0, "aDC_Ag": 20.0}
    out = tcell_activation_rhs(state, PARAMS)
    alpha, eta, T0 = (PARAMS.value(p) for p in ("alpha", "eta", "T0"))
    expected_dT = alpha * 20.0 * 100.0 / (100.0 + 20.0) - eta * (100.0 - T0)
    assert out["T"] == pytest.approx(expected_dT)


def test_science_tcell_equation_zero_aDC_Ag_and_T_at_baseline_gives_zero():
    # At the published initial condition (T=T0, aDC_Ag=0), dT/dt = 0.
    T0 = PARAMS.value("T0")
    state = {"T": T0, "aDC_Ag": 0.0}
    out = tcell_activation_rhs(state, PARAMS)
    assert out["T"] == pytest.approx(0.0)


def test_science_tfh_equation():
    # d[TFH]/dt = eta*(T-T0)   (Eq.7)
    state = {"T": 150.0}
    out = tfh_rhs(state, PARAMS)
    eta, T0 = PARAMS.value("eta"), PARAMS.value("T0")
    assert out["TFH"] == pytest.approx(eta * (150.0 - T0))


def test_science_no_k_affinity_parameter_exists():
    # Flag S1: the published Science model has no K term. Locking this in
    # as a test so nobody silently reintroduces K into the Science
    # reproduction (K belongs only to Mayer/integrated, Phases 5 and 7).
    assert "K" not in PARAMS
