"""Tests for StateSpec: the state-name <-> array-index mapping."""

from __future__ import annotations

import numpy as np
import pytest

from vaccine_tcell_model.core import INTEGRATED_STATES, MAYER_STATES, SCIENCE_STATES, StateSpec


def test_science_states_match_published_eq_1_to_7():
    # Supplement Eq. 1-7: Ag, Adj, TC, DC, aDC_Ag, T, TFH. No pMHC, no K.
    assert SCIENCE_STATES.names == ("Ag", "Adj", "TC", "DC", "aDC_Ag", "T", "TFH")
    assert SCIENCE_STATES.size == 7


def test_mayer_states():
    assert MAYER_STATES.names == ("T", "C")
    assert MAYER_STATES.size == 2


def test_integrated_states_insert_pmhc():
    assert INTEGRATED_STATES.names == (
        "Ag", "Adj", "TC", "DC", "aDC_Ag", "pMHC", "T", "TFH"
    )
    assert INTEGRATED_STATES.size == 8


def test_index_lookup():
    assert SCIENCE_STATES.index("aDC_Ag") == 4
    with pytest.raises(KeyError):
        SCIENCE_STATES.index("pMHC")  # not a Science state


def test_duplicate_names_rejected():
    with pytest.raises(ValueError):
        StateSpec(names=("Ag", "Ag"))


def test_to_array_and_back_roundtrip():
    values = {"Ag": 1.0, "Adj": 2.0, "TC": 0.0, "DC": 0.0, "aDC_Ag": 0.0, "T": 28.0, "TFH": 0.0}
    arr = SCIENCE_STATES.to_array(values)
    assert isinstance(arr, np.ndarray)
    assert arr[SCIENCE_STATES.index("T")] == 28.0

    back = SCIENCE_STATES.to_dict(arr)
    assert back == values


def test_to_array_missing_key_raises():
    incomplete = {"Ag": 1.0}
    with pytest.raises(ValueError):
        SCIENCE_STATES.to_array(incomplete)


def test_to_array_extra_key_raises():
    values = {n: 0.0 for n in SCIENCE_STATES.names}
    values["pMHC"] = 0.0  # not a Science state
    with pytest.raises(ValueError):
        SCIENCE_STATES.to_array(values)


def test_to_dict_wrong_length_raises():
    with pytest.raises(ValueError):
        SCIENCE_STATES.to_dict(np.zeros(3))
