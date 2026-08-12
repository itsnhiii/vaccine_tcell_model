"""Tests for the parameter provenance system (master spec rule 11)."""

from __future__ import annotations

import pytest

from vaccine_tcell_model.parameters import Parameter, ParameterSet, SourceType


def test_parameter_requires_valid_source_type():
    with pytest.raises(TypeError):
        Parameter(
            name="d_Ag",
            value=3.0,
            units="1/day",
            source="Bhagchandani et al. 2024 Table S2",
            source_type="literature",  # not a SourceType member -> reject
        )


def test_parameter_requires_nonempty_source():
    with pytest.raises(ValueError):
        Parameter(
            name="d_Ag",
            value=3.0,
            units="1/day",
            source="",
            source_type=SourceType.LITERATURE,
        )


def test_parameter_accepts_all_allowed_source_types():
    for st in SourceType:
        p = Parameter(
            name="x",
            value=1.0,
            units="dimensionless",
            source="unit test",
            source_type=st,
        )
        assert p.source_type == st


def test_parameter_is_immutable():
    p = Parameter(
        name="d_Ag", value=3.0, units="1/day", source="Table S2",
        source_type=SourceType.LITERATURE,
    )
    with pytest.raises(Exception):
        p.value = 999.0  # frozen dataclass


def test_parameter_set_key_must_match_parameter_name():
    p = Parameter(
        name="d_Ag", value=3.0, units="1/day", source="Table S2",
        source_type=SourceType.LITERATURE,
    )
    with pytest.raises(ValueError):
        ParameterSet("science", {"wrong_key": p})


def test_parameter_set_value_and_dict_access():
    p1 = Parameter(
        name="d_Ag", value=3.0, units="1/day", source="Table S2",
        source_type=SourceType.LITERATURE,
    )
    p2 = Parameter(
        name="k_p", value=1.0, units="1/day", source="model extension",
        source_type=SourceType.MODEL_EXTENSION,
    )
    ps = ParameterSet("demo", {"d_Ag": p1, "k_p": p2})

    assert ps.value("d_Ag") == 3.0
    assert ps["k_p"].source_type is SourceType.MODEL_EXTENSION
    assert ps.as_dict() == {"d_Ag": 3.0, "k_p": 1.0}
    assert len(ps) == 2
    assert "d_Ag" in ps
    assert "missing" not in ps
