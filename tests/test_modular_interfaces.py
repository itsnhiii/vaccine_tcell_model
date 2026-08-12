"""Phase 6 success criterion: replace one module with a toy/mock module
without changing the solver or other modules.

This test defines a mock T-cell component entirely within the test file
(not in src/ -- it is not part of any real model) and substitutes it into
the Science model's composition via `simulate_science(..., model=...)`.
Nothing in solvers/ or in the untouched components (ScienceUpstreamModel,
ScienceTfhModel) is edited to make this work.
"""

from __future__ import annotations

import numpy as np
import pytest

from vaccine_tcell_model.core import (
    ComposedModel,
    InnateDCModel,
    PresentationModel,
    SCIENCE_STATES,
    TCellActivationModel,
    TfhModel,
)
from vaccine_tcell_model.dosing import DoseSchedule
from vaccine_tcell_model.models.mayer import MayerPresentationModel, MayerTCellModel
from vaccine_tcell_model.models.science import (
    ScienceTCellModel,
    ScienceTfhModel,
    ScienceUpstreamModel,
    simulate_science,
)
from vaccine_tcell_model.parameters import default_science_parameters, science_default_initial_conditions

PARAMS = default_science_parameters()
IC = science_default_initial_conditions(PARAMS)


class MockConstantTCellModel(TCellActivationModel):
    """Toy/mock T-cell component: T never changes, regardless of aDC_Ag.

    Exists only in this test to prove pluggability -- not a scientific
    model of anything.
    """

    @property
    def owned_states(self) -> tuple[str, ...]:
        return ("T",)

    def rhs(self, t: float, state: dict[str, float]) -> dict[str, float]:
        return {"T": 0.0}


def test_role_interfaces_are_satisfied_by_the_real_components():
    assert isinstance(ScienceUpstreamModel(PARAMS), InnateDCModel)
    assert isinstance(ScienceTCellModel(PARAMS), TCellActivationModel)
    assert isinstance(ScienceTfhModel(PARAMS), TfhModel)
    assert isinstance(MayerPresentationModel(PARAMS), PresentationModel)
    assert isinstance(MayerTCellModel(PARAMS), TCellActivationModel)


def test_composed_model_rejects_unowned_state():
    with pytest.raises(ValueError, match="No component owns"):
        ComposedModel(SCIENCE_STATES, [ScienceUpstreamModel(PARAMS)])  # missing T, TFH


def test_composed_model_rejects_duplicate_ownership():
    with pytest.raises(ValueError, match="claimed by both"):
        ComposedModel(
            SCIENCE_STATES,
            [
                ScienceUpstreamModel(PARAMS),
                ScienceTCellModel(PARAMS),
                ScienceTCellModel(PARAMS),  # T claimed twice
                ScienceTfhModel(PARAMS),
            ],
        )


def test_composed_model_rejects_states_outside_the_spec():
    with pytest.raises(ValueError, match="not in"):
        ComposedModel(
            SCIENCE_STATES,
            [
                ScienceUpstreamModel(PARAMS),
                ScienceTCellModel(PARAMS),
                ScienceTfhModel(PARAMS),
                MayerPresentationModel(PARAMS),  # owns "C", not a Science state
            ],
        )


def test_swapping_the_tcell_module_leaves_upstream_states_unaffected():
    schedule = DoseSchedule.from_exponential_escalation(numshot=7, k=1.0, duration=12)
    t_eval = np.linspace(0, 21, 211)

    default_result = simulate_science(PARAMS, IC, schedule, t_end=21.0, t_eval=t_eval)

    mock_model = ComposedModel(
        SCIENCE_STATES,
        [
            ScienceUpstreamModel(PARAMS),  # untouched -- same instance shape as default
            MockConstantTCellModel(),  # <-- the swap
            ScienceTfhModel(PARAMS),  # untouched
        ],
    )
    mock_result = simulate_science(PARAMS, IC, schedule, t_end=21.0, t_eval=t_eval, model=mock_model)

    # The upstream states don't depend on T or TFH at all, so swapping the
    # T-cell component must leave them numerically equivalent to solver
    # tolerance. Not bit-for-bit: RK45's adaptive step control looks at
    # the error estimate across the whole coupled state vector, so a
    # different T/TFH trajectory perturbs step sizes by ~1e-8-ish
    # relative noise even for analytically-decoupled states like Ag --
    # that's a property of solving one coupled system, not evidence the
    # swap leaked into the upstream equations themselves.
    for name in ["Ag", "Adj", "TC", "DC", "aDC_Ag"]:
        assert np.allclose(
            default_result.trajectory(name), mock_result.trajectory(name), rtol=1e-5, atol=1e-12
        ), name

    # The mock component holds T exactly constant at its initial value...
    assert np.all(mock_result.trajectory("T") == IC["T"])
    # ...and since T never leaves T0, ScienceTfhModel (unmodified) never
    # produces any Tfh cells either -- purely a consequence of composing
    # the untouched Tfh component with the mock, not a special case coded
    # into the mock itself.
    assert np.all(mock_result.trajectory("TFH") == 0.0)

    # And the default (real) run shows real T-cell/Tfh expansion, so the
    # difference above is actually due to the swap, not a no-op schedule.
    assert default_result.trajectory("T").max() > IC["T"]
    assert default_result.trajectory("TFH")[-1] > 0.0
