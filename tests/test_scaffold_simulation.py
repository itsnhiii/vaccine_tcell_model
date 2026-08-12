"""Phase 1 success criterion: "one trivial simulation can be constructed."

This does not exercise any published Science/Mayer equation (those come
in Phases 3 and 5, with their own solver wrapper in Phase 2). It only
proves that StateSpec, Parameter/ParameterSet, and SimulationResult
interoperate end-to-end: a trivial single-state exponential-decay ODE,
solved directly with scipy, wrapped into a SimulationResult, and checked
against its analytic solution.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from vaccine_tcell_model.core import SimulationResult, StateSpec
from vaccine_tcell_model.parameters import Parameter, ParameterSet, SourceType


def test_trivial_exponential_decay_simulation_matches_analytic_solution():
    states = StateSpec(names=("X",))

    decay_rate = Parameter(
        name="d_X",
        value=0.5,
        units="1/day",
        source="scaffold smoke test, not a scientific parameter",
        source_type=SourceType.USER_DEFINED,
    )
    params = ParameterSet("trivial_decay", {"d_X": decay_rate})

    x0 = 10.0
    t_end = 5.0

    def rhs(t, y):
        return [-params.value("d_X") * y[0]]

    sol = solve_ivp(
        rhs,
        t_span=(0.0, t_end),
        y0=states.to_array({"X": x0}),
        t_eval=np.linspace(0.0, t_end, 50),
        method="RK45",
        rtol=1e-8,
        atol=1e-10,
    )
    assert sol.success

    data = pd.DataFrame({"time": sol.t, "X": sol.y[0]})
    result = SimulationResult(
        data=data,
        states=states,
        parameters=params,
        dose_schedule=None,
        model_name="trivial_scaffold_check",
    )

    analytic = x0 * np.exp(-decay_rate.value * result.time)
    assert np.allclose(result.trajectory("X"), analytic, rtol=1e-6)
    assert result.time[0] == 0.0
    assert result.time[-1] == t_end
    assert "trivial_scaffold_check" in repr(result)
