"""High-level solver wrapper: a dose-aware `simulate()` that produces a
SimulationResult from a named-state RHS function (master spec Section 11).

Model authors write `rhs(t, state_dict) -> derivative_dict` against named
states, never array indices; this wrapper handles the array<->dict
conversion and delegates the actual piecewise/event integration to
solvers.events.integrate_with_events.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from vaccine_tcell_model.core.events import StateJumpEvent
from vaccine_tcell_model.core.result import SimulationResult
from vaccine_tcell_model.core.state import StateSpec

from .events import integrate_with_events

RHSFunc = Callable[[float, dict[str, float]], dict[str, float]]


def simulate(
    rhs: RHSFunc,
    state_spec: StateSpec,
    initial_conditions: dict[str, float],
    t_end: float,
    *,
    t0: float = 0.0,
    dose_schedule=None,
    antigen_state: str = "Ag",
    adjuvant_state: str = "Adj",
    extra_events: list[StateJumpEvent] | None = None,
    t_eval: np.ndarray | None = None,
    method: str = "RK45",
    rtol: float = 1e-8,
    atol: float = 1e-10,
    max_step: float = np.inf,
    parameters=None,
    model_name: str = "",
) -> SimulationResult:
    """Integrate a dict-based RHS from t0 to t_end and return a tidy SimulationResult.

    Args:
        rhs: `rhs(t, state_dict) -> {state_name: d(state_name)/dt}`. Must
            return a derivative for every name in `state_spec`.
        state_spec: defines which states exist and their order.
        initial_conditions: {state_name: value} for every state in
            `state_spec` (master spec Section 14 -- must be supplied
            explicitly, never silently defaulted).
        t_end: end of the integration window; `t0` defaults to 0.
        dose_schedule: a dosing.DoseSchedule, or None for models with no
            dosing concept (e.g. a bare Mayer T/C run). Its events are
            applied as exact jumps to `antigen_state`/`adjuvant_state`.
        extra_events: additional StateJumpEvents beyond the dose schedule
            (e.g. for models that jump a state other than Ag/Adj).
        method: passed to scipy.integrate.solve_ivp ("RK45", "BDF",
            "LSODA", ...).
        parameters, model_name: attached to the returned SimulationResult
            for provenance/bookkeeping; not used by the solver itself.
    """
    def array_rhs(t: float, y: np.ndarray) -> np.ndarray:
        state = state_spec.to_dict(y)
        derivatives = rhs(t, state)
        missing = set(state_spec.names) - set(derivatives)
        if missing:
            raise ValueError(f"rhs() did not return derivatives for: {sorted(missing)}")
        return np.array([derivatives[n] for n in state_spec.names], dtype=float)

    events = list(extra_events) if extra_events else []
    if dose_schedule is not None:
        events += dose_schedule.to_state_jump_events(
            antigen_state=antigen_state, adjuvant_state=adjuvant_state
        )

    y0 = state_spec.to_array(initial_conditions)

    t, y = integrate_with_events(
        array_rhs,
        state_spec,
        y0,
        (t0, t_end),
        events=events,
        t_eval=t_eval,
        method=method,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
    )

    data = pd.DataFrame({"time": t})
    for i, name in enumerate(state_spec.names):
        data[name] = y[i]

    return SimulationResult(
        data=data,
        states=state_spec,
        parameters=parameters,
        dose_schedule=dose_schedule,
        model_name=model_name,
    )
