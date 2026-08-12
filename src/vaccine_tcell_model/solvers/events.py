"""Event-aware ODE integration: solve piecewise between jump events,
applying each jump exactly at its boundary time -- no numerical smearing
(master spec rule 8).

Design note on which value is reported at a boundary time: when a jump
occurs at time T, this module reports the POST-jump ("right-continuous")
value at T. That is a deliberate choice, not incidental -- it means a
dose administered at t=0 is directly visible as the dosed value at the
very first row of the output, and a dose at an interior time T is
visible as an exact discontinuity between the row at T and the row
immediately before it, rather than being folded into the first
integration step after T (which is what the reference MATLAB
implementation does -- see docs/equations.md; we chose the more exact
representation here).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from scipy.integrate import solve_ivp

from vaccine_tcell_model.core.events import StateJumpEvent
from vaccine_tcell_model.core.state import StateSpec


def integrate_with_events(
    rhs: Callable[[float, np.ndarray], np.ndarray],
    state_spec: StateSpec,
    y0: np.ndarray,
    t_span: tuple[float, float],
    events: Sequence[StateJumpEvent] = (),
    *,
    t_eval: np.ndarray | None = None,
    method: str = "RK45",
    rtol: float = 1e-8,
    atol: float = 1e-10,
    max_step: float = np.inf,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate `rhs` over t_span, applying `events` as exact instantaneous
    additive jumps to the state.

    Implementation: split [t_span[0], t_span[1]] into segments at every
    event time, integrate each segment independently with
    `scipy.integrate.solve_ivp`, and add each event's jump to the state
    before integrating the segment that starts at that event's time. The
    jump size is therefore exact floating-point addition, never an
    artifact of solver step size.

    Returns (t, y) with t of shape (n,) and y of shape (len(state_spec), n).
    """
    t0, t_end = t_span
    if t_end <= t0:
        raise ValueError(f"t_span must have t_end > t0, got {t_span}")

    events = sorted(events, key=lambda e: e.time)
    for e in events:
        if not (t0 <= e.time <= t_end):
            raise ValueError(f"Event time {e.time} is outside the integration span {t_span}")

    boundaries = sorted({t0, t_end, *(e.time for e in events)})
    t_eval_arr = np.asarray(t_eval, dtype=float) if t_eval is not None else None

    t_all: list[np.ndarray] = []
    y_all: list[np.ndarray] = []
    y_current = np.array(y0, dtype=float)

    for seg_start, seg_end in zip(boundaries[:-1], boundaries[1:]):
        for e in events:
            if e.time == seg_start:
                y_current = _apply_jump(y_current, state_spec, e)

        if t_eval_arr is not None:
            mask = (t_eval_arr >= seg_start) & (t_eval_arr <= seg_end)
            seg_t_eval = np.unique(np.concatenate(([seg_start], t_eval_arr[mask], [seg_end])))
        else:
            seg_t_eval = None

        sol = solve_ivp(
            rhs,
            (seg_start, seg_end),
            y_current,
            method=method,
            t_eval=seg_t_eval,
            rtol=rtol,
            atol=atol,
            max_step=max_step,
        )
        if not sol.success:
            raise RuntimeError(
                f"ODE integration failed on segment [{seg_start}, {seg_end}]: {sol.message}"
            )

        if t_all:
            # Drop the previous segment's stored boundary point (the
            # pre-jump value) so the retained value at this boundary is
            # this segment's first point (post-jump / right-continuous).
            t_all[-1] = t_all[-1][:-1]
            y_all[-1] = y_all[-1][:, :-1]

        t_all.append(sol.t)
        y_all.append(sol.y)
        y_current = sol.y[:, -1]

    # A jump scheduled exactly at t_end never starts a new segment (t_end
    # is always a seg_end, never a seg_start) -- apply it directly to the
    # final stored point.
    for e in events:
        if e.time == t_end:
            y_current = _apply_jump(y_current, state_spec, e)
            y_all[-1][:, -1] = y_current

    t = np.concatenate(t_all)
    y = np.concatenate(y_all, axis=1)
    return t, y


def _apply_jump(y: np.ndarray, state_spec: StateSpec, event: StateJumpEvent) -> np.ndarray:
    y = y.copy()
    for name, amount in event.jumps.items():
        y[state_spec.index(name)] += amount
    return y
