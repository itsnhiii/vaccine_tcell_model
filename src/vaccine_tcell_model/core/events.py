"""Generic state-jump event: the solver-level primitive dosing builds on.

Model-agnostic on purpose: the solver only needs to know "add these
deltas to these named states at this time." DoseEvent
(dosing/events.py) is the vaccine-specific specialization that knows
about antigen/adjuvant and produces these.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class StateJumpEvent:
    """An additive, instantaneous jump applied to named states at `time`.

    Represents a published Dirac-delta dose term (e.g. Science supplement
    Eq. 1-2) as an explicit discontinuity rather than a literal delta
    function passed to an ODE solver (master spec rule 8).
    """

    time: float
    jumps: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.time < 0:
            raise ValueError(f"StateJumpEvent.time must be non-negative, got {self.time}")
