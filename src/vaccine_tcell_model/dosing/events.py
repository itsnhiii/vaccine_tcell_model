"""DoseEvent: a single antigen/adjuvant dose administered at an instant in time.

The vaccine-specific specialization of core.events.StateJumpEvent --
this is where "antigen" and "adjuvant" as concepts enter the dosing
layer; the solver layer only ever sees the generic StateJumpEvent.
"""

from __future__ import annotations

from dataclasses import dataclass

from vaccine_tcell_model.core.events import StateJumpEvent


@dataclass(frozen=True)
class DoseEvent:
    """A discrete antigen and/or adjuvant dose administered at `time`.

    Represents the published Dirac-delta dose terms (Science supplement
    Eq. 1-2) as an explicit state jump, not a literal delta function
    inside the solver (master spec rule 8).
    """

    time: float
    antigen_jump: float = 0.0
    adjuvant_jump: float = 0.0

    def __post_init__(self) -> None:
        if self.time < 0:
            raise ValueError(f"DoseEvent.time must be non-negative, got {self.time}")
        if self.antigen_jump < 0 or self.adjuvant_jump < 0:
            raise ValueError(
                "DoseEvent jumps cannot be negative, got "
                f"antigen_jump={self.antigen_jump}, adjuvant_jump={self.adjuvant_jump}"
            )

    def to_state_jump_event(
        self, antigen_state: str = "Ag", adjuvant_state: str = "Adj"
    ) -> StateJumpEvent:
        """Translate to the solver's generic, name-agnostic event type."""
        jumps: dict[str, float] = {}
        if self.antigen_jump:
            jumps[antigen_state] = self.antigen_jump
        if self.adjuvant_jump:
            jumps[adjuvant_state] = self.adjuvant_jump
        return StateJumpEvent(time=self.time, jumps=jumps)
