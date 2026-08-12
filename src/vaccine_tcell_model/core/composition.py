"""Compose independent RHSComponent objects into a single, solver-facing RHS.

This is the mechanism that makes a model pluggable (master spec Section
9): swap out any component -- a different PresentationModel, a mock
TCellActivationModel for testing -- without touching the solver or any
other component, as long as the replacement owns the same state names
and reads whatever it needs from the shared `state` dict.
"""

from __future__ import annotations

from collections.abc import Sequence

from .interfaces import RHSComponent
from .state import StateSpec


class ComposedModel:
    """A full ODE right-hand side assembled from independent RHSComponents.

    Validates at construction time that every state in `state_spec` is
    owned by exactly one component: no silent gaps (a state nobody
    computes a derivative for) and no silent overlaps (two components
    both writing the same state, where whichever runs last would win
    unnoticed).
    """

    def __init__(self, state_spec: StateSpec, components: Sequence[RHSComponent]):
        self.state_spec = state_spec
        self.components = list(components)

        owned_by: dict[str, str] = {}
        for component in self.components:
            for name in component.owned_states:
                if name in owned_by:
                    raise ValueError(
                        f"State {name!r} is claimed by both "
                        f"{owned_by[name]} and {type(component).__name__} -- "
                        "each state must be owned by exactly one component"
                    )
                owned_by[name] = type(component).__name__

        missing = set(state_spec.names) - set(owned_by)
        if missing:
            raise ValueError(
                f"No component owns these states from {state_spec}: {sorted(missing)}"
            )
        extra = set(owned_by) - set(state_spec.names)
        if extra:
            raise ValueError(f"Components own states not in {state_spec}: {sorted(extra)}")

    def rhs(self, t: float, state: dict[str, float]) -> dict[str, float]:
        derivatives: dict[str, float] = {}
        for component in self.components:
            derivatives.update(component.rhs(t, state))
        return derivatives

    def __repr__(self) -> str:
        names = [type(c).__name__ for c in self.components]
        return f"ComposedModel({self.state_spec.names}, components={names})"
