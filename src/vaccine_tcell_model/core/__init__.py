"""Core: state definitions, interfaces, and simulation results.

Kept separate from numerical integration (solvers/), plotting
(plotting/), fitting (analysis/), and dosing (dosing/) per master-spec
rule 10.
"""

from .composition import ComposedModel
from .events import StateJumpEvent
from .interfaces import (
    InnateDCModel,
    PresentationModel,
    RHSComponent,
    TCellActivationModel,
    TfhModel,
)
from .result import SimulationResult
from .state import INTEGRATED_STATES, MAYER_STATES, SCIENCE_STATES, StateSpec

__all__ = [
    "StateSpec",
    "SCIENCE_STATES",
    "MAYER_STATES",
    "INTEGRATED_STATES",
    "SimulationResult",
    "RHSComponent",
    "InnateDCModel",
    "PresentationModel",
    "TCellActivationModel",
    "TfhModel",
    "StateJumpEvent",
    "ComposedModel",
]
