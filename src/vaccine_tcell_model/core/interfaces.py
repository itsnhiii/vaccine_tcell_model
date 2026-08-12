"""Abstract interfaces for pluggable ODE right-hand-side components.

This module fixes the *shape* every component must have, so the solver
and dosing code never depend on a specific implementation (master spec
Section 9) -- a future ExplicitPeptideProcessingModel can replace
SimplePmhcProductionModel without the T-cell or DC code changing.

Phase 6 ("modular interfaces") adds four named role subclasses --
InnateDCModel, PresentationModel, TCellActivationModel, TfhModel --
matching the four pluggable slots master spec Section 3/9 names
explicitly. They add no new abstract methods; they exist so a component's
conceptual role is checkable (`isinstance(x, TCellActivationModel)`) and
documented at the type level, and so `IntegratedModel` (Phase 7) can
require "one of each role" rather than "four arbitrary RHSComponents."
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class RHSComponent(ABC):
    """One piece of a larger ODE system's right-hand side.

    A component computes time-derivatives only for the state names it
    owns, but may *read* any state in the full system state dict (e.g.
    a T-cell component reads "aDC_Ag", a state owned by the DC
    component) so components can be composed into a full model without
    depending on each other's implementations.
    """

    @property
    @abstractmethod
    def owned_states(self) -> tuple[str, ...]:
        """State names this component computes derivatives for."""
        raise NotImplementedError

    @abstractmethod
    def rhs(self, t: float, state: dict[str, float]) -> dict[str, float]:
        """Return {state_name: d(state_name)/dt} for this component's states.

        `state` contains the full current system state (all components'
        states), keyed by name, at time `t`.
        """
        raise NotImplementedError


class InnateDCModel(RHSComponent):
    """Upstream innate/DC dynamics: antigen/adjuvant kinetics through
    activated antigen-loaded DCs. Expected to be the sole producer of
    whatever state a PresentationModel (or, for models with no separate
    presentation step, a TCellActivationModel) reads as its antigen input."""


class PresentationModel(RHSComponent):
    """pMHC / antigen-presentation dynamics. Reads an InnateDCModel's
    output (e.g. aDC_Ag) and owns the presented-antigen state (e.g. pMHC).

    Not every model has a distinct presentation step -- the published
    Science model folds presentation into aDC_Ag itself (docs/equations.md
    Flag S1) and has no separate PresentationModel component."""


class TCellActivationModel(RHSComponent):
    """T-cell proliferation dynamics. Reads a PresentationModel's (or
    InnateDCModel's) output and owns the T-cell state (e.g. T)."""


class TfhModel(RHSComponent):
    """Tfh differentiation dynamics. Reads a TCellActivationModel's
    output (e.g. T) and owns the Tfh state (e.g. TFH)."""
