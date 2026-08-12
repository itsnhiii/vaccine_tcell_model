"""Mayer et al. 2019 PNAS T-cell expansion model.

Implements main text Eq. 1-2 exactly (the approximate, saturating-
competition form used in every figure -- docs/equations.md Section 2),
as two independently pluggable components (Phase 6): MayerTCellModel,
MayerPresentationModel, composed by build_mayer_model().
"""

from .full_model import build_mayer_model, mayer_rhs, simulate_mayer
from .pmhc import MayerPresentationModel, pmhc_rhs
from .tcell import MayerTCellModel, tcell_rhs

__all__ = [
    "pmhc_rhs",
    "tcell_rhs",
    "MayerPresentationModel",
    "MayerTCellModel",
    "build_mayer_model",
    "mayer_rhs",
    "simulate_mayer",
]
