"""Bhagchandani et al. 2024 Science Immunology T-cell priming model.

Implements supplement Eq. 1-7 exactly (docs/equations.md Section 1), as
three independently pluggable components (Phase 6): ScienceUpstreamModel,
ScienceTCellModel, ScienceTfhModel, composed by build_science_model().
No K/affinity term -- confirmed absent from the published equations.
"""

from .antigen_adjuvant import antigen_adjuvant_rhs
from .full_model import build_science_model, science_rhs, simulate_science
from .innate_dc import innate_dc_rhs
from .tcell import ScienceTCellModel, ScienceTfhModel, tcell_activation_rhs, tfh_rhs
from .upstream import ScienceUpstreamModel

__all__ = [
    "antigen_adjuvant_rhs",
    "innate_dc_rhs",
    "tcell_activation_rhs",
    "tfh_rhs",
    "ScienceUpstreamModel",
    "ScienceTCellModel",
    "ScienceTfhModel",
    "build_science_model",
    "science_rhs",
    "simulate_science",
]
