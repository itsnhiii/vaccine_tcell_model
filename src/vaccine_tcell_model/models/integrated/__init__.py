"""This project's integrated model extension: Ag -> DC -> aDC_Ag -> pMHC -> T -> TFH.

Combines Bhagchandani et al. 2024's upstream/Tfh equations (reused
unchanged) with this project's own pMHC-production and hybrid T-cell
equations (docs/equations.md Section 3). NEVER described as published
verbatim by either source paper -- see pmhc.py and tcell.py docstrings.
"""

from .full_model import build_integrated_model, integrated_rhs, simulate_integrated
from .pmhc import SimplePmhcProductionModel, pmhc_production_rhs
from .tcell import IntegratedTCellModel, integrated_tcell_rhs
from .tfh import ScienceTfhModel, tfh_rhs

__all__ = [
    "SimplePmhcProductionModel",
    "pmhc_production_rhs",
    "IntegratedTCellModel",
    "integrated_tcell_rhs",
    "ScienceTfhModel",
    "tfh_rhs",
    "build_integrated_model",
    "integrated_rhs",
    "simulate_integrated",
]
