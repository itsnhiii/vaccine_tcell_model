"""This project's cancer/TCR-signaling model: Ag -> DC -> aDC_Ag -> pMHC_density -> T,
plus a closed-form TCR signaling-strength calculation (docs/cancer_tcr_model.md).

Built per Zach's request (a cancer-mouse/patient dosing-comparison tool,
not the virus/humoral-antibody pipeline the Science/Mayer/Integrated
models target): reuses Bhagchandani et al. 2024's upstream DC/adjuvant
pharmacokinetics unchanged, replaces the Tfh/germinal-center output with
nothing (no Tfh compartment at all), and adds a kinetic-proofreading TCR
signaling-strength layer built from Chakraborty & Weiss 2014. NEVER
described as published verbatim by any source paper -- see
pmhc_density.py, tcell.py, and signal.py docstrings for exactly which
piece comes from where.
"""

from .full_model import build_cancer_tcr_model, cancer_tcr_rhs, simulate_cancer_tcr
from .pmhc_density import PMHCDensityModel, pmhc_density_rhs
from .signal import compute_tcr_signal, tcr_signal_from_arrays
from .tcell import CancerTCellModel, cancer_tcell_rhs

__all__ = [
    "PMHCDensityModel",
    "pmhc_density_rhs",
    "CancerTCellModel",
    "cancer_tcell_rhs",
    "build_cancer_tcr_model",
    "cancer_tcr_rhs",
    "simulate_cancer_tcr",
    "compute_tcr_signal",
    "tcr_signal_from_arrays",
]
