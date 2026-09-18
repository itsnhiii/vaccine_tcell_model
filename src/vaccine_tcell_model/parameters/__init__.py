"""Parameter metadata and provenance system.

Concrete parameter sets (ScienceParameters, MayerParameters,
IntegratedParameters) are added in Phases 3, 5, and 7 respectively.
This subpackage only defines the shared Parameter/ParameterSet/SourceType
machinery every parameter set is built from.
"""

from .cancer_tcr import (
    cancer_tcr_default_initial_conditions,
    default_cancer_tcr_parameters,
    default_tcr_signal_parameters,
)
from .integrated import default_integrated_parameters, integrated_default_initial_conditions
from .mayer import (
    MAYER_FIG2_LIGAND_K,
    default_mayer_parameters,
    mayer_demo_initial_conditions,
    mayer_parameters_fig1_cd4,
    mayer_parameters_fig2_cd8,
    mayer_parameters_fig4_dosing,
)
from .metadata import Parameter, ParameterSet
from .provenance import SourceType
from .science import default_science_parameters, science_default_initial_conditions

__all__ = [
    "Parameter",
    "ParameterSet",
    "SourceType",
    "default_science_parameters",
    "science_default_initial_conditions",
    "default_mayer_parameters",
    "mayer_parameters_fig1_cd4",
    "mayer_parameters_fig2_cd8",
    "mayer_parameters_fig4_dosing",
    "mayer_demo_initial_conditions",
    "MAYER_FIG2_LIGAND_K",
    "default_integrated_parameters",
    "integrated_default_initial_conditions",
    "default_cancer_tcr_parameters",
    "cancer_tcr_default_initial_conditions",
    "default_tcr_signal_parameters",
]
