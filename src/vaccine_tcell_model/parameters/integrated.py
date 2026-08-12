"""IntegratedParameters: this project's own parameter set for the
integrated model extension (docs/equations.md Section 3).

Explicitly constructed, never a silent merge of ScienceParameters and
MayerParameters (master spec rule 4 / Section 12): every value below is
individually sourced, even the ones that numerically equal a Science or
Mayer literature value, and k_p/mu_pMHC/K are explicitly
source_type=model_extension (or user_defined when overridden) since no
literature value exists for them in this combined equation.
"""

from __future__ import annotations

from .metadata import Parameter, ParameterSet
from .provenance import SourceType


def _extension_parameter(name: str, value: float, default: float, units: str, description: str) -> Parameter:
    """Build a model_extension Parameter, downgrading to user_defined
    provenance when the caller has overridden the illustrative default --
    the same pattern used for K, applied uniformly to k_p/mu_pMHC/K so
    sweeping any of them (e.g. in tests/validation/analysis.sweep_parameter)
    never mislabels an overridden value as if it were still the default.
    """
    if value == default:
        source = (
            f"No literature value exists for this project's integrated-model "
            f"extension (docs/equations.md Section 3); {value:g} is an "
            "illustrative default, not fit to any dataset"
        )
        source_type = SourceType.MODEL_EXTENSION
    else:
        source = (
            f"User-supplied {name}, overriding this project's illustrative "
            f"default of {default:g} (no literature value exists for this "
            "integrated-model extension parameter)"
        )
        source_type = SourceType.USER_DEFINED
    return Parameter(name=name, value=value, units=units, source=source, source_type=source_type, description=description)


def default_integrated_parameters(K: float = 10.0, k_p: float = 1.0, mu_pMHC: float = 1.0) -> ParameterSet:
    """Default parameters for the integrated model.

    Upstream (Ag/Adj/TC/DC/aDC_Ag) parameters are Science's Table S2
    values, reused unchanged since ScienceUpstreamModel is reused
    unchanged (Phase 6/7). alpha/eta are likewise Science's values
    (themselves from Mayer et al. 2019 Fig.1, see docs/equations.md Flag
    S3), reused for the integrated T-cell equation's proliferation-rate
    and differentiation-sink terms respectively. k_p, mu_pMHC, and K have
    no literature source for this combined equation -- they are this
    project's own extension parameters, each independently overridable
    (e.g. for the Phase 8 sensitivity sweeps in
    tests/test_integrated_validation.py) via this function's keyword
    arguments rather than by mutating a ParameterSet after construction
    (ParameterSet deliberately has no such API -- see metadata.py).
    """

    params = {
        "d_Ag": Parameter(
            name="d_Ag", value=3.0, units="1/day",
            source="Bhagchandani et al. 2024 Table S2, reused unchanged via ScienceUpstreamModel",
            source_type=SourceType.LITERATURE,
            description="Antigen first-order decay rate",
        ),
        "d_Adj": Parameter(
            name="d_Adj", value=3.0, units="1/day",
            source="Bhagchandani et al. 2024 Table S2 ('taken to be identical to d_Ag'), reused unchanged via ScienceUpstreamModel",
            source_type=SourceType.ASSUMED,
            description="Adjuvant first-order decay rate",
        ),
        "S_Adj": Parameter(
            name="S_Adj", value=0.1, units="dimensionless (normalized dose units, total dose = 1)",
            source="Bhagchandani et al. 2024 Table S2, reused unchanged via ScienceUpstreamModel",
            source_type=SourceType.ASSUMED,
            description="Half-max adjuvant activity concentration",
        ),
        "k": Parameter(
            name="k", value=10.0, units="dimensionless",
            source="Bhagchandani et al. 2024 Table S2, reused unchanged via ScienceUpstreamModel",
            source_type=SourceType.ASSUMED,
            description="Max fold-increase in antigen uptake rate conferred by adjuvant",
        ),
        "mu": Parameter(
            name="mu", value=1.2, units="1/day",
            source="Mayer et al. 2019 Fig.1 fit, via Bhagchandani et al. 2024 Table S2, reused unchanged via ScienceUpstreamModel",
            source_type=SourceType.LITERATURE,
            description="Innate immune cell (TC, DC, aDC_Ag) death/decay rate",
        ),
        "D0": Parameter(
            name="D0", value=1.8e6, units="1/day",
            source="Bhagchandani et al. 2024 Table S2, fitted to Fig. 3F, reused unchanged via ScienceUpstreamModel",
            source_type=SourceType.FITTED,
            description="Rate of DC recruitment by tissue-resident innate cells (TC)",
        ),
        "T0": Parameter(
            name="T0", value=28.0, units="cells (arbitrary/normalized units)",
            source="Bhagchandani et al. 2024 Table S2, fitted jointly with D0 to Fig. 3F, reused unchanged via ScienceUpstreamModel and ScienceTfhModel",
            source_type=SourceType.FITTED,
            description="Baseline number of antigen-specific T cells",
        ),
        "alpha": Parameter(
            name="alpha", value=1.5, units="1/day",
            source="Mayer et al. 2019 Fig.1 fit, via Bhagchandani et al. 2024 Table S2, reused for the integrated T-cell equation's proliferation term",
            source_type=SourceType.LITERATURE,
            description="Maximum T-cell proliferation rate",
        ),
        "eta": Parameter(
            name="eta", value=0.22, units="1/day",
            source=(
                "Numerically identical to Mayer et al. 2019 Fig.1 fitted delta "
                "(T-cell death rate); Bhagchandani et al. 2024 reinterprets it as "
                "a T-to-Tfh differentiation rate (docs/equations.md Flag S3), "
                "reused unchanged for the integrated model's T-cell sink term and Tfh source term"
            ),
            source_type=SourceType.LITERATURE,
            description="T-cell to Tfh differentiation rate",
        ),
        "k_p": _extension_parameter(
            "k_p", k_p, default=1.0, units="1/day",
            description="pMHC production rate from aDC_Ag",
        ),
        "mu_pMHC": _extension_parameter(
            "mu_pMHC", mu_pMHC, default=1.0, units="1/day",
            description="pMHC decay rate",
        ),
        "K": _extension_parameter(
            "K", K, default=10.0, units="same units as pMHC (arbitrary/normalized)",
            description="Functional avidity/affinity parameter for the integrated T-cell equation",
        ),
    }
    return ParameterSet("integrated", params)


def integrated_default_initial_conditions(params: ParameterSet) -> dict[str, float]:
    """All states zero except T(0) = T0, matching ScienceParameters'
    initial conditions with pMHC(0) = 0 inserted (master spec rule 14 --
    literature-derived for the reused Science states; the added pMHC
    state defaults to 0 as its only physically sensible starting value
    before any antigen presentation has occurred).
    """
    return {
        "Ag": 0.0,
        "Adj": 0.0,
        "TC": 0.0,
        "DC": 0.0,
        "aDC_Ag": 0.0,
        "pMHC": 0.0,
        "T": params.value("T0"),
        "TFH": 0.0,
    }
