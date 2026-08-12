"""MayerParameters: Mayer et al. 2019 fitted parameter presets.

Three fits appear in the paper, each for a different experimental
system; they are never merged (master spec rule 4; docs/equations.md
Section 2.2 table):

    Fig.1 -- CD4 T cells vs. cytochrome C (Quiel et al. data), K fixed at 0
    Fig.2 -- CD8 OT-1 T cells vs. Listeria variants (Zehn et al. data), K = ligand EC50
    Fig.4 -- antigen-dosing-kinetics simulation, alpha/mu/delta as Fig.2, K=10

Each is exposed as its own function returning an independent ParameterSet.
"""

from __future__ import annotations

from .metadata import Parameter, ParameterSet
from .provenance import SourceType


def mayer_parameters_fig1_cd4() -> ParameterSet:
    """Fig.1 legend fit: CD4 T cells responding to cytochrome C.

    K is fixed at 0 in this fit (upper bound from fit ~700) -- not useful
    for demonstrating K-affinity direction on its own; use the fig2/fig4
    presets for that.
    """
    params = {
        "alpha": Parameter(
            name="alpha", value=1.5, units="1/day",
            source="Mayer et al. 2019 Fig. 1 legend fit (alpha = 1.5 +/- 0.3/d)",
            source_type=SourceType.LITERATURE,
            description="Maximum T-cell proliferation rate",
        ),
        "mu": Parameter(
            name="mu", value=1.2, units="1/day",
            source="Mayer et al. 2019 Fig. 1 legend fit (mu = 1.2 +/- 0.5/d)",
            source_type=SourceType.LITERATURE,
            description="pMHC decay rate",
        ),
        "delta": Parameter(
            name="delta", value=0.22, units="1/day",
            source="Mayer et al. 2019 Fig. 1 legend fit (delta = 0.22 +/- 0.21/d)",
            source_type=SourceType.LITERATURE,
            description="T-cell death rate",
        ),
        "K": Parameter(
            name="K", value=0.0, units="same units as C (arbitrary/normalized)",
            source="Mayer et al. 2019 Fig. 1 legend, fixed at 0 (upper bound from fit ~700)",
            source_type=SourceType.LITERATURE,
            description="Functional avidity/affinity parameter (fixed, not free, in this fit)",
        ),
    }
    return ParameterSet("mayer_fig1_cd4", params)


MAYER_FIG2_LIGAND_K: dict[str, float] = {
    "SIINFEKL": 1.0,
    "SAINFEKL": 2.7,
    "SIYNFEKL": 4.1,
    "SIIQFEKL": 18.3,
    "SIITFEKL": 70.7,
    "SIIVFEKL": 680.0,
}
"""Relative EC50 (used directly as K) per ligand -- Mayer et al. 2019 Fig. 2 legend table."""


def mayer_parameters_fig2_cd8(K: float = 1.0) -> ParameterSet:
    """Fig.2 legend fit: CD8 OT-1 T cells vs. Listeria ligands of differing affinity.

    K is set per-ligand to the measured relative EC50 -- pass one of
    MAYER_FIG2_LIGAND_K's values. Default (K=1.0) is SIINFEKL, the
    highest-affinity ligand.
    """
    is_tabulated = K in MAYER_FIG2_LIGAND_K.values()
    k_source = (
        "Mayer et al. 2019 Fig. 2 legend: 'K was set equal to the relative "
        "EC50 values of the different ligands' (see MAYER_FIG2_LIGAND_K)"
        if is_tabulated
        else "User-supplied K, outside the six ligand EC50 values tabulated "
        "in Mayer et al. 2019 Fig. 2 (MAYER_FIG2_LIGAND_K)"
    )
    params = {
        "alpha": Parameter(
            name="alpha", value=2.47, units="1/day",
            source="Mayer et al. 2019 Fig. 2 legend fit (alpha = 2.47 +/- 0.13/d)",
            source_type=SourceType.LITERATURE,
            description="Maximum T-cell proliferation rate",
        ),
        "mu": Parameter(
            name="mu", value=3.1, units="1/day",
            source="Mayer et al. 2019 Fig. 2 legend fit (mu = 3.1 +/- 0.3/d)",
            source_type=SourceType.LITERATURE,
            description="pMHC decay rate",
        ),
        "delta": Parameter(
            name="delta", value=0.23, units="1/day",
            source="Mayer et al. 2019 Fig. 2 legend fit (delta = 0.23 +/- 0.06/d)",
            source_type=SourceType.LITERATURE,
            description="T-cell death rate",
        ),
        "K": Parameter(
            name="K", value=K, units="relative EC50 (dimensionless, SIINFEKL-normalized)",
            source=k_source,
            source_type=SourceType.LITERATURE if is_tabulated else SourceType.USER_DEFINED,
            description="Functional avidity, set to a ligand's measured relative EC50",
        ),
    }
    return ParameterSet("mayer_fig2_cd8", params)


def mayer_parameters_fig4_dosing(K: float = 10.0) -> ParameterSet:
    """Fig.4 legend: antigen-dosing-kinetics simulation.

    alpha/mu/delta as Fig.2 (CD8). K=10 is the paper's value; K is left
    freely overridable since Fig.4's own point is exploring dosing
    kinetics at a fixed, tunable K -- this is also the package's default
    Mayer preset (below) since, unlike Fig.1, K is genuinely a free
    parameter here rather than fixed at 0.
    """
    if K == 10.0:
        k_source = "Mayer et al. 2019 Fig. 4 legend, K=10"
        k_source_type = SourceType.LITERATURE
    else:
        k_source = (
            "User-supplied K, overriding Mayer et al. 2019 Fig. 4's literature "
            "value of 10 (used here to sweep functional avidity)"
        )
        k_source_type = SourceType.USER_DEFINED

    params = {
        "alpha": Parameter(
            name="alpha", value=2.47, units="1/day",
            source="Mayer et al. 2019 Fig. 4 legend ('alpha, mu, and delta as in Fig. 2')",
            source_type=SourceType.LITERATURE,
            description="Maximum T-cell proliferation rate",
        ),
        "mu": Parameter(
            name="mu", value=3.1, units="1/day",
            source="Mayer et al. 2019 Fig. 4 legend ('alpha, mu, and delta as in Fig. 2')",
            source_type=SourceType.LITERATURE,
            description="pMHC decay rate",
        ),
        "delta": Parameter(
            name="delta", value=0.23, units="1/day",
            source="Mayer et al. 2019 Fig. 4 legend ('alpha, mu, and delta as in Fig. 2')",
            source_type=SourceType.LITERATURE,
            description="T-cell death rate",
        ),
        "K": Parameter(
            name="K", value=K, units="same units as C (arbitrary/normalized)",
            source=k_source,
            source_type=k_source_type,
            description="Functional avidity/affinity parameter",
        ),
    }
    return ParameterSet("mayer_fig4_dosing", params)


def default_mayer_parameters(K: float = 10.0) -> ParameterSet:
    """The Fig.4 preset, used as this package's general-purpose Mayer
    default -- unlike Fig.1 (K fixed at 0), Fig.4 treats K as a free,
    tunable parameter, making it the natural default for K-affinity sweeps.
    """
    return mayer_parameters_fig4_dosing(K=K)


def mayer_demo_initial_conditions(T0: float = 300.0, C0: float = 10.0**6.7) -> dict[str, float]:
    """Example initial conditions from Fig. 1C (T0=300 or 30000 precursor
    T cells; C(0)=10^(6.7+/-1.1) pMHC).

    NOT a universal biological default -- T0 and C0 are exactly the
    quantities this model (and the paper) sweeps across figures. Labeled
    demo/example values only, per master spec rule 14.
    """
    return {"T": T0, "C": C0}
