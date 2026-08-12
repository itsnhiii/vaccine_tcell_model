"""ScienceParameters: Bhagchandani et al. 2024 Table S2 values, with full
provenance. See docs/equations.md Section 1.2 for the complete
source-to-parameter mapping and every flag noted below.

Every value here is either the literature value from Table S2, or (for
d_Adj, S_Adj, k) the explicit modeling assumption the paper itself makes
-- never a value we chose independently.
"""

from __future__ import annotations

from .metadata import Parameter, ParameterSet
from .provenance import SourceType


def default_science_parameters() -> ParameterSet:
    """Table S2 parameter values for the Science T-cell priming model."""
    params = {
        "d_Ag": Parameter(
            name="d_Ag",
            value=3.0,
            units="1/day",
            source=(
                "Bhagchandani et al. 2024 Table S2 ('Our data', Table S1). "
                "Table S1's raw MLE is ~2.7/day (half-life 6.2 hr); Table S2 "
                "records 3/day -- see docs/equations.md Flag S2, not reconciled."
            ),
            source_type=SourceType.LITERATURE,
            description="Antigen first-order decay rate",
        ),
        "d_Adj": Parameter(
            name="d_Adj",
            value=3.0,
            units="1/day",
            source="Bhagchandani et al. 2024 Table S2, 'taken to be identical to d_Ag'",
            source_type=SourceType.ASSUMED,
            description="Adjuvant first-order decay rate",
        ),
        "S_Adj": Parameter(
            name="S_Adj",
            value=0.1,
            units="dimensionless (normalized dose units, total dose = 1)",
            source=(
                "Bhagchandani et al. 2024 Table S2, chosen qualitatively so "
                "that DC recruitment saturates at low adjuvant dose (Fig. 3H)"
            ),
            source_type=SourceType.ASSUMED,
            description="Half-max adjuvant activity concentration",
        ),
        "k": Parameter(
            name="k",
            value=10.0,
            units="dimensionless",
            source="Bhagchandani et al. 2024 Table S2, 'we set k to be large'",
            source_type=SourceType.ASSUMED,
            description="Max fold-increase in antigen uptake rate conferred by adjuvant",
        ),
        "mu": Parameter(
            name="mu",
            value=1.2,
            units="1/day",
            source=(
                "Mayer et al. 2019 Fig. 1 legend fit (CD4/cytochrome-C system), "
                "reused as-is via Bhagchandani et al. 2024 Table S2 ('From ref. 37')"
            ),
            source_type=SourceType.LITERATURE,
            description="Innate immune cell (TC, DC, aDC_Ag) death/decay rate",
        ),
        "alpha": Parameter(
            name="alpha",
            value=1.5,
            units="1/day",
            source=(
                "Mayer et al. 2019 Fig. 1 legend fit (CD4/cytochrome-C system), "
                "reused as-is via Bhagchandani et al. 2024 Table S2 ('From ref. 37')"
            ),
            source_type=SourceType.LITERATURE,
            description="Maximum T-cell proliferation rate",
        ),
        "eta": Parameter(
            name="eta",
            value=0.22,
            units="1/day",
            source=(
                "Numerically identical to Mayer et al. 2019 Fig. 1 fitted delta "
                "(T-cell death rate, CD4/cytochrome-C system). Bhagchandani et al. "
                "2024 explicitly reinterprets this value as a T-to-Tfh "
                "differentiation rate instead of a death rate (supplement p.3: "
                "'rather than accounting for death, we model the process as T "
                "cells proliferating and differentiating'). Confirmed in the "
                "reference MATLAB code: the variable is literally named 'delta' "
                "throughout odeInnate.m, never renamed. See docs/equations.md Flag S3."
            ),
            source_type=SourceType.LITERATURE,
            description="T-cell to Tfh differentiation rate (NOT an independently-fit differentiation rate)",
        ),
        "D0": Parameter(
            name="D0",
            value=1.8e6,
            units="1/day",
            source=(
                "Bhagchandani et al. 2024 Table S2, fitted to Fig. 3F Tfh counts "
                "across dose-number regimens, using the model's T-cell/Tfh state "
                "AT DAY 21 (near-plateau), per the 2026 erratum to the Fig. 3F "
                "caption -- see docs/equations.md Flag S4. Raw fit value in the "
                "reference code (optimized_parameters.txt): 1.7826e6."
            ),
            source_type=SourceType.FITTED,
            description="Rate of DC recruitment by tissue-resident innate cells (TC)",
        ),
        "T0": Parameter(
            name="T0",
            value=28.0,
            units="cells (arbitrary/normalized units, same scale as D0)",
            source=(
                "Bhagchandani et al. 2024 Table S2, fitted jointly with D0 to "
                "Fig. 3F (day-21 model value, see Flag S4). Raw fit value in the "
                "reference code: 28.091."
            ),
            source_type=SourceType.FITTED,
            description="Baseline number of antigen-specific T cells",
        ),
    }
    return ParameterSet("science", params)


def science_default_initial_conditions(params: ParameterSet) -> dict[str, float]:
    """Table S2 initial conditions: all states zero except T(0) = T0.

    Literature-derived (Table S2 explicitly specifies these), not an
    invented default (master spec rule 14).
    """
    return {
        "Ag": 0.0,
        "Adj": 0.0,
        "TC": 0.0,
        "DC": 0.0,
        "aDC_Ag": 0.0,
        "T": params.value("T0"),
        "TFH": 0.0,
    }
