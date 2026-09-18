"""CancerTcrParameters: parameters for the cancer/TCR-signaling model
(docs/cancer_tcr_model.md).

Two INDEPENDENT ParameterSets, never merged (master-spec rule 4, same
discipline as parameters/integrated.py):

- "cancer_tcr": the ODE population dynamics (Ag/Adj/TC/DC/aDC_Ag reused
  unchanged from Science; pMHC_density and the T-cell equation are this
  project's own extension, replacing Mayer/Science's Tfh-oriented
  equations for a cancer-context, non-humoral readout -- see
  models/cancer_tcr/).
- "tcr_signal": the closed-form kinetic-proofreading TCR-signal
  calculation (Chakraborty & Weiss 2014), a pure function of the ODE's
  already-simulated trajectories, not itself part of the ODE.

They are kept separate deliberately: the ODE parameters describe cell
population kinetics (units: cells, days), the TCR-signal parameters
describe single-molecule binding biophysics (units: uM, seconds). Nothing
in this file combines them arithmetically.
"""

from __future__ import annotations

from .metadata import Parameter, ParameterSet
from .provenance import SourceType


def _extension_parameter(
    name: str, value: float, default: float, units: str, description: str
) -> Parameter:
    """Same pattern as parameters/integrated.py's _extension_parameter:
    build a model_extension Parameter, downgrading to user_defined
    provenance when the caller overrides the illustrative default, so a
    sweep never mislabels an overridden value as if it were still the
    default."""
    if value == default:
        source = (
            f"No literature value exists for this project's cancer/TCR-signaling "
            f"extension (docs/cancer_tcr_model.md); {value:g} is an illustrative "
            "default, not fit to any dataset"
        )
        source_type = SourceType.MODEL_EXTENSION
    else:
        source = (
            f"User-supplied {name}, overriding this project's illustrative "
            f"default of {default:g} (no literature value exists for this "
            "extension parameter)"
        )
        source_type = SourceType.USER_DEFINED
    return Parameter(name=name, value=value, units=units, source=source, source_type=source_type, description=description)


def default_cancer_tcr_parameters(
    k_load: float = 1.0, mu_pMHC: float = 1.0, K_T: float = 10.0
) -> ParameterSet:
    """Default parameters for the cancer_tcr population-dynamics ODE.

    d_Ag/d_Adj/S_Adj/k/mu/D0 are Science's Table S2 values, reused
    unchanged since ScienceUpstreamModel is reused unchanged (this layer
    of vaccine-adjuvant/DC pharmacokinetics is not virus-specific).
    alpha/delta are Mayer et al. 2019's own Fig.1-fitted values, reused
    with Mayer's OWN interpretation (a plain proliferation rate and a
    plain death rate) -- NOT Science's reinterpretation of delta as a
    Tfh-differentiation rate (docs/equations.md Flag S3), since this
    model has no Tfh compartment at all. k_load, mu_pMHC, and K_T have no
    literature source for this cancer-context pMHC_density/T-cell
    equation pairing -- they are this project's own extension
    parameters, independently overridable via this function's keyword
    arguments (ParameterSet has no in-place mutation API -- see
    metadata.py).
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
            source="Bhagchandani et al. 2024 Table S2, fitted jointly with D0 to Fig. 3F -- reused here ONLY as this model's initial T-cell count, not as a differentiation floor (this model has no Tfh sink)",
            source_type=SourceType.FITTED,
            description="Initial antigen-specific T-cell count",
        ),
        "alpha": Parameter(
            name="alpha", value=1.5, units="1/day",
            source="Mayer et al. 2019 Fig.1 fit (main text Eq.1's own alpha), reused for this model's T-cell proliferation term",
            source_type=SourceType.LITERATURE,
            description="Maximum T-cell proliferation rate",
        ),
        "delta": Parameter(
            name="delta", value=0.22, units="1/day",
            source=(
                "Numeric value borrowed from Mayer et al. 2019 Fig.1 fit (main "
                "text Eq.1's own delta) as an order-of-magnitude T-cell turnover "
                "rate -- but used in the loss term -delta*(T-T0), a homeostatic "
                "relaxation toward baseline T0, NOT Mayer's own unconditional "
                "-delta*T death term (that made the antigen-free baseline unstable "
                "-- caught by tests/test_cancer_tcr_model.py, see "
                "models/cancer_tcr/tcell.py's docstring) and NOT Science's "
                "Tfh-differentiation reinterpretation either (no Tfh compartment "
                "exists here to differentiate into -- docs/equations.md Flag S3)"
            ),
            source_type=SourceType.LITERATURE,
            description="T-cell homeostatic turnover rate (relaxation toward baseline T0)",
        ),
        "k_load": _extension_parameter(
            "k_load", k_load, default=1.0, units="1/day",
            description="pMHC_density loading rate from activated antigen-loaded DCs (aDC_Ag)",
        ),
        "mu_pMHC": _extension_parameter(
            "mu_pMHC", mu_pMHC, default=1.0, units="1/day",
            description="pMHC_density turnover/decay rate (internalization + degradation)",
        ),
        "K_T": _extension_parameter(
            "K_T", K_T, default=10.0, units="same units as pMHC_density (arbitrary/normalized)",
            description=(
                "T-cell growth saturation constant. Deliberately a SEPARATE symbol "
                "from the TCR-signal module's K_D (docs/cancer_tcr_model.md) -- do "
                "not equate K_T to an experimentally measured biochemical K_D, same "
                "warning as models/integrated/tcell.py's K"
            ),
        ),
    }
    return ParameterSet("cancer_tcr", params)


def cancer_tcr_default_initial_conditions(params: ParameterSet) -> dict[str, float]:
    """All states zero except T(0) = T0 -- same convention as Science/Integrated."""
    return {
        "Ag": 0.0,
        "Adj": 0.0,
        "TC": 0.0,
        "DC": 0.0,
        "aDC_Ag": 0.0,
        "pMHC_density": 0.0,
        "T": params.value("T0"),
    }


def default_tcr_signal_parameters(
    K_D: float = 50.0,
    k_on: float = 0.01,
    k_p: float = 0.3,
    N: float = 3.0,
    include_serial_triggering: float = 0.0,
) -> ParameterSet:
    """Default parameters for the closed-form kinetic-proofreading
    TCR-signal calculation (models/cancer_tcr/signal.py).

    K_D: the model's REQUIRED scientific input -- antigen affinity, in
    uM. Default 50 uM is illustrative only (mid-range "tumor
    self-antigen" per Stone, Chervin & Kranz 2009, Immunology
    126:165-176) -- always override with the real measured or assumed
    affinity of the antigen being modeled.

    k_on: FIXED literature constant (both source documents agree TCR-pMHC
    k_on varies little across peptides; k_off carries the affinity
    difference -- Stone/Chervin/Kranz 2009; Chakraborty & Weiss 2014's own
    kinetic-proofreading formalism assumes konAg=konEn explicitly). Default
    0.01 /uM/s = 1e4 /M/s, a representative order-of-magnitude 3D-SPR value
    (typical reported range ~1e3-1e5 /M/s). Units chosen as uM^-1 s^-1
    (not M^-1 s^-1) specifically so k_off = K_D*k_on falls out in s^-1
    with NO separate uM->M conversion factor floating around in the code.

    k_p, N: the kinetic-proofreading chain parameters (see
    models/cancer_tcr/signal.py docstring for what they mean physically).
    Defaults N=3 steps, k_p=0.3/s are literature-ANCHORED order-of-magnitude
    values (Tischer & Weiner 2019; Yousefi et al. 2019 optogenetic
    dwell-time-tuning experiments report ~2-4 effective steps at ~seconds
    each), not measured for any specific tumor-antigen/cDC1 system --
    treat both as FITTABLE against real experimental TCR-signal data,
    not as fixed literature constants.

    include_serial_triggering: RESERVED, not yet implemented (0.0=off,
    the only currently-supported value). Chakraborty & Weiss 2014 find
    "very little evidence" for the bell-curve/optimal-dwell-time
    prediction (Holler & Kranz 2003's TCR-affinity-variant data shows a
    monotonic response with no observed optimum) -- so this is left off
    by default rather than silently assumed. Setting it to a nonzero
    value raises NotImplementedError with this same explanation.
    """
    params = {
        "K_D": _extension_parameter(
            "K_D", K_D, default=50.0, units="uM",
            description=(
                "TCR-pMHC antigen affinity (dissociation constant), REQUIRED "
                "scientific input -- the antigen actually being modeled, not "
                "an illustrative value. Also used directly (unconverted) as "
                "the occupancy saturation "
                "constant against pMHC_density -- see the K_D-units caveat in "
                "models/cancer_tcr/signal.py and docs/cancer_tcr_model.md"
            ),
        ),
        "k_on": Parameter(
            name="k_on", value=k_on, units="1/uM/s",
            source=(
                "Order-of-magnitude literature constant (~1e4 /M/s typical 3D-SPR "
                "value; Stone, Chervin & Kranz 2009, Immunology 126:165-176). Fixed "
                "per both source documents' finding that k_on varies little across "
                "peptides -- not intended to be swept per-antigen"
            ),
            source_type=SourceType.LITERATURE,
            description="TCR-pMHC association rate constant, treated as antigen-independent",
        ),
        "k_p": _extension_parameter(
            "k_p", k_p, default=0.3, units="1/s",
            description="Kinetic-proofreading chain forward step rate (fittable; literature-anchored default)",
        ),
        "N": _extension_parameter(
            "N", N, default=3.0, units="dimensionless (integer number of steps)",
            description="Number of kinetic-proofreading steps (fittable; literature-anchored default)",
        ),
        "include_serial_triggering": Parameter(
            name="include_serial_triggering", value=include_serial_triggering,
            units="boolean (0.0=off, 1.0=reserved/not implemented)",
            source="Reserved flag -- see default_tcr_signal_parameters docstring and docs/cancer_tcr_model.md",
            source_type=SourceType.MODEL_EXTENSION,
            description="Whether to apply a serial-triggering/optimal-dwell-time correction (NOT YET IMPLEMENTED)",
        ),
    }
    return ParameterSet("tcr_signal", params)
