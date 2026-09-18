"""Typed state containers and the state-name <-> array-index mapping.

The numerical solver operates on plain numpy arrays (master-spec rule
5: "the numerical solver may internally use numpy arrays"). StateSpec is
the single source of truth mapping a named biological quantity (e.g.
"aDC_Ag") to its index in that array, so no model or solver ever
hard-codes "index 4 is aDC_Ag".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StateSpec:
    """The ordered list of state-variable names for a model."""

    names: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(set(self.names)) != len(self.names):
            raise ValueError(f"Duplicate state names in StateSpec: {self.names}")
        if len(self.names) == 0:
            raise ValueError("StateSpec must have at least one state name")

    @property
    def size(self) -> int:
        return len(self.names)

    def index(self, name: str) -> int:
        try:
            return self.names.index(name)
        except ValueError as exc:
            raise KeyError(
                f"Unknown state name {name!r}. Known states: {self.names}"
            ) from exc

    def to_array(self, values: dict[str, float]) -> np.ndarray:
        """Build an ordered numpy array from a {name: value} dict.

        Every name in this spec must be present in `values`, and `values`
        must not contain names outside this spec -- both directions raise,
        so a typo in an initial-condition dict fails loudly instead of
        silently defaulting a state to zero.
        """
        missing = set(self.names) - set(values)
        if missing:
            raise ValueError(f"Missing values for states: {sorted(missing)}")
        extra = set(values) - set(self.names)
        if extra:
            raise ValueError(f"Unknown state names not in this StateSpec: {sorted(extra)}")
        return np.array([values[n] for n in self.names], dtype=float)

    def to_dict(self, arr: np.ndarray) -> dict[str, float]:
        """Inverse of to_array: a flat array back to a {name: value} dict."""
        if len(arr) != self.size:
            raise ValueError(
                f"Array length {len(arr)} does not match StateSpec size {self.size}"
            )
        return {n: float(v) for n, v in zip(self.names, arr)}

    def __repr__(self) -> str:
        return f"StateSpec({self.names})"


# ---------------------------------------------------------------------------
# Published state definitions (master spec Section 5 / docs/equations.md)
# ---------------------------------------------------------------------------

SCIENCE_STATES = StateSpec(names=("Ag", "Adj", "TC", "DC", "aDC_Ag", "T", "TFH"))
"""Bhagchandani et al. 2024 T-cell priming model (supplement Eq. 1-7).

No pMHC state and no K (affinity) parameter -- both are absent from the
published Science equations. See docs/equations.md Flag 1.
"""

MAYER_STATES = StateSpec(names=("T", "C"))
"""Mayer et al. 2019 (main text Eq. 1-2): T = antigen-specific T cells,
C = cognate presented antigen / pMHC."""

INTEGRATED_STATES = StateSpec(
    names=("Ag", "Adj", "TC", "DC", "aDC_Ag", "pMHC", "T", "TFH")
)
"""This project's integrated extension (master spec Section 8).

Identical to SCIENCE_STATES except pMHC is inserted between aDC_Ag and
T. The pMHC production/decay equation and the Mayer-shaped T-cell
denominator that consumes it are a model_extension -- not published
verbatim by either source paper. See docs/equations.md Section 3.
"""

CANCER_TCR_STATES = StateSpec(names=("Ag", "Adj", "TC", "DC", "aDC_Ag", "pMHC_density", "T"))
"""This project's cancer/TCR-signaling extension (docs/cancer_tcr_model.md).

Reuses Science's Ag/Adj/TC/DC/aDC_Ag upstream states unchanged (they are
generic vaccine-adjuvant/DC pharmacokinetics, not virus-specific). Drops
TFH entirely -- there is no humoral/germinal-center output in a cancer
CD8-killing context. Replaces the bulk `pMHC` pool with `pMHC_density`,
an intensive (per-DC) quantity, and the T-cell equation loses the
`-eta*(T-T0)` Tfh-differentiation sink (see models/cancer_tcr/tcell.py).
The TCR-signal itself (S_contact, S_pop) is deliberately NOT a state
here -- it is a closed-form post-processing function of these states,
not an ODE quantity (see models/cancer_tcr/signal.py).
"""
