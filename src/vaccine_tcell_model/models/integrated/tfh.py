"""Tfh differentiation for the integrated model.

    dTFH/dt = eta*(T-T0)

Identical to Science supplement Eq.7 -- literally reused, not
reimplemented (docs/equations.md Section 3: only the pMHC production
equation and the T-cell equation are this project's extension; Tfh
differentiation is unchanged). Re-exported here (rather than imported
directly from models.science in full_model.py) so models/integrated/ is
self-documenting about which of the four composed components is
genuinely new and which is reused verbatim.
"""

from __future__ import annotations

from vaccine_tcell_model.models.science import ScienceTfhModel, tfh_rhs

__all__ = ["ScienceTfhModel", "tfh_rhs"]
