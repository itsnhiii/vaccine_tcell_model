"""Isolates whether the documented 7-ED/6-ED reordering (Integrated model
vs. Science-only, docs/equations.md Section 7 / test_integrated_validation.py)
is caused by K (the affinity parameter Science's own equation doesn't
have) or by the pMHC accumulation layer (a first-order low-pass filter
sitting between aDC_Ag and T) -- or both.

Science's own T equation has no K at all:
    dT/dt = alpha*aDC_Ag*T/(T+aDC_Ag) - eta*(T-T0)                 (Eq.6)
The integrated model's T equation is:
    dT/dt = alpha*T*pMHC/(K+T+pMHC) - eta*(T-T0)
Setting K->0 removes K's own contribution to the denominator, leaving
the ONLY structural difference from Science's equation as "T+pMHC"
instead of "T+aDC_Ag" -- i.e. whatever divergence survives at K=0 must
be attributable to the pMHC filtering layer alone, not to K.

Run with:  python k_isolation_check.py
Saves:     k_isolation_check.png
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
OUTDIR = Path(__file__).parent

from vaccine_tcell_model.models.integrated import simulate_integrated
from vaccine_tcell_model.models.science import simulate_science
from vaccine_tcell_model.parameters import (
    default_integrated_parameters,
    default_science_parameters,
    integrated_default_initial_conditions,
    science_default_initial_conditions,
)
from vaccine_tcell_model.validation.science import REFERENCE_DOSE_SCHEMES, tfh_at_day

ORDER = ["7-ED", "6-ED", "4-ED", "3-ED", "2-ED", "Bolus"]
DOSE_COUNTS = {"7-ED": 7, "6-ED": 6, "4-ED": 4, "3-ED": 3, "2-ED": 2, "Bolus": 1}
T_EVAL = np.linspace(0, 21, 211)

# -- Science alone (no K, ground truth ordering) -----------------------------
sci_params = default_science_parameters()
sci_ic = science_default_initial_conditions(sci_params)
science_tfh = {}
for name in ORDER:
    result = simulate_science(sci_params, sci_ic, REFERENCE_DOSE_SCHEMES[name], t_end=21.0, t_eval=T_EVAL)
    science_tfh[name] = tfh_at_day(result, 21.0)

# -- Integrated at K=10 (default, matches the documented reordering) --------
int_params_default = default_integrated_parameters(K=10.0)
int_ic = integrated_default_initial_conditions(int_params_default)
integrated_k10_tfh = {}
for name in ORDER:
    result = simulate_integrated(int_params_default, int_ic, REFERENCE_DOSE_SCHEMES[name], t_end=21.0, t_eval=T_EVAL)
    integrated_k10_tfh[name] = tfh_at_day(result, 21.0)

# -- Integrated at K~0 (isolates the pMHC-filter effect from K's effect) ----
int_params_k0 = default_integrated_parameters(K=1e-9)
integrated_k0_tfh = {}
for name in ORDER:
    result = simulate_integrated(int_params_k0, int_ic, REFERENCE_DOSE_SCHEMES[name], t_end=21.0, t_eval=T_EVAL)
    integrated_k0_tfh[name] = tfh_at_day(result, 21.0)


def report(label, tfh):
    ratio = tfh["6-ED"] / tfh["7-ED"]
    order_str = " > ".join(f"{n}({tfh[n]:,.0f})" for n in sorted(ORDER, key=lambda n: -tfh[n]))
    flipped = tfh["6-ED"] > tfh["7-ED"]
    print(f"{label:28s} 6-ED/7-ED = {ratio:.4f}  {'*** FLIPPED ***' if flipped else '(7-ED > 6-ED, as Science)'}")
    print(f"    {order_str}")


print("=" * 78)
print("K-ISOLATION CHECK: is the 7-ED/6-ED reordering caused by K, or by the pMHC filter?")
print("=" * 78)
report("Science (no K, no pMHC)", science_tfh)
report("Integrated, K=10 (default)", integrated_k10_tfh)
report("Integrated, K~0 (isolates filter)", integrated_k0_tfh)

flip_at_k10 = integrated_k10_tfh["6-ED"] > integrated_k10_tfh["7-ED"]
flip_at_k0 = integrated_k0_tfh["6-ED"] > integrated_k0_tfh["7-ED"]
print()
if flip_at_k0:
    print("CONCLUSION: the reordering persists even at K~0. It is caused by the pMHC")
    print("accumulation layer's own low-pass-filter dynamics, NOT by the K parameter.")
else:
    print("CONCLUSION: the reordering disappears at K~0. K itself is a necessary")
    print("ingredient in the reordering, not just the pMHC filter alone.")
if flip_at_k10 and not flip_at_k0:
    print("(K=10 flips it, K~0 does not -- K and the filter interact.)")

# -- Plot ---------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
for ax, (label, tfh) in zip(
    axes, [("Science\n(no K, no pMHC)", science_tfh), ("Integrated, K=10\n(default)", integrated_k10_tfh),
           ("Integrated, K~0\n(isolates pMHC filter)", integrated_k0_tfh)]
):
    colors = ["C3" if n in ("6-ED", "7-ED") else "C0" for n in ORDER]
    ax.bar(ORDER, [tfh[n] for n in ORDER], color=colors)
    ax.set_title(label, fontsize=10)
    ax.set_ylabel("TFH(day 21)")
    ax.tick_params(axis="x", rotation=45)
fig.suptitle("Is the 7-ED/6-ED reordering caused by K or by the pMHC accumulation layer? (6-ED/7-ED bars in red)")
fig.tight_layout()
fig.savefig(OUTDIR / "k_isolation_check.png", dpi=150)
print("\nSaved k_isolation_check.png")
