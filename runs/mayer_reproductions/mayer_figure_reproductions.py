"""Reproduces Mayer et al. 2019 Fig. 1C and Fig. 2A/2B -- the two published
figures we validated the model's qualitative behavior against but had
never actually plotted in the same layout as the paper.

Run with:  python mayer_figure_reproductions.py
Saves:     mayer_fig1c_reproduction.png
           mayer_fig2ab_reproduction.png
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
OUTDIR = Path(__file__).parent

from vaccine_tcell_model.models.mayer import simulate_mayer
from vaccine_tcell_model.parameters import MAYER_FIG2_LIGAND_K, mayer_parameters_fig1_cd4, mayer_parameters_fig2_cd8

# ============================================================
# Fig. 1C: T cell & pMHC number vs. time, T(0)=300 and T(0)=30,000
# Paper's fitted params (Fig.1 legend): alpha=1.5, mu=1.2, delta=0.22,
# K~0, C(0)=10^6.7
# ============================================================
params = mayer_parameters_fig1_cd4()
C0 = 10.0 ** 6.7
t_eval = np.linspace(0, 10, 201)

fig1, ax1 = plt.subplots(figsize=(6, 5))

# pMHC doesn't depend on T0 -- one curve, like the paper's single dotted line.
pmhc_result = simulate_mayer(params, {"T": 300.0, "C": C0}, t_end=10.0, t_eval=t_eval)
ax1.plot(pmhc_result.time, pmhc_result.trajectory("C"), "r:", linewidth=2, label="pMHCs")

for T0, style in [(300.0, "b-"), (30000.0, "c-")]:
    result = simulate_mayer(params, {"T": T0, "C": C0}, t_end=10.0, t_eval=t_eval)
    ax1.plot(result.time, result.trajectory("T"), style, linewidth=2, label=f"T cells, T(0)={T0:g}")

ax1.set_yscale("log")
ax1.set_ylim(1e2, 1e7)
ax1.set_xlabel("Time in days")
ax1.set_ylabel("Number")
ax1.set_title("Reproduction of Mayer et al. 2019 Fig. 1C")
ax1.legend(fontsize=9)
fig1.tight_layout()
fig1.savefig(OUTDIR / "mayer_fig1c_reproduction.png", dpi=150)
print("Saved mayer_fig1c_reproduction.png")

# ============================================================
# Fig. 2A: fold expansion T(7)/T(4) vs. relative EC50 (affinity), 6 ligands
# Fig. 2B: T cell number vs time (day 4-8) per ligand
# Paper's fitted params (Fig.2 legend): alpha=2.47, mu=3.1, delta=0.23,
# C(4)=10^3.22, T(4)=0.92, K = ligand's relative EC50
# ============================================================
C4 = 10.0 ** 3.22
T4 = 0.92
t_eval_2 = np.linspace(4, 8, 161)

ligand_order = ["SIINFEKL", "SAINFEKL", "SIYNFEKL", "SIIQFEKL", "SIITFEKL", "SIIVFEKL"]

fold_expansions = {}
trajectories = {}
for ligand in ligand_order:
    K = MAYER_FIG2_LIGAND_K[ligand]
    params_2 = mayer_parameters_fig2_cd8(K=K)
    result = simulate_mayer(params_2, {"T": T4, "C": C4}, t_end=8.0, t_eval=t_eval_2)
    trajectories[ligand] = result
    t_at_7 = result.trajectory("T")[np.argmin(np.abs(result.time - 7.0))]
    fold_expansions[ligand] = t_at_7 / T4

fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(11, 4.5))

rel_ec50 = [MAYER_FIG2_LIGAND_K[lig] for lig in ligand_order]
folds = [fold_expansions[lig] for lig in ligand_order]
ax2a.scatter(rel_ec50, folds, s=50, zorder=3)
for lig, x, y in zip(ligand_order, rel_ec50, folds):
    ax2a.annotate(lig, (x, y), fontsize=7, xytext=(4, 4), textcoords="offset points")
ax2a.set_xscale("log")
ax2a.set_yscale("log")
ax2a.set_xlabel("rel. pMHC concentration for half max. response")
ax2a.set_ylabel("Fold expansion")
ax2a.set_title("Reproduction of Fig. 2A")

for ligand in ligand_order:
    result = trajectories[ligand]
    ax2b.plot(result.time, result.trajectory("T"), label=ligand)
ax2b.set_yscale("log")
ax2b.set_xlabel("Time in days")
ax2b.set_ylabel("T cell number")
ax2b.set_title("Reproduction of Fig. 2B")
ax2b.legend(fontsize=7)

fig2.tight_layout()
fig2.savefig(OUTDIR / "mayer_fig2ab_reproduction.png", dpi=150)
print("Saved mayer_fig2ab_reproduction.png")

print("\nFold expansion T(7)/T(4) per ligand (should decrease as rel. EC50 increases):")
for lig in ligand_order:
    print(f"  {lig:10s} K={MAYER_FIG2_LIGAND_K[lig]:6.1f}  fold={fold_expansions[lig]:.2f}")
