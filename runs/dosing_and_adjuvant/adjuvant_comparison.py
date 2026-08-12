"""Compares full vs. reduced vs. zero adjuvant dosing, same antigen schedule throughout.

Run with:  python adjuvant_comparison.py
Saves:     adjuvant_comparison.png
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
OUTDIR = Path(__file__).parent

from vaccine_tcell_model.dosing import DoseSchedule
from vaccine_tcell_model.models.integrated import simulate_integrated
from vaccine_tcell_model.parameters import default_integrated_parameters, integrated_default_initial_conditions

params = default_integrated_parameters(K=10.0)
ic = integrated_default_initial_conditions(params)
t_eval = np.linspace(0, 21, 211)

conditions = {
    "Full adjuvant (1.0)": 1.0,
    "Reduced adjuvant (0.05)": 0.05,
    "Zero adjuvant (0.0)": 0.0,
}

results = {}
for label, total_adj in conditions.items():
    schedule = DoseSchedule(
        times=[0, 10],
        antigen_fractions=[0.2, 0.8],
        adjuvant_fractions=[0.2, 0.8],
        total_adjuvant_dose=total_adj,
    )
    results[label] = simulate_integrated(params, ic, schedule, t_end=21.0, t_eval=t_eval)

fig, axes = plt.subplots(2, 2, figsize=(9, 7))
for ax, state in zip(axes.flat, ["DC", "aDC_Ag", "T", "TFH"]):
    for label, result in results.items():
        ax.plot(result.time, result.trajectory(state) + 1e-6, label=label)  # +1e-6 so log-scale zeros still show
    ax.set_title(state)
    ax.set_xlabel("time (days)")
    ax.set_yscale("log")

axes.flat[0].legend(fontsize=8)
fig.suptitle("Effect of adjuvant dose (antigen schedule unchanged: 20% day 0, 80% day 10)")
fig.tight_layout()
fig.savefig(OUTDIR / "adjuvant_comparison.png", dpi=150)
print("Saved adjuvant_comparison.png")

for label, result in results.items():
    print(f"{label:26s} max DC={result.trajectory('DC').max():.3e}  final TFH={result.trajectory('TFH')[-1]:.3e}")
