"""Compares several dosing SCHEDULES that all deliver the SAME total
antigen + adjuvant dose (1.0 each), to isolate the effect of timing/
spread from the effect of total dose amount.

Run with:  python dosing_schedule_comparison_run.py
Saves:     dosing_schedule_comparison_run.png
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
OUTDIR = Path(__file__).parent

from vaccine_tcell_model.analysis import sweep_dose_schedules
from vaccine_tcell_model.dosing import DoseSchedule
from vaccine_tcell_model.models.integrated import simulate_integrated
from vaccine_tcell_model.parameters import default_integrated_parameters, integrated_default_initial_conditions

params = default_integrated_parameters(K=10.0)
ic = integrated_default_initial_conditions(params)
t_eval = np.linspace(0, 21, 211)

# All schedules deliver total_antigen_dose=1.0, total_adjuvant_dose=1.0
# (the defaults) -- same total vaccine, different timing/spread.
schedules = {
    "Bolus (all on day 0)":
        DoseSchedule.bolus(),
    "2-dose 20/80 (d0, d7)":
        DoseSchedule(times=[0, 7], antigen_fractions=[0.2, 0.8], adjuvant_fractions=[0.2, 0.8]),
    "2-dose 50/50 (d0, d7)":
        DoseSchedule(times=[0, 7], antigen_fractions=[0.5, 0.5], adjuvant_fractions=[0.5, 0.5]),
    "7-dose equal (d0-d12)":
        DoseSchedule.equal_doses(times=[0, 2, 4, 6, 8, 10, 12]),
    "7-dose escalating (d0-d12)":
        DoseSchedule.from_exponential_escalation(numshot=7, k=1.0, duration=12),
}

# Sanity check: confirm every schedule really does deliver the same total.
for label, s in schedules.items():
    assert np.isclose(s.total_antigen_dose, 1.0) and np.isclose(s.total_adjuvant_dose, 1.0), label

results = {label: simulate_integrated(params, ic, s, t_end=21.0, t_eval=t_eval) for label, s in schedules.items()}

# --- Figure: dosing patterns (top, spanning full width) + trajectories (below) ---
colors = {label: f"C{i}" for i, label in enumerate(schedules)}
fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1])

ax_dose = fig.add_subplot(gs[0, :])
width = 0.15
for i, (label, s) in enumerate(schedules.items()):
    offset = (i - len(schedules) / 2) * width
    ax_dose.bar(np.array(s.antigen_times) + offset, s.antigen_doses, width=width, label=label, color=colors[label])
ax_dose.set_xlabel("day")
ax_dose.set_ylabel("antigen dose fraction")
ax_dose.set_title("Same total dose (1.0), different timing")
ax_dose.legend(fontsize=8, ncol=2)

# Trajectories for Ag, DC, aDC_Ag, pMHC, T, TFH -- two rows of three panels.
trajectory_axes = [fig.add_subplot(gs[1, c]) for c in range(3)] + [fig.add_subplot(gs[2, c]) for c in range(3)]
for ax, state in zip(trajectory_axes, ["Ag", "DC", "aDC_Ag", "pMHC", "T", "TFH"]):
    for label, result in results.items():
        ax.plot(result.time, result.trajectory(state), label=label, color=colors[label])
    ax.set_title(state)
    ax.set_xlabel("time (days)")
    ax.set_yscale("log")

trajectory_axes[0].legend(fontsize=7)
fig.tight_layout()
fig.savefig(OUTDIR / "dosing_schedule_comparison_run.png", dpi=150)
print("Saved dosing_schedule_comparison_run.png")

# --- Tidy metrics table (same total dose, different outcomes) ---
df = sweep_dose_schedules(model="integrated", schedules=schedules, t_end=21.0, t_eval=t_eval)
print()
print(df[["schedule", "T_max", "TFH_max", "TFH_final", "AUC_TFH", "fold_expansion"]].to_string(index=False))
