"""Finds the optimal 7-dose schedule (same total dose, same 12-day window)
to maximize final/peak TFH, and compares it against the hand-picked
schedules from the earlier comparison run.

Run with:  python optimal_dosing_run.py
Saves:     optimal_dosing_run.png
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
OUTDIR = Path(__file__).parent

from vaccine_tcell_model.analysis import optimize_dose_schedule, sweep_dose_schedules
from vaccine_tcell_model.dosing import DoseSchedule
from vaccine_tcell_model.models.integrated import simulate_integrated
from vaccine_tcell_model.parameters import default_integrated_parameters, integrated_default_initial_conditions

params = default_integrated_parameters(K=10.0)
ic = integrated_default_initial_conditions(params)
t_eval = np.linspace(0, 21, 211)

# SOLVE for the best 7-dose schedule: dose TIMES fixed at evenly-spaced
# days 0-12 (same window as "7-dose equal/escalating" above), dose
# FRACTIONS free to vary (still summing to 1 -- same total vaccine).
opt_result = optimize_dose_schedule(
    "integrated",
    num_doses=7,
    time_window=12.0,
    metric="TFH_max",   # TFH never decreases in this model, so TFH_max == TFH_final
    t_end=21.0,
    t_eval=t_eval,
)

schedules = {
    "Bolus": DoseSchedule.bolus(),
    "2-dose 20/80": DoseSchedule(times=[0, 7], antigen_fractions=[0.2, 0.8], adjuvant_fractions=[0.2, 0.8]),
    "2-dose 50/50": DoseSchedule(times=[0, 7], antigen_fractions=[0.5, 0.5], adjuvant_fractions=[0.5, 0.5]),
    "7-dose equal": DoseSchedule.equal_doses(times=[0, 2, 4, 6, 8, 10, 12]),
    "7-dose escalating": DoseSchedule.from_exponential_escalation(numshot=7, k=1.0, duration=12),
    "OPTIMIZED (7-dose)": opt_result.schedule,
}

df = sweep_dose_schedules(model="integrated", schedules=schedules, t_end=21.0, t_eval=t_eval)
print(df[["schedule", "T_max", "TFH_max", "AUC_TFH", "fold_expansion"]].to_string(index=False))

print("\nOptimized dose fractions:")
for t, f in zip(opt_result.schedule.antigen_times, opt_result.fractions):
    print(f"  day {t:5.1f}: {f * 100:5.1f}%")
print(f"\nConstraints satisfied: {opt_result.constraints_satisfied}")
print(f"Optimizer success: {opt_result.success} ({opt_result.n_iterations} iterations)")

# --- Plot ---
results = {label: simulate_integrated(params, ic, s, t_end=21.0, t_eval=t_eval) for label, s in schedules.items()}
colors = {label: f"C{i}" for i, label in enumerate(schedules)}

fig = plt.figure(figsize=(13, 7))
gs = fig.add_gridspec(2, 3)

ax_dose = fig.add_subplot(gs[0, :])
width = 0.13
for i, (label, s) in enumerate(schedules.items()):
    offset = (i - len(schedules) / 2) * width
    ax_dose.bar(np.array(s.antigen_times) + offset, s.antigen_doses, width=width, label=label, color=colors[label])
ax_dose.set_xlabel("day")
ax_dose.set_ylabel("antigen dose fraction")
ax_dose.set_title("Same total dose, different timing -- including the SOLVED optimum")
ax_dose.legend(fontsize=8, ncol=2)

for ax, state in zip([fig.add_subplot(gs[1, c]) for c in range(3)], ["DC", "T", "TFH"]):
    for label, result in results.items():
        lw = 3 if label.startswith("OPTIMIZED") else 1.5
        ax.plot(result.time, result.trajectory(state), label=label, color=colors[label], linewidth=lw)
    ax.set_title(state)
    ax.set_xlabel("time (days)")
    ax.set_yscale("log")
fig.axes[1].legend(fontsize=7)

fig.tight_layout()
fig.savefig(OUTDIR / "optimal_dosing_run.png", dpi=150)
print("\nSaved optimal_dosing_run.png")
