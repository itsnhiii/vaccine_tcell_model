import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
OUTDIR = Path(__file__).parent
from vaccine_tcell_model.dosing import DoseSchedule
from vaccine_tcell_model.models.integrated import simulate_integrated
from vaccine_tcell_model.parameters import default_integrated_parameters, integrated_default_initial_conditions

# INPUT 1: dose schedule — when doses are given and what fraction each is
schedule = DoseSchedule(times=[0, 10], antigen_fractions=[0.2, 0.8], adjuvant_fractions=[0.2, 0.8])

# INPUT 2: parameters — K (affinity)
params = default_integrated_parameters(K=10.0)
ic = integrated_default_initial_conditions(params)  # T starts at baseline T0, everything else 0

# RUN
result = simulate_integrated(params, ic, schedule, t_end=21.0, t_eval=np.linspace(0, 21, 211))

# PLOT — dosing schedule + trajectories, like Science Fig. 3 / Mayer Fig. 1
fig, axes = plt.subplots(2, 3, figsize=(12, 6))

# Dosing schedule panel
ax_dose = axes.flat[0]
bar_width = 0.35
ag_t, ag_d = np.array(schedule.antigen_times), np.array(schedule.antigen_doses)
adj_t, adj_d = np.array(schedule.adjuvant_times), np.array(schedule.adjuvant_doses)
ax_dose.bar(ag_t - bar_width / 2, ag_d, width=bar_width, label="antigen", color="C0")
ax_dose.bar(adj_t + bar_width / 2, adj_d, width=bar_width, label="adjuvant", color="C1")
ax_dose.set_title("Dosing schedule")
ax_dose.set_xlabel("time (days)")
ax_dose.set_ylabel("dose amount")
ax_dose.legend(fontsize=8)

# Trajectory panels
for ax, state in zip(axes.flat[1:], ["DC", "aDC_Ag", "T", "TFH"]):
    ax.plot(result.time, result.trajectory(state))
    ax.set_title(state); ax.set_yscale("log")
    ax.set_xlabel("time (days)")

axes.flat[-1].axis("off")  # unused 6th slot

fig.tight_layout()
fig.savefig(OUTDIR / "my_run.png")
print("Saved my_run.png")
