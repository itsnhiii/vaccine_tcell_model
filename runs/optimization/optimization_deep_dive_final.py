"""Final, corrected plot combining all three analyses with multi-start-
verified numbers throughout (see optimization_deep_dive.py and
optimization_deep_dive_followup.py for how the single-start artifacts
were caught and fixed).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pathlib import Path
OUTDIR = Path(__file__).parent

from vaccine_tcell_model.dosing import DoseSchedule
from vaccine_tcell_model.models.integrated import simulate_integrated
from vaccine_tcell_model.parameters import default_integrated_parameters, integrated_default_initial_conditions

params = default_integrated_parameters(K=10.0)
ic = integrated_default_initial_conditions(params)
t_eval = np.linspace(0, 21, 211)

# --- Multi-start search results (from optimization_deep_dive.py, panel 1) ---
starting_point_results = {
    "uniform": 131_372.8,
    "escalating k=0.5": 165_286.5,
    "escalating k=1.0 (default)": 161_397.1,
    "escalating k=2.0": 120_839.4,
    "reverse-escalating\n(front-loaded)": 113_604.4,
    "bolus-like": 115_526.8,
    "extreme late-load k=3.0": 167_035.8,
    "random (seed 42)": 164_027.7,
    "random (seed 7)": 99_741.4,
}
best_label = "extreme late-load k=3.0"
best_fractions_7dose = np.array([0.053, 0.0848, 0.0981, 0.2509, 0.3229, 0.1903, 0.0])
best_schedule_7dose = DoseSchedule(
    times=[0, 2, 4, 6, 8, 10, 12], antigen_fractions=list(best_fractions_7dose),
    adjuvant_fractions=list(best_fractions_7dose),
)

# --- Multi-start-verified dose-count sweep ---
dose_count_tfh = {1: 14_027.7, 2: 27_565.4, 3: 47_946.1, 4: 76_555.4, 5: 114_208.4,
                   6: 136_229.2, 7: 167_035.8, 8: 171_016.8, 9: 186_926.6, 10: 199_859.2}

# --- Exact winning schedule (day: fraction) behind every dose-count-sweep
# point above, multi-start verified (re-derived independently and confirmed
# to reproduce the TFH values in dose_count_tfh exactly). Dose days follow
# round(linspace(0, 12, N)); fractions are each N's best-of-9-starts result.
dose_count_schedules = {
    1: ([0], [1.0]),
    2: ([0, 12], [0.387, 0.613]),
    3: ([0, 6, 12], [0.158, 0.394, 0.448]),
    4: ([0, 4, 8, 12], [0.159, 0.372, 0.469, 0.0]),
    5: ([0, 3, 6, 9, 12], [0.092, 0.194, 0.372, 0.342, 0.0]),
    6: ([0, 2, 5, 7, 10, 12], [0.035, 0.065, 0.140, 0.228, 0.276, 0.256]),
    7: ([0, 2, 4, 6, 8, 10, 12], [0.053, 0.0848, 0.0981, 0.2509, 0.3229, 0.1903, 0.0]),
    8: ([0, 2, 3, 5, 7, 9, 10, 12], [0.025, 0.030, 0.051, 0.109, 0.176, 0.190, 0.247, 0.172]),
    9: ([0, 2, 3, 4, 6, 8, 9, 10, 12], [0.030, 0.065, 0.0, 0.088, 0.190, 0.199, 0.187, 0.230, 0.011]),
    10: ([0, 1, 3, 4, 5, 7, 8, 9, 11, 12], [0.039, 0.037, 0.081, 0.067, 0.149, 0.167, 0.158, 0.302, 0.0, 0.0]),
}

# --- Multi-start-verified AUC_TFH vs TFH_max comparison ---
auc_fractions = np.array([0.0713, 0.2006, 0.1407, 0.2806, 0.3068, 0.0, 0.0])
auc_schedule = DoseSchedule(
    times=[0, 2, 4, 6, 8, 10, 12], antigen_fractions=list(auc_fractions), adjuvant_fractions=list(auc_fractions)
)
result_auc_sched = simulate_integrated(params, ic, auc_schedule, t_end=21.0, t_eval=t_eval)
result_tfhmax_sched = simulate_integrated(params, ic, best_schedule_7dose, t_end=21.0, t_eval=t_eval)

fig = plt.figure(figsize=(16, 14))
outer = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1.25], hspace=0.5)
gs = outer[0].subgridspec(1, 3)

ax1 = fig.add_subplot(gs[0, :2])
labels = list(starting_point_results.keys())
values = list(starting_point_results.values())
colors = ["C2" if l == best_label else "C0" for l in labels]
ax1.barh(labels, values, color=colors)
ax1.set_xlabel("TFH achieved")
ax1.set_title("1. Multi-start search (N=7, 12-day window): result depends on starting point\n(green = best of 9 starts)")

ax2 = fig.add_subplot(gs[0, 2])
ax2.bar(best_schedule_7dose.antigen_times, best_fractions_7dose, color="C2")
ax2.set_xlabel("day")
ax2.set_ylabel("dose fraction")
ax2.set_title("Best 7-dose schedule found\n(from 'extreme late-load' start)")

gs2 = outer[1].subgridspec(1, 3)

ax3 = fig.add_subplot(gs2[0, 0])
ns = list(dose_count_tfh.keys())
vals = list(dose_count_tfh.values())
ax3.plot(ns, vals, "o-")
ax3.set_xticks(ns)
ax3.set_xlabel("number of doses")
ax3.set_ylabel("best achievable TFH (multi-start)")
ax3.set_title("2. Diminishing returns vs. dose count\n(all points multi-start verified)")

ax4 = fig.add_subplot(gs2[0, 1])
tfh_at_7 = dose_count_tfh[7]
pct = [v / tfh_at_7 * 100 for v in vals]
ax4.plot(ns, pct, "o-", color="C3")
ax4.set_xticks(ns)
ax4.axhline(100, color="gray", linestyle="--", linewidth=1)
for n, p in zip(ns, pct):
    ax4.annotate(f"{p:.0f}%", (n, p), textcoords="offset points", xytext=(0, 6), fontsize=7, ha="center")
ax4.set_xlabel("number of doses")
ax4.set_ylabel("% of 7-dose optimum")
ax4.set_title("2. Same data, normalized to 7-dose result")

ax5 = fig.add_subplot(gs2[0, 2])
ax5.plot(result_auc_sched.time, result_auc_sched.trajectory("TFH"), label="AUC_TFH-optimized")
ax5.plot(result_tfhmax_sched.time, result_tfhmax_sched.trajectory("TFH"), label="TFH_max-optimized")
ax5.set_yscale("log")
ax5.set_xlabel("time (days)")
ax5.set_ylabel("TFH")
ax5.set_title("3. AUC-optimal vs TFH_max-optimal\n(both multi-start verified)")
ax5.legend(fontsize=8)

# Panel row 3: grid of the exact winning schedule (day vs. fraction) behind
# every point in panel 2's dose-count sweep, N=1..10.
gs3 = outer[2].subgridspec(2, 5, hspace=0.6, wspace=0.4)
for i, n in enumerate(range(1, 11)):
    axg = fig.add_subplot(gs3[i // 5, i % 5])
    days, fracs = dose_count_schedules[n]
    axg.bar(days, fracs, width=0.8, color="C0")
    axg.set_xlim(-1, 13)
    axg.set_ylim(0, 1.05)
    axg.xaxis.set_major_locator(MaxNLocator(integer=True))
    axg.set_title(f"N={n}  (TFH={dose_count_tfh[n]:,.0f})", fontsize=8)
    axg.tick_params(labelsize=7)
    if i % 5 == 0:
        axg.set_ylabel("dose fraction", fontsize=7)
    if i // 5 == 1:
        axg.set_xlabel("day", fontsize=7)
fig.tight_layout()
fig.subplots_adjust(left=0.16)
fig.text(0.5, 0.365,
          "4. Exact optimized schedule behind every dose-count-sweep point (panel 2)",
          ha="center", fontsize=10, fontweight="bold")
fig.savefig(OUTDIR / "optimization_deep_dive_final.png", dpi=150)
print("Saved optimization_deep_dive_final.png")
