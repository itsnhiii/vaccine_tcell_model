"""Three follow-up optimization analyses:

  1. Multi-start search for the 7-dose/12-day schedule (is the earlier
     result actually a good optimum, or an artifact of one starting point?)
  2. Sweep number of doses 1-10 -- how much of the benefit of "many doses"
     is captured by just 3 or 5?
  3. Optimize for AUC_TFH (total exposure over time) instead of TFH_max
     (final/peak level) -- does the "best" schedule change?

Run with:  python optimization_deep_dive.py
Saves:     optimization_deep_dive.png
"""

import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
OUTDIR = Path(__file__).parent

from vaccine_tcell_model.analysis import optimize_dose_schedule
from vaccine_tcell_model.models.integrated import simulate_integrated
from vaccine_tcell_model.parameters import default_integrated_parameters, integrated_default_initial_conditions

params = default_integrated_parameters(K=10.0)
ic = integrated_default_initial_conditions(params)
t_eval = np.linspace(0, 21, 106)
TIME_WINDOW = 12.0
T_END = 21.0

t0_total = time.time()

# =============================================================================
# 1. MULTI-START: is the single-start (exponential-escalation seed) result
#    from before actually a good optimum?
# =============================================================================
print("=" * 70)
print("1. MULTI-START SEARCH (num_doses=7, time_window=12, metric=TFH_max)")
print("=" * 70)

N = 7
rng42 = np.random.default_rng(42)
rng7 = np.random.default_rng(7)


def _norm(w):
    return w / w.sum()


starting_points = {
    "uniform": np.full(N, 1 / N),
    "escalating k=0.5": _norm(np.exp(np.arange(N) * 0.5)),
    "escalating k=1.0 (default)": _norm(np.exp(np.arange(N) * 1.0)),
    "escalating k=2.0": _norm(np.exp(np.arange(N) * 2.0)),
    "reverse-escalating (front-loaded) k=1.0": _norm(np.exp(np.arange(N)[::-1] * 1.0)),
    "bolus-like (90% on dose 1)": _norm(np.array([0.9, *([0.1 / (N - 1)] * (N - 1))])),
    "extreme late-load k=3.0": _norm(np.exp(np.arange(N) * 3.0)),
    "random (seed 42)": rng42.dirichlet(np.ones(N)),
    "random (seed 7)": rng7.dirichlet(np.ones(N)),
}

multi_start_results = {}
t0 = time.time()
for label, x0 in starting_points.items():
    result = optimize_dose_schedule(
        "integrated", num_doses=N, time_window=TIME_WINDOW, metric="TFH_max",
        t_end=T_END, t_eval=t_eval, initial_fractions=x0,
    )
    multi_start_results[label] = result
    print(f"  {label:42s} -> TFH = {result.objective_value:12,.1f}  ({result.n_iterations} iters, success={result.success})")

best_label = max(multi_start_results, key=lambda k: multi_start_results[k].objective_value)
best_multistart = multi_start_results[best_label]
worst_label = min(multi_start_results, key=lambda k: multi_start_results[k].objective_value)
print(f"\n  BEST : {best_label!r} -> TFH = {best_multistart.objective_value:,.1f}")
print(f"  WORST: {worst_label!r} -> TFH = {multi_start_results[worst_label].objective_value:,.1f}")
print(f"  spread: {(best_multistart.objective_value / multi_start_results[worst_label].objective_value - 1) * 100:.1f}% "
      f"difference between best and worst starting point")
print(f"  time: {time.time() - t0:.1f}s")

# =============================================================================
# 2. NUMBER-OF-DOSES SWEEP: diminishing returns?
# =============================================================================
print()
print("=" * 70)
print("2. NUMBER-OF-DOSES SWEEP (time_window=12, metric=TFH_max)")
print("=" * 70)

dose_counts = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
dose_count_results = {}
t0 = time.time()
for n in dose_counts:
    result = optimize_dose_schedule(
        "integrated", num_doses=n, time_window=TIME_WINDOW, metric="TFH_max",
        t_end=T_END, t_eval=t_eval,
    )
    dose_count_results[n] = result
    print(f"  num_doses={n:2d}  -> best TFH = {result.objective_value:12,.1f}")
print(f"  time: {time.time() - t0:.1f}s")

best_overall = dose_count_results[max(dose_count_results, key=lambda n: dose_count_results[n].objective_value)]
tfh_at_7 = dose_count_results[7].objective_value
for n in dose_counts:
    pct_of_7dose = dose_count_results[n].objective_value / tfh_at_7 * 100
    print(f"    num_doses={n:2d}: {pct_of_7dose:5.1f}% of the 7-dose result")

# =============================================================================
# 3. ALTERNATE OBJECTIVE: AUC_TFH instead of TFH_max
# =============================================================================
print()
print("=" * 70)
print("3. ALTERNATE OBJECTIVE: optimize AUC_TFH instead of TFH_max")
print("=" * 70)

t0 = time.time()
auc_opt = optimize_dose_schedule(
    "integrated", num_doses=N, time_window=TIME_WINDOW, metric="AUC_TFH",
    t_end=T_END, t_eval=t_eval,
)
print(f"  AUC_TFH-optimized fractions: {np.round(auc_opt.fractions, 4)}")
print(f"  TFH_max-optimized fractions: {np.round(best_multistart.fractions, 4)}")

# Cross-evaluate: how does each schedule score on the OTHER metric?
result_auc_sched = simulate_integrated(params, ic, auc_opt.schedule, t_end=T_END, t_eval=t_eval)
result_tfhmax_sched = simulate_integrated(params, ic, best_multistart.schedule, t_end=T_END, t_eval=t_eval)

from vaccine_tcell_model.analysis import auc as auc_metric, peak_value

print(f"\n  {'schedule':22s} {'TFH_max':>12s} {'AUC_TFH':>14s}")
print(f"  {'AUC_TFH-optimized':22s} {peak_value(result_auc_sched, 'TFH'):12,.1f} {auc_metric(result_auc_sched, 'TFH'):14,.1f}")
print(f"  {'TFH_max-optimized':22s} {peak_value(result_tfhmax_sched, 'TFH'):12,.1f} {auc_metric(result_tfhmax_sched, 'TFH'):14,.1f}")
print(f"  time: {time.time() - t0:.1f}s")

print(f"\nTOTAL runtime: {time.time() - t0_total:.1f}s (all local CPU, no cluster/GPU needed)")

# =============================================================================
# PLOTS
# =============================================================================
fig = plt.figure(figsize=(15, 9))
gs = fig.add_gridspec(2, 3)

# Panel 1: multi-start spread (bar chart of objective value per starting point)
ax1 = fig.add_subplot(gs[0, :2])
labels = list(multi_start_results.keys())
values = [multi_start_results[l].objective_value for l in labels]
bar_colors = ["C2" if l == best_label else "C0" for l in labels]
ax1.barh(labels, values, color=bar_colors)
ax1.set_xlabel("TFH achieved")
ax1.set_title("1. Multi-start search: final TFH by starting point (green = best)")

# Panel 2: best fractions found (multi-start winner) vs day
ax2 = fig.add_subplot(gs[0, 2])
ax2.bar(best_multistart.schedule.antigen_times, best_multistart.fractions, color="C2")
ax2.set_xlabel("day")
ax2.set_ylabel("dose fraction")
ax2.set_title(f"Best schedule found\n({best_label})")

# Panel 3: dose-count sweep
ax3 = fig.add_subplot(gs[1, 0])
ns = list(dose_count_results.keys())
vals = [dose_count_results[n].objective_value for n in ns]
ax3.plot(ns, vals, "o-")
ax3.set_xticks(ns)
ax3.set_xlabel("number of doses")
ax3.set_ylabel("best achievable TFH")
ax3.set_title("2. Diminishing returns vs. dose count")

# Panel 4: dose-count sweep as % of 7-dose result
ax4 = fig.add_subplot(gs[1, 1])
pct = [dose_count_results[n].objective_value / tfh_at_7 * 100 for n in ns]
ax4.plot(ns, pct, "o-", color="C3")
ax4.axhline(100, color="gray", linestyle="--", linewidth=1)
ax4.set_xticks(ns)
ax4.set_xlabel("number of doses")
ax4.set_ylabel("% of 7-dose optimum")
ax4.set_title("2. Same data, normalized to 7-dose result")

# Panel 5: AUC-optimal vs TFH_max-optimal schedules + TFH trajectories
ax5 = fig.add_subplot(gs[1, 2])
ax5.plot(result_auc_sched.time, result_auc_sched.trajectory("TFH"), label="AUC_TFH-optimized")
ax5.plot(result_tfhmax_sched.time, result_tfhmax_sched.trajectory("TFH"), label="TFH_max-optimized")
ax5.set_yscale("log")
ax5.set_xlabel("time (days)")
ax5.set_ylabel("TFH")
ax5.set_title("3. AUC-optimal vs TFH_max-optimal")
ax5.legend(fontsize=8)

fig.tight_layout()
fig.savefig(OUTDIR / "optimization_deep_dive.png", dpi=150)
print("\nSaved optimization_deep_dive.png")
