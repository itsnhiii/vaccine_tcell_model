"""Three-part dose-schedule optimization report:
  1. Multi-start search (how much can we trust the optimum found earlier?)
  2. Dose-count sweep (diminishing returns -- how many doses is "enough"?)
  3. Metric sweep (does the optimal SHAPE change depending on what you optimize for?)

All local, no cluster/GPU needed (~8s per optimization run, ~17 runs total).

Run with:  python optimization_report.py
Saves:     optimization_report_1_multistart.png
           optimization_report_2_dosecount.png
           optimization_report_3_metrics.png
"""

import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
OUTDIR = Path(__file__).parent

from vaccine_tcell_model.analysis import optimize_dose_schedule
from vaccine_tcell_model.dosing import DoseSchedule
from vaccine_tcell_model.models.integrated import simulate_integrated
from vaccine_tcell_model.parameters import default_integrated_parameters, integrated_default_initial_conditions

params = default_integrated_parameters(K=10.0)
ic = integrated_default_initial_conditions(params)
T_EVAL = np.linspace(0, 21, 106)
TIME_WINDOW = 12.0
T_END = 21.0

t0 = time.time()

# ============================================================
# PART 1: multi-start search for the 7-dose optimum
# ============================================================
print("=" * 70)
print("PART 1: multi-start search (num_doses=7, metric=TFH_max)")
print("=" * 70)

N = 7
starting_points = {
    "uniform": np.full(N, 1.0 / N),
    "exp escalation k=1 (default)": np.exp(np.arange(N) * 1.0) / np.sum(np.exp(np.arange(N) * 1.0)),
    "exp escalation k=0.5": np.exp(np.arange(N) * 0.5) / np.sum(np.exp(np.arange(N) * 0.5)),
    "exp escalation k=2": np.exp(np.arange(N) * 2.0) / np.sum(np.exp(np.arange(N) * 2.0)),
    "front-loaded (reverse escalation)": np.exp(np.arange(N)[::-1] * 1.0) / np.sum(np.exp(np.arange(N) * 1.0)),
    "bell-shaped (peak in middle)": None,  # filled below
    "random (seed=0)": None,  # filled below
}
mid = np.exp(-((np.arange(N) - N // 2) ** 2) / 4.0)
starting_points["bell-shaped (peak in middle)"] = mid / mid.sum()
rng = np.random.default_rng(0)
rand_raw = rng.uniform(0.1, 1.0, size=N)
starting_points["random (seed=0)"] = rand_raw / rand_raw.sum()

multistart_results = {}
for label, x0 in starting_points.items():
    result = optimize_dose_schedule(
        "integrated", num_doses=N, time_window=TIME_WINDOW, metric="TFH_max",
        t_end=T_END, t_eval=T_EVAL, initial_fractions=x0,
    )
    multistart_results[label] = result
    print(f"  {label:35s} -> TFH_max={result.objective_value:12.1f}  ({result.n_iterations} iters, success={result.success})")

best_label = max(multistart_results, key=lambda k: multistart_results[k].objective_value)
best_result = multistart_results[best_label]
worst_val = min(r.objective_value for r in multistart_results.values())
best_val = best_result.objective_value
print(f"\n  BEST: {best_label!r} -> TFH_max={best_val:.1f}")
print(f"  Spread across starts: {worst_val:.1f} to {best_val:.1f} ({(best_val/worst_val - 1)*100:.1f}% range)")

fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(12, 4.5))
labels = list(multistart_results.keys())
values = [multistart_results[l].objective_value for l in labels]
colors = ["C2" if l == best_label else "C0" for l in labels]
ax1a.barh(labels, values, color=colors)
ax1a.set_xlabel("achieved TFH_max")
ax1a.set_title("Multi-start: objective reached from each starting point")

for label in labels:
    r = multistart_results[label]
    ax1b.plot(r.schedule.antigen_times, r.fractions, "o-",
              label=label, linewidth=3 if label == best_label else 1, alpha=1.0 if label == best_label else 0.5)
ax1b.set_xlabel("day")
ax1b.set_ylabel("optimized dose fraction")
ax1b.set_title("Optimized fraction pattern found from each start")
ax1b.legend(fontsize=7)
fig1.tight_layout()
fig1.savefig(OUTDIR / "optimization_report_1_multistart.png", dpi=150)
print("Saved optimization_report_1_multistart.png")

# ============================================================
# PART 2: dose-count sweep (diminishing returns)
# ============================================================
print("\n" + "=" * 70)
print("PART 2: dose-count sweep (metric=TFH_max, best-of-2-starts each)")
print("=" * 70)

dose_counts = [1, 2, 3, 4, 5, 6, 7, 10]
dosecount_results = {}
for n in dose_counts:
    candidates = []
    for x0 in [np.full(n, 1.0 / n), np.exp(np.arange(n) * 1.0) / np.sum(np.exp(np.arange(n) * 1.0))]:
        r = optimize_dose_schedule(
            "integrated", num_doses=n, time_window=TIME_WINDOW, metric="TFH_max",
            t_end=T_END, t_eval=T_EVAL, initial_fractions=x0,
        )
        candidates.append(r)
    best = max(candidates, key=lambda r: r.objective_value)
    dosecount_results[n] = best
    print(f"  num_doses={n:2d}  best TFH_max={best.objective_value:12.1f}")

fig2, ax2 = plt.subplots(figsize=(7, 5))
ns = list(dosecount_results.keys())
vals = [dosecount_results[n].objective_value for n in ns]
ax2.plot(ns, vals, "o-", color="C0")
max_val = max(vals)
ax2.axhline(max_val, color="gray", linestyle="--", linewidth=1, label=f"max achieved ({max_val:.0f})")
ax2.axhline(max_val * 0.9, color="gray", linestyle=":", linewidth=1, label="90% of max")
ax2.set_xticks(ns)
ax2.set_xlabel("number of doses")
ax2.set_ylabel("optimized TFH_max")
ax2.set_title("Diminishing returns: optimal TFH_max vs. number of doses")
ax2.legend(fontsize=8)
fig2.tight_layout()
fig2.savefig(OUTDIR / "optimization_report_2_dosecount.png", dpi=150)
print("Saved optimization_report_2_dosecount.png")

# ============================================================
# PART 3: does the optimal SHAPE change with the objective metric?
# ============================================================
print("\n" + "=" * 70)
print("PART 3: metric sweep (num_doses=7) -- does optimizing for a")
print("different outcome change the SHAPE of the optimal schedule?")
print("=" * 70)

metrics = ["TFH_max", "AUC_TFH", "T_max"]
metric_results = {}
for metric in metrics:
    # reuse the same multi-start diversity for fairness
    candidates = [
        optimize_dose_schedule(
            "integrated", num_doses=N, time_window=TIME_WINDOW, metric=metric,
            t_end=T_END, t_eval=T_EVAL, initial_fractions=x0,
        )
        for x0 in [starting_points["uniform"], starting_points["exp escalation k=1 (default)"]]
    ]
    best = max(candidates, key=lambda r: r.objective_value)
    metric_results[metric] = best
    print(f"  optimizing for {metric:10s} -> value={best.objective_value:12.1f}  fractions={np.round(best.fractions, 3).tolist()}")

fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 4.5))
for metric, r in metric_results.items():
    ax3a.plot(r.schedule.antigen_times, r.fractions, "o-", label=f"optimized for {metric}")
ax3a.set_xlabel("day")
ax3a.set_ylabel("dose fraction")
ax3a.set_title("Optimal schedule SHAPE depends on the chosen objective")
ax3a.legend(fontsize=8)

for metric, r in metric_results.items():
    result = simulate_integrated(params, ic, r.schedule, t_end=T_END, t_eval=T_EVAL)
    ax3b.plot(result.time, result.trajectory("TFH"), label=f"optimized for {metric}")
ax3b.set_xlabel("time (days)")
ax3b.set_ylabel("TFH")
ax3b.set_yscale("log")
ax3b.set_title("Resulting TFH trajectory")
ax3b.legend(fontsize=8)
fig3.tight_layout()
fig3.savefig(OUTDIR / "optimization_report_3_metrics.png", dpi=150)
print("Saved optimization_report_3_metrics.png")

print(f"\nTotal runtime: {time.time() - t0:.1f}s")
