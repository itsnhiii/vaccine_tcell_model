"""Follow-up: two results from optimization_deep_dive.py looked like
single-start artifacts rather than real findings:
  - AUC_TFH-optimized schedule scored WORSE than the TFH_max-optimized
    schedule on its own metric (AUC_TFH) -- shouldn't happen if the
    AUC_TFH optimization actually found a good optimum.
  - num_doses=6 scored worse than num_doses=5, breaking the otherwise
    monotonic trend.

Both were single-start runs. Re-check both with the same multi-start
approach used for the headline 7-dose/TFH_max result.

Run with:  python optimization_deep_dive_followup.py
"""

import time

import numpy as np

from vaccine_tcell_model.analysis import auc as auc_metric, optimize_dose_schedule, peak_value
from vaccine_tcell_model.models.integrated import simulate_integrated
from vaccine_tcell_model.parameters import default_integrated_parameters, integrated_default_initial_conditions

params = default_integrated_parameters(K=10.0)
ic = integrated_default_initial_conditions(params)
t_eval = np.linspace(0, 21, 106)
TIME_WINDOW = 12.0
T_END = 21.0

t0_total = time.time()


def starting_points(n):
    rng42 = np.random.default_rng(42)
    rng7 = np.random.default_rng(7)

    def norm(w):
        return w / w.sum()

    return {
        "uniform": np.full(n, 1 / n),
        "escalating k=0.5": norm(np.exp(np.arange(n) * 0.5)),
        "escalating k=1.0": norm(np.exp(np.arange(n) * 1.0)),
        "escalating k=2.0": norm(np.exp(np.arange(n) * 2.0)),
        "escalating k=3.0": norm(np.exp(np.arange(n) * 3.0)),
        "reverse-escalating k=1.0": norm(np.exp(np.arange(n)[::-1] * 1.0)),
        "bolus-like": norm(np.array([0.9, *([0.1 / max(n - 1, 1)] * (n - 1))])),
        "random (seed 42)": rng42.dirichlet(np.ones(n)),
        "random (seed 7)": rng7.dirichlet(np.ones(n)),
    }


def multi_start_best(model, num_doses, time_window, metric, t_end, t_eval, params, ic):
    best = None
    for x0 in starting_points(num_doses).values():
        result = optimize_dose_schedule(
            model, num_doses=num_doses, time_window=time_window, metric=metric,
            t_end=t_end, t_eval=t_eval, initial_fractions=x0,
            params=params, initial_conditions=ic,
        )
        if best is None or result.objective_value > best.objective_value:
            best = result
    return best


# =============================================================================
# Re-check 1: AUC_TFH objective, multi-start, N=7
# =============================================================================
print("=" * 70)
print("RE-CHECK 1: AUC_TFH objective with multi-start (N=7)")
print("=" * 70)
t0 = time.time()
best_auc = multi_start_best("integrated", 7, TIME_WINDOW, "AUC_TFH", T_END, t_eval, params, ic)
best_tfhmax = multi_start_best("integrated", 7, TIME_WINDOW, "TFH_max", T_END, t_eval, params, ic)
print(f"  multi-start best AUC_TFH objective     : {best_auc.objective_value:,.1f}")
print(f"  multi-start best TFH_max objective     : {best_tfhmax.objective_value:,.1f}")

result_auc_sched = simulate_integrated(params, ic, best_auc.schedule, t_end=T_END, t_eval=t_eval)
result_tfhmax_sched = simulate_integrated(params, ic, best_tfhmax.schedule, t_end=T_END, t_eval=t_eval)

print(f"\n  {'schedule':24s} {'TFH_max':>12s} {'AUC_TFH':>14s}")
print(f"  {'AUC_TFH-optimized':24s} {peak_value(result_auc_sched, 'TFH'):12,.1f} {auc_metric(result_auc_sched, 'TFH'):14,.1f}")
print(f"  {'TFH_max-optimized':24s} {peak_value(result_tfhmax_sched, 'TFH'):12,.1f} {auc_metric(result_tfhmax_sched, 'TFH'):14,.1f}")
print(f"  AUC_TFH-optimized fractions: {np.round(best_auc.fractions, 4)}")
print(f"  TFH_max-optimized fractions: {np.round(best_tfhmax.fractions, 4)}")
print(f"  time: {time.time() - t0:.1f}s")

# =============================================================================
# Re-check 2: num_doses=5 and 6, multi-start
# =============================================================================
print()
print("=" * 70)
print("RE-CHECK 2: num_doses=5 and 6 with multi-start")
print("=" * 70)
t0 = time.time()
for n in [4, 5, 6, 7]:
    best_n = multi_start_best("integrated", n, TIME_WINDOW, "TFH_max", T_END, t_eval, params, ic)
    print(f"  num_doses={n}: multi-start best TFH = {best_n.objective_value:,.1f}")
print(f"  time: {time.time() - t0:.1f}s")

print(f"\nTOTAL runtime: {time.time() - t0_total:.1f}s")
