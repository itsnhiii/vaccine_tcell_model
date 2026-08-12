"""Sanity-check the multi-start SLSQP results against a population-based
global optimizer (differential evolution), targeted at N=10 -- the one
point in optimization_nonzero_sweep.py's sweep where the result looked
suspicious: N=10's unconstrained TFH (188,413) came out *below* N=9's
(191,942.5), which should be structurally impossible. The model's dosing
equations form a monotone/cooperative dynamical system (see
docs/model_specification.md, "The model cannot represent 'more doses can
be worse'") -- with a 10th dose slot available, the optimizer could
always fall back to replicating the 9-dose solution (park the extra slot
near 0), so the true N=10 optimum must be >= the true N=9 optimum. A
worse N=10 result means our search got stuck, not that the model
misbehaves.

Why differential_evolution and not e.g. Adam: Adam's main advantage is
averaging out noisy gradients from mini-batch sampling in high-dimensional
problems (its usual home, neural net training). This objective is a
deterministic, cheap-to-evaluate ODE simulation in only 1-10 dimensions --
there's no batch noise to average out, and Adam is still a purely local,
gradient-following method, no better than SLSQP at escaping basins here.
differential_evolution is a population-based *global* search: no gradient
needed, and it explores the whole box rather than descending from a
single point, which is the actual property we want to test for.

Run with:  python optimization_global_check.py
"""

import time

import numpy as np
from scipy.optimize import differential_evolution, minimize

from vaccine_tcell_model.dosing import DoseSchedule
from vaccine_tcell_model.models.integrated import simulate_integrated
from vaccine_tcell_model.parameters import default_integrated_parameters, integrated_default_initial_conditions
from vaccine_tcell_model.analysis import peak_value

params = default_integrated_parameters(K=10.0)
ic = integrated_default_initial_conditions(params)
t_eval = np.linspace(0, 21, 106)
TIME_WINDOW = 12.0
T_END = 21.0
MIN_FRACTION = 0.01

# Current multi-start SLSQP results, for comparison (optimization_nonzero_sweep.py).
PRIOR = {
    (9, 0.0): 191_942.5,
    (9, MIN_FRACTION): 189_415.1,
    (10, 0.0): 188_413.3,
    (10, MIN_FRACTION): 187_994.7,
}


def dose_times_for(n):
    return tuple(float(t) for t in np.round(np.linspace(0.0, TIME_WINDOW, n)))


def build_schedule(fractions, dose_times, floor):
    fracs = np.clip(np.asarray(fractions, dtype=float), floor, None)
    fracs = fracs / fracs.sum()
    return DoseSchedule(times=list(dose_times), antigen_fractions=list(fracs), adjuvant_fractions=list(fracs))


def neg_tfh(fractions, dose_times, floor):
    schedule = build_schedule(fractions, dose_times, floor)
    result = simulate_integrated(params, ic, schedule, t_end=T_END, t_eval=t_eval)
    return -peak_value(result, "TFH")


def polish(n, x0, floor):
    dose_times = dose_times_for(n)
    opt = minimize(
        neg_tfh, x0=x0, args=(dose_times, floor), method="SLSQP",
        bounds=[(floor, 1.0)] * n,
        constraints=[{"type": "eq", "fun": lambda f: np.sum(f) - 1.0}],
        options={"maxiter": 200, "ftol": 1e-10},
    )
    fracs = np.clip(opt.x, floor, None)
    return fracs / fracs.sum(), float(-opt.fun)


def global_search(n, floor, seed):
    dose_times = dose_times_for(n)
    t0 = time.time()
    # build_schedule() already clips-and-renormalizes any raw vector, so
    # DE can search the box [floor, 1]^n directly with no equality
    # constraint needed -- any point in the box maps to a valid schedule.
    de = differential_evolution(
        neg_tfh, bounds=[(floor, 1.0)] * n, args=(dose_times, floor),
        seed=seed, maxiter=50, popsize=12, tol=1e-6, mutation=(0.5, 1.5),
        recombination=0.7, polish=False,
    )
    fracs = np.clip(de.x, floor, None)
    fracs = fracs / fracs.sum()
    de_tfh = -de.fun
    # Finish with an SLSQP polish from DE's best point for a crisp local optimum.
    fracs2, tfh2 = polish(n, fracs, floor)
    elapsed = time.time() - t0
    return fracs2, tfh2, de_tfh, elapsed


print("=" * 70)
print("GLOBAL-OPTIMIZER CHECK (differential_evolution + SLSQP polish)")
print("=" * 70)
for n in (9, 10):
    for floor in (0.0, MIN_FRACTION):
        fracs, tfh, de_raw, elapsed = global_search(n, floor, seed=0)
        prior = PRIOR[(n, floor)]
        beat = tfh > prior + 1.0
        label = "IMPROVED" if beat else ("tied" if abs(tfh - prior) <= 1.0 else "worse (multi-start SLSQP still best)")
        print(f"  N={n:2d} floor={floor:.2f}: DE+polish TFH={tfh:12,.1f}  (raw DE={de_raw:,.1f})  "
              f"vs prior={prior:,.1f}  -> {label}  [{elapsed:.1f}s]")
        if beat:
            days = dose_times_for(n)
            frac_str = ", ".join(f"d{int(d)}={f * 100:.1f}%" for d, f in zip(days, fracs))
            print(f"      new schedule: {frac_str}")
