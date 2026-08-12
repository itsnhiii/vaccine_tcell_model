"""Dose-count sweep under a genuine non-zero-dose constraint, searched
with a global optimizer.

`optimize_dose_schedule` (the package's built-in optimizer, see
analysis/optimization.py) bounds each fraction to [0, 1] -- nothing stops
SLSQP from parking a fraction at exactly 0. Several of the "N-dose"
winners in optimization_deep_dive_final.py's dose-count sweep do exactly
that (e.g. the N=4 winner puts 0% on day 12, N=9 puts 0% on day 3, N=10
puts 0% on days 11-12) -- so those aren't honestly N-dose schedules,
they're (N-1)-dose schedules wearing an N-dose label.

This script sweeps num_doses=1..10 (12-day window, TFH_max objective,
integrated model) twice, differing only in the lower bound on each
fraction:

  - "unconstrained": bound = 0              (matches optimize_dose_schedule)
  - "nonzero":        bound = MIN_FRACTION  (every one of the N doses
                       must carry >= MIN_FRACTION of the total dose)

Search method -- why differential_evolution instead of multi-start SLSQP:
this module originally used a 9-10-point multi-start SLSQP search (still
scipy's recommended local method here, see analysis/optimization.py's
docstring). An earlier version of this exact script surfaced a case that
search missed: N=10's unconstrained result came out *below* N=9's, which
is structurally impossible for this model (the dosing equations form a
monotone/cooperative dynamical system, docs/model_specification.md --
with an extra dose slot available, the optimizer can always fall back to
the (N-1)-dose solution by parking the extra slot near 0, so true
TFH*(N) must be non-decreasing in N). Checking that specific case
(optimization_global_check.py) against scipy.optimize.differential_evolution
-- a population-based global search that doesn't follow a gradient from a
single point -- found a schedule 5.3% better than every multi-start SLSQP
attempt at N=10, and improvements at every other point checked too. So
this script now uses differential_evolution (bounds-only, no gradient) as
the primary search, followed by an SLSQP polish step for a crisp local
optimum at whatever basin DE lands in. (Adam was considered and rejected:
its benefit is averaging noisy mini-batch gradients in high-dimensional
problems, which doesn't apply here -- 1-10 dimensions, deterministic,
cheap to evaluate -- and it's still a purely local, gradient-following
method with no particular advantage over SLSQP at escaping basins.)

Run with:  python optimization_nonzero_sweep.py
Saves:     optimization_nonzero_sweep.png
"""

import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pathlib import Path
OUTDIR = Path(__file__).parent

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
MIN_FRACTION = 0.01  # every one of the N doses must carry >= 1% of the total dose


def dose_times_for(n):
    return tuple(float(t) for t in np.round(np.linspace(0.0, TIME_WINDOW, n)))


def build_schedule(fractions, dose_times, floor):
    # Clips-and-renormalizes any raw vector into a valid schedule, so DE
    # can search the box [floor, 1]^n directly -- no equality constraint
    # needed, every point in the box maps to something valid.
    fracs = np.clip(np.asarray(fractions, dtype=float), floor, None)
    fracs = fracs / fracs.sum()
    return DoseSchedule(times=list(dose_times), antigen_fractions=list(fracs), adjuvant_fractions=list(fracs))


def neg_tfh(fractions, dose_times, floor):
    schedule = build_schedule(fractions, dose_times, floor)
    result = simulate_integrated(params, ic, schedule, t_end=T_END, t_eval=t_eval)
    return -peak_value(result, "TFH")


def slsqp_polish(n, x0, floor):
    dose_times = dose_times_for(n)
    opt = minimize(
        neg_tfh, x0=x0, args=(dose_times, floor), method="SLSQP",
        bounds=[(floor, 1.0)] * n,
        constraints=[{"type": "eq", "fun": lambda f: np.sum(f) - 1.0}],
        options={"maxiter": 200, "ftol": 1e-10},
    )
    fracs = np.clip(opt.x, floor, None)
    fracs = fracs / fracs.sum()
    return fracs, float(-opt.fun)


def global_optimize(n, floor, seed, extra_x0=None):
    """DE global search + SLSQP polish. `extra_x0`, if given, seeds DE's
    initial population with one specific point (e.g. a neighboring N's
    winning schedule) alongside its own random population, in addition to
    the search DE does on its own."""
    dose_times = dose_times_for(n)
    rng = np.random.default_rng(seed)
    popsize = 12
    pop_n = popsize * n
    init_pop = rng.uniform(floor, 1.0, size=(pop_n, n))
    if extra_x0 is not None:
        init_pop[0] = np.clip(extra_x0, floor, 1.0)

    de = differential_evolution(
        neg_tfh, bounds=[(floor, 1.0)] * n, args=(dose_times, floor),
        init=init_pop, maxiter=50, tol=1e-6, mutation=(0.5, 1.5),
        recombination=0.7, polish=False, seed=seed,
    )
    fracs, tfh = slsqp_polish(n, de.x, floor)
    schedule = build_schedule(fracs, dose_times, floor)
    return schedule, fracs, tfh


print("=" * 70)
print("PASS 1/2: UNCONSTRAINED (floor=0), differential_evolution + SLSQP polish")
print("=" * 70)
t0_total = time.time()
unconstrained_results = {}
for n in range(1, 11):
    t0 = time.time()
    schedule, fracs, tfh = global_optimize(n, floor=0.0, seed=n)
    unconstrained_results[n] = (schedule, fracs, tfh)
    print(f"  N={n:2d}  TFH={tfh:12,.1f}  ({time.time() - t0:5.1f}s)")

print()
print("=" * 70)
print(f"PASS 2/2: NON-ZERO-CONSTRAINED (every fraction >= {MIN_FRACTION:.0%} of total dose)")
print("=" * 70)
results = {}
for n in range(1, 11):
    t0 = time.time()
    # Seed DE's population with pass 1's unconstrained winner (clipped
    # into the feasible region) -- cheap extra information, since it's
    # already near-optimal everywhere except whatever pass 1 zeroed out.
    unc_fracs = unconstrained_results[n][1]
    schedule, fracs, tfh = global_optimize(n, floor=MIN_FRACTION, seed=100 + n, extra_x0=unc_fracs)
    results[n] = (schedule, fracs, tfh)
    days = schedule.antigen_times
    frac_str = ", ".join(f"d{int(d)}={f * 100:.1f}%" for d, f in zip(days, fracs))
    print(f"  N={n:2d}  TFH={tfh:12,.1f}  ({time.time() - t0:5.1f}s)   {frac_str}")
print(f"\nTOTAL runtime: {time.time() - t0_total:.1f}s")

unconstrained_tfh = {n: unconstrained_results[n][2] for n in unconstrained_results}
nonzero_tfh = {n: results[n][2] for n in results}

# Repair pass 1: a constrained-feasible schedule (every fraction >=
# MIN_FRACTION > 0) is automatically feasible for the unconstrained
# problem too (its feasible region is a strict superset). If pass 2 ever
# beat pass 1 at some N, pass 1's search wasn't thorough enough there --
# polish pass 2's winner directly against the unconstrained problem.
print()
print("=" * 70)
print("REPAIR PASS 1: nonzero-beats-unconstrained check")
print("=" * 70)
any_repair = False
for n in range(1, 11):
    if nonzero_tfh[n] > unconstrained_tfh[n]:
        any_repair = True
        fracs, tfh = slsqp_polish(n, results[n][1], floor=0.0)
        if tfh > unconstrained_tfh[n]:
            dose_times = dose_times_for(n)
            unconstrained_results[n] = (build_schedule(fracs, dose_times, 0.0), fracs, tfh)
            unconstrained_tfh[n] = tfh
        print(f"  N={n}: nonzero was {nonzero_tfh[n]:,.1f} > unconstrained {unconstrained_results[n][2] if n in unconstrained_results else 0:,.1f} -> updated unconstrained to {unconstrained_tfh[n]:,.1f}")
if not any_repair:
    print("  none needed")

# Repair pass 2: the model is structurally guaranteed to never do worse
# with more dose slots available (monotone/cooperative dynamical system,
# docs/model_specification.md). If unconstrained_tfh dips going from N-1
# to N, that's a search-quality miss, not a real result -- retry N with
# a second DE seed and with N-1's winner (padded to N dose slots by
# splitting its largest dose in two) as an extra seed point.
print()
print("=" * 70)
print("REPAIR PASS 2: monotonicity-in-N check (unconstrained)")
print("=" * 70)
for n in range(2, 11):
    if unconstrained_tfh[n] < unconstrained_tfh[n - 1] - 1.0:
        print(f"  N={n}: {unconstrained_tfh[n]:,.1f} < N={n - 1}: {unconstrained_tfh[n - 1]:,.1f} -- retrying with extra seed")
        prev_fracs = unconstrained_results[n - 1][1]
        i = int(np.argmax(prev_fracs))
        padded = np.delete(np.insert(prev_fracs, i, prev_fracs[i] / 2), i + 1)
        padded[i] = prev_fracs[i] / 2
        padded = np.append(padded, prev_fracs[i] / 2) if len(padded) < n else padded[:n]
        schedule, fracs, tfh = global_optimize(n, floor=0.0, seed=n + 500, extra_x0=padded if len(padded) == n else None)
        if tfh > unconstrained_tfh[n]:
            unconstrained_results[n] = (schedule, fracs, tfh)
            unconstrained_tfh[n] = tfh
        print(f"    -> N={n} unconstrained now {unconstrained_tfh[n]:,.1f}")
    if nonzero_tfh[n] > unconstrained_tfh[n]:
        # keep the superset guarantee intact after any updates above
        fracs, tfh = slsqp_polish(n, results[n][1], floor=0.0)
        if tfh > unconstrained_tfh[n]:
            dose_times = dose_times_for(n)
            unconstrained_results[n] = (build_schedule(fracs, dose_times, 0.0), fracs, tfh)
            unconstrained_tfh[n] = tfh

for n in range(1, 11):
    assert nonzero_tfh[n] <= unconstrained_tfh[n] + 1.0, (
        f"N={n}: nonzero ({nonzero_tfh[n]:,.1f}) beat unconstrained ({unconstrained_tfh[n]:,.1f}) "
        f"-- should be impossible, search rigor mismatch"
    )
print("\nSanity check passed: nonzero_tfh[n] <= unconstrained_tfh[n] for every N.")
non_monotone = [n for n in range(2, 11) if unconstrained_tfh[n] < unconstrained_tfh[n - 1] - 1.0]
if non_monotone:
    print(f"NOTE: unconstrained still non-monotonic at N={non_monotone} after repair -- residual search-quality gap, not a model property.")
else:
    print("Unconstrained sweep is monotonically non-decreasing in N, as the model structurally guarantees.")

# =============================================================================
# PLOTS
# =============================================================================
fig = plt.figure(figsize=(16, 11))
outer = fig.add_gridspec(2, 1, height_ratios=[1, 1.3], hspace=0.55)
gs1 = outer[0].subgridspec(1, 2)

ns = list(range(1, 11))

ax1 = fig.add_subplot(gs1[0, 0])
ax1.plot(ns, [unconstrained_tfh[n] for n in ns], "o--", color="gray", label="unconstrained (fractions can hit 0)")
ax1.plot(ns, [nonzero_tfh[n] for n in ns], "o-", color="C0", label=f"every fraction $\\geq$ {MIN_FRACTION:.0%}")
ax1.set_xticks(ns)
ax1.set_xlabel("number of doses (N)")
ax1.set_ylabel("best achievable TFH")
ax1.set_title("Dose-count sweep: honest N-dose schedules\nvs. the unconstrained optimizer (both DE + SLSQP-polish)")
ax1.legend(fontsize=8)

ax2 = fig.add_subplot(gs1[0, 1])
pct = [nonzero_tfh[n] / unconstrained_tfh[n] * 100 for n in ns]
ax2.plot(ns, pct, "o-", color="C3")
ax2.axhline(100, color="gray", linestyle="--", linewidth=1)
ax2.set_xticks(ns)
ax2.set_ylim(min(pct) - 2, 101)
for n, p in zip(ns, pct):
    ax2.annotate(f"{p:.1f}%", (n, p), textcoords="offset points", xytext=(0, 6), fontsize=7, ha="center")
ax2.set_xlabel("number of doses (N)")
ax2.set_ylabel("% of unconstrained optimum")
ax2.set_title("Cost of requiring all N doses to be real\n(100% = no cost; unconstrained already used all N)")

gs2 = outer[1].subgridspec(2, 5, hspace=0.7, wspace=0.4)
for i, n in enumerate(ns):
    axg = fig.add_subplot(gs2[i // 5, i % 5])
    schedule, fracs, tfh = results[n]
    axg.bar(schedule.antigen_times, fracs, width=0.8, color="C0")
    axg.set_xlim(-1, 13)
    axg.set_ylim(0, 1.05)
    axg.xaxis.set_major_locator(MaxNLocator(integer=True))
    axg.set_title(f"N={n}  (TFH={tfh:,.0f})", fontsize=8)
    axg.tick_params(labelsize=7)
    if i % 5 == 0:
        axg.set_ylabel("dose fraction", fontsize=7)
    if i // 5 == 1:
        axg.set_xlabel("day", fontsize=7)

fig.tight_layout()
fig.subplots_adjust(left=0.08)
row3_top = outer[1].get_position(fig).y1
fig.text(0.5, row3_top + 0.02, f"Non-zero-constrained schedule per N (every bar $\\geq$ {MIN_FRACTION:.0%})",
          ha="center", fontsize=10, fontweight="bold")
fig.savefig(OUTDIR / "optimization_nonzero_sweep.png", dpi=150)
print("\nSaved optimization_nonzero_sweep.png")
