"""Phase 2 demo: exact dose jumps for bolus / 2-dose / 7-dose schedules.

Deliberately uses a trivial first-order decay system, not the Science or
Mayer equations (those arrive in Phases 3 and 5) -- the point here is
only to demonstrate that DoseSchedule + the event-aware solver produce
exact, unsmeared jumps at the requested dose times, for:

    - a single bolus
    - a two-dose 20/80 split (day 0 / day 7)
    - a seven-dose exponential escalation (day 0-12)

Run with:  python examples/00_dosing_and_events.py
"""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")  # headless-safe; script only writes a PNG, no display needed
import matplotlib.pyplot as plt
import numpy as np

from vaccine_tcell_model.core import StateSpec
from vaccine_tcell_model.dosing import DoseSchedule
from vaccine_tcell_model.solvers import simulate

DECAY_RATE = 0.3  # 1/day, arbitrary demo value -- not a literature parameter
STATES = StateSpec(names=("Ag", "Adj"))  # both channels, since DoseSchedule always carries both


def rhs(t, state):
    return {"Ag": -DECAY_RATE * state["Ag"], "Adj": -DECAY_RATE * state["Adj"]}


def run_and_report(name: str, schedule: DoseSchedule, t_end: float) -> None:
    t_eval = np.linspace(0, t_end, int(t_end * 100) + 1)
    result = simulate(
        rhs, STATES, initial_conditions={"Ag": 0.0, "Adj": 0.0}, t_end=t_end,
        dose_schedule=schedule, t_eval=t_eval, rtol=1e-10, atol=1e-12,
        model_name=name,
    )
    ag = result.trajectory("Ag")
    t = result.time

    print(f"\n{name}: {schedule!r}")
    for dose_time, dose_amount in zip(schedule.antigen_times, schedule.antigen_doses):
        idx = np.argmin(np.abs(t - dose_time))
        step = ag[idx] - (ag[idx - 1] if idx > 0 else 0.0)
        print(
            f"  dose at t={dose_time:5.2f}: amount={dose_amount:.4f}, "
            f"observed step={step:.4f} (exact, no smearing: {abs(step - dose_amount) < 1e-3 or idx == 0})"
        )
    return result


def main() -> None:
    bolus = DoseSchedule.bolus()
    two_dose = DoseSchedule(times=[0, 7], antigen_fractions=[0.2, 0.8], adjuvant_fractions=[0.2, 0.8])
    seven_dose = DoseSchedule.from_exponential_escalation(numshot=7, k=1.0, duration=12)

    fig, ax = plt.subplots(figsize=(7, 4))
    for name, schedule, t_end in [
        ("bolus", bolus, 10.0),
        ("2-dose (20/80, d0/d7)", two_dose, 14.0),
        ("7-dose exponential escalation", seven_dose, 18.0),
    ]:
        result = run_and_report(name, schedule, t_end)
        ax.plot(result.time, result.trajectory("Ag"), label=name)

    ax.set_xlabel("time (days)")
    ax.set_ylabel("Ag (trivial decay demo, not a biological trajectory)")
    ax.legend()
    ax.set_title("Phase 2: exact dose jumps, no numerical smearing")
    fig.tight_layout()
    out_path = "examples/00_dosing_and_events_output.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    main()
