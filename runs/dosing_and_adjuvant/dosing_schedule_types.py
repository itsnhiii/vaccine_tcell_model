"""Illustrates every DoseSchedule construction type supported by the package.

Run with:  python dosing_schedule_types.py
Saves:     dosing_schedule_types.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
OUTDIR = Path(__file__).parent
import numpy as np

from vaccine_tcell_model.dosing import DoseSchedule

BAR_WIDTH = 0.35

schedules = {
    "1. Bolus\nDoseSchedule.bolus()":
        DoseSchedule.bolus(),

    "2. Arbitrary fractions\nDoseSchedule(times=[0,7], antigen_fractions=[.2,.8])":
        DoseSchedule(times=[0, 7], antigen_fractions=[0.2, 0.8], adjuvant_fractions=[0.2, 0.8]),

    "3. Equal doses\nDoseSchedule.equal_doses(times=[0,2,...,12])":
        DoseSchedule.equal_doses(times=[0, 2, 4, 6, 8, 10, 12]),

    "4. Exponential escalation\nfrom_exponential_escalation(numshot=7, k=1, duration=12)":
        DoseSchedule.from_exponential_escalation(numshot=7, k=1.0, duration=12),

    "5. Absolute doses\nantigen_doses=[2,8], adjuvant_doses=[1,4]":
        DoseSchedule(times=[0,5,7], antigen_doses=[2.0, 8.0, 10.0], adjuvant_doses=[1.0, 4.0, 8.0]),

    "6. Independent Ag/Adj timing\nAg: 7-dose escalation | Adj: single bolus":
        DoseSchedule.from_exponential_escalation(
            numshot=7, k=1.0, duration=12,
            adjuvant_numshot=1, adjuvant_k=0.0, adjuvant_duration=0.0,
        ),
}

fig, axes = plt.subplots(2, 3, figsize=(15, 7))

for ax, (title, schedule) in zip(axes.flat, schedules.items()):
    ag_t, ag_d = np.array(schedule.antigen_times), np.array(schedule.antigen_doses)
    adj_t, adj_d = np.array(schedule.adjuvant_times), np.array(schedule.adjuvant_doses)

    ax.bar(ag_t - BAR_WIDTH / 2, ag_d, width=BAR_WIDTH, label="antigen", color="C0")
    ax.bar(adj_t + BAR_WIDTH / 2, adj_d, width=BAR_WIDTH, label="adjuvant", color="C1")

    ax.set_title(title, fontsize=9)
    ax.set_xlabel("time (days)")
    ax.set_ylabel("dose amount")
    ax.set_xlim(-1, 13)  # shared range across all panels (bolus has only 1 bar,
                          # so it would otherwise auto-zoom to just that point)
    ax.legend(fontsize=8)

fig.suptitle("DoseSchedule construction types", fontsize=13)
fig.tight_layout()
fig.savefig(OUTDIR / "dosing_schedule_types.png", dpi=150)
print("Saved dosing_schedule_types.png")
