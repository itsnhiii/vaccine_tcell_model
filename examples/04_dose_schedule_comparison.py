"""Phase 9 demo: dose-schedule comparison via the analysis API.

Compares bolus, 2-dose (20/80), and 7-dose (exponential escalation)
regimens for the integrated model using sweep_dose_schedules(), which
returns a tidy DataFrame -- the master-spec Section 19 pattern -- rather
than one-off hand-written loops like earlier examples used.

Run with:  python examples/04_dose_schedule_comparison.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vaccine_tcell_model.analysis import sweep_dose_schedules
from vaccine_tcell_model.dosing import DoseSchedule


def main() -> None:
    schedules = {
        "Bolus": DoseSchedule.bolus(name="Bolus"),
        "2-ED (20/80, d0/d7)": DoseSchedule(
            times=[0, 7], antigen_fractions=[0.2, 0.8], adjuvant_fractions=[0.2, 0.8], name="2-ED"
        ),
        "7-ED": DoseSchedule.from_exponential_escalation(numshot=7, k=1.0, duration=12, name="7-ED"),
    }

    df = sweep_dose_schedules(
        model="integrated", schedules=schedules, t_end=21.0, t_eval=np.linspace(0, 21, 211)
    )
    print(df.to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    for ax, metric in zip(axes, ["T_max", "AUC_TFH", "fold_expansion"]):
        ax.bar(df["schedule"], df[metric])
        ax.set_title(metric)
        ax.set_yscale("log")
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Dose schedule comparison (integrated model)")
    fig.tight_layout()
    out_path = "examples/04_dose_schedule_comparison_output.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    main()
