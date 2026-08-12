"""Phase 3 demo: reproduce the Science (Bhagchandani et al. 2024) T-cell
priming model (supplement Eq. 1-7) end-to-end, with default (Table S2)
parameters, for bolus vs. 2-ED (20/80, day 0/7) vs. 7-ED (exponential
escalation) dosing regimens.

Plots DC, aDC_Ag, T, and TFH trajectories, per Phase 3's success
criterion. This is NOT a claim of exact numerical parity with the paper's
Fig. 3 -- solver tolerances, t_eval grid, and (per docs/equations.md
Flag S4) the exact day used for any D0/T0 refit are not reconciled here.
It demonstrates that the published equations, as implemented, produce
qualitatively sensible, correctly-ordered trajectories.

Run with:  python examples/01_science_reproduction.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vaccine_tcell_model.dosing import DoseSchedule
from vaccine_tcell_model.models.science import simulate_science
from vaccine_tcell_model.parameters import default_science_parameters, science_default_initial_conditions

REGIMENS = {
    "Bolus": DoseSchedule.bolus(name="Bolus"),
    "2-ED (20/80, d0/d7)": DoseSchedule(
        times=[0, 7], antigen_fractions=[0.2, 0.8], adjuvant_fractions=[0.2, 0.8],
        name="2-ED",
    ),
    "7-ED (exponential escalation)": DoseSchedule.from_exponential_escalation(
        numshot=7, k=1.0, duration=12, name="7-ED",
    ),
}

STATES_TO_PLOT = ["DC", "aDC_Ag", "T", "TFH"]


def main() -> None:
    params = default_science_parameters()
    ic = science_default_initial_conditions(params)
    t_end = 21.0
    t_eval = np.linspace(0, t_end, 421)

    results = {}
    for label, schedule in REGIMENS.items():
        result = simulate_science(params, ic, schedule, t_end=t_end, t_eval=t_eval)
        results[label] = result
        print(
            f"{label:32s} peak DC={result.trajectory('DC').max():.3e}  "
            f"peak aDC_Ag={result.trajectory('aDC_Ag').max():.3e}  "
            f"peak T={result.trajectory('T').max():.3e}  "
            f"final TFH={result.trajectory('TFH')[-1]:.3e}"
        )

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for ax, state_name in zip(axes.flat, STATES_TO_PLOT):
        for label, result in results.items():
            ax.plot(result.time, result.trajectory(state_name), label=label)
        ax.set_ylabel(f"# {state_name}")
        ax.set_yscale("log")
        ax.set_xlabel("time (days)")
        ax.set_title(state_name)

    axes.flat[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Science model (Bhagchandani et al. 2024, Eq. 1-7) -- Phase 3 reproduction")
    fig.tight_layout()
    out_path = "examples/01_science_reproduction_output.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    main()
