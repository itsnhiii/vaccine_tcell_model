"""Phase 7 demo: the integrated model extension end-to-end.

    Ag -> DC -> aDC_Ag -> pMHC -> T -> TFH

Ag/Adj/TC/DC/aDC_Ag come from ScienceUpstreamModel (Bhagchandani et al.
2024 Eq.1-5, reused unchanged). pMHC production and the T-cell equation
are THIS PROJECT'S EXTENSION (docs/equations.md Section 3) -- not
published verbatim by either source paper. TFH is Science's Eq.7, reused
unchanged.

Demonstrates:
  1. The full 6-state pipeline for a single dosing regimen.
  2. Affinity direction (K) in the integrated T-cell equation.
  3. Dose schedule changes propagate through to pMHC and T (Section 15.C).

Run with:  python examples/03_integrated_model.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vaccine_tcell_model.dosing import DoseSchedule
from vaccine_tcell_model.models.integrated import simulate_integrated
from vaccine_tcell_model.parameters import default_integrated_parameters, integrated_default_initial_conditions
from vaccine_tcell_model.validation.science import REFERENCE_DOSE_SCHEMES, tfh_at_day

PIPELINE_STATES = ["Ag", "DC", "aDC_Ag", "pMHC", "T", "TFH"]


def plot_pipeline(fig) -> None:
    params = default_integrated_parameters(K=10.0)
    ic = integrated_default_initial_conditions(params)
    schedule = DoseSchedule.from_exponential_escalation(numshot=7, k=1.0, duration=12, name="7-ED")
    t_eval = np.linspace(0, 21, 421)
    result = simulate_integrated(params, ic, schedule, t_end=21.0, t_eval=t_eval)

    axes = fig.subplots(2, 3)
    for ax, name in zip(axes.flat, PIPELINE_STATES):
        ax.plot(result.time, result.trajectory(name))
        ax.set_title(name)
        ax.set_xlabel("time (days)")
        ax.set_yscale("log")
    fig.suptitle("Integrated model pipeline (7-ED dosing): Ag -> DC -> aDC_Ag -> pMHC -> T -> TFH")


def plot_affinity_and_schedule(fig) -> None:
    ax_k, ax_schedule = fig.subplots(1, 2)

    schedule = DoseSchedule.from_exponential_escalation(numshot=7, k=1.0, duration=12)
    t_eval = np.linspace(0, 21, 211)
    for K in [1.0, 10.0, 100.0, 1000.0]:
        params = default_integrated_parameters(K=K)
        ic = integrated_default_initial_conditions(params)
        result = simulate_integrated(params, ic, schedule, t_end=21.0, t_eval=t_eval)
        ax_k.plot(result.time, result.trajectory("T"), label=f"K={K:g}")
    ax_k.set_yscale("log")
    ax_k.set_xlabel("time (days)")
    ax_k.set_ylabel("# T")
    ax_k.set_title("Affinity direction (K)")
    ax_k.legend(fontsize=8)

    params = default_integrated_parameters(K=10.0)
    ic = integrated_default_initial_conditions(params)
    for label, sched in [
        ("Bolus", DoseSchedule.bolus(name="Bolus")),
        ("2-ED (20/80, d0/d7)", DoseSchedule(times=[0, 7], antigen_fractions=[0.2, 0.8], adjuvant_fractions=[0.2, 0.8])),
        ("7-ED", schedule),
    ]:
        result = simulate_integrated(params, ic, sched, t_end=21.0, t_eval=t_eval)
        ax_schedule.plot(result.time, result.trajectory("pMHC"), label=label)
    ax_schedule.set_yscale("log")
    ax_schedule.set_xlabel("time (days)")
    ax_schedule.set_ylabel("# pMHC")
    ax_schedule.set_title("Dose schedule changes pMHC")
    ax_schedule.legend(fontsize=8)


def plot_phase8_validation(fig) -> None:
    """Phase 8 comparison plots: dose response and schedule response."""
    ax_dose, ax_schedule = fig.subplots(1, 2)

    params = default_integrated_parameters(K=10.0)
    ic = integrated_default_initial_conditions(params)
    doses = [0.5, 1.0, 2.0, 5.0, 10.0]
    t_max, tfh_final = [], []
    for total_dose in doses:
        schedule = DoseSchedule.bolus(total_antigen_dose=total_dose, total_adjuvant_dose=total_dose)
        result = simulate_integrated(params, ic, schedule, t_end=21.0, t_eval=np.linspace(0, 21, 211))
        t_max.append(result.trajectory("T").max())
        tfh_final.append(result.trajectory("TFH")[-1])
    ax_dose.plot(doses, t_max, "o-", label="T_max")
    ax_dose.plot(doses, tfh_final, "s-", label="TFH_final")
    ax_dose.set_xscale("log")
    ax_dose.set_yscale("log")
    ax_dose.set_xlabel("total antigen/adjuvant dose")
    ax_dose.set_ylabel("response")
    ax_dose.set_title("Dose response")
    ax_dose.legend(fontsize=8)

    # Schedule response: TFH(day 21) across the six Fig.1B/3F regimens.
    # Note the 7-ED/6-ED near-tie swap vs. the Science-only ordering --
    # see docs/equations.md Section 7 and
    # tests/test_integrated_validation.py::test_schedule_response_broadly_matches_validated_science_ordering.
    order = ["7-ED", "6-ED", "4-ED", "3-ED", "2-ED", "Bolus"]
    t_eval = np.linspace(0, 21, 211)
    tfh_values = []
    for name in order:
        result = simulate_integrated(params, ic, REFERENCE_DOSE_SCHEMES[name], t_end=21.0, t_eval=t_eval)
        tfh_values.append(tfh_at_day(result, 21.0))
    ax_schedule.bar(order, tfh_values, color=["C2" if n in ("7-ED", "6-ED") else "C0" for n in order])
    ax_schedule.set_yscale("log")
    ax_schedule.set_ylabel("TFH(day 21)")
    ax_schedule.set_title("Schedule response (green = near-tied pair)")
    ax_schedule.tick_params(axis="x", rotation=45)


def main() -> None:
    fig1 = plt.figure(figsize=(11, 6))
    plot_pipeline(fig1)
    fig1.tight_layout()
    fig1.savefig("examples/03_integrated_model_pipeline.png", dpi=150)

    fig2 = plt.figure(figsize=(9, 4))
    plot_affinity_and_schedule(fig2)
    fig2.tight_layout()
    fig2.savefig("examples/03_integrated_model_affinity_schedule.png", dpi=150)

    fig3 = plt.figure(figsize=(9, 4))
    plot_phase8_validation(fig3)
    fig3.tight_layout()
    fig3.savefig("examples/03_integrated_model_phase8_validation.png", dpi=150)

    print("Saved examples/03_integrated_model_pipeline.png")
    print("Saved examples/03_integrated_model_affinity_schedule.png")
    print("Saved examples/03_integrated_model_phase8_validation.png")


if __name__ == "__main__":
    main()
