"""Cancer/TCR-signaling model demo (docs/cancer_tcr_model.md).

    Ag -> DC -> aDC_Ag -> pMHC_density -> T  ->  [Theta, A, S_contact, S_pop]

Ag/Adj/TC/DC/aDC_Ag come from ScienceUpstreamModel (Bhagchandani et al.
2024 Eq.1-5, reused unchanged). pMHC_density and the T-cell equation are
THIS PROJECT'S cancer-context extension (no Tfh/humoral output). The TCR
signaling strength (Theta, A, S_contact, S_pop) is a closed-form
kinetic-proofreading calculation built on Chakraborty & Weiss 2014 --
see models/cancer_tcr/signal.py for every parameter/caveat.

Demonstrates:
  1. The full pipeline for a single dosing regimen, including the new
     TCR-signal columns.
  2. Affinity direction (K_D): stronger antigen -> more TCR signal, for a
     FIXED dosing schedule.
  3. The model's actual purpose -- comparing TCR signaling strength
     across dosing schedules, for a FIXED antigen affinity.

Run with:  python examples/06_cancer_tcr_model.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vaccine_tcell_model.dosing import DoseSchedule
from vaccine_tcell_model.models.cancer_tcr import compute_tcr_signal, simulate_cancer_tcr
from vaccine_tcell_model.parameters import (
    cancer_tcr_default_initial_conditions,
    default_cancer_tcr_parameters,
    default_tcr_signal_parameters,
)

T_EVAL = np.linspace(0, 21, 421)
PIPELINE_STATES = ["Ag", "aDC_Ag", "pMHC_density", "T"]
PIPELINE_YLABELS = {
    "Ag": "Antigen (dose units)",
    "aDC_Ag": "Number of dendritic cells",
    "pMHC_density": "Peptide-MHC density",
    "T": "Number of T cells",
}

UNITS_FOOTNOTE = (
    "Cell counts and antigen dose units are this model's own fitted, "
    "normalized scale, not independently validated absolute values."
)


def add_units_footnote(fig) -> None:
    fig.text(0.5, 0.01, UNITS_FOOTNOTE, ha="center", va="bottom", fontsize=8, style="italic", wrap=True)


def plot_pipeline(fig) -> None:
    params = default_cancer_tcr_parameters()
    ic = cancer_tcr_default_initial_conditions(params)
    schedule = DoseSchedule.from_exponential_escalation(numshot=7, k=1.0, duration=12, name="7-ED")
    result = simulate_cancer_tcr(params, ic, schedule, t_end=21.0, t_eval=T_EVAL)
    signal_df = compute_tcr_signal(result, default_tcr_signal_parameters())

    axes = fig.subplots(1, 5)
    for ax, name in zip(axes, PIPELINE_STATES):
        ax.plot(result.time, result.trajectory(name))
        ax.set_title(name)
        ax.set_xlabel("time (days)")
        ax.set_ylabel(PIPELINE_YLABELS[name])
        if name == "Ag":
            # Linear 0-1 scale (plain floats) rather than log/scientific
            # notation -- antigen's own dynamic range is small enough
            # (default total dose = 1.0) that log scale just obscures it
            # behind tiny decay-tail values instead of showing the doses.
            ax.set_ylim(bottom=0)
        else:
            ax.set_yscale("log")

    axes[-1].plot(signal_df["time"], signal_df["S_pop"], color="C3")
    axes[-1].set_title("S_pop\n(TCR signal, population)")
    axes[-1].set_xlabel("time (days)")
    axes[-1].set_ylabel("TCR signal")

    fig.suptitle("Cancer/TCR-signaling model pipeline (7-ED dosing)")
    add_units_footnote(fig)


def plot_affinity_direction(fig) -> None:
    """Fixed dosing schedule, varying antigen affinity (K_D)."""
    params = default_cancer_tcr_parameters()
    ic = cancer_tcr_default_initial_conditions(params)
    schedule = DoseSchedule.from_exponential_escalation(numshot=7, k=1.0, duration=12)
    result = simulate_cancer_tcr(params, ic, schedule, t_end=21.0, t_eval=T_EVAL)

    ax_contact, ax_pop = fig.subplots(1, 2)
    for K_D in [1.0, 10.0, 50.0, 200.0]:
        signal_df = compute_tcr_signal(result, default_tcr_signal_parameters(K_D=K_D))
        ax_contact.plot(signal_df["time"], signal_df["S_contact"], label=f"K_D = {K_D:g} micromolar")
        ax_pop.plot(signal_df["time"], signal_df["S_pop"], label=f"K_D = {K_D:g} micromolar")

    ax_contact.set_title("S_contact (per-cell signal)")
    ax_contact.set_xlabel("time (days)")
    ax_contact.set_ylabel("TCR signal per cell")

    ax_pop.set_title("S_pop (population signal)")
    ax_pop.set_xlabel("time (days)")
    ax_pop.set_ylabel("TCR signal, population")
    ax_pop.set_yscale("log")

    # One shared legend outside both axes -- both panels use the same
    # color-to-K_D mapping, and placing it outside the plot area (rather
    # than inline) keeps it from covering the data lines.
    handles, labels = ax_contact.get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)

    fig.suptitle("Affinity direction: stronger antigen (lower K_D) -> more TCR signal, same dosing")
    add_units_footnote(fig)


def plot_dosing_schedule_comparison(fig) -> None:
    """The model's actual purpose: comparing TCR signal across dosing
    schedules, for a FIXED antigen affinity."""
    params = default_cancer_tcr_parameters()
    ic = cancer_tcr_default_initial_conditions(params)
    signal_params = default_tcr_signal_parameters(K_D=50.0)

    ax_pmhc, ax_pop = fig.subplots(1, 2)
    schedules = [
        ("Bolus", DoseSchedule.bolus(name="Bolus")),
        ("2-ED (20/80, d0/d7)", DoseSchedule(times=[0, 7], antigen_fractions=[0.2, 0.8], adjuvant_fractions=[0.2, 0.8])),
        ("7-ED", DoseSchedule.from_exponential_escalation(numshot=7, k=1.0, duration=12, name="7-ED")),
    ]
    peak_signal = {}
    for label, schedule in schedules:
        result = simulate_cancer_tcr(params, ic, schedule, t_end=21.0, t_eval=T_EVAL)
        signal_df = compute_tcr_signal(result, signal_params)
        ax_pmhc.plot(result.time, result.trajectory("pMHC_density"), label=label)
        ax_pop.plot(signal_df["time"], signal_df["S_pop"], label=label)
        peak_signal[label] = signal_df["S_pop"].max()

    ax_pmhc.set_title("pMHC_density")
    ax_pmhc.set_xlabel("time (days)")
    ax_pmhc.set_ylabel("Peptide-MHC density")
    ax_pmhc.set_yscale("log")

    ax_pop.set_title("S_pop (population TCR signal)")
    ax_pop.set_xlabel("time (days)")
    ax_pop.set_ylabel("TCR signal, population")

    # One shared legend outside both axes, same reasoning as
    # plot_affinity_direction -- keeps the legend off the data lines.
    handles, labels = ax_pmhc.get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)

    peak_str = ", ".join(f"{label}={value:.1f}" for label, value in peak_signal.items())
    fig.suptitle(f"Dosing schedule comparison (K_D = 50 micromolar) -- peak S_pop: {peak_str}", fontsize=10)
    add_units_footnote(fig)


def main() -> None:
    fig1 = plt.figure(figsize=(15, 3.8))
    plot_pipeline(fig1)
    fig1.tight_layout(rect=(0, 0.06, 1, 1))
    fig1.savefig("examples/06_cancer_tcr_model_pipeline.png", dpi=150)

    fig2 = plt.figure(figsize=(10.5, 4.3))
    plot_affinity_direction(fig2)
    fig2.tight_layout(rect=(0, 0.07, 0.85, 1))
    fig2.savefig("examples/06_cancer_tcr_model_affinity_direction.png", dpi=150, bbox_inches="tight")

    fig3 = plt.figure(figsize=(10.5, 4.3))
    plot_dosing_schedule_comparison(fig3)
    fig3.tight_layout(rect=(0, 0.07, 0.85, 0.93))
    fig3.savefig("examples/06_cancer_tcr_model_dosing_comparison.png", dpi=150, bbox_inches="tight")

    print("Saved examples/06_cancer_tcr_model_pipeline.png")
    print("Saved examples/06_cancer_tcr_model_affinity_direction.png")
    print("Saved examples/06_cancer_tcr_model_dosing_comparison.png")


if __name__ == "__main__":
    main()
