"""Cancer/TCR-signaling model: antigen vs. adjuvant dosing schedule effects.

`DoseSchedule` tracks antigen and adjuvant as two independent channels
(dosing/schedules.py) -- each can have its own timing, fractions, and
total dose. This example uses that independence to isolate what each
channel's schedule contributes on its own, holding the other fixed,
using the cancer/TCR-signaling model (docs/cancer_tcr_model.md).

Demonstrates:
  1. Fixed antigen schedule (7-dose escalation), varying adjuvant
     delivery: matched escalation, single bolus, or none at all.
  2. Fixed adjuvant schedule (single bolus), varying antigen delivery:
     bolus, 2-dose escalation, or 7-dose escalation.
  3. A summary comparing peak population TCR signal across every
     combination tested.

Run with:  python examples/07_cancer_tcr_dosing_and_adjuvant.py
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
PARAMS = default_cancer_tcr_parameters()
IC = cancer_tcr_default_initial_conditions(PARAMS)
SIGNAL_PARAMS = default_tcr_signal_parameters(K_D=50.0)

UNITS_FOOTNOTE = (
    "Cell counts and antigen dose units are this model's own fitted, "
    "normalized scale, not independently validated absolute values."
)

# A reference 7-dose antigen escalation, reused below so every schedule
# that says "fixed antigen schedule" uses the exact same antigen timing
# and doses, not a recomputed approximation of it.
REFERENCE_7ED = DoseSchedule.from_exponential_escalation(numshot=7, k=1.0, duration=12, name="7-ED")


def add_units_footnote(fig) -> None:
    fig.text(0.5, 0.01, UNITS_FOOTNOTE, ha="center", va="bottom", fontsize=8, style="italic", wrap=True)


def add_shared_legend(fig, ax) -> None:
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)


def run(schedule: DoseSchedule):
    result = simulate_cancer_tcr(PARAMS, IC, schedule, t_end=21.0, t_eval=T_EVAL)
    signal_df = compute_tcr_signal(result, SIGNAL_PARAMS)
    return result, signal_df


# -- Figure 1: fixed antigen schedule, varying adjuvant delivery ----------------

ADJUVANT_VARIANTS = {
    "Adjuvant matches antigen escalation": REFERENCE_7ED,
    "Adjuvant as a single bolus": DoseSchedule(
        antigen_times=REFERENCE_7ED.antigen_times,
        antigen_doses=REFERENCE_7ED.antigen_doses,
        adjuvant_times=[0.0],
        adjuvant_fractions=[1.0],
        name="7-ED antigen, adjuvant bolus",
    ),
    "No adjuvant at all": DoseSchedule(
        antigen_times=REFERENCE_7ED.antigen_times,
        antigen_doses=REFERENCE_7ED.antigen_doses,
        adjuvant_times=[0.0],
        adjuvant_doses=[0.0],
        name="7-ED antigen, no adjuvant",
    ),
}


def plot_adjuvant_effect(fig) -> None:
    axes = fig.subplots(1, 3)
    for label, schedule in ADJUVANT_VARIANTS.items():
        result, signal_df = run(schedule)
        axes[0].plot(result.time, result.trajectory("pMHC_density"), label=label)
        axes[1].plot(result.time, result.trajectory("T"), label=label)
        axes[2].plot(signal_df["time"], signal_df["S_pop"], label=label)

    axes[0].set_title("Peptide-MHC density")
    axes[0].set_ylabel("Peptide-MHC density")
    axes[0].set_yscale("log")

    axes[1].set_title("Number of T cells")
    axes[1].set_ylabel("Number of T cells")
    axes[1].set_yscale("log")

    axes[2].set_title("Population TCR signal")
    axes[2].set_ylabel("TCR signal, population")

    for ax in axes:
        ax.set_xlabel("time (days)")

    add_shared_legend(fig, axes[0])
    fig.suptitle("Fixed antigen schedule (7-dose escalation) -- effect of adjuvant delivery")
    add_units_footnote(fig)


# -- Figure 2: fixed adjuvant schedule, varying antigen delivery ----------------

ANTIGEN_VARIANTS = {
    "Antigen as a single bolus": DoseSchedule(
        antigen_times=[0.0], antigen_fractions=[1.0],
        adjuvant_times=[0.0], adjuvant_fractions=[1.0],
        name="Antigen bolus, adjuvant bolus",
    ),
    "Antigen split 2 doses (20/80)": DoseSchedule(
        antigen_times=[0.0, 7.0], antigen_fractions=[0.2, 0.8],
        adjuvant_times=[0.0], adjuvant_fractions=[1.0],
        name="Antigen 2-ED, adjuvant bolus",
    ),
    "Antigen split 7 doses (escalating)": DoseSchedule(
        antigen_times=REFERENCE_7ED.antigen_times, antigen_doses=REFERENCE_7ED.antigen_doses,
        adjuvant_times=[0.0], adjuvant_fractions=[1.0],
        name="Antigen 7-ED, adjuvant bolus",
    ),
}


def plot_antigen_effect(fig) -> None:
    axes = fig.subplots(1, 3)
    for label, schedule in ANTIGEN_VARIANTS.items():
        result, signal_df = run(schedule)
        axes[0].plot(result.time, result.trajectory("pMHC_density"), label=label)
        axes[1].plot(result.time, result.trajectory("T"), label=label)
        axes[2].plot(signal_df["time"], signal_df["S_pop"], label=label)

    axes[0].set_title("Peptide-MHC density")
    axes[0].set_ylabel("Peptide-MHC density")
    axes[0].set_yscale("log")

    axes[1].set_title("Number of T cells")
    axes[1].set_ylabel("Number of T cells")
    axes[1].set_yscale("log")

    axes[2].set_title("Population TCR signal")
    axes[2].set_ylabel("TCR signal, population")

    for ax in axes:
        ax.set_xlabel("time (days)")

    add_shared_legend(fig, axes[0])
    fig.suptitle("Fixed adjuvant schedule (single bolus) -- effect of antigen delivery")
    add_units_footnote(fig)


# -- Figure 3: summary across every combination tested --------------------------


def plot_summary(fig) -> None:
    ax = fig.subplots(1, 1)
    all_variants = {**ADJUVANT_VARIANTS, **ANTIGEN_VARIANTS}
    peak_signal = {}
    for label, schedule in all_variants.items():
        _, signal_df = run(schedule)
        peak_signal[label] = signal_df["S_pop"].max()

    labels = list(peak_signal.keys())
    values = [peak_signal[label] for label in labels]
    colors = ["C0" if label in ADJUVANT_VARIANTS else "C1" for label in labels]
    ax.bar(range(len(labels)), values, color=colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Peak population TCR signal")
    ax.set_title("Peak population TCR signal across every schedule tested")

    from matplotlib.patches import Patch

    handles = [
        Patch(color="C0", label="Fixed antigen, varying adjuvant"),
        Patch(color="C1", label="Fixed adjuvant, varying antigen"),
    ]
    add_shared_legend_from_handles(fig, handles)
    add_units_footnote(fig)


def add_shared_legend_from_handles(fig, handles) -> None:
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)


def main() -> None:
    fig1 = plt.figure(figsize=(15.5, 4.3))
    plot_adjuvant_effect(fig1)
    fig1.tight_layout(rect=(0, 0.08, 0.85, 1))
    fig1.savefig("examples/07_cancer_tcr_dosing_and_adjuvant_adjuvant_effect.png", dpi=150, bbox_inches="tight")

    fig2 = plt.figure(figsize=(15.5, 4.3))
    plot_antigen_effect(fig2)
    fig2.tight_layout(rect=(0, 0.08, 0.85, 1))
    fig2.savefig("examples/07_cancer_tcr_dosing_and_adjuvant_antigen_effect.png", dpi=150, bbox_inches="tight")

    fig3 = plt.figure(figsize=(9, 5.5))
    plot_summary(fig3)
    fig3.tight_layout(rect=(0, 0.08, 0.72, 1))
    fig3.savefig("examples/07_cancer_tcr_dosing_and_adjuvant_summary.png", dpi=150, bbox_inches="tight")

    print("Saved examples/07_cancer_tcr_dosing_and_adjuvant_adjuvant_effect.png")
    print("Saved examples/07_cancer_tcr_dosing_and_adjuvant_antigen_effect.png")
    print("Saved examples/07_cancer_tcr_dosing_and_adjuvant_summary.png")


if __name__ == "__main__":
    main()
