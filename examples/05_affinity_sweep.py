"""Phase 9 demo: affinity (K) sweep via the analysis API.

Reproduces the master-spec Section 28 pattern:

    results = sweep_parameter(
        model="integrated", parameter="K", values=np.logspace(-2, 3, 50),
    )

and the four comparison plots Section 17 asks for: K vs T_max, K vs
TFH_max, K vs AUC_T, K vs AUC_TFH.

Run with:  python examples/05_affinity_sweep.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vaccine_tcell_model.analysis import sweep_parameter
from vaccine_tcell_model.dosing import DoseSchedule


def main() -> None:
    schedule = DoseSchedule.from_exponential_escalation(numshot=7, k=1.0, duration=12)
    df = sweep_parameter(
        model="integrated",
        parameter="K",
        values=np.logspace(-2, 3, 50),
        dose_schedule=schedule,
        t_end=21.0,
        t_eval=np.linspace(0, 21, 211),
    )
    print(df.head().to_string(index=False))
    print(f"... ({len(df)} rows total)")

    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    panels = [
        ("T_max", axes[0, 0]),
        ("TFH_max", axes[0, 1]),
        ("AUC_T", axes[1, 0]),
        ("AUC_TFH", axes[1, 1]),
    ]
    for metric, ax in panels:
        ax.plot(df["value"], df[metric], "o-", markersize=3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("K")
        ax.set_ylabel(metric)
        ax.set_title(f"K vs {metric}")
    fig.suptitle("Affinity sweep (integrated model, 7-ED dosing)")
    fig.tight_layout()
    out_path = "examples/05_affinity_sweep_output.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    main()
