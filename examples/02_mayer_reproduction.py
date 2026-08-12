"""Phase 5 demo: reproduce the Mayer et al. 2019 T-cell/pMHC model
(main text Eq. 1-2) end-to-end.

Demonstrates:
  1. T and C trajectories for a single run.
  2. Affinity direction: lower K -> stronger T-cell expansion.
  3. The paper's headline finding: fold expansion T(t*)/T(0) declines as
     precursor number T(0) increases (inverse power law, Fig. 1B).

Run with:  python examples/02_mayer_reproduction.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vaccine_tcell_model.models.mayer import simulate_mayer
from vaccine_tcell_model.parameters import default_mayer_parameters, mayer_demo_initial_conditions


def plot_trajectories(ax_T, ax_C) -> None:
    ic = mayer_demo_initial_conditions(T0=300.0, C0=10.0**6.7)
    t_eval = np.linspace(0, 10, 201)
    # K values chosen comparable to C0 (~5e6) so the affinity-limited
    # regime is actually visible -- K << C0 (e.g. K=1..100) is
    # competition-limited and K barely matters at all (by design; that's
    # the correct behavior of Eq.1, not a bug -- see docs/equations.md
    # and Mayer et al.'s own affinity-limited-regime discussion).
    for K in [1e5, 1e6, 1e7]:
        params = default_mayer_parameters(K=K)
        result = simulate_mayer(params, ic, t_end=10.0, t_eval=t_eval)
        ax_T.plot(result.time, result.trajectory("T"), label=f"K={K:.0e}")
        if K == 1e6:
            ax_C.plot(result.time, result.trajectory("C"), color="k", label="pMHC (C)")

    ax_T.set_yscale("log")
    ax_T.set_xlabel("time (days)")
    ax_T.set_ylabel("# T cells")
    ax_T.set_title("T-cell expansion vs. K (affinity)")
    ax_T.legend(fontsize=8)

    ax_C.set_yscale("log")
    ax_C.set_xlabel("time (days)")
    ax_C.set_ylabel("# pMHC")
    ax_C.set_title("pMHC decay (independent of K)")
    ax_C.legend(fontsize=8)


def plot_power_law(ax) -> None:
    C0 = 10.0**6.7
    t_eval = np.linspace(0, 10, 401)
    params = default_mayer_parameters(K=10.0)

    T0_values = np.logspace(0, 4.5, 15)
    fold_expansions = []
    for T0 in T0_values:
        result = simulate_mayer(params, {"T": T0, "C": C0}, t_end=10.0, t_eval=t_eval)
        fold_expansions.append(result.trajectory("T").max() / T0)

    ax.plot(T0_values, fold_expansions, "o-")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("initial T-cell number T(0)")
    ax.set_ylabel("fold expansion T_max / T(0)")
    ax.set_title("Inverse power law (Mayer et al. 2019 Fig. 1B)")


def main() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    plot_trajectories(axes[0], axes[1])
    plot_power_law(axes[2])
    fig.suptitle("Mayer model (Mayer et al. 2019, Eq. 1-2) -- Phase 5 reproduction")
    fig.tight_layout()
    out_path = "examples/02_mayer_reproduction_output.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
