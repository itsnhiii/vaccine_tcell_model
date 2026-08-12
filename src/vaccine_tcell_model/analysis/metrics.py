"""Trajectory summary metrics (master spec Section 18).

Generic single-state metrics (peak_value, time_to_peak, auc, final_value,
fold_expansion, duration_above_threshold) plus per-state-group summary
functions (summarize_tcell, summarize_tfh, summarize_pmhc) matching the
exact metric names master spec Section 18 lists for T, TFH, and pMHC.
"""

from __future__ import annotations

import numpy as np

from vaccine_tcell_model.core.result import SimulationResult


def peak_value(result: SimulationResult, state: str) -> float:
    return float(np.max(result.trajectory(state)))


def time_to_peak(result: SimulationResult, state: str) -> float:
    traj = result.trajectory(state)
    return float(result.time[int(np.argmax(traj))])


def auc(result: SimulationResult, state: str) -> float:
    """Area under the trajectory, via trapezoidal integration over `result.time`."""
    return float(np.trapezoid(result.trajectory(state), result.time))


def final_value(result: SimulationResult, state: str) -> float:
    return float(result.trajectory(state)[-1])


def fold_expansion(result: SimulationResult, state: str, baseline: float) -> float:
    """peak_value / baseline (e.g. T_max/T0 -- master spec Section 18)."""
    return peak_value(result, state) / baseline


def duration_above_threshold(result: SimulationResult, state: str, threshold: float) -> float:
    """Total time (in `result.time`'s units) the trajectory spends strictly
    above `threshold`, via trapezoidal integration of the indicator function.
    """
    above = (result.trajectory(state) > threshold).astype(float)
    if not np.any(above):
        return 0.0
    return float(np.trapezoid(above, result.time))


def summarize_tcell(result: SimulationResult, baseline: float | None = None, state: str = "T") -> dict[str, float]:
    """T_max, t_T_max, T_final, AUC_T, fold_expansion (master spec Section 18)."""
    t0 = baseline if baseline is not None else float(result.trajectory(state)[0])
    return {
        f"{state}_max": peak_value(result, state),
        f"t_{state}_max": time_to_peak(result, state),
        f"{state}_final": final_value(result, state),
        f"AUC_{state}": auc(result, state),
        "fold_expansion": fold_expansion(result, state, t0),
    }


def summarize_tfh(result: SimulationResult, state: str = "TFH") -> dict[str, float]:
    """TFH_max, t_TFH_max, TFH_final, AUC_TFH (master spec Section 18)."""
    return {
        f"{state}_max": peak_value(result, state),
        f"t_{state}_max": time_to_peak(result, state),
        f"{state}_final": final_value(result, state),
        f"AUC_{state}": auc(result, state),
    }


def summarize_pmhc(
    result: SimulationResult, threshold: float | None = None, state: str = "pMHC"
) -> dict[str, float]:
    """pMHC_max, pMHC_AUC, and (if `threshold` given) duration_above_threshold
    (master spec Section 18). `state` defaults to "pMHC" (integrated model)
    but can be set to "C" for the Mayer model's cognate-pMHC state.
    """
    out = {
        f"{state}_max": peak_value(result, state),
        f"{state}_AUC": auc(result, state),
    }
    if threshold is not None:
        out["duration_above_threshold"] = duration_above_threshold(result, state, threshold)
    return out
