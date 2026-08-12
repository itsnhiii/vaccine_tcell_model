"""Trajectory metrics, parameter sweeps, sensitivity, and fitting
(master spec Sections 18-21)."""

from .fitting import FitResult, Observable, fit_parameters
from .metrics import (
    auc,
    duration_above_threshold,
    final_value,
    fold_expansion,
    peak_value,
    summarize_pmhc,
    summarize_tcell,
    summarize_tfh,
    time_to_peak,
)
from .optimization import (
    OptimizationResult,
    compare_dose_schedules_including_optimized,
    optimize_dose_schedule,
)
from .sensitivity import local_sensitivity, sensitivity_analysis
from .sweeps import sweep_dose_schedules, sweep_parameter

__all__ = [
    "peak_value",
    "time_to_peak",
    "auc",
    "final_value",
    "fold_expansion",
    "duration_above_threshold",
    "summarize_tcell",
    "summarize_tfh",
    "summarize_pmhc",
    "sweep_parameter",
    "sweep_dose_schedules",
    "local_sensitivity",
    "sensitivity_analysis",
    "fit_parameters",
    "Observable",
    "FitResult",
    "optimize_dose_schedule",
    "compare_dose_schedules_including_optimized",
    "OptimizationResult",
]
