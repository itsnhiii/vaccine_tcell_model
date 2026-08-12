"""Parameter fitting (master spec Section 21).

Fits a subset of a model's kinetic parameters to time-course data via
scipy.optimize.least_squares -- bounded, multi-observable, residual-based
nonlinear least squares, with optional log-transformation so positive
parameters spanning orders of magnitude (e.g. D0 ~ 1e6) are well
conditioned and never go negative during optimization.

Never touches model validation (Phases 3-8) and never overwrites a
model's literature-default ParameterSet (master spec Section 21: "Do not
overwrite literature defaults. Fitted parameter sets should be saved
separately.") -- fit_parameters always returns a NEW ParameterSet inside
its FitResult, with only the fitted names marked source_type=FITTED and
every other parameter carried over unchanged from the base set.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from vaccine_tcell_model.dosing import DoseSchedule
from vaccine_tcell_model.parameters import ParameterSet, SourceType

from ._common import default_dose_schedule, default_params_and_ic, run_model


@dataclass(frozen=True)
class Observable:
    """One observed time course to fit against, e.g. measured T(t).

    `weight` scales this observable's contribution to the residual
    vector -- use it to balance observables on very different numeric
    scales (master spec Section 21: "multiple observables").
    """

    state: str
    time: np.ndarray
    value: np.ndarray
    weight: float = 1.0

    def __post_init__(self) -> None:
        if len(self.time) != len(self.value):
            raise ValueError(
                f"Observable({self.state!r}): time ({len(self.time)}) and "
                f"value ({len(self.value)}) must have the same length"
            )


@dataclass
class FitResult:
    """Result of fit_parameters(). Carries its own provenance-tagged
    ParameterSet rather than ever mutating the base/default one."""

    model: str
    param_names: tuple[str, ...]
    fitted_values: dict[str, float]
    initial_guess: dict[str, float]
    bounds: dict[str, tuple[float, float]]
    fitted_params: ParameterSet
    success: bool
    cost: float
    message: str
    n_function_evals: int

    def __repr__(self) -> str:
        return (
            f"FitResult(model={self.model!r}, success={self.success}, "
            f"cost={self.cost:.6g}, fitted={self.fitted_values})"
        )


def fit_parameters(
    model: str,
    param_names: Sequence[str],
    observables: Sequence[Observable],
    *,
    base_params: ParameterSet | None = None,
    initial_conditions: dict[str, float] | None = None,
    dose_schedule: DoseSchedule | None = None,
    t_end: float = 21.0,
    bounds: dict[str, tuple[float, float]] | None = None,
    initial_guess: dict[str, float] | None = None,
    log_transform: bool = True,
    method: str = "trf",
) -> FitResult:
    """Fit `param_names` to `observables` via bounded nonlinear least squares.

    `base_params` supplies every non-fitted parameter's value (defaults
    to the model's literature defaults) and is never mutated -- the
    result's `fitted_params` is a new ParameterSet built from it via
    repeated `with_value` calls.

    Default bounds, if not supplied for a given parameter, are an order
    of magnitude below/above its initial guess -- a permissive bracket,
    not a scientific claim; always pass explicit `bounds` for real fits.
    """
    if not observables:
        raise ValueError("fit_parameters requires at least one Observable")
    if not param_names:
        raise ValueError("fit_parameters requires at least one parameter name")

    default_params, default_ic = default_params_and_ic(model)
    params0 = base_params if base_params is not None else default_params
    ic = initial_conditions if initial_conditions is not None else default_ic
    schedule = default_dose_schedule(model, dose_schedule)

    bounds = dict(bounds or {})
    initial_guess = dict(initial_guess or {})

    guesses = []
    lo_bounds, hi_bounds = [], []
    for name in param_names:
        guess = initial_guess.get(name, params0.value(name))
        if log_transform and guess <= 0:
            raise ValueError(
                f"log_transform=True requires a strictly positive initial guess "
                f"for {name!r}, got {guess}"
            )
        lo, hi = bounds.get(name, (guess * 0.1, guess * 10.0))
        guesses.append(guess)
        lo_bounds.append(lo)
        hi_bounds.append(hi)

    all_times = np.unique(np.concatenate([np.asarray(obs.time, dtype=float) for obs in observables]))

    def to_optimizer_space(values: Sequence[float]) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        return np.log(arr) if log_transform else arr

    def from_optimizer_space(x: np.ndarray) -> np.ndarray:
        return np.exp(x) if log_transform else x

    def residuals(x: np.ndarray) -> np.ndarray:
        values = from_optimizer_space(x)
        params = params0
        for name, value in zip(param_names, values):
            params = params.with_value(name, float(value))
        result = run_model(model, params, ic, schedule, t_end, all_times)

        res = []
        for obs in observables:
            traj = result.trajectory(obs.state)
            idx = np.searchsorted(result.time, obs.time)
            predicted = traj[idx]
            res.append(obs.weight * (predicted - np.asarray(obs.value, dtype=float)))
        return np.concatenate(res)

    x0 = to_optimizer_space(guesses)
    x_lo = to_optimizer_space(lo_bounds)
    x_hi = to_optimizer_space(hi_bounds)

    opt = least_squares(residuals, x0=x0, bounds=(x_lo, x_hi), method=method)

    fitted_arr = from_optimizer_space(opt.x)
    fitted_values = {name: float(v) for name, v in zip(param_names, fitted_arr)}

    fitted_params = params0
    for name, value in fitted_values.items():
        fitted_params = fitted_params.with_value(
            name,
            value,
            source=f"Fitted via analysis.fitting.fit_parameters to {len(observables)} observable(s)",
            source_type=SourceType.FITTED,
        )

    return FitResult(
        model=model,
        param_names=tuple(param_names),
        fitted_values=fitted_values,
        initial_guess=dict(zip(param_names, guesses)),
        bounds=dict(zip(param_names, zip(lo_bounds, hi_bounds))),
        fitted_params=fitted_params,
        success=bool(opt.success),
        cost=float(opt.cost),
        message=str(opt.message),
        n_function_evals=int(opt.nfev),
    )
