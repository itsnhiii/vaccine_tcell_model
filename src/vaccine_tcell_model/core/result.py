"""SimulationResult: the structured output of a simulation run."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .state import StateSpec


@dataclass
class SimulationResult:
    """Structured output of a simulation run.

    Attributes:
        data: tidy DataFrame with a "time" column plus one column per
            state name in `states`.
        states: the StateSpec describing which columns are states.
        parameters: the ParameterSet (or dict) used for this run.
        dose_schedule: the DoseSchedule used, or None for models with no
            dosing concept (e.g. a bare Mayer T/C run).
        model_name: identifier, e.g. "science", "mayer", "integrated".
    """

    data: pd.DataFrame
    states: StateSpec
    parameters: Any
    dose_schedule: Any
    model_name: str

    def __post_init__(self) -> None:
        if "time" not in self.data.columns:
            raise ValueError("SimulationResult.data must have a 'time' column")
        missing = [n for n in self.states.names if n not in self.data.columns]
        if missing:
            raise ValueError(f"SimulationResult.data is missing state columns: {missing}")

    def trajectory(self, name: str) -> np.ndarray:
        """Time series for a single state, as a numpy array."""
        if name not in self.states.names:
            raise KeyError(f"Unknown state {name!r}. Known: {self.states.names}")
        return self.data[name].to_numpy()

    @property
    def time(self) -> np.ndarray:
        return self.data["time"].to_numpy()

    def __repr__(self) -> str:
        t0, t1 = self.data["time"].iloc[0], self.data["time"].iloc[-1]
        return (
            f"SimulationResult(model={self.model_name!r}, "
            f"states={self.states.names}, t=[{t0:g}, {t1:g}], "
            f"n_points={len(self.data)})"
        )
