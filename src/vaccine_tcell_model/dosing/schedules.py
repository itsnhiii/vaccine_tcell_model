"""DoseSchedule: reusable description of a vaccine dosing regimen.

Master spec Section 10. Antigen and adjuvant are tracked as two
independent channels (independently timed and dosed if needed) and only
merged into a unified, time-sorted event list at the point of consumption
(`to_dose_events` / `to_state_jump_events`).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .events import DoseEvent
from .validation import validate_fraction_sum, validate_matching_lengths, validate_times


class DoseSchedule:
    """A vaccine dosing regimen: when antigen/adjuvant doses are given and how much.

    Construction (per channel, pick exactly one of fractions/doses, or
    neither for a single bolus of the full total dose):

        # fractions of a total dose, validated to sum to 1
        DoseSchedule(times=[0, 7], antigen_fractions=[0.2, 0.8],
                     adjuvant_fractions=[0.2, 0.8])

        # absolute doses
        DoseSchedule(times=[0, 7], antigen_doses=[2.0, 8.0],
                     adjuvant_doses=[1.0, 4.0])

        # independently timed antigen and adjuvant
        DoseSchedule(antigen_times=[0, 7], antigen_fractions=[0.2, 0.8],
                     adjuvant_times=[0], adjuvant_fractions=[1.0])

    Convenience constructors: `bolus`, `equal_doses`,
    `from_exponential_escalation` (matches the escalating-dose generator
    observed in the Science reference implementation -- see
    docs/equations.md Flag S5 -- so 7-ED/6-ED/... style regimens don't
    need to be hand-typed as fraction lists).
    """

    def __init__(
        self,
        times: Sequence[float] | None = None,
        antigen_fractions: Sequence[float] | None = None,
        adjuvant_fractions: Sequence[float] | None = None,
        *,
        antigen_times: Sequence[float] | None = None,
        adjuvant_times: Sequence[float] | None = None,
        antigen_doses: Sequence[float] | None = None,
        adjuvant_doses: Sequence[float] | None = None,
        total_antigen_dose: float = 1.0,
        total_adjuvant_dose: float = 1.0,
        name: str = "",
    ):
        self.name = name

        ag_times = antigen_times if antigen_times is not None else times
        adj_times = adjuvant_times if adjuvant_times is not None else times
        if ag_times is None:
            raise ValueError("Must supply `times` or `antigen_times` for the antigen channel")
        if adj_times is None:
            raise ValueError("Must supply `times` or `adjuvant_times` for the adjuvant channel")

        self.antigen_times, self.antigen_doses = self._resolve_channel(
            "antigen", ag_times, antigen_fractions, antigen_doses, total_antigen_dose
        )
        self.adjuvant_times, self.adjuvant_doses = self._resolve_channel(
            "adjuvant", adj_times, adjuvant_fractions, adjuvant_doses, total_adjuvant_dose
        )

    @staticmethod
    def _resolve_channel(
        label: str,
        chan_times: Sequence[float],
        fractions: Sequence[float] | None,
        doses: Sequence[float] | None,
        total_dose: float,
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        chan_times = tuple(float(t) for t in chan_times)
        validate_times(chan_times, f"{label}_times")

        if fractions is not None and doses is not None:
            raise ValueError(f"Supply either {label}_fractions or {label}_doses, not both")

        if fractions is not None:
            fractions = tuple(float(f) for f in fractions)
            validate_matching_lengths(chan_times, fractions, f"{label} fractions")
            validate_fraction_sum(fractions, f"{label}_fractions")
            resolved = tuple(f * total_dose for f in fractions)
        elif doses is not None:
            resolved = tuple(float(d) for d in doses)
            validate_matching_lengths(chan_times, resolved, f"{label} doses")
        else:
            if len(chan_times) != 1:
                raise ValueError(
                    f"Must supply {label}_fractions or {label}_doses when "
                    f"{label}_times has more than one entry"
                )
            resolved = (float(total_dose),)

        return chan_times, resolved

    # -- convenience constructors --------------------------------------------

    @classmethod
    def bolus(
        cls,
        time: float = 0.0,
        total_antigen_dose: float = 1.0,
        total_adjuvant_dose: float = 1.0,
        name: str = "bolus",
    ) -> "DoseSchedule":
        """A single dose of the full antigen and adjuvant amount at `time`."""
        return cls(
            times=[time],
            antigen_fractions=[1.0],
            adjuvant_fractions=[1.0],
            total_antigen_dose=total_antigen_dose,
            total_adjuvant_dose=total_adjuvant_dose,
            name=name,
        )

    @classmethod
    def equal_doses(
        cls,
        times: Sequence[float],
        total_antigen_dose: float = 1.0,
        total_adjuvant_dose: float = 1.0,
        name: str = "",
    ) -> "DoseSchedule":
        """`n` equally-sized antigen and adjuvant doses at `times` (e.g. 7-dose equal)."""
        n = len(times)
        fractions = [1.0 / n] * n
        return cls(
            times=times,
            antigen_fractions=fractions,
            adjuvant_fractions=fractions,
            total_antigen_dose=total_antigen_dose,
            total_adjuvant_dose=total_adjuvant_dose,
            name=name or f"{n}-dose equal",
        )

    @classmethod
    def from_exponential_escalation(
        cls,
        numshot: int,
        k: float,
        duration: float,
        *,
        adjuvant_numshot: int | None = None,
        adjuvant_k: float | None = None,
        adjuvant_duration: float | None = None,
        total_antigen_dose: float = 1.0,
        total_adjuvant_dose: float = 1.0,
        round_times: bool = True,
        name: str = "",
    ) -> "DoseSchedule":
        """Escalating-dose schedule built the way the Science reference
        implementation generates it (docs/equations.md Flag S5):

            dose_t = round(linspace(0, duration, numshot))
            dose_i = exp(k*i) / sum(exp(k*.))   for i = 0 .. numshot-1

        By default antigen and adjuvant share (numshot, k, duration); pass
        the `adjuvant_*` overrides for a regimen like "7-ED (adjuvant
        bolus)" (Science supplement fig. S2B: antigen escalates over 7
        shots while adjuvant is a single bolus at t=0 --
        adjuvant_numshot=1, adjuvant_k=0.0, adjuvant_duration=0.0).

        Reproduces the reference 2-ED split exactly: numshot=2, k=log(4),
        duration=7 -> antigen_fractions=[0.2, 0.8] at times [0, 7].
        """
        ag_times, ag_fracs = cls._exponential_fractions(numshot, k, duration, round_times)

        a_numshot = adjuvant_numshot if adjuvant_numshot is not None else numshot
        a_k = adjuvant_k if adjuvant_k is not None else k
        a_duration = adjuvant_duration if adjuvant_duration is not None else duration
        adj_times, adj_fracs = cls._exponential_fractions(a_numshot, a_k, a_duration, round_times)

        return cls(
            antigen_times=ag_times,
            antigen_fractions=ag_fracs,
            adjuvant_times=adj_times,
            adjuvant_fractions=adj_fracs,
            total_antigen_dose=total_antigen_dose,
            total_adjuvant_dose=total_adjuvant_dose,
            name=name or f"{numshot}-dose exponential escalation (k={k:g})",
        )

    @staticmethod
    def _exponential_fractions(
        numshot: int, k: float, duration: float, round_times: bool
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        if numshot < 1:
            raise ValueError(f"numshot must be >= 1, got {numshot}")
        raw_times = np.linspace(0.0, duration, numshot)
        times = np.round(raw_times) if round_times else raw_times
        weights = np.exp(np.arange(numshot) * k)
        fractions = weights / weights.sum()
        return tuple(float(t) for t in times), tuple(float(f) for f in fractions)

    # -- conversion to solver-facing events -----------------------------------

    def to_dose_events(self) -> list[DoseEvent]:
        """Merge the antigen and adjuvant channels into one sorted event list.

        Mirrors the reference implementation's union/intersect merge of
        antigen and adjuvant dose times (`getInnateDynamics.m`) for the
        case where the two channels are scheduled independently.
        """
        all_times = sorted(set(self.antigen_times) | set(self.adjuvant_times))
        ag_map = dict(zip(self.antigen_times, self.antigen_doses))
        adj_map = dict(zip(self.adjuvant_times, self.adjuvant_doses))
        return [
            DoseEvent(
                time=t,
                antigen_jump=ag_map.get(t, 0.0),
                adjuvant_jump=adj_map.get(t, 0.0),
            )
            for t in all_times
        ]

    def to_state_jump_events(self, antigen_state: str = "Ag", adjuvant_state: str = "Adj"):
        """DoseEvents translated into the solver's generic StateJumpEvent."""
        return [
            e.to_state_jump_event(antigen_state=antigen_state, adjuvant_state=adjuvant_state)
            for e in self.to_dose_events()
        ]

    @property
    def total_antigen_dose(self) -> float:
        return sum(self.antigen_doses)

    @property
    def total_adjuvant_dose(self) -> float:
        return sum(self.adjuvant_doses)

    def __repr__(self) -> str:
        label = f"{self.name!r} " if self.name else ""
        ag = list(zip(self.antigen_times, self.antigen_doses))
        adj = list(zip(self.adjuvant_times, self.adjuvant_doses))
        return f"DoseSchedule({label}antigen={ag}, adjuvant={adj})"
