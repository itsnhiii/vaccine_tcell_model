"""Validation helpers for dosing schedules (master spec Section 10)."""

from __future__ import annotations

from collections.abc import Sequence


def validate_fraction_sum(fractions: Sequence[float], label: str, tol: float = 1e-8) -> None:
    """Raise if `fractions` (e.g. antigen_fractions) does not sum to 1."""
    total = sum(fractions)
    if abs(total - 1.0) > tol:
        raise ValueError(
            f"{label} must sum to 1 (got {total!r} from {list(fractions)!r}). "
            "Use absolute doses (antigen_doses=/adjuvant_doses=) instead of "
            "fractions if the total is not meant to be 1."
        )


def validate_times(times: Sequence[float], label: str) -> None:
    """Raise if `times` is empty, negative, or has within-channel duplicates."""
    if len(times) == 0:
        raise ValueError(f"{label} must have at least one dose time")
    if any(t < 0 for t in times):
        raise ValueError(f"{label} must be non-negative, got {list(times)!r}")
    if len(set(times)) != len(times):
        raise ValueError(
            f"{label} contains duplicate dose times {list(times)!r}; "
            "combine repeated doses at the same time into a single entry."
        )


def validate_matching_lengths(times: Sequence[float], amounts: Sequence[float], label: str) -> None:
    if len(times) != len(amounts):
        raise ValueError(
            f"{label}: times ({len(times)}) and doses ({len(amounts)}) "
            "must have the same length"
        )
