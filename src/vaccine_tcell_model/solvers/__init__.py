"""Dose-event-aware ODE solving (master spec Section 11)."""

from .events import integrate_with_events
from .ode import simulate

__all__ = ["integrate_with_events", "simulate"]
