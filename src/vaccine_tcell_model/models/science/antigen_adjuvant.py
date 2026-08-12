"""Antigen and adjuvant kinetics -- Science supplement Eq. 1-2 (decay term only).

Published equations:

    d[Ag]/dt  = sum_i f_i,Ag  * delta(t-t_i) - d_Ag  * [Ag]
    d[Adj]/dt = sum_i f_i,Adj * delta(t-t_i) - d_Adj * [Adj]

Per master-spec rule 8, the Dirac-delta dose term is represented as an
explicit state jump applied by the solver (dosing.DoseSchedule +
solvers.events.integrate_with_events), never as a literal delta inside
an ODE right-hand side. This module therefore implements only the decay
term; callers supply the dose term separately via `dose_schedule=` when
calling solvers.simulate() / simulate_science().
"""

from __future__ import annotations

from vaccine_tcell_model.parameters import ParameterSet


def antigen_adjuvant_rhs(state: dict[str, float], params: ParameterSet) -> dict[str, float]:
    """dAg/dt = -d_Ag*Ag, dAdj/dt = -d_Adj*Adj (decay-only; dosing is a solver-level jump)."""
    d_Ag = params.value("d_Ag")
    d_Adj = params.value("d_Adj")
    return {
        "Ag": -d_Ag * state["Ag"],
        "Adj": -d_Adj * state["Adj"],
    }
