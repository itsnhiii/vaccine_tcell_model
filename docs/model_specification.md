# Model specification

This document is a readable overview of *what* this package models and
*why*. For the equation-by-equation provenance audit (what came from
which paper, every discrepancy found against the reference code), see
[`equations.md`](equations.md). For parameter values, see
[`parameter_table.md`](parameter_table.md). For software design, see
[`architecture.md`](architecture.md).

## Scientific question

Given a fixed total antigen dose, an adjuvant dose, a dosing schedule,
and a T-cell functional avidity/affinity, how do antigen presentation
kinetics determine T-cell and Tfh expansion?

## The pipeline

```
vaccine dose + schedule + adjuvant
        |
        v
   Ag(t), Adj(t)          <- first-order decay + discrete dose jumps
        |
        v
      TC(t)                <- tissue-resident innate cells, adjuvant-recruited
        |
        v
      DC(t)                <- dendritic cells, recruited by TC, consume Ag
        |
        v
    aDC_Ag(t)               <- activated antigen-loaded DCs
        |
        v
     [pMHC(t)]              <- integrated model only; absent from Science
        |
        v
      T(t)                  <- antigen-specific T cells
        |
        v
     TFH(t)                 <- T follicular helper cells
```

## Three models, not one

This package deliberately keeps three separate, independently-validated
models rather than one blended model, per the project's non-negotiable
rule against silently combining sources:

| | States | Source | Status |
|---|---|---|---|
| **Science** | Ag, Adj, TC, DC, aDC_Ag, T, TFH | Bhagchandani et al. 2024, Sci. Immunol. supplement Eq. 1-7 | Reproduced exactly, no equation altered |
| **Mayer** | T, C (pMHC) | Mayer et al. 2019, PNAS main text Eq. 1-2 | Reproduced exactly (the approximate/saturating form used in every figure) |
| **Integrated** | Ag, Adj, TC, DC, aDC_Ag, pMHC, T, TFH | This project | Combines Science's upstream/Tfh equations (reused unchanged) with a new pMHC layer and a new hybrid T-cell equation |

**The Science model has no pMHC state and no K (affinity) parameter.**
This was confirmed by direct inspection of the published equations, the
reference MATLAB parameter list, and the reference code's ODE
right-hand side — none of them include K anywhere. This is the single
most consequential fact this project's specification rests on: K enters
the modeling only through Mayer and through the integrated extension,
never through the Science reproduction.

**Mayer has no dosing schedule.** Antigen enters Mayer's model purely
through the initial pMHC concentration `C(0)`; there is no `DoseSchedule`
concept for Mayer, and `simulate_mayer()` doesn't take one.

**The integrated model's two new equations are not published anywhere.**
`dpMHC/dt = k_p*aDC_Ag - mu_pMHC*pMHC` has no counterpart in either
paper. The T-cell equation `dT/dt = alpha*T*pMHC/(K+T+pMHC) -
eta*(T-T0)` structurally resembles Mayer's Eq.1 (same saturating-
competition numerator/denominator) but keeps Science's `-eta*(T-T0)`
sink term instead of Mayer's own `-delta*T` death term — a combination
neither paper makes. Both are `source_type=model_extension` in the
parameter provenance system.

## Initial conditions

All three models start every state at 0 except the baseline T-cell
population:

- Science / Integrated: `T(0) = T0` (28, Table S2's fitted value); all
  other states 0.
- Mayer: no universal default — `T(0)` and `C(0)` are exactly the
  quantities the model sweeps across figures (e.g. Fig.1B/C's precursor-
  number sweep). `mayer_demo_initial_conditions()` provides a clearly-
  labeled example value (`T0=300`, `C0=10^6.7`, from Fig.1C), never
  presented as a literature default.

## Dosing

Doses are represented as exact, instantaneous state jumps at specified
times — never as a narrow Gaussian/box pulse approximating a delta
function, and never smeared across solver steps (`DoseSchedule` +
`solvers.events.integrate_with_events`, Phase 2). Antigen and adjuvant
can be scheduled independently. Fraction-based schedules
(`antigen_fractions=[0.2, 0.8]`) and an exponential-escalation
constructor (`DoseSchedule.from_exponential_escalation`, matching the
reference code's actual dose generator) are both supported.

## Validated behaviors

Documented in detail in `equations.md` Sections 6-7, briefly:

- **Science**: qualitative dose-schedule ordering (more/earlier-escalating
  doses → larger TFH response) matches the paper across all six of its
  own canonical regimens (7-ED, 6-ED, 4-ED, 3-ED, 2-ED, Bolus).
- **Mayer**: affinity direction (lower K → stronger expansion), pMHC-
  exposure dependence, and the paper's own headline inverse-power-law
  relationship between fold expansion and precursor number all hold.
- **Integrated**: the full Ag→DC→aDC_Ag→pMHC→T→TFH causal chain runs
  correctly end-to-end; affinity direction and pMHC-parameter
  sensitivity hold; a real conservation-style invariant
  (`d(T+TFH)/dt = alpha*T*pMHC/(K+T+pMHC) >= 0` always) was proven and
  verified numerically. One genuine, documented behavior change from the
  Science-only model: the near-tied 7-ED/6-ED ordering (they differ by
  <1% in the Science model) swaps once the pMHC accumulation layer is
  added — a real, mechanistically-explained consequence of adding a
  low-pass-filter-like compartment, not a bug (`equations.md` Section 7).

## What this package does not do (yet)

Explicitly out of scope, per the master specification: the Science
B-cell/germinal-center model (Eqs. 8-25 of the supplement), explicit
peptide processing/MHC loading, multiple epitopes/clones, spatial or
stochastic dynamics, cytokines, and lymph-node compartments. The
architecture (pluggable `RHSComponent`s composed via `ComposedModel`) is
designed so these can be added later without rewriting existing modules
— see `architecture.md`.

### The model cannot represent "more doses can be worse"

Real vaccine dosing literature (and general immunology) documents cases
where spreading a fixed total dose across *too many* doses, or dosing
too frequently, actively *hurts* the response rather than just plateauing
— e.g. via antibody-mediated clearance of later doses before they can be
presented, T-cell exhaustion from chronic stimulation, or reactogenicity
constraints. **This model cannot reproduce that.** The dose-count
optimization sweep (`runs/optimization/optimization_deep_dive_final.py`)
is monotonically non-decreasing all the way from 1 to 10 doses — more
doses is always at least as good, never worse.

This is not an optimizer or search-quality issue; it is structural.
Every equation in the Science/Integrated model is what's called a
*monotone* (or cooperative) dynamical system: every production term is a
non-negative, non-decreasing function of its inputs (`Adj/(S_Adj+Adj)`,
`DC*Ag`, `k_p*aDC_Ag`, ...), and every loss term is a plain decay. There
is no term anywhere by which *more* antigen or adjuvant can actively
*suppress* a downstream state. Consequently no search over dosing
schedules within this model can ever discover a "too much hurts" result
— the mechanism that would produce one (antibody/immune-complex
interference, exhaustion, competition) simply isn't present in the
equations, most of which live in the Science B-cell/GC model this
package deliberately does not implement (see above).

This is consistent with, not contradicted by, the two source papers:
Bhagchandani et al.'s own mouse data only tested up to 7 doses and rose
monotonically the entire time (bolus < 2-dose < ... < 7-dose) — neither
paper's own equations or data demonstrate a downturn either. The
"more isn't always better" effect is real, but it requires mechanisms
outside what either published model represents.
