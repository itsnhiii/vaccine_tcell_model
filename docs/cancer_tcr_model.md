# Cancer/TCR-signaling model

A readable overview of the fourth model in this package, added per Zach's
request: a tool for comparing **TCR signaling strength across dosing
schedules** in a cancer (mouse/patient) context, rather than the
virus/humoral-antibody pipeline the Science/Mayer/Integrated models
target. For equation-by-equation code, see
`src/vaccine_tcell_model/models/cancer_tcr/` and
`src/vaccine_tcell_model/parameters/cancer_tcr.py` -- every equation and
parameter below is documented again, in more depth, in those files'
docstrings. This document is the one-time narrative version; don't let
the two drift apart.

## Why a fourth model, not a fifth equation bolted onto the integrated one

The Science/Mayer/Integrated models all terminate in **TFH** (T follicular
helper cells), i.e. a germinal-center/antibody readout. That's the right
output for the viral/vaccine-humoral-immunity context those two papers
were built for, and the wrong output for a cancer CD8-killing context,
which has no germinal-center step at all. Rather than force a Tfh output
to zero everywhere, this is a genuinely separate model: it reuses the
Science upstream pharmacokinetics (Ag/Adj/TC/DC/aDC_Ag -- generic
vaccine-adjuvant/DC biology, not virus-specific) and replaces everything
downstream of `aDC_Ag`.

## The pipeline

```
dose schedule + antigen affinity (K_D)
        |
        v
   Ag(t), Adj(t)          <- UNCHANGED: same dosing/decay machinery as every other model
        |
        v
      TC(t), DC(t)        <- UNCHANGED: Science Eq. 3-4, reused verbatim
        |
        v
    aDC_Ag(t)              <- UNCHANGED: Science Eq. 5, reused verbatim -- "number of DCs"
        |
        v
  pMHC_density(t)          <- NEW: per-DC antigen-presentation density (this project's extension)
        |
        v
      T(t)                  <- NEW: antigen-specific T-cell count, no Tfh sink (this project's extension)
        |
        v
  [ Theta, A, S_contact, S_pop ]   <- NEW: closed-form TCR signaling strength
                                       (Chakraborty & Weiss 2014 kinetic-proofreading mechanism)
```

`S_contact`/`S_pop` are **not** ODE states -- they are a closed-form
post-processing function of the already-simulated `pMHC_density`, `T`,
and `aDC_Ag` trajectories (`models/cancer_tcr/signal.py`), computed by a
separate call (`compute_tcr_signal`) after `simulate_cancer_tcr` returns.

## The two new equations

**pMHC density per DC** (`models/cancer_tcr/pmhc_density.py`):

```
dpMHC_density/dt = k_load*aDC_Ag - mu_pMHC*pMHC_density
```

This went through one iteration during testing, worth recording: the
first draft drove production directly from ambient antigen `Ag(t)`
(reasoning: pMHC density is an *intensive*, per-DC quantity, so
shouldn't scale with the *number* of DCs). `tests/test_cancer_tcr_model.py`
caught that this was physically backwards -- `Ag` decays independently
of DC recruitment, so pMHC_density could rise and peak *before* `aDC_Ag`
had even built up, i.e. antigen apparently "displayed" before any DC had
captured it. The fix ties production to `aDC_Ag` instead -- the same
causal dependency, and the same "low-pass filter on the upstream signal"
structure, as the integrated model's own `dpMHC/dt = k_p*aDC_Ag -
mu_pMHC*pMHC`. The honest cost: `pMHC_density` is no longer a strictly
single-DC intensive quantity independent of DC count -- "per
representative DC" here means "the ensemble-average antigen-presentation
signal," not a literally-tracked single cell's surface density. Revisit
this equation first if real per-DC pMHC copy-number data ever becomes
available.

**T-cell expansion, homeostatic baseline, no Tfh sink**
(`models/cancer_tcr/tcell.py`):

```
dT/dt = alpha*T*pMHC_density/(K_T + T + pMHC_density) - delta*(T - T0)
```

Same saturating-competition proliferation shape as the integrated model.
The loss term went through one iteration during testing, worth recording
so it isn't silently reverted: the first draft used Mayer's own
unconditional `-delta*T` death term (reasoning: "no Tfh compartment, so
nothing to differentiate into"). `tests/test_cancer_tcr_model.py` caught
that this makes the antigen-free baseline **unstable** -- T decays
exponentially to zero over 21 days even with zero antigen ever presented,
which contradicts basic T-cell biology (baseline circulating T cells are
maintained by homeostatic IL-7/IL-15-driven turnover, not left to die off
absent antigen). The fix, `-delta*(T-T0)`, is a homeostatic relaxation
toward baseline `T0` -- mathematically the same *form* as Science's
`-eta*(T-T0)`, but not the same *biological claim*: there is no Tfh state
here receiving that flux, so it means "net turnover pulls the population
back to a set point," not "cells differentiate into Tfh." `K_T` is a
growth-saturation constant, kept a distinct symbol from the TCR-signal
module's `K_D` (see below) -- one shapes population growth, the other
shapes single-TCR signal strength; conflating them would hide two
physically different knobs behind one number.

## The TCR signaling-strength calculation

Built from Chakraborty & Weiss 2014 (*Nat. Immunol.* 15:797-807), the
paper Zach specifically flagged as more relevant than either of the
original two papers. Two pieces, both closed-form (no new ODE
integration):

**1. Occupancy** -- the TCR+pMHC binding step is bimolecular
(second-order), but because pMHC sits at a site with roughly constant
local TCR density, it reduces to pseudo-first-order saturation:

```
Theta(t) = pMHC_density(t) / (pMHC_density(t) + K_D)
```

**2. Kinetic-proofreading amplification** -- McKeithan 1995's model
(reviewed and extended in Chakraborty & Weiss 2014): after binding, `N`
first-order forward steps (rate `k_p` each) must complete before a
downstream signal is emitted; the ligand dissociating at any point
(rate `k_off`) resets the whole chain. The steady-state probability of
surviving all `N` steps:

```
k_off = K_D * k_on
A     = (k_p / (k_p + k_off)) ** N
```

Combined:

```
S_contact(t) = Theta(t) * A                            # per-cell / per-engagement signal
contacts(t)  = T(t) * aDC_Ag(t) / (T(t) + aDC_Ag(t))    # saturating T-DC meeting rate
S_pop(t)     = S_contact(t) * contacts(t)               # population-aggregate signal
```

Both outputs are exposed (per your decision to keep both, documented
here so it's not a mystery later): `S_contact` isolates "how hard is
each engaged TCR firing" (driven by affinity and pMHC density alone);
`S_pop` folds in "how many T cells and DCs are actually meeting" (driven
by the dosing schedule via the upstream population dynamics). A low
`S_pop` is therefore always traceable to one of exactly two causes,
readable directly from these two numbers.

## What `k_p` and `N` mean, and why they're fittable, not fixed

- `k_p`: the forward rate of *each* biochemical step in the proofreading
  chain (ITAM phosphorylation, ZAP-70 activation, LAT/SLP-76
  phosphorylation) -- how fast the kinase machinery advances the
  complex, assuming the ligand is still bound. One shared rate for all
  steps is the standard simplification; there's no literature consensus
  on distinct per-step rates.
- `N`: the number of such steps required before a signal fires.

Literature anchors (not a hard consensus, both papers say so
explicitly): optogenetic dwell-time-tuning experiments (Tischer & Weiner
2019; Yousefi et al. 2019) put the effective chain at **N ~ 2-4 steps**,
each taking **on the order of seconds** (so `k_p` ~ 0.1-1/s). Defaults
here are `N=3`, `k_p=0.3/s` -- documented as `model_extension`,
illustrative, and meant to be **fit against Zach's own experimental
TCR-signal readout**, not treated as ground truth.

## `K_D` and `k_on`

Both source documents agree `k_on` varies little across peptides --
`k_off` (equivalently affinity, `K_D`) carries essentially all the
discriminating information. So: `K_D` is the model's **required**
scientific input (antigen affinity, in uM -- always override the
illustrative default), and `k_on` is a **fixed** literature constant
(default `0.01 /uM/s` = `1e4 /M/s`, a representative order-of-magnitude
3D-SPR value), from which `k_off = K_D * k_on` is derived rather than
supplied independently.

## Quasi-steady-state justification (read before trusting the closed form)

Using the closed form `(k_p/(k_p+k_off))^N` instead of integrating the
`N`-state linear ODE chain explicitly assumes the chain equilibrates
"instantly" relative to everything else in the model. This is not a
claim either source paper makes about *this* modeling context -- it's a
deduction from the numbers they *do* report:

- Proofreading chain timescale: bond half-lives of ~0.5-10s for typical
  agonists, per-step delays of "seconds" for a 2-4 step chain -- so
  binding-to-signal completes in **seconds to tens of seconds**.
- This package's own timescales: `d_Ag = 3/day` (Ag half-life ~8h),
  DC/T-cell population dynamics evolving over the course of a
  multi-day dosing schedule.

That's a **>1000-fold separation**, the standard condition for a valid
quasi-steady-state reduction. Two caveats to keep in mind, not just
gloss over:

1. Single-molecule literature (Huang et al. 2013, cited in Chakraborty &
   Weiss 2014) shows downstream cytokine secretion, once a cell *is*
   triggered, unfolds over **hours** and is stochastic/digital rather
   than continuously graded per cell. If "TCR signaling strength" is
   ever meant to capture that sustained commitment rather than the
   instantaneous proofreading output, that's a slower, separate
   integration step -- out of scope for this model (see "Deferred" below).
2. At a discontinuous dose-jump instant, the QSS assumption is
   momentarily invalid (the chain hasn't had its few seconds to
   re-equilibrate). Invisible at this model's day-scale output
   resolution, but worth knowing it's there.

## The K_D-units caveat (the single biggest unvalidated assumption here)

`pMHC_density` is an arbitrary, uncalibrated model quantity -- there is
no established conversion to real molecule counts (the upstream Science
pharmacokinetics were never calibrated to absolute antigen/pMHC copy
numbers either). `K_D` (supplied in uM) is used **directly**, unconverted,
as the occupancy saturation constant against `pMHC_density`. This
implicitly assumes `K_D`'s numeric value sits on the same normalized
scale as `pMHC_density` -- not literal molar concentration. It does
**not** affect `k_off` (used only in the proofreading term `A`), which is
computed from `K_D` and `k_on` in real, internally consistent units.
**Trust relative comparisons (across affinities/dosing schedules) from
this module; don't trust absolute `S_contact`/`S_pop` magnitudes** until
real pMHC copy-number data lets us calibrate `pMHC_density`'s scale.

## The serial-triggering question -- deliberately not implemented

Zach's background-reading report presents a bell-shaped
response-vs-dwell-time curve (proofreading rewards long dwell, serial
triggering rewards short dwell, product peaks at an intermediate
affinity). Chakraborty & Weiss 2014 are explicitly skeptical of this as
a general phenomenon: they cite Holler & Kranz 2003's TCR-affinity-variant
data (half-lives spanning 30-1500s) showing a **monotonic** response with
**no observed optimum**, and state "very little evidence exists in clear
support" of the bell-curve prediction. Since this model exists
specifically to be validated against Zach's own data, the bell curve is
left as an **untested hypothesis**, not baked into the default equations:
`tcr_signal_params.value("include_serial_triggering")` is a reserved
flag that raises `NotImplementedError` if set to a nonzero value, rather
than silently doing nothing. Implement it only if Zach's data actually
shows the predicted downturn.

## Deferred: downstream fate (signal 2/3, NFAT/AP-1, anergy/exhaustion)

Zach's background report also covers CD28 costimulation (signal 2),
inflammatory cytokines (signal 3), and the NFAT/AP-1 partnership that
determines full effector differentiation vs. anergy/exhaustion. None of
that is implemented here -- per your explicit instruction, this model
stops at `S_contact`/`S_pop` (the proximal TCR-triggering signal), and a
"Module F" covering downstream fate is deferred until/unless Zach wants
it. If it's added later, it should consume `S_contact`/`S_pop` as an
input rather than being fused into this module.
