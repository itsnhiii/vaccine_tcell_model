# runs/

Ad-hoc exploration scripts and their output plots, run interactively
during the conversation (as opposed to `examples/`, which is the
package's formal, numbered example set from the original build).

Each script saves its `.png` output(s) next to itself regardless of
where you run it from (`python runs/optimization/optimal_dosing_run.py`
from the project root works the same as running it from inside that
folder) — every script sets `OUTDIR = Path(__file__).parent` and saves
via `OUTDIR / "name.png"`.

- **`mayer_reproductions/`** — recreates Mayer et al. 2019 Fig. 1C and
  Fig. 2A/2B in the paper's own layout.
- **`dosing_and_adjuvant/`** — first exploratory runs: a basic
  simulate-and-plot script (`my_run.py`), an illustration of every
  `DoseSchedule` construction type, an adjuvant dose-response comparison,
  and a same-total-dose/different-timing schedule comparison.
- **`optimization/`** — the dose-schedule optimization deep dive:
  multi-start search, number-of-doses sweep, and TFH_max-vs-AUC_TFH
  objective comparison. `optimization_deep_dive_final.py` is the
  corrected, canonical version (see its docstring and
  `optimization_deep_dive_followup.py` for how two single-start
  artifacts in the first pass were caught and fixed).
  - `archive_superseded_draft/` — an earlier, less rigorous pass at the
    same three-part analysis (2-start dose-count sweep instead of
    9-start). Kept for reference, not the numbers to cite — use
    `optimization_deep_dive_final.py`'s instead.
