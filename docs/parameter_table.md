# Parameter table

Every parameter in every `ParameterSet` this package ships, with value,
units, provenance category, and a short description. Generated directly
from `parameters/*.py` by calling each builder function and reading its
output (not transcribed by hand), so this table cannot drift from the
code without being regenerated:

```bash
python -c "
from vaccine_tcell_model.parameters import (
    default_science_parameters, mayer_parameters_fig1_cd4,
    mayer_parameters_fig2_cd8, mayer_parameters_fig4_dosing,
    default_integrated_parameters,
)
for ps in [default_science_parameters(), mayer_parameters_fig1_cd4(),
           mayer_parameters_fig2_cd8(), mayer_parameters_fig4_dosing(),
           default_integrated_parameters()]:
    for p in ps:
        print(ps.name, p.name, p.value, p.units, p.source_type.value)
"
```

Full source strings (the detailed provenance/reasoning behind each
value) are given below each table rather than inline, since several are
multi-sentence. For the equation each parameter feeds into, see
[`equations.md`](equations.md).

For source_type definitions, see `parameters/provenance.py`: every value
is exactly one of `literature`, `fitted`, `assumed`, `user_defined`,
`derived`, `model_extension`.

---

## Science (`default_science_parameters()`)

| Parameter | Value | Units | Type | Description |
|---|---|---|---|---|
| `d_Ag` | 3 | 1/day | literature | Antigen first-order decay rate |
| `d_Adj` | 3 | 1/day | assumed | Adjuvant first-order decay rate |
| `S_Adj` | 0.1 | dimensionless (normalized dose units, total dose = 1) | assumed | Half-max adjuvant activity concentration |
| `k` | 10 | dimensionless | assumed | Max fold-increase in antigen uptake rate conferred by adjuvant |
| `mu` | 1.2 | 1/day | literature | Innate immune cell (TC, DC, aDC_Ag) death/decay rate |
| `alpha` | 1.5 | 1/day | literature | Maximum T-cell proliferation rate |
| `eta` | 0.22 | 1/day | literature | T-cell to Tfh differentiation rate (see note below — NOT an independently-fit differentiation rate) |
| `D0` | 1.8e6 | 1/day | fitted | Rate of DC recruitment by tissue-resident innate cells (TC) |
| `T0` | 28 | cells (arbitrary/normalized units) | fitted | Baseline number of antigen-specific T cells |

Initial conditions (`science_default_initial_conditions`): all states 0
except `T(0) = T0`.

<details>
<summary>Full source strings</summary>

- **d_Ag**: Bhagchandani et al. 2024 Table S2 ('Our data', Table S1). Table S1's raw MLE is ~2.7/day (half-life 6.2 hr); Table S2 records 3/day — see `equations.md` Flag S2, not reconciled.
- **d_Adj**: Bhagchandani et al. 2024 Table S2, "taken to be identical to d_Ag".
- **S_Adj**: Bhagchandani et al. 2024 Table S2, chosen qualitatively so that DC recruitment saturates at low adjuvant dose (Fig. 3H).
- **k**: Bhagchandani et al. 2024 Table S2, "we set k to be large".
- **mu**: Mayer et al. 2019 Fig. 1 legend fit (CD4/cytochrome-C system), reused as-is via Bhagchandani et al. 2024 Table S2 ("From ref. 37").
- **alpha**: Mayer et al. 2019 Fig. 1 legend fit (CD4/cytochrome-C system), reused as-is via Bhagchandani et al. 2024 Table S2 ("From ref. 37").
- **eta**: Numerically identical to Mayer et al. 2019 Fig. 1 fitted delta (T-cell death rate, CD4/cytochrome-C system). Bhagchandani et al. 2024 explicitly reinterprets this value as a T-to-Tfh differentiation rate instead of a death rate (supplement p.3: "rather than accounting for death, we model the process as T cells proliferating and differentiating"). Confirmed in the reference MATLAB code: the variable is literally named `delta` throughout `odeInnate.m`, never renamed. See `equations.md` Flag S3.
- **D0**: Bhagchandani et al. 2024 Table S2, fitted to Fig. 3F Tfh counts across dose-number regimens, using the model's T-cell/Tfh state AT DAY 21 (near-plateau), per the 2026 erratum to the Fig. 3F caption — see `equations.md` Flag S4. Raw fit value in the reference code (`optimized_parameters.txt`): 1.7826e6.
- **T0**: Bhagchandani et al. 2024 Table S2, fitted jointly with D0 to Fig. 3F (day-21 model value, see Flag S4). Raw fit value in the reference code: 28.091.

</details>

---

## Mayer — three independent presets, never merged

### `mayer_parameters_fig1_cd4()` — CD4 T cells vs. cytochrome C (Quiel et al. data)

| Parameter | Value | Units | Type | Description |
|---|---|---|---|---|
| `alpha` | 1.5 | 1/day | literature | Maximum T-cell proliferation rate |
| `mu` | 1.2 | 1/day | literature | pMHC decay rate |
| `delta` | 0.22 | 1/day | literature | T-cell death rate |
| `K` | 0 | same units as C (arbitrary/normalized) | literature | Functional avidity/affinity parameter (fixed, not free, in this fit; upper bound from fit ~700) |

### `mayer_parameters_fig2_cd8(K=1.0)` — CD8 OT-1 T cells vs. Listeria ligands (Zehn et al. data)

| Parameter | Value (default) | Units | Type | Description |
|---|---|---|---|---|
| `alpha` | 2.47 | 1/day | literature | Maximum T-cell proliferation rate |
| `mu` | 3.1 | 1/day | literature | pMHC decay rate |
| `delta` | 0.23 | 1/day | literature | T-cell death rate |
| `K` | 1.0 (SIINFEKL) | relative EC50 (dimensionless) | literature | Functional avidity — set per ligand to its measured relative EC50 |

`MAYER_FIG2_LIGAND_K`: `SIINFEKL`=1.0, `SAINFEKL`=2.7, `SIYNFEKL`=4.1,
`SIIQFEKL`=18.3, `SIITFEKL`=70.7, `SIIVFEKL`=680.0 (Mayer et al. 2019
Fig. 2 legend table).

### `mayer_parameters_fig4_dosing(K=10.0)` — antigen-dosing-kinetics simulation

| Parameter | Value (default) | Units | Type | Description |
|---|---|---|---|---|
| `alpha` | 2.47 | 1/day | literature | Maximum T-cell proliferation rate ("as in Fig. 2") |
| `mu` | 3.1 | 1/day | literature | pMHC decay rate ("as in Fig. 2") |
| `delta` | 0.23 | 1/day | literature | T-cell death rate ("as in Fig. 2") |
| `K` | 10 | same units as C (arbitrary/normalized) | literature | Functional avidity/affinity parameter |

`default_mayer_parameters()` aliases this preset — the only one of the
three where K is a genuinely free parameter (Fig.1 fixes K=0), making it
the natural default for K-affinity sweeps.

No universal default initial conditions — `mayer_demo_initial_conditions
(T0=300, C0=10**6.7)` provides a clearly-labeled *example* (Fig.1C),
never a literature default (T0/C0 are exactly what Mayer's own figures
sweep).

---

## Integrated (`default_integrated_parameters(K=10.0, k_p=1.0, mu_pMHC=1.0)`)

Explicitly constructed — never a silent merge of ScienceParameters and
MayerParameters (master spec rule 4). Upstream parameters (`d_Ag`
through `T0`) are Science's Table S2 values, reused unchanged since
`ScienceUpstreamModel` is reused unchanged.

| Parameter | Value | Units | Type | Description |
|---|---|---|---|---|
| `d_Ag` | 3 | 1/day | literature | Antigen first-order decay rate (via ScienceUpstreamModel) |
| `d_Adj` | 3 | 1/day | assumed | Adjuvant first-order decay rate (via ScienceUpstreamModel) |
| `S_Adj` | 0.1 | dimensionless (normalized dose units) | assumed | Half-max adjuvant activity concentration (via ScienceUpstreamModel) |
| `k` | 10 | dimensionless | assumed | Max fold-increase in antigen uptake rate (via ScienceUpstreamModel) |
| `mu` | 1.2 | 1/day | literature | Innate immune cell death/decay rate (via ScienceUpstreamModel) |
| `D0` | 1.8e6 | 1/day | fitted | Rate of DC recruitment (via ScienceUpstreamModel) |
| `T0` | 28 | cells (arbitrary/normalized units) | fitted | Baseline number of antigen-specific T cells |
| `alpha` | 1.5 | 1/day | literature | Maximum T-cell proliferation rate (integrated T-cell equation) |
| `eta` | 0.22 | 1/day | literature | T-cell/Tfh differentiation rate (integrated T-cell + Tfh equations) |
| `k_p` | 1.0 (illustrative) | 1/day | model_extension | pMHC production rate from aDC_Ag — **no literature value exists** |
| `mu_pMHC` | 1.0 (illustrative) | 1/day | model_extension | pMHC decay rate — **no literature value exists** |
| `K` | 10.0 (illustrative) | same units as pMHC (arbitrary/normalized) | model_extension | Functional avidity for the integrated T-cell equation — **no literature value exists** |

`k_p`, `mu_pMHC`, and `K` automatically become `source_type=user_defined`
(instead of `model_extension`) if overridden away from their illustrative
defaults, via the shared `_extension_parameter()` helper in
`parameters/integrated.py` — so a swept/fitted value is never
mislabeled as if it were still the untouched default.

Initial conditions (`integrated_default_initial_conditions`): all states
0 except `T(0) = T0`, matching Science's initial conditions with
`pMHC(0) = 0` inserted.

---

## Cross-model parameter reuse — traceable, never silent

| Value | Appears in | Relationship |
|---|---|---|
| `alpha=1.5, mu=1.2, eta/delta=0.22` | Science, Mayer Fig.1, Integrated | Science's `alpha`/`mu`/`eta` are Mayer's Fig.1 (CD4) fit, reused via Table S2. Integrated reuses Science's values again. `eta` is Mayer's `delta` (death rate) reinterpreted as a differentiation rate — see `equations.md` Flag S3. |
| `alpha=2.47, mu=3.1, delta=0.23` | Mayer Fig.2, Mayer Fig.4 | Fig.4 explicitly reuses Fig.2's CD8 fit ("alpha, mu, and delta as in Fig. 2") — same paper, same fit, not a cross-paper merge. |
| `d_Ag=3, d_Adj=3, S_Adj=0.1, k=10, D0, T0` | Science, Integrated | Integrated's upstream is `ScienceUpstreamModel`, unchanged — these values are reused, not re-derived. |

No numeric value ever crosses from Science's `ParameterSet` into Mayer's
or vice versa except through this explicit, documented, one-directional
provenance chain (Mayer → Science, both → Integrated) — never a runtime
merge of two `ParameterSet` objects, which the API does not support.
