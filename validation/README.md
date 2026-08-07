# Verification

## Python against R

`reference_R.R` runs the original R implementation on `simdata.csv` across ten
model specifications — all five estimators, four CDF estimators, an
interaction, and JAMS with and without conditioning — and writes the
coefficients to `reference_R.csv` and the raw CDF estimates to
`reference_R_cdf.csv`.

`compare_python.py` fits the same ten models with the Python port and compares
them term by term.

```bash
COPREG_FN_DIR=/path/to/Copreg_functions Rscript reference_R.R
python3 compare_python.py
```

`COPREG_FN_DIR` is a directory holding `Copreg_core.R`, `Copreg_pg.R`,
`Copreg_2scope.R`, `Copreg_ima.R`, `Copreg_jams.R` and `Copreg_bmw.R` from the
`functions` branch of
<https://github.com/HashtagHaschka/Copula-based-endogeneity-corrections>.
The R side needs the `Formula` package.

Bootstrap standard errors are not comparable across languages — different RNGs
draw different resamples — so only the point estimates, ρ, and the CDF
estimators themselves are compared. Those are deterministic.

**Result:** all 75 matched terms agree to within 1e-12; the six closed-form CDF
estimators agree to machine precision.

## Stata

`stata_selftest.do` runs every estimator, checks the copula transformations
against the same R reference to 1e-8, and prints the coefficients for
comparison with `reference_R.csv`.

```stata
do stata_selftest.do
```

**This has never been run.** The Stata licence on the machine the port was
written on had expired. Run it before relying on the Stata results.

## Data

- `simdata.csv` — n = 800, used by the R/Python/Stata comparisons. Columns
  `y`, `P` (endogenous), `x`, `w` (exogenous), `g` (a three-level category).
- `vignette_data.csv` — n = 2000, the dataset behind the results table in
  `docs/vignette.pdf`.

Both come from `copulaendog.sim_endog()`: P = exp(0.8v + 0.6w) with
corr(v, ξ) = 0.6, and y = 1 + 2P + 1.5x − 0.5w + ξ. The true coefficient on P
is 2. The 0.6w term makes corr(W, P\*) ≠ 0, so the Park & Gupta assumption
fails by construction and the validity report should say so.
