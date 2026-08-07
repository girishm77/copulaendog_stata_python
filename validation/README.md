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

`stata_selftest.do` runs every estimator, the validity report, factor variables
and interactions in `exog()`, two endogenous regressors, `generate()` and
`predict`; then checks all seven marginal CDF estimators and every coefficient
and ρ against `reference_R_cdf.csv` and `reference_R.csv` at a 1e-8 tolerance.

```stata
do stata_selftest.do
```

It ends in `ALL CHECKS PASSED`, or names the checks that failed and how far off
they were.

**Result:** all 75 matched terms agree to within 1.4e-12; the seven CDF
estimators agree to within 1.9e-13. Verified on StataNow 19.5 SE.

Two things worth knowing if you extend this file:

- Every `import delimited` here passes `asdouble`. Without it Stata stores
  numeric columns as `float`, and seven significant digits put a ~1e-8 floor
  under every comparison — which is the tolerance being tested.
- The mata functions live inside `copulaendog.ado` and are not visible at the
  top-level mata namespace, so `ce_cdf()` cannot be called directly from a
  do-file. The CDF check goes through the public interface instead: for PG the
  copula term is exactly Φ⁻¹(F(P)), so `normal(p_cop)` after `generate()`
  recovers the marginal CDF.

## Data

- `simdata.csv` — n = 800, used by the R/Python/Stata comparisons. Columns
  `y`, `P` (endogenous), `x`, `w` (exogenous), `g` (a three-level category).
- `vignette_data.csv` — n = 2000, the dataset behind the results table in
  `docs/vignette.pdf`.

Both come from `copulaendog.sim_endog()`: P = exp(0.8v + 0.6w) with
corr(v, ξ) = 0.6, and y = 1 + 2P + 1.5x − 0.5w + ξ. The true coefficient on P
is 2. The 0.6w term makes corr(W, P\*) ≠ 0, so the Park & Gupta assumption
fails by construction and the validity report should say so.
