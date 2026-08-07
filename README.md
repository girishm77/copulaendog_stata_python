# copulaendog — instrument-free copula corrections, in Stata and Python

[![PyPI](https://img.shields.io/pypi/v/copulaendog)](https://pypi.org/project/copulaendog/)
[![Python](https://img.shields.io/pypi/pyversions/copulaendog)](https://pypi.org/project/copulaendog/)
[![Licence](https://img.shields.io/badge/licence-GPL--3.0--or--later-blue)](LICENSE)

Corrects endogenous regressors in a linear model when you have no instrument,
or none you trust. The dependence between the regressor and the structural
error is modelled with a Gaussian copula and the resulting control function is
added to the regression.

Five cross-sectional estimators behind one interface:

| Estimator | Reference | Idea |
|---|---|---|
| `pg` | Park & Gupta (2012) | add Φ⁻¹(F̂(P)) to the regression |
| `2scope` | Yang, Qian & Xie (2025) | transform W too, residualise P\* on W\* |
| `ima` | Haschka (2025a) | the same, without an intercept in the first stage |
| `bmw` | Breitung, Mayer & Wied (2024) | residualise first, rank-transform the residual |
| `jams` | Liengaard et al. (2025) | let the copula structure vary across categories |

---

## Credit

**This is a port. The econometrics and the reference code are not ours.**

- **[Copula-based endogeneity corrections in R](https://github.com/HashtagHaschka/Copula-based-endogeneity-corrections)**
  — Rouven E. Haschka ([ORCID 0000-0002-2916-9745](https://orcid.org/0000-0002-2916-9745)).
  The reference implementation. Every estimator, all seven CDF estimators, the
  bootstrap, the diagnostics and the validity report are ported from it.
- **[endogCopula](https://github.com/ashgreat/endogCopula)** — Ashwin Malshe.
  The packaged R version of the same estimators, which shaped how this port is
  organised and what its documented interface looks like.

If this code contributes to published work, please cite both of those and the
paper behind the estimator you used.

### Authorship

This work was inspired by, and is derived from, the R implementations above.
The Stata command and the Python package were developed by **Girish
Mallapragada**
([LinkedIn](https://www.linkedin.com/in/girishmallapragada/) ·
[Google Scholar](https://scholar.google.com/citations?user=CixA1fgAAAAJ&hl=en))
using [Claude](https://claude.ai). Everything here is a
translation of someone else's econometrics into two more languages, checked
line by line against their results; the credit for the methods belongs to the
authors named above and to the papers they implement.

---

## Try it without installing anything

Download one file and run it. Both do-files fetch the command straight from
this repository into a temporary directory, so there is nothing else to
download, no working directory to set, and nothing left on your machine
afterwards.

- **[`stata/quickstart.do`](stata/quickstart.do)** — the short tour. One
  dataset, every estimator, `validity` throughout.
- **[`stata/examples.do`](stata/examples.do)** — the full reference. Every
  option of every estimator spelled out, defaults included, plus `auto.dta`.

```stata
do quickstart.do
```

---

## Install

### Stata

```stata
ssc install copulaendog
```

Or straight from this repository, no SSC required:

```stata
net install copulaendog, ///
    from("https://raw.githubusercontent.com/girishm77/copulaendog_stata_python/main/stata") ///
    replace
```

Or, from a clone, without installing:

```stata
adopath ++ "/path/to/copulaendog_stata_python/stata"
help copulaendog
```

Requires Stata 16 or later. No Python needed — the estimators are written in
Mata and the command is self-contained.

### Python

```bash
pip install copulaendog
```

Or straight from this repository:

```bash
pip install git+https://github.com/girishm77/copulaendog_stata_python
```

Or, from a clone, without installing:

```bash
pip install numpy scipy pandas patsy
export PYTHONPATH="/path/to/copulaendog_stata_python/python:$PYTHONPATH"
```

Declared for Python 3.9 and later. The test suite has been run on 3.13 against
two dependency stacks: numpy 2.3 / scipy 1.16 / pandas 2.3, and numpy 2.5 /
scipy 1.18 / pandas 3.0.

> **SSC is still pending.** The package has been submitted; SSC is reviewed by
> hand, so `ssc install copulaendog` starts working only once it is accepted.
> Until then use the `net install` form above, which installs exactly the same
> files. PyPI is live: `pip install copulaendog` works now.

---

## Use it

### Stata

```stata
copulaendog y price, exog(feature display) method(2scope) nboots(499) seed(1)
copulaendog y price, exog(feature display) method(pg) seed(1) validity
copulaendog y price, exog(feature i.store) method(jams) conditional(store) seed(1)
copulaendog y price adspend, exog(feature) method(bmw) seed(1)

predict yhat                  // structural prediction; copula terms excluded
predict xi, residuals
```

Position decides. Variables after the dependent variable are endogenous and
each gets its own copula term; everything in `exog()` is exogenous and gets
none. Endogenous regressors must be plain numeric variables — a factor
variable expands into several columns and has no single copula term, so put
categorical controls in `exog()`, where `i.` notation is allowed.

### Python

```python
from copulaendog import copreg, sim_endog

df  = sim_endog(n=2000, seed=123)
fit = copreg("y ~ P | x + w", df, method="2scope", nboots=499, seed=1)

print(fit.summary())
print(fit.validity())

fit.params, fit.bse, fit.vcov
fit.conf_int(0.95, kind="percentile")
fit.rho, fit.icon
fit.predict(newdata)
```

Terms before the `|` are endogenous, terms after it exogenous. Everything
patsy accepts works: `np.log(price)`, `price:feat`, `I(x**2)`, `C(store)`, and
`- 1` to drop the intercept.

---

## What the options mean

`cdf()` picks how the marginal CDF entering the copula transformation is
estimated. The literature disagrees on this, so all of the proposals are
implemented and the default follows each estimator's own paper.

| Value | Source |
|---|---|
| `kde.silverman` | integrated Epanechnikov density, Silverman bandwidth; Park & Gupta (2012) |
| `kde.cv` | the same, cross-validated bandwidth; Li, Li & Racine (2017) — Python only |
| `kde.plugin` | Gaussian kernel CDF, Polansky & Baker (2000) plug-in bandwidth |
| `ecdf.fixed` | ECDF with replaced boundary; Becker, Proksch & Ringle (2022) |
| `ecdf.adj` | adjusted ECDF; Liengaard et al. (2025) |
| `rank.n` | Rank/n with a top correction; Qian, Koschmann & Xie (2025) |
| `rank.n1` | Rank/(n+1); Breitung, Mayer & Wied (2024) |

BMW is the one place where the choice is not free: its Proposition 3.1 is
derived for Rank/(n+1), so anything else warns.

Standard errors come from a pairs bootstrap, because the copula term is a
generated regressor and the textbook OLS standard errors are wrong for the
structural coefficients. No seed is set unless you set one.

`validity` walks the identification requirements: non-normality of P (or, for
BMW, of the first-stage residuals) on both the Yang and the Becker criteria,
the uncorrelatedness assumption where it applies, the shape of the structural
error, and ICON — the standard error inflation relative to uncorrected OLS,
which flags weak identification above 6.

---

## Documentation

`docs/vignette.pdf` is the full write-up: the model, what separates the five
estimators, how to read the validity report, and a worked example. In Stata,
`help copulaendog`.

---

## Verification status

**Python: verified against the R reference.** On a common dataset, across ten
model specifications — all five estimators, four CDF estimators, an
interaction, and JAMS with and without conditioning — all 75 coefficients and
ρ values agree with the R implementation to within 1e-12, and the six
closed-form CDF estimators agree to machine precision. Reproduce it with:

```bash
cd validation
COPREG_FN_DIR=/path/to/Copreg_functions Rscript reference_R.R
python3 compare_python.py
```

**Stata: verified against the R reference.** On the same ten specifications,
all 75 coefficients and ρ values agree with the R implementation to within
1.4e-12, and all six marginal CDF estimators it implements agree to within
1.9e-13. Run it yourself with:

```stata
cd validation
do stata_selftest.do
```

The self-test exercises every estimator, the validity report, factor variables
and interactions in `exog()`, two endogenous regressors, `generate()` and
`predict`, then checks the copula transformations and every coefficient
against the R reference. It ends in `ALL CHECKS PASSED` or names what failed.

Verified on StataNow 19.5 SE. One thing to know if you write your own
comparison: `import delimited` stores numeric columns as `float` unless you
pass `asdouble`, and seven significant digits alone put a ~1e-8 floor under
any agreement you try to measure.

---

## Not ported

Three estimators from the R implementation are out of scope; use the R code
for them.

- **2sCOPE-np** (Hu, Qian & Xie 2025) — replaces the first stage with a
  nonparametric conditional CDF, and is the only member of the family that
  tolerates discrete endogenous regressors.
- **PANEL** (Haschka 2022) — fixed-effects panel model by maximum likelihood.
- **BAYES** (Haschka 2025b) — the marginal CDFs are drawn rather than plugged in.

---

## Layout

```
python/copulaendog/     the Python package
  cdf.py                seven marginal CDF estimators and their bandwidths
  formula.py            two-part formula handling
  constructors.py       one copula control function per estimator
  core.py               fitting engine, bootstrap, results object, validity
  diagnostics.py        normality tests and model diagnostics
  estimators.py         the five user-facing estimators and copreg()
  simulate.py           a simulated dataset built for the assumptions
python/tests/           the Python test suite
stata/                  copulaendog.ado, copulaendog_p.ado, copulaendog.sthlp
docs/vignette.{tex,pdf} the write-up
validation/             R reference run, Python comparison, Stata self-test
pyproject.toml          build file; at the root so that pip install git+... works
```

---

## Licence

The R reference implementation is GPL-3-or-later with an additional
attribution term under section 7(b), which requires that the author
attribution travel with any material conveying it. This port is a derived work
and is released under the same terms; see [LICENSE](LICENSE).

---

## References

Becker, J.-M., D. Proksch, and C. M. Ringle (2022). Revisiting Gaussian
copulas to handle endogenous regressors. *Journal of the Academy of Marketing
Science* 50, 46–66.

Breitung, J., A. Mayer, and D. Wied (2024). Asymptotic properties of
endogeneity corrections using nonlinear transformations. *The Econometrics
Journal* 27(3), 362–383.

Haschka, R. E. (2022). Handling endogenous regressors using copulas: A
generalization to linear panel models with fixed effects and correlated
regressors. *Journal of Marketing Research* 59(4), 861–880.

Haschka, R. E. (2025a). Robustness of copula-correction models in causal
analysis: Exploiting between-regressor correlation. *IMA Journal of Management
Mathematics* 36(1), 161–180.

Haschka, R. E. (2025b). Bayesian inference for joint estimation models using
copulas to handle endogenous regressors. *Oxford Bulletin of Economics and
Statistics*.

Hu, X., Y. Qian, and H. Xie (2025). Correcting endogeneity via nonparametric
copula control functions. NBER Working Paper 33607.

Li, Q., J. Li, and J. S. Racine (2017). Cross-validated mixed-datatype
bandwidth selection for nonparametric cumulative distribution/survivor
functions. *Econometric Reviews* 36, 970–987.

Liengaard, B. D., J.-M. Becker, M. Bennedsen, P. Heiler, L. N. Taylor, and
C. M. Ringle (2025). Dealing with regression models' endogeneity by means of
an adjusted estimator for the Gaussian copula approach. *Journal of the Academy
of Marketing Science* 53, 279–299.

Park, S., and S. Gupta (2012). Handling endogenous regressors by joint
estimation using copulas. *Marketing Science* 31(4), 567–586.

Polansky, A. M., and E. R. Baker (2000). Multistage plug-in bandwidth
selection for kernel distribution function estimates. *Journal of Statistical
Computation and Simulation* 65, 63–80.

Qian, Y., A. Koschmann, and H. Xie (2025). A practical guide to endogeneity
correction using copulas. *Journal of Marketing*.

Yang, F., Y. Qian, and H. Xie (2025). Addressing endogeneity using a two-stage
copula generated regressor approach. *Journal of Marketing Research* 62(4),
601–623.
