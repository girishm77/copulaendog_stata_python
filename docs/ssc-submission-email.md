# SSC submission

**To:** baum@bc.edu
**Subject:** SSC submission: copulaendog — instrument-free copula corrections for endogenous regressors
**Attach:** `copulaendog.zip` (on your Desktop as `copulaendog-ssc-submission.zip`)

---

Dear Professor Baum,

I would like to submit a new package, `copulaendog`, to the SSC archive. The
zip archive is attached and contains `copulaendog.ado`, `copulaendog_p.ado`,
`copulaendog.sthlp`, `copulaendog.pkg`, `stata.toc`, and two ancillary
do-files, `quickstart.do` and `examples.do`.

`copulaendog` corrects endogenous regressors in a linear model when no
instrument is available. The dependence between the regressor and the
structural error is modelled with a Gaussian copula, and the resulting control
function is added to the regression. Five cross-sectional estimators sit behind
one interface: Park and Gupta (2012), the two-stage copula generated regressor
approach of Yang, Qian and Xie (2025), the between-regressor correlation
variant of Haschka (2025), the nonlinear-transformation approach of Breitung,
Mayer and Wied (2024), and the adjusted estimator of Liengaard et al. (2025).
Six marginal CDF estimators are available for the copula transformation,
inference is by pairs bootstrap, and a `validity` option reports the
identification requirements the literature states — non-normality of the
endogenous regressor on both the Yang and the Becker criteria, the
uncorrelatedness assumption where it applies, and the ICON standard-error
inflation of Qian, Koschmann and Xie (2025).

I should be explicit that this is a port rather than original work. The
estimators, and the reference code every one of them is derived from, are due
to Rouven E. Haschka, whose R implementation is at

  https://github.com/HashtagHaschka/Copula-based-endogeneity-corrections

and to Ashwin Malshe, whose packaged R version, `endogCopula`, is at

  https://github.com/ashgreat/endogCopula

Both are credited in the package description, in the help file, and in the
header of every source file; the original code is GPL-3 with an additional
attribution term under section 7(b), which this port preserves and is
distributed under. I have kept the attribution in the `.pkg` description so
that it appears in `ssc describe`.

The port is validated against the R implementation rather than only against
itself. On a common dataset and across ten model specifications — all five
estimators, four CDF estimators, an interaction term, and the JAMS estimator
with and without conditioning — all 75 coefficients and endogeneity
correlations agree with the R results to within 1.4e-12, and the six marginal
CDF estimators agree to within 1.9e-13. The do-file that performs this
comparison, together with the R reference output it checks against, is in the
repository under `validation/`. Testing was done under StataNow 19.5 SE; the
command requires Stata 16 or later and has no dependencies beyond official
Stata.

The package, a companion Python implementation, and a vignette are at

  https://github.com/girishm77/copulaendog_stata_python

Please let me know if you would like anything changed in the package files or
the description.

With thanks for maintaining the archive,

Girish Mallapragada
Indiana University
https://scholar.google.com/citations?user=CixA1fgAAAAJ&hl=en

---

## Before you send

- [ ] Attach the zip (`~/Desktop/copulaendog-ssc-submission.zip`, 29K)
- [ ] Check the affiliation line — the help file and this email both say
      "Indiana University"; add the school if you want it named
- [ ] Decide whether to send before or after the PyPI upload. It makes no
      difference to Kit, but if PyPI is already live you may want to mention it

## What happens next

Kit Baum reviews submissions by hand and usually replies within a few days to
a couple of weeks. He may ask for changes to the help file or the `.pkg`
description. Once accepted he installs it and announces it on Statalist, after
which `ssc install copulaendog` works and `ssc describe copulaendog` shows the
description from the `.pkg` file.

At that point, delete the "Status of the two package archives" note from
`README.md`, since `ssc install copulaendog` becomes true.

## Updating later

Send Kit the revised files with a note saying what changed, and bump
`Distribution-Date` in `copulaendog.pkg` — it is currently `20260807`. He
replaces the package in place; the name and install command do not change.
