# Changelog

Notable changes to `copulaendog`. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning is
[semantic](https://semver.org/spec/v2.0.0.html).

Development history, the defects found while building this, and the reasoning
behind the design choices are in [docs/PROJECT-LOG.md](docs/PROJECT-LOG.md).
How to check any claim made here is in
[docs/VERIFICATION.md](docs/VERIFICATION.md).

## [Unreleased]

Nothing yet. The commits after `v0.1.0` are documentation only and change no
released code.

## [0.1.0] — 2026-08-07

First release. Stata and Python ports of the Gaussian copula endogeneity
corrections, from the R reference implementation of Rouven E. Haschka and the
`endogCopula` package of Ashwin Malshe.

### Added

**Estimators** — five cross-sectional Gaussian copula corrections behind one
interface, in both languages:

| Method | Reference |
|---|---|
| `pg` | Park & Gupta (2012) |
| `2scope` | Yang, Qian & Xie (2025) |
| `ima` | Haschka (2025a) |
| `bmw` | Breitung, Mayer & Wied (2024) |
| `jams` | Liengaard et al. (2025) |

**Marginal CDF estimators** — `kde.silverman`, `kde.plugin`, `ecdf.fixed`,
`ecdf.adj`, `rank.n`, `rank.n1` in both languages, plus `kde.cv` in Python
only. Defaults follow each estimator's own paper.

**Inference** — pairs bootstrap, because the copula term is a generated
regressor and the textbook OLS standard errors are wrong for the structural
coefficients. Resamples that lose a factor level, or leave a JAMS cell too
thin, are rejected and redrawn, and the command reports when that happens
often.

**Diagnostics** — a `validity` report covering non-normality of the endogenous
regressor on both the Yang and the Becker criteria, the uncorrelatedness
assumption where it applies, the shape of the structural error, and the ICON
standard-error inflation of Qian, Koschmann & Xie (2025). For `bmw` the
non-normality requirement is tested on the first-stage residuals, per their
Theorem 2.1, and a Durbin–Hausman–Wu test is reported alongside the bootstrap
one per their Corollary 3.2.

**Stata** — `copulaendog.ado`, `copulaendog_p.ado` (`predict`), a help file,
and `copulaendog.pkg` / `stata.toc` for `net install` and SSC. Factor
variables and interactions are supported in `exog()`.

**Python** — the `copulaendog` package with a two-part formula interface
(`"y ~ endog | exog"`), `copreg()` plus the five named estimators, a results
object with `summary()`, `validity()`, `conf_int()` and `predict()`, and
`sim_endog()` for a dataset whose truth is known.

**Run-it-and-go** — `stata/quickstart.do` and `stata/examples.do` fetch the
command into a temporary directory and run, mirroring how the R package is
used: download one file, nothing else to fetch, nothing installed.

**Documentation** — an eight-page vignette (`docs/vignette.pdf`), the Stata
help file, a verification runbook and a project log.

### Verified

Against the R reference implementation on a shared dataset, across ten model
specifications covering all five estimators, four CDF estimators, an
interaction, and JAMS with and without conditioning:

| | |
|---|---|
| Python vs R | 75 terms agree to **9.4e-13**; CDF estimators to between 0 and 2.4e-13 |
| Stata vs R | 75 terms agree to **1.34e-12**; six CDF estimators to **1.85e-13** |
| Python tests | 40 pass on numpy 2.3/scipy 1.16/pandas 2.3 and on numpy 2.5/scipy 1.18/pandas 3.0 |
| Stata self-test | `ALL CHECKS PASSED`, zero errors, StataNow/SE 19.5 |

Bootstrap standard errors are not compared across languages and cannot be —
different RNGs draw different resamples. Only deterministic quantities are
diffed.

### Not included

`2sCOPE-np` (Hu, Qian & Xie 2025), `PANEL` (Haschka 2022) and `BAYES`
(Haschka 2025b) are not ported; use the R implementation. `kde.cv` is Python
only — the Stata command rejects it rather than substituting another
bandwidth.

### Licence

GPL-3.0-or-later, with the additional attribution term under section 7(b)
carried from upstream: the attribution to Rouven E. Haschka must be preserved
in anything conveying this material. Both licence files ship inside the Python
distribution.

[Unreleased]: https://github.com/girishm77/copulaendog_stata_python/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/girishm77/copulaendog_stata_python/releases/tag/v0.1.0
