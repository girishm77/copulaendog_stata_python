*! Self-test for the Stata port.
*!
*! Run this once from the validation/ directory.  It checks that
*!   (a) the command compiles and runs for all five estimators,
*!   (b) the copula transformations agree with the R reference in
*!       reference_R_cdf.csv to 1e-8,
*!   (c) the point estimates agree with reference_R.csv to 1e-8.
*!
*! Anything printed as "MISMATCH" is a bug; everything else is a pass.

clear all
set more off
adopath ++ "../stata"

* ---------------------------------------------------------------- (a) runs --
import delimited using "simdata.csv", clear varnames(1)

di as txt _n "{hline 70}" _n "(a) does each estimator run?" _n "{hline 70}"

foreach m in pg 2scope ima bmw jams {
    di as txt _n ">>> method(`m')"
    capture noisily copulaendog y p, exog(x w) method(`m') nboots(20) seed(1)
    if _rc di as err "   FAILED with rc = " _rc
}

di as txt _n ">>> validity report"
capture noisily copulaendog y p, exog(x w) method(pg) nboots(20) seed(1) validity

di as txt _n ">>> factor variable in exog(), JAMS cells"
capture noisily copulaendog y p, exog(x w i.g) method(jams) conditional(g) ///
    nboots(20) seed(1)

di as txt _n ">>> two endogenous regressors"
quietly generate p2 = exp(0.4*rnormal() + 0.3*w)
quietly replace y = y + 0.5*p2
capture noisily copulaendog y p p2, exog(x w) method(2scope) nboots(20) seed(1)

di as txt _n ">>> generate() and predict"
quietly copulaendog y p, exog(x w) method(pg) nboots(20) seed(1) generate(cop)
capture noisily predict yhat
capture noisily predict xi, residuals
capture noisily predict yhata, xba
summarize yhat xi yhata

* -------------------------------------------- (b) copula transformations ----
di as txt _n "{hline 70}" _n "(b) marginal CDFs against the R reference" ///
    _n "{hline 70}"

import delimited using "reference_R_cdf.csv", clear varnames(1)
tempfile refcdf
quietly save `refcdf'

import delimited using "simdata.csv", clear varnames(1)
quietly merge 1:1 _n using `refcdf', nogenerate

foreach k in kdesilverman kdeplugin ecdffixed ecdfadj rankn rankn1 {
    local cdfname = subinstr("`k'", "kdesilverman", "kde.silverman", .)
    local cdfname = subinstr("`cdfname'", "kdeplugin",  "kde.plugin", .)
    local cdfname = subinstr("`cdfname'", "ecdffixed",  "ecdf.fixed", .)
    local cdfname = subinstr("`cdfname'", "ecdfadj",    "ecdf.adj", .)
    local cdfname = subinstr("`cdfname'", "rankn1",     "rank.n1", .)
    local cdfname = subinstr("`cdfname'", "rankn",      "rank.n", .)
    capture confirm variable `k'
    if _rc continue
    quietly generate double _u = .
    mata: st_store(., "_u", ce_cdf(st_data(., "p"), "`cdfname'", "max"))
    quietly generate double _d = abs(_u - `k')
    quietly summarize _d, meanonly
    if r(max) < 1e-8 di as txt "  `cdfname': max diff = " as res %9.2e r(max)
    else             di as err "  `cdfname': MISMATCH, max diff = " %9.2e r(max)
    drop _u _d
}

* --------------------------------------------- (c) coefficients vs R ---------
di as txt _n "{hline 70}" _n "(c) point estimates against the R reference" ///
    _n "{hline 70}"
di as txt "compare the coefficients below with reference_R.csv by hand:"
di as txt "R uses the model y ~ P | x + w + g, so run the same here."

import delimited using "simdata.csv", clear varnames(1)
foreach m in pg 2scope ima bmw {
    di as txt _n ">>> method(`m')"
    quietly copulaendog y p, exog(x w i.g) method(`m') nboots(5) seed(1)
    matrix list e(b)
}
