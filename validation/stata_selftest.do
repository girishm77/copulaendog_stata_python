*! Self-test for the Stata port.
*!
*! Run this from the validation/ directory.  It checks that
*!   (a) the command compiles and runs for all five estimators,
*!   (b) the marginal CDFs agree with reference_R_cdf.csv to 1e-8,
*!   (c) the point estimates and rho agree with reference_R.csv to 1e-8.
*!
*! Anything printed as MISMATCH or FAILED is a bug; a run that ends with
*! "ALL CHECKS PASSED" is clean.
*!
*! Note on asdouble: import delimited stores numeric columns as float unless
*! told otherwise, and float carries about seven significant digits.  That
*! alone puts a ~1e-8 floor under every comparison below, which is exactly the
*! tolerance being tested, so every import here is asdouble.

clear all
set more off
adopath ++ "../stata"

local TOL = 1e-8
local nfail = 0

* ---------------------------------------------------------------- (a) runs --
import delimited using "simdata.csv", clear varnames(1) asdouble

di as txt _n "{hline 70}" _n "(a) does each estimator run?" _n "{hline 70}"

foreach m in pg 2scope ima bmw jams {
    di as txt _n ">>> method(`m')"
    capture noisily copulaendog y p, exog(x w) method(`m') nboots(20) seed(1)
    if _rc {
        di as err "   FAILED with rc = " _rc
        local ++nfail
    }
}

* cdf(kde.plugin) is the one estimator whose code path depends on the sample
* size: ce_cdf_gauss() and ce_kfe() evaluate the kernel exactly at or below
* ce_exactmax() = 1500 observations and bin it on a grid above.  The exact path
* is the one that was broken, and everything else in this file runs at n = 800,
* so the gate is crossed here from both sides.  Any n that fails but whose
* neighbour passes points at one of those two branches.
di as txt _n ">>> cdf(kde.plugin) either side of ce_exactmax() = 1500"
foreach n in 400 1499 1500 1501 2500 {
    preserve
    quietly {
        clear
        set obs `n'
        set seed 42
        generate double w = rnormal()
        generate double x = rnormal()
        generate double p = exp(0.8*rnormal() + 0.6*w)
        generate double y = 1 + 2*p + 1.5*x - 0.5*w + rnormal()
    }
    capture quietly copulaendog y p, exog(x w) method(pg) cdf(kde.plugin) ///
        nboots(5) seed(1)
    if _rc {
        di as err "   n = `n': FAILED with rc = " _rc
        local ++nfail
    }
    else di as txt "   n = `n': ok, b[p] = " %7.4f _b[p]
    restore
}

di as txt _n ">>> validity report"
capture noisily copulaendog y p, exog(x w) method(pg) nboots(20) seed(1) validity
if _rc {
    di as err "   FAILED with rc = " _rc
    local ++nfail
}

di as txt _n ">>> factor variable in exog(), JAMS cells"
capture noisily copulaendog y p, exog(x w i.g) method(jams) conditional(g) ///
    nboots(20) seed(1)
if _rc {
    di as err "   FAILED with rc = " _rc
    local ++nfail
}

di as txt _n ">>> interaction in exog()"
capture noisily copulaendog y p, exog(x w c.x#c.w) method(pg) nboots(20) seed(1)
if _rc {
    di as err "   FAILED with rc = " _rc
    local ++nfail
}

di as txt _n ">>> two endogenous regressors"
quietly generate double p2 = exp(0.4*rnormal() + 0.3*w)
quietly replace y = y + 0.5*p2
capture noisily copulaendog y p p2, exog(x w) method(2scope) nboots(20) seed(1)
if _rc {
    di as err "   FAILED with rc = " _rc
    local ++nfail
}

di as txt _n ">>> generate() and predict"
quietly copulaendog y p, exog(x w) method(pg) nboots(20) seed(1) generate(cop)
capture noisily predict yhat
capture noisily predict xi, residuals
capture noisily predict yhata, xba
summarize yhat xi yhata

* -------------------------------------------- (b) copula transformations ----
* The mata functions live inside the ado file and are not visible at the top
* level, so ce_cdf() cannot be called directly from here.  For PG the copula
* term is exactly invnormal(F(P)), so generate() plus normal() recovers the
* marginal CDF through the public interface instead.
di as txt _n "{hline 70}" _n "(b) marginal CDFs against the R reference" ///
    _n "{hline 70}"

import delimited using "reference_R_cdf.csv", clear varnames(1) asdouble
tempfile refcdf
quietly save `refcdf'

import delimited using "simdata.csv", clear varnames(1) asdouble
quietly merge 1:1 _n using `refcdf', nogenerate

foreach pair in "kdesilverman kde.silverman" "kdeplugin kde.plugin"  ///
                "ecdffixed ecdf.fixed"       "ecdfadj ecdf.adj"      ///
                "rankn rank.n"               "rankn1 rank.n1" {
    local col     : word 1 of `pair'
    local cdfname : word 2 of `pair'
    capture confirm variable `col'
    if _rc continue
    capture drop p_cop
    capture quietly copulaendog y p, exog(x w) method(pg) cdf(`cdfname') ///
        nboots(2) seed(1) generate(c)
    if _rc {
        di as err "  `cdfname': FAILED with rc = " _rc
        local ++nfail
        continue
    }
    quietly generate double _d = abs(normal(p_cop) - `col')
    quietly summarize _d, meanonly
    if r(max) < `TOL' di as txt "  `cdfname': max diff = " as res %9.2e r(max)
    else {
        di as err "  `cdfname': MISMATCH, max diff = " %9.2e r(max)
        local ++nfail
    }
    drop _d
}

* --------------------------------------------- (c) coefficients vs R ---------
di as txt _n "{hline 70}" _n "(c) point estimates against the R reference" ///
    _n "{hline 70}"

* Load reference_R.csv into locals keyed by model and term.  strtoname() turns
* "(Intercept)" and "P_cop.g0" into legal local names.
import delimited using "reference_R.csv", clear varnames(1) asdouble ///
    stringcols(1 2)
local nref = _N
forvalues i = 1/`nref' {
    local rm = model[`i']
    local rt = term[`i']
    local R_`rm'_`=strtoname("`rt'")' = estimate[`i']
    local T_`rm' `T_`rm'' `rt'
}

* R term name -> the matching Stata coefficient
program define ce_sval, rclass
    args t
    if "`t'" == "rho.P" {
        tempname R
        matrix `R' = e(rho)
        return scalar v = el(`R', 1, 1)
        exit
    }
    local s "`t'"
    if      "`s'" == "(Intercept)" local s "_cons"
    else if "`s'" == "P"           local s "p"
    else if "`s'" == "P_cop"       local s "p_cop"
    else if "`s'" == "g1"          local s "1.g"
    else if "`s'" == "g2"          local s "2.g"
    else if "`s'" == "x:w"         local s "c.x#c.w"
    else if "`s'" == "P_cop.g0"    local s "p_cop1"
    else if "`s'" == "P_cop.g1"    local s "p_cop2"
    else if "`s'" == "P_cop.g2"    local s "p_cop3"
    return scalar v = _b[`s']
end

import delimited using "simdata.csv", clear varnames(1) asdouble

* the ten specifications of reference_R.R, in its own naming
foreach spec in "pg|exog(x w i.g) method(pg)"                   ///
                "s2cope|exog(x w i.g) method(2scope)"           ///
                "ima|exog(x w i.g) method(ima)"                 ///
                "bmw|exog(x w i.g) method(bmw)"                 ///
                "jams|exog(x w i.g) method(jams) conditional(g)" ///
                "jams_f|exog(x w i.g) method(jams)"             ///
                "pg_ecdf|exog(x w) method(pg) cdf(ecdf.adj)"    ///
                "pg_rank|exog(x w) method(pg) cdf(rank.n)"      ///
                "pg_plug|exog(x w) method(pg) cdf(kde.plugin)"  ///
                "pg_int|exog(x w c.x#c.w) method(pg)" {

    local m    = substr("`spec'", 1, strpos("`spec'", "|") - 1)
    local opts = substr("`spec'", strpos("`spec'", "|") + 1, .)

    capture quietly copulaendog y p, `opts' nboots(5) seed(1)
    if _rc {
        di as err "  `m': FAILED with rc = " _rc
        local ++nfail
        continue
    }

    local worst = 0
    local wterm ""
    foreach t of local T_`m' {
        capture ce_sval "`t'"
        if _rc {
            di as err "  `m': could not read the Stata coefficient for `t'"
            local ++nfail
            continue
        }
        local d = abs(`R_`m'_`=strtoname("`t'")'' - r(v))
        if `d' > `worst' {
            local worst = `d'
            local wterm "`t'"
        }
    }
    local nt : word count `T_`m''
    if `worst' < `TOL' {
        di as txt "  `m': `nt' terms, max diff = " as res %9.2e `worst'
    }
    else {
        di as err "  `m': MISMATCH over `nt' terms, max diff = " ///
            %9.2e `worst' " at `wterm'"
        local ++nfail
    }
}

* ------------------------------------------------------------------ verdict --
di as txt _n "{hline 70}"
if `nfail' == 0 di as res "ALL CHECKS PASSED"
else            di as err "`nfail' CHECK(S) FAILED"
di as txt "{hline 70}"
