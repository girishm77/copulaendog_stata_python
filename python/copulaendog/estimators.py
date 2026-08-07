"""The five cross-sectional estimators, and the generic copreg() entry point.

    CopRegPG        Park & Gupta (2012)
    CopReg2sCOPE    Yang, Qian & Xie (2025)
    CopRegIMA       Haschka (2025a)
    CopRegBMW       Breitung, Mayer & Wied (2024)
    CopRegJAMS      Liengaard et al. (2025)

All five share the same arguments:

    formula   "y ~ endog_1 + endog_2 + ... | exog_1 + exog_2 + ..."
    data      a pandas DataFrame
    cdf       how the marginal CDF is estimated; the default follows each
              estimator's own paper
    ties      "max" for the counting function, "average" for midranks
    nboots    bootstrap replicates for the standard errors
    subset    boolean mask, as in lm()
    seed      seed for the bootstrap; None leaves the draws unseeded
    verbose   emit the warnings about ties, thin levels and dropped columns
"""
#
# ---------------------------------------------------------------------------
# Ported from the R reference implementation
#
#   Copula-based endogeneity corrections in R
#   Copyright (C) 2026 Rouven E. Haschka
#   ORCID: https://orcid.org/0000-0002-2916-9745
#   https://github.com/HashtagHaschka/Copula-based-endogeneity-corrections
#
# and from the packaged version of it, endogCopula by Ashwin Malshe,
#   https://github.com/ashgreat/endogCopula
#
# This file is free software under the GNU General Public License v3 or later,
# with the additional term under section 7(b) stated by the upstream author:
# the author attribution above must be preserved in any material conveying
# this code, modified versions and larger works included.  See LICENSE and
# LICENSE.ADDITIONAL-TERMS.md.
# ---------------------------------------------------------------------------


from __future__ import annotations

import warnings

from .constructors import (
    BMWConstructor,
    JAMSConstructor,
    PGConstructor,
    TwoStageConstructor,
)
from .core import CopregResult, copreg_fit


def _redirect_to_pg(name, formula, data, cdf, ties, nboots, subset, seed,
                    verbose, default_cdf):
    warnings.warn(
        f"{name} without exogenous regressors is identical to Park & Gupta "
        "(the first stage has nothing to project on). Redirecting to "
        f'CopRegPG() with cdf = "{default_cdf}".',
        stacklevel=3,
    )
    return CopRegPG(formula, data, cdf=default_cdf, ties=ties, nboots=nboots,
                    subset=subset, seed=seed, verbose=verbose)


def _has_exog(formula: str) -> bool:
    parts = formula.split("~", 1)[1].split("|")
    return len(parts) > 1 and bool(parts[1].strip())


# ---------------------------------------------------------------------------
def CopRegPG(formula, data, cdf="kde.silverman", ties="max", nboots=199,
             subset=None, seed=None, verbose=True) -> CopregResult:
    """Park & Gupta (2012).

    The original.  Transforms each endogenous regressor to a normal score,
    C = Phi^-1(Fhat(P)), and adds it to the regression.  Assumes the endogenous
    and exogenous regressors are uncorrelated; summary() and validity() test
    that assumption, because violating it is what motivated the later
    estimators.
    """
    return copreg_fit(
        formula, data, ctor=PGConstructor(),
        method="PG (Park & Gupta 2012)",
        cdf=cdf, ties=ties, nboots=nboots, subset=subset, seed=seed,
        assumption5=True, verbose=verbose,
    )


def CopReg2sCOPE(formula, data, cdf="rank.n", ties="max", nboots=199,
                 subset=None, seed=None, verbose=True) -> CopregResult:
    """Yang, Qian & Xie (2025).

    Relaxes the uncorrelatedness assumption.  Transforms the exogenous
    regressors as well, runs a first-stage regression of P* on W* and uses its
    residual as the copula term, so the correction is orthogonal to W by
    construction.  The first stage carries an intercept, following note 8 of
    Qian, Koschmann & Xie (2025).
    """
    if not _has_exog(formula):
        return _redirect_to_pg("2sCOPE", formula, data, cdf, ties, nboots,
                               subset, seed, verbose, "kde.silverman")
    return copreg_fit(
        formula, data, ctor=TwoStageConstructor(intercept=True, verbose=verbose),
        method="2sCOPE (Yang, Qian & Xie 2025)",
        cdf=cdf, ties=ties, nboots=nboots, subset=subset, seed=seed,
        assumption5=False, verbose=verbose,
    )


def CopRegIMA(formula, data, cdf="rank.n", ties="max", nboots=199,
              subset=None, seed=None, verbose=True) -> CopregResult:
    """Haschka (2025a).

    2sCOPE without an intercept in the first stage, which is what the paper's
    derivation of rho requires: the slope of P* on W* is then the Pearson
    correlation between the two.  The two are usually very close; the gap
    widens the further the transformed exogenous regressors sit from mean zero.
    """
    if not _has_exog(formula):
        return _redirect_to_pg("IMA", formula, data, cdf, ties, nboots,
                               subset, seed, verbose, "kde.silverman")
    return copreg_fit(
        formula, data, ctor=TwoStageConstructor(intercept=False, verbose=verbose),
        method="IMA (Haschka 2025a)",
        cdf=cdf, ties=ties, nboots=nboots, subset=subset, seed=seed,
        assumption5=False, verbose=verbose,
    )


def CopRegBMW(formula, data, cdf="rank.n1", ties="max", nboots=199,
              subset=None, seed=None, verbose=True) -> CopregResult:
    """Breitung, Mayer & Wied (2024).

    Inverts the order: the first stage runs on the raw variables and the rank
    transform is applied to its residuals.  Two consequences show in the
    output.  The identification requirement falls on the first-stage residuals
    rather than on P, so validity() tests those; and their Corollary 3.2 makes
    the textbook t statistic valid for testing rho = 0, so summary() reports a
    Durbin-Hausman-Wu test next to the bootstrap one.

    cdf is not a free choice here: the asymptotic theory of Proposition 3.1 is
    derived for Rank/(n+1), so anything else warns.
    """
    if not _has_exog(formula):
        return _redirect_to_pg("BMW", formula, data, cdf, ties, nboots,
                               subset, seed, verbose, "kde.silverman")
    if cdf != "rank.n1":
        warnings.warn(
            "Proposition 3.1 of Breitung, Mayer & Wied (2024) is derived for "
            'the rank transformation of their Equation 2.3, cdf = "rank.n1". '
            f'With cdf = "{cdf}" the point estimates remain sensible but the '
            "reported standard errors are no longer covered by the published "
            "theory.",
            stacklevel=2,
        )
    return copreg_fit(
        formula, data, ctor=BMWConstructor(),
        method="BMW (Breitung, Mayer & Wied 2024)",
        cdf=cdf, ties=ties, nboots=nboots, subset=subset, seed=seed,
        assumption5=False, dhw=True, verbose=verbose,
    )


def CopRegJAMS(formula, data, cdf="ecdf.adj", ties="max", nboots=199,
               conditional=True, subset=None, seed=None,
               verbose=True) -> CopregResult:
    """Liengaard et al. (2025).

    Lets the copula structure differ across the categories of the discrete
    exogenous regressors.

    conditional=True   the structure is estimated separately per joint category
                       of the categorical regressors in the exogenous part
                       (Equations 20 and 21)
    conditional=False  one common structure (Equation 18)
    a list of names    the variables whose categories the structure may vary over
    """
    return copreg_fit(
        formula, data,
        ctor=JAMSConstructor(conditional=conditional, verbose=verbose),
        method="JAMS (Liengaard et al. 2025)",
        cdf=cdf, ties=ties, nboots=nboots, subset=subset, seed=seed,
        assumption5=False, verbose=verbose,
    )


# ---------------------------------------------------------------------------
_REGISTRY = {
    "pg": CopRegPG,
    "2scope": CopReg2sCOPE,
    "ima": CopRegIMA,
    "bmw": CopRegBMW,
    "jams": CopRegJAMS,
}


def copreg(formula, data, method="pg", **kwargs) -> CopregResult:
    """Generic entry point.

    Only dispatches to the estimator's own function, so both routes behave
    identically down to the warnings.  method is one of "pg", "2scope", "ima",
    "bmw", "jams".
    """
    method = str(method).lower()
    if method not in _REGISTRY:
        raise ValueError(
            f"Estimator {method!r} is not implemented. Choose one of "
            f"{sorted(_REGISTRY)}."
        )
    return _REGISTRY[method](formula, data, **kwargs)
