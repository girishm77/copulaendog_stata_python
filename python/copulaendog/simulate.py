"""A simulated dataset for testing and demonstration.

The design is the one the estimators are built for: the endogenous regressor is
strictly monotone in a normal latent variable, so the Gaussian copula between
it and the structural error holds exactly, while its marginal distribution is
lognormal and therefore strongly non-normal -- which is what identifies the
correction (Park & Gupta 2012; Breitung, Mayer & Wied 2024, Theorem 2.1).

    (e, v)  ~ bivariate normal with corr(e, v) = rho
    w, x    ~ N(0, 1), independent of both
    P       = exp(a * v + g * w)
    y       = b0 + b1 * P + b2 * x + b3 * w + sigma * e

Because P is monotone in a*v + g*w, the copula data P* = Phi^-1(F(P)) equals
(a*v + g*w) / sd(.).  Two consequences make this dataset useful:

    corr(P*, e) != 0   P is endogenous, so OLS is biased
    corr(W, P*) != 0   whenever g != 0, so Assumption 5 of Park & Gupta fails
                       and PG stays biased while 2sCOPE, IMA and BMW do not

Set g=0 to get the Park & Gupta case where all of them are consistent.
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

import numpy as np
import pandas as pd


def sim_endog(
    n: int = 2000,
    rho: float = 0.6,
    a: float = 0.8,
    g: float = 0.6,
    beta=(1.0, 2.0, 1.5, -0.5),
    sigma: float = 1.0,
    seed: int | None = 123,
) -> pd.DataFrame:
    """Simulate y, an endogenous regressor P, and exogenous x and w.

    Parameters
    ----------
    n      sample size
    rho    correlation between the structural error and the latent driver of P
    a      loading of P on that latent driver
    g      loading of P on the exogenous regressor w; g=0 satisfies Assumption 5
    beta   (intercept, coefficient on P, on x, on w)
    sigma  standard deviation of the structural error
    seed   RNG seed

    Returns a DataFrame with columns y, P, x, w.  The true coefficient on P is
    beta[1].
    """
    rng = np.random.default_rng(seed)
    b0, b1, b2, b3 = beta

    cov = np.array([[1.0, rho], [rho, 1.0]])
    ev = rng.multivariate_normal([0.0, 0.0], cov, size=n)
    e, v = ev[:, 0], ev[:, 1]

    w = rng.standard_normal(n)
    x = rng.standard_normal(n)

    P = np.exp(a * v + g * w)
    y = b0 + b1 * P + b2 * x + b3 * w + sigma * e

    return pd.DataFrame({"y": y, "P": P, "x": x, "w": w})
