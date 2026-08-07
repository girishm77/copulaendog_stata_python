"""copulaendog -- instrument-free copula corrections for endogenous regressors.

A Python port of the R reference implementation

    Haschka, R. E. (2026). Copula-based endogeneity corrections in R.
    https://github.com/HashtagHaschka/Copula-based-endogeneity-corrections

and of the packaged version of it,

    Malshe, A. endogCopula. https://github.com/ashgreat/endogCopula

Five cross-sectional estimators behind one interface:

    CopRegPG        Park & Gupta (2012)
    CopReg2sCOPE    Yang, Qian & Xie (2025)
    CopRegIMA       Haschka (2025a)
    CopRegBMW       Breitung, Mayer & Wied (2024)
    CopRegJAMS      Liengaard et al. (2025)

Quick start
-----------
    import pandas as pd
    from copulaendog import copreg

    fit = copreg("y ~ z_endog | x_exog + w_instr", df, method="2scope",
                 nboots=199, seed=1)
    print(fit.summary())
    print(fit.validity())

Not ported: 2sCOPE-np (Hu, Qian & Xie 2025), PANEL (Haschka 2022) and BAYES
(Haschka 2025b).  Use the R implementation for those.
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


from .cdf import (
    CDF_CHOICES,
    TIES_CHOICES,
    cdf_estimate,
    copula_transform,
    copula_transform_matrix,
)
from .core import CopregResult, Validity
from .estimators import (
    CopReg2sCOPE,
    CopRegBMW,
    CopRegIMA,
    CopRegJAMS,
    CopRegPG,
    copreg,
)
from .formula import build_model
from .simulate import sim_endog

__all__ = [
    "copreg",
    "CopRegPG",
    "CopReg2sCOPE",
    "CopRegIMA",
    "CopRegBMW",
    "CopRegJAMS",
    "CopregResult",
    "Validity",
    "cdf_estimate",
    "copula_transform",
    "copula_transform_matrix",
    "build_model",
    "sim_endog",
    "CDF_CHOICES",
    "TIES_CHOICES",
]

__version__ = "0.1.0"
