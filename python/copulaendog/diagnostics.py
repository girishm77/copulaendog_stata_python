"""Normality tests, model diagnostics and the validity check.

Ported from sections 4 and 8 of Copreg_core.R.
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
from scipy import stats


# ---------------------------------------------------------------------------
# normality tests
# ---------------------------------------------------------------------------
def ad_test(x: np.ndarray):
    """Anderson-Darling test for composite normality (D'Agostino & Stephens 1986).

    Returns (A, p).
    """
    x = np.sort(np.asarray(x, dtype=float)[np.isfinite(x)])
    n = x.size
    if n < 8:
        return np.nan, np.nan
    p = stats.norm.cdf((x - x.mean()) / np.std(x, ddof=1))
    p = np.clip(p, 1e-15, 1 - 1e-15)
    h = (2 * np.arange(1, n + 1) - 1) * (np.log(p) + np.log(1 - p[::-1]))
    A = -n - h.mean()
    AA = A * (1 + 0.75 / n + 2.25 / n**2)
    if AA < 0.2:
        pv = 1 - np.exp(-13.436 + 101.14 * AA - 223.73 * AA**2)
    elif AA < 0.34:
        pv = 1 - np.exp(-8.318 + 42.796 * AA - 59.938 * AA**2)
    elif AA < 0.6:
        pv = np.exp(0.9177 - 4.279 * AA - 1.38 * AA**2)
    elif AA < 10:
        pv = np.exp(1.2937 - 5.709 * AA + 0.0186 * AA**2)
    else:
        pv = 3.7e-24
    return float(A), float(pv)


def cvm_test(x: np.ndarray):
    """Cramer-von Mises test for composite normality (Stephens 1986)."""
    x = np.sort(np.asarray(x, dtype=float)[np.isfinite(x)])
    n = x.size
    if n < 8:
        return np.nan, np.nan
    p = stats.norm.cdf((x - x.mean()) / np.std(x, ddof=1))
    W = 1 / (12 * n) + np.sum((p - (2 * np.arange(1, n + 1) - 1) / (2 * n)) ** 2)
    WW = W * (1 + 0.5 / n)
    if WW < 0.0275:
        pv = 1 - np.exp(-13.953 + 775.5 * WW - 12542.61 * WW**2)
    elif WW < 0.051:
        pv = 1 - np.exp(-5.903 + 179.546 * WW - 1515.29 * WW**2)
    elif WW < 0.092:
        pv = np.exp(0.886 - 31.62 * WW + 10.897 * WW**2)
    elif WW < 1.1:
        pv = np.exp(1.111 - 34.242 * WW + 12.832 * WW**2)
    else:
        pv = 7.37e-10
    return float(W), float(pv)


def ks_normal(v: np.ndarray):
    """Kolmogorov-Smirnov test against a fitted normal.  Returns (D, p).

    The sample is standardised and compared with the standard normal rather
    than passing the fitted mean and standard deviation through kstest's
    args=.  The two are the same test -- the empirical CDF of (v - m)/s
    against Phi is the empirical CDF of v against Phi((. - m)/s) -- but scipy
    1.18 broke the args= form for the named "norm" distribution, and this
    form works on every version.
    """
    v = np.asarray(v, dtype=float)
    s = np.std(v, ddof=1)
    if not np.isfinite(s) or s <= 0:
        return np.nan, np.nan
    k = stats.kstest((v - v.mean()) / s, "norm")
    return float(k.statistic), float(k.pvalue)


def skewness(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    m = x - x.mean()
    return float((np.sum(m**3) / x.size) / (np.sum(m**2) / x.size) ** 1.5)


def ex_kurtosis(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    m = x - x.mean()
    return float((np.sum(m**4) / x.size) / (np.sum(m**2) / x.size) ** 2 - 3)


def fisher_z_p(r: float, n: int) -> float:
    """Fisher z test for a single correlation."""
    if not np.isfinite(r) or n < 4:
        return np.nan
    r = min(max(r, -1 + 1e-12), 1 - 1e-12)
    z = np.arctanh(r) * np.sqrt(n - 3)
    return float(2 * stats.norm.cdf(-abs(z)))


def holm(p: np.ndarray) -> np.ndarray:
    """Holm step-down adjustment, matching R's p.adjust(method='holm')."""
    p = np.asarray(p, dtype=float)
    n = p.size
    o = np.argsort(p)
    adj = np.maximum.accumulate((n - np.arange(n)) * p[o])
    out = np.empty(n)
    out[o] = np.minimum(adj, 1.0)
    return out


# ---------------------------------------------------------------------------
# model diagnostics
# ---------------------------------------------------------------------------
def _ols(D, v):
    coef, *_ = np.linalg.lstsq(D, v, rcond=None)
    resid = v - D @ coef
    return coef, resid, np.linalg.matrix_rank(D)


def model_diagnostics(X, C, Cstar, info, groups=None, cnames=None) -> dict:
    """(a) non-normality of P, (b) corr(W, P*), (c) collinearity of the copula
    terms with the rest of the design."""
    n = X.shape[0]
    ec = info.endo_cols
    nm = info.endo_names
    K = len(nm)

    # (a) non-normality of the endogenous regressors
    rows = []
    for k in range(K):
        v = X[:, ec[k]]
        A, ap = ad_test(v)
        _, kp = ks_normal(v)
        rows.append({"AD": A, "AD p": ap, "KS p": kp})
    nonnorm = pd.DataFrame(rows, index=nm)

    # (b) correlation of the exogenous regressors with the copula data.
    # Assumption 5 of Park & Gupta requires corr(W, P*) = 0.  Reported are the
    # largest single correlation with a Holm-adjusted p value -- it is a maximum
    # over many regressors, so the raw p value would be far too small -- and a
    # joint test over the whole set, which is what the assumption is about.
    drop = {0} if info.has_intercept else set()
    wcols = [j for j in info.exo_cols
             if j not in drop and np.unique(X[:, j]).size > 1]

    corrW = corrM = None
    if wcols:
        W = X[:, wcols]
        wnames = [info.xnames[j] for j in wcols]
        corrM = pd.DataFrame(
            np.array([[_corr(W[:, m], Cstar[:, k]) for m in range(W.shape[1])]
                      for k in range(K)]),
            index=nm, columns=wnames,
        )
        Wd = np.column_stack([np.ones(n), W])
        rows = []
        for k in range(K):
            r = corrM.values[k]
            pa = holm(np.array([fisher_z_p(v, n) for v in r]))
            i = int(np.nanargmax(np.abs(r)))
            _, res, rank = _ols(Wd, Cstar[:, k])
            df1, df2 = rank - 1, n - rank
            if df1 >= 1 and df2 >= 1:
                r2 = 1 - np.sum(res**2) / np.sum(
                    (Cstar[:, k] - Cstar[:, k].mean()) ** 2)
                Fs = (r2 / df1) / ((1 - r2) / df2)
                pj = float(stats.f.sf(Fs, df1, df2))
            else:
                r2 = pj = np.nan
            rows.append({"max |corr|": r[i], "with": wnames[i],
                         "p (Holm)": pa[i], "joint R2": r2, "joint p": pj})
        corrW = pd.DataFrame(rows, index=nm)

    # (c) collinearity: omega = 1 - R^2 of the copula term on all other
    # regressors.  For the bivariate model this reduces to 1 - corr(P, C(P))^2,
    # the bias inflation factor of Liengaard et al. (2025, Eq. 4).
    A = np.column_stack([X, C])
    kC = X.shape[1] + np.arange(C.shape[1])
    src = ([nm.index(g[0]) for g in groups] if groups is not None
           else list(range(C.shape[1])))

    omega, rPC = np.empty(C.shape[1]), np.empty(C.shape[1])
    for k in range(C.shape[1]):
        rest = np.delete(A, kC[k], axis=1)
        _, res, _ = _ols(rest, A[:, kC[k]])
        r2 = 1 - np.sum(res**2) / np.sum((A[:, kC[k]] - A[:, kC[k]].mean()) ** 2)
        omega[k] = 1 - r2
        # a JAMS column is zero outside its own cell, so the correlation is
        # taken over the rows where the column is supported
        sup = np.flatnonzero(C[:, k] != 0) if C.shape[1] > K else np.arange(C.shape[0])
        rPC[k] = _corr(X[sup, ec[src[k]]], C[sup, k]) if sup.size > 2 else np.nan

    coll = pd.DataFrame({"corr(P, C)": rPC, "omega": omega},
                        index=cnames if cnames is not None
                        else [f"C{k}" for k in range(C.shape[1])])

    return {"nonnormality": nonnorm, "exog.correlation": corrW,
            "exog.correlation.matrix": corrM, "collinearity": coll}


def _corr(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


# ---------------------------------------------------------------------------
# validity
# ---------------------------------------------------------------------------
def becker_thresholds(n: int, power: float) -> dict:
    """Becker, Proksch & Ringle (2022, Fig. 8) boundary conditions.

    Their C5.0 decision tree on Study 4, for a target power of the copula term.
    """
    if power not in (0.8, 0.9):
        raise ValueError(
            "'power' must be 0.8 or 0.9; Becker et al. report thresholds for "
            "these two levels only."
        )
    if power == 0.8:
        sk = np.inf if n <= 200 else 1.932 if n <= 1000 else 0.774 if n <= 2000 else 0.0
        return {"skewness": sk, "AD": 18.964, "CvM": 3.488}
    sk = np.inf if n <= 600 else 1.974 if n <= 2000 else 0.998
    return {
        "skewness": sk,
        "AD": 67.875 if n <= 2000 else 46.832,
        "CvM": 12.246 if n <= 2000 else 7.994,
    }


def nonnormality_table(M: np.ndarray, cols, names, n: int, power: float):
    th = becker_thresholds(n, power)
    rows = []
    for j, nm in zip(cols, names):
        v = M[:, j]
        sk = skewness(v)
        A, _ = ad_test(v)
        W, _ = cvm_test(v)
        _, kp = ks_normal(v)
        rows.append({
            "skewness": round(sk, 3),
            "ex.kurtosis": round(ex_kurtosis(v), 3),
            "AD": round(A, 3),
            "CvM": round(W, 3),
            "KS p": kp,
            "Yang ok": kp < 0.05,
            "Becker ok": (abs(sk) >= th["skewness"]) or (A > th["AD"])
            or (W > th["CvM"]),
        })
    return pd.DataFrame(rows, index=list(names)), th
