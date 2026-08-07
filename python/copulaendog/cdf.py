"""Marginal CDF estimators for the copula transformation C(x) = Phi^-1(Fhat(x)).

Ported from Copreg_core.R (Haschka 2026) / endogCopula (Malshe).  Seven
estimators, one dispatcher.  Every function returns u in (0, 1) so that
qnorm(u) is finite.

    "kde.silverman"  Epanechnikov kernel CDF, Silverman bandwidth
                     Park & Gupta (2012), Eq. 3
    "kde.cv"         Epanechnikov kernel CDF, least-squares CV bandwidth
                     Li, Li & Racine (2017)
    "kde.plugin"     Gaussian kernel CDF, Polansky & Baker (2000) plug-in
    "ecdf.fixed"     ECDF with replaced boundary, Becker et al. (2022)
    "ecdf.adj"       adjusted ECDF, Liengaard et al. (2025), Eq. 9
    "rank.n"         Rank/n with a top correction, Qian, Koschmann & Xie (2025)
    "rank.n1"        Rank/(n+1), Breitung, Mayer & Wied (2024), Eq. 2.3
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
from scipy import stats

CDF_CHOICES = (
    "kde.silverman",
    "kde.cv",
    "kde.plugin",
    "ecdf.fixed",
    "ecdf.adj",
    "rank.n",
    "rank.n1",
)

TIES_CHOICES = ("max", "average")


# ---------------------------------------------------------------------------
# ranks
# ---------------------------------------------------------------------------
def _counts_le(x: np.ndarray, ties: str = "max") -> np.ndarray:
    """Counting function.

    ties="max" reproduces F(x) = (1/n) sum I(X_i <= x) literally, which is how
    every one of the papers writes it.  ties="average" uses midranks, the
    convention of the wider copula literature.
    """
    method = "max" if ties == "max" else "average"
    return stats.rankdata(x, method=method).astype(float)


# ---------------------------------------------------------------------------
# bandwidths
# ---------------------------------------------------------------------------
def _sd(x: np.ndarray) -> float:
    """R's sd(): denominator n - 1."""
    return float(np.std(x, ddof=1))


def bw_silverman(x: np.ndarray) -> float:
    """Silverman's rule as used by Park & Gupta (2012, p. 571).

    b = 0.9 * n^(-1/5) * min(s, IQR/1.34)
    """
    n = x.size
    s = _sd(x)
    iq = float(np.subtract(*np.percentile(x, [75, 25]))) / 1.34
    sc = min(s, iq)
    if not np.isfinite(sc) or sc <= 0:
        sc = s if np.isfinite(s) and s > 0 else 1.0
    return 0.9 * n ** (-1 / 5) * sc


def _hermite(z: np.ndarray, r: int) -> np.ndarray:
    if r == 2:
        return z**2 - 1
    if r == 4:
        return z**4 - 6 * z**2 + 3
    if r == 6:
        return z**6 - 15 * z**4 + 45 * z**2 - 15
    raise ValueError(f"Hermite polynomial of order {r} not needed here.")


def _dnorm_deriv(x: np.ndarray, sigma: float, r: int) -> np.ndarray:
    """r-th derivative of the N(0, sigma^2) density."""
    z = np.asarray(x, dtype=float) / sigma
    return (-1.0) ** r * _hermite(z, r) * stats.norm.pdf(z) / sigma ** (r + 1)


def _psins(r: int, sigma: float) -> float:
    """Normal-scale reference value of psi_r."""
    from math import factorial, pi, sqrt

    return (-1.0) ** (r // 2) * factorial(r) / (
        (2 * sigma) ** (r + 1) * factorial(r // 2) * sqrt(pi)
    )


def _wbin(idx: np.ndarray, w: np.ndarray, M: int) -> np.ndarray:
    """Weighted bin counts on 0..M-1."""
    return np.bincount(idx, weights=w, minlength=M)[:M]


def _kfe(x: np.ndarray, g: float, r: int, exact_max: int = 1500, M: int = 4096) -> float:
    """Kernel functional estimate psi_r.

    Exact double sum for moderate n; above that a linear binning plus an FFT
    autocorrelation, which is what ks does internally and costs O(M log M).
    """
    n = x.size

    if n <= exact_max:
        s = 0.0
        for a in range(0, n, 1000):
            b = min(a + 1000, n)
            s += float(np.sum(_dnorm_deriv(x[a:b, None] - x[None, :], g, r)))
        return s / n**2

    lo0, hi0 = float(x.min()), float(x.max())
    if hi0 <= lo0:
        return float(_dnorm_deriv(np.array([0.0]), g, r)[0])
    delta = (hi0 - lo0) / (M - 1)

    idx = (x - lo0) / delta
    lo = np.clip(np.floor(idx).astype(int), 0, M - 1)
    w = idx - lo
    hi = np.minimum(lo + 1, M - 1)
    cnt = _wbin(np.concatenate([lo, hi]), np.concatenate([1 - w, w]), M)

    L = int(2 ** np.ceil(np.log2(2 * M)))
    f = np.fft.fft(np.concatenate([cnt, np.zeros(L - M)]))
    ac = np.real(np.fft.ifft(f * np.conj(f)))
    sl = ac[:M]

    k = _dnorm_deriv(np.arange(M) * delta, g, r)
    return float((k[0] * sl[0] + 2 * np.sum(k[1:] * sl[1:])) / n**2)


def bw_plugin(x: np.ndarray) -> float:
    """Polansky & Baker (2000) plug-in bandwidth for the distribution function.

    This is what ks::hpi.kcde computes.
    """
    n = x.size
    K2 = float(_dnorm_deriv(np.array([0.0]), 1.0, 2)[0])
    K4 = float(_dnorm_deriv(np.array([0.0]), 1.0, 4)[0])
    m1 = (4 * np.pi) ** -0.5
    sx = _sd(x)
    if not np.isfinite(sx) or sx <= 0:
        raise ValueError("Cannot choose a bandwidth for a constant variable.")
    psi6 = _psins(6, sx)
    g4 = (2 * K4 / (-psi6 * n)) ** (1 / 7)
    psi4 = _kfe(x, g4, 4)
    g2 = (2 * K2 / (-psi4 * n)) ** (1 / 5)
    psi2 = _kfe(x, g2, 2)
    return float((2 * m1 / (-psi2 * n)) ** (1 / 3))


def bw_cv(x: np.ndarray, n_grid: int = 40) -> float:
    """Least-squares cross-validated bandwidth for the CDF.

    Li, Li & Racine (2017).  Minimises the leave-one-out criterion

        CV(h) = sum_i sum_{j != i} [ I(X_j <= X_i) - G((X_i - X_j)/h) ]^2

    with G the integrated Epanechnikov kernel, over a log grid anchored on the
    Silverman bandwidth.  O(n^2) per evaluation, so this is by far the slowest
    of the seven and is not a sensible choice inside a large bootstrap.
    """
    n = x.size
    b0 = bw_silverman(x)
    grid = b0 * np.exp(np.linspace(np.log(0.15), np.log(6.0), n_grid))
    ind = (x[None, :] <= x[:, None]).astype(float)
    np.fill_diagonal(ind, 0.0)
    d = x[:, None] - x[None, :]
    eye = np.eye(n, dtype=bool)

    best, best_h = np.inf, b0
    for h in grid:
        G = _pepan(d / h)
        G[eye] = 0.0
        val = float(np.sum((ind - G) ** 2))
        if val < best:
            best, best_h = val, float(h)
    return best_h


# ---------------------------------------------------------------------------
# kernel CDFs
# ---------------------------------------------------------------------------
def _pepan(u: np.ndarray) -> np.ndarray:
    """CDF of the Epanechnikov kernel K(t) = 0.75 (1 - t^2) I(|t| <= 1)."""
    u = np.clip(u, -1.0, 1.0)
    return 0.75 * (u - u**3 / 3) + 0.5


def cdf_kde_epan(x: np.ndarray, b: float) -> np.ndarray:
    """F(x_j) = (1/n) sum_i G((x_j - x_i)/b) with G the Epanechnikov CDF.

    Park & Gupta integrate the density numerically; the Epanechnikov kernel
    integrates in closed form, so the exact antiderivative is used instead.
    Same estimator, no quadrature error, and O(n log n) rather than O(n^2):
    G is a cubic on its support, so the window sums come from prefix sums of
    x, x^2, x^3.
    """
    n = x.size
    if not np.isfinite(b) or b <= 0:
        raise ValueError("Non-positive bandwidth in kernel CDF estimation.")

    # affine equivariance: F_x(x_j; b) = F_z(z_j; b/s) for z = (x - m)/s
    m = float(np.mean(x))
    s = _sd(x)
    if not np.isfinite(s) or s <= 0:
        s = 1.0
    z = (x - m) / s
    bb = b / s

    o = np.argsort(z, kind="mergesort")
    zs = z[o]

    lo = np.searchsorted(zs, zs - bb, side="right")
    hi = np.searchsorted(zs, zs + bb, side="right")

    S1 = np.concatenate([[0.0], np.cumsum(zs)])
    S2 = np.concatenate([[0.0], np.cumsum(zs**2)])
    S3 = np.concatenate([[0.0], np.cumsum(zs**3)])

    m0 = (hi - lo).astype(float)
    m1 = S1[hi] - S1[lo]
    m2 = S2[hi] - S2[lo]
    m3 = S3[hi] - S3[lo]

    sd1 = (m0 * zs - m1) / bb
    sd3 = (m0 * zs**3 - 3 * zs**2 * m1 + 3 * zs * m2 - m3) / bb**3

    out = (lo + 0.75 * sd1 - 0.25 * sd3 + 0.5 * m0) / n

    res = np.empty(n)
    res[o] = out
    return np.clip(res, 0.0, 1.0)


def cdf_kde_gauss(
    x: np.ndarray, h: float, exact_max: int = 1500, M: int = 4096
) -> np.ndarray:
    """F(x_j) = n^-1 sum_i Phi((x_j - X_i)/h).

    Exact for moderate n, otherwise binned on a grid and interpolated back,
    the way ks::kcde and its predict method work.
    """
    n = x.size
    if not np.isfinite(h) or h <= 0:
        raise ValueError("Non-positive bandwidth in kernel CDF estimation.")

    if n <= exact_max:
        out = np.empty(n)
        for a in range(0, n, 1000):
            b = min(a + 1000, n)
            out[a:b] = np.mean(stats.norm.cdf((x[a:b, None] - x[None, :]) / h), axis=1)
        return out

    lo0 = float(x.min()) - 8 * h
    hi0 = float(x.max()) + 8 * h
    gr = np.linspace(lo0, hi0, M)
    delta = gr[1] - gr[0]

    idx = (x - lo0) / delta
    lo = np.clip(np.floor(idx).astype(int), 0, M - 1)
    w = idx - lo
    hi = np.minimum(lo + 1, M - 1)
    cnt = _wbin(np.concatenate([lo, hi]), np.concatenate([1 - w, w]), M)

    L = int(2 ** np.ceil(np.log2(2 * M)))
    ker = stats.norm.cdf((np.arange(L) - (L // 2)) * delta / h)
    cv = np.real(
        np.fft.ifft(np.fft.fft(np.concatenate([cnt, np.zeros(L - M)])) * np.fft.fft(ker))
    )
    Fg = cv[(L // 2) : (L // 2) + M] / n

    Fg = np.maximum.accumulate(np.clip(Fg, 0.0, 1.0))
    return np.interp(x, gr, Fg)


# ---------------------------------------------------------------------------
# dispatcher
# ---------------------------------------------------------------------------
def cdf_estimate(x: np.ndarray, cdf: str, ties: str = "max") -> np.ndarray:
    """Estimated marginal CDF at the sample points.  Returns u in (0, 1)."""
    x = np.asarray(x, dtype=float)
    n = x.size

    if cdf == "rank.n":
        # Yang, Qian & Xie (2025); Qian, Koschmann & Xie (2025), Eq. 9
        u = _counts_le(x, ties) / n
        u[x == x.max()] = n / (n + 1)
        return u

    if cdf == "rank.n1":
        # Breitung, Mayer & Wied (2024), Eq. 2.3
        return _counts_le(x, ties) / (n + 1)

    if cdf == "ecdf.fixed":
        # Becker, Proksch & Ringle (2022)
        u = _counts_le(x, ties) / n
        u[u <= 0] = 1e-7
        u[u >= 1] = 1 - 1e-7
        return u

    if cdf == "ecdf.adj":
        # Liengaard et al. (2025), Eq. 9
        return 1 / (2 * n) + (n - 1) / n**2 * _counts_le(x, ties)

    if cdf == "kde.silverman":
        # Park & Gupta (2012), Eq. 3
        return cdf_kde_epan(x, bw_silverman(x))

    if cdf == "kde.cv":
        # Li, Li & Racine (2017)
        return cdf_kde_epan(x, bw_cv(x))

    if cdf == "kde.plugin":
        # Gaussian kernel, Polansky & Baker (2000) plug-in bandwidth
        return cdf_kde_gauss(x, bw_plugin(x))

    raise ValueError(f"Unknown cdf: {cdf!r}. Choose one of {CDF_CHOICES}.")


def copula_transform(
    x: np.ndarray, cdf: str, ties: str = "max", varname: str = "regressor"
) -> np.ndarray:
    """C(x) = Phi^-1(Fhat(x))."""
    u = cdf_estimate(x, cdf, ties)
    cc = stats.norm.ppf(u)
    if not np.all(np.isfinite(cc)):
        raise ValueError(
            f"Copula transformation of {varname!r} produced non-finite values "
            "(the estimated CDF hit 0 or 1). Try a different 'cdf'."
        )
    return cc


def copula_transform_matrix(M: np.ndarray, cdf: str, ties: str = "max",
                            names=None) -> np.ndarray:
    """Column-wise copula transformation."""
    M = np.asarray(M, dtype=float)
    out = np.empty_like(M)
    for j in range(M.shape[1]):
        nm = names[j] if names is not None else f"column {j}"
        out[:, j] = copula_transform(M[:, j], cdf, ties, nm)
    return out
