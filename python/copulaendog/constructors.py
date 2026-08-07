"""Copula control-function constructors, one per estimator.

Each constructor is a callable

    ctor(X, info, cdf, ties) -> CopulaTerms

returning

    C      n x K matrix of copula control functions entering the regression
    Cstar  n x K matrix of the plain copula data Phi^-1(Fhat(P)), used for the
           endogeneity measure rho = corr(xi, P*)
    Wstar  optional, the copula transformation of the exogenous regressors
    resid1 optional, first-stage residuals (BMW)
    groups optional, which endogenous regressor and which cell each copula
           column belongs to (JAMS)

They are classes rather than closures so that the bootstrap can hand them to
worker processes.  State derived once from the full sample -- which exogenous
columns survive the first stage, the JAMS cell layout -- is cached on the
instance and must not be recomputed per resample: doing so would renumber the
cells and silently permute the copula columns between replicates.
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
from dataclasses import dataclass, field

import numpy as np

from .cdf import copula_transform_matrix
from .formula import exog_cols, term_variables


@dataclass
class CopulaTerms:
    C: np.ndarray
    Cstar: np.ndarray
    names: list
    Wstar: np.ndarray | None = None
    resid1: np.ndarray | None = None
    groups: list | None = None          # list of (endogenous, cell) per column
    groups_reference: bool = False


def _lstsq_resid(D: np.ndarray, v: np.ndarray, what: str) -> np.ndarray:
    q, r = np.linalg.qr(D)
    if np.linalg.matrix_rank(D) < D.shape[1]:
        raise np.linalg.LinAlgError(f"The first stage of {what} is rank deficient.")
    return v - q @ (q.T @ v)


# ---------------------------------------------------------------------------
# Park & Gupta (2012)
# ---------------------------------------------------------------------------
class PGConstructor:
    """y = mu + P alpha + W beta + sum_k gamma_k Phi^-1(Fhat(P_k)) + u.

    Exogenous regressors are not transformed and do not enter the copula term,
    so the control function and the plain copula data coincide.
    """

    def __call__(self, X, info, cdf, ties) -> CopulaTerms:
        P = X[:, info.endo_cols]
        C = copula_transform_matrix(P, cdf, ties, info.endo_names)
        return CopulaTerms(
            C=C, Cstar=C.copy(),
            names=[f"{v}_cop" for v in info.endo_names],
        )


# ---------------------------------------------------------------------------
# 2sCOPE (Yang, Qian & Xie 2025) and IMA (Haschka 2025a)
# ---------------------------------------------------------------------------
class TwoStageConstructor:
    """Transform every regressor to normal scores, then residualise.

        C_k = P*_k - delta_k' W*

    Each P*_k is regressed on W* only, separately, never on the other
    endogenous regressors.  The two estimators differ in exactly one place:
    Equation 11 of Yang et al. carries no intercept in the first stage and IMA
    implements it that way, because the slope then equals the Pearson
    correlation between P* and W*, which is what Haschka uses to recover rho.
    2sCOPE adds an intercept, following note 8 of Qian, Koschmann & Xie (2025).
    """

    def __init__(self, intercept: bool = True, verbose: bool = True):
        self.intercept = intercept
        self.verbose = verbose
        self.cols = None

    def _first_stage_cols(self, X, info, cdf, ties):
        """Which exogenous columns carry independent information after the
        copula transformation?

        The rank-based CDFs are invariant to strictly monotone transformations,
        so Phi^-1(Fhat(w)) and Phi^-1(Fhat(w**2)) are the same column whenever
        w is positive and the first stage would be singular.  The kernel-based
        CDFs make them near rather than exactly collinear, which is no better.
        Redundant columns are dropped from the first stage -- they stay in the
        structural model -- and the choice is made once and reused, so the
        design does not change between bootstrap replicates.
        """
        wc = exog_cols(info, X)
        if wc.size < 2:
            return wc
        Ws = copula_transform_matrix(X[:, wc], cdf, ties,
                                     [info.xnames[j] for j in wc])
        Wc = Ws - Ws.mean(axis=0)
        rank = np.linalg.matrix_rank(Wc, tol=1e-7)
        if rank == wc.size:
            return wc
        # pivoted QR to pick a maximal independent subset
        _, _, piv = _qr_pivot(Wc)
        sel = np.sort(piv[:rank])
        dropped = [info.xnames[j] for j in wc[np.setdiff1d(np.arange(wc.size), sel)]]
        if self.verbose:
            warnings.warn(
                "Left out of the first stage because they carry no independent "
                "information after the copula transformation, which is invariant "
                "to monotone transformations: " + ", ".join(dropped)
                + ". They remain regressors in the structural model.",
                stacklevel=2,
            )
        return wc[sel]

    def __call__(self, X, info, cdf, ties) -> CopulaTerms:
        if self.cols is None:
            self.cols = self._first_stage_cols(X, info, cdf, ties)
        wc = self.cols

        P = X[:, info.endo_cols]
        Cstar = copula_transform_matrix(P, cdf, ties, info.endo_names)
        names = [f"{v}_cop" for v in info.endo_names]

        # no exogenous regressors: the first stage has nothing to project on
        # and the estimator collapses to Park & Gupta.  The wrappers redirect
        # before reaching this point; this branch only guards the bootstrap.
        if wc.size == 0:
            return CopulaTerms(C=Cstar.copy(), Cstar=Cstar, names=names)

        Wstar = copula_transform_matrix(X[:, wc], cdf, ties,
                                        [info.xnames[j] for j in wc])
        D = np.column_stack([np.ones(X.shape[0]), Wstar]) if self.intercept else Wstar

        C = np.empty_like(Cstar)
        for k in range(Cstar.shape[1]):
            C[:, k] = _lstsq_resid(D, Cstar[:, k], "the two-stage estimator")
        return CopulaTerms(C=C, Cstar=Cstar, names=names, Wstar=Wstar)


# ---------------------------------------------------------------------------
# BMW (Breitung, Mayer & Wied 2024)
# ---------------------------------------------------------------------------
class BMWConstructor:
    """Inverts the order of the two operations.

        1. regress z on x by OLS and keep the residuals ehat
        2. etahat = Phi^-1(Fhat(ehat)) with Fhat = rank/(n + 1)
        3. regress y on x, z and etahat

    The rank transform is applied to the first-stage residuals rather than to
    z, which is what separates this from 2sCOPE.  Each z_j is regressed on x on
    its own (Remark 2.1).  The first stage always carries an intercept, whether
    or not the structural model does, because Assumption A4 requires E[e] = 0.
    """

    def __init__(self):
        self.cols = None

    def __call__(self, X, info, cdf, ties) -> CopulaTerms:
        if self.cols is None:
            self.cols = exog_cols(info, X)
        wc = self.cols

        P = X[:, info.endo_cols]
        if wc.size == 0:
            # delta is empty and the residual is just z centred, so the
            # estimator coincides with Park & Gupta on rank/(n+1)
            E = P - P.mean(axis=0)
        else:
            D = np.column_stack([np.ones(X.shape[0]), X[:, wc]])
            E = np.column_stack(
                [_lstsq_resid(D, P[:, k], "BMW") for k in range(P.shape[1])]
            )

        C = copula_transform_matrix(E, cdf, ties, info.endo_names)
        return CopulaTerms(
            C=C, Cstar=C.copy(),
            names=[f"{v}_cop" for v in info.endo_names],
            Wstar=None if wc.size == 0 else X[:, wc],
            resid1=E,
        )


# ---------------------------------------------------------------------------
# JAMS (Liengaard et al. 2025)
# ---------------------------------------------------------------------------
@dataclass
class _JamsLayout:
    cells: np.ndarray
    levels: list
    J: int
    wc: list
    clab: list
    need: np.ndarray


class JAMSConstructor:
    """Equation 17.  With d_P endogenous P and d_W continuous exogenous W, all
    transformed to normal scores,

        C(P, W) = ( C(P)' C(W)' ) Sigma^-1 [ I_dP ; 0 ]

    with Sigma the variance-covariance matrix of (C(P), C(W)) -- the covariance
    matrix, not the correlation matrix.

    Equations 20 and 21 let the structure differ across the joint categories of
    the discrete exogenous regressors: within each cell the CDFs and Sigma are
    estimated from that cell's observations only, and every copula column is
    zero outside its own cell.  conditional=False collapses this to Equation 18.
    """

    def __init__(self, conditional=True, verbose: bool = True):
        self.conditional = conditional
        self.verbose = verbose
        self.layout = None

    # -- layout, derived once from the original design matrix ---------------
    def _build_layout(self, info) -> _JamsLayout:
        X = info.X
        n, ncol = X.shape
        exo = np.array(
            [j for j in range(ncol) if info.part[j] >= 2 and info.order[j] == 1]
        )
        fac = np.array([j for j in exo if j in set(info.factor_cols.tolist())])

        if isinstance(self.conditional, (list, tuple, set)) or isinstance(
            self.conditional, str
        ):
            wanted = (
                {self.conditional}
                if isinstance(self.conditional, str)
                else set(self.conditional)
            )
            cells = np.array(
                [j for j in exo if any(w in info.term_label[j] for w in wanted)]
            )
            hit = {w for w in wanted
                   if any(w in info.term_label[j] for j in cells)}
            missing = wanted - hit
            if missing:
                raise ValueError(
                    "Variable(s) named in 'conditional' are not exogenous "
                    "first-order regressors: " + ", ".join(sorted(missing))
                )
            left = np.setdiff1d(fac, cells)
            if left.size and self.verbose:
                warnings.warn(
                    "Discrete regressor(s) not conditioned on and therefore left "
                    "out of the copula terms as well, while staying in the model: "
                    + ", ".join(sorted({info.term_label[j] for j in left})) + ".",
                    stacklevel=2,
                )
        elif self.conditional is True:
            cells = fac
            few = [
                j
                for j in np.setdiff1d(exo, fac)
                if np.unique(X[:, j]).size <= 10
            ]
            if few and self.verbose:
                warnings.warn(
                    "Treated as continuous although they take few distinct "
                    "values: " + ", ".join(info.xnames[j] for j in few)
                    + ". Wrap them in C() or name them in 'conditional' to let "
                    "the copula structure vary over their categories.",
                    stacklevel=2,
                )
        else:
            cells = np.array([], dtype=int)

        disc = np.union1d(fac, cells).astype(int)

        if cells.size == 0:
            levels, cell = None, np.zeros(n, dtype=int)
        else:
            keys = _row_keys(X[:, cells])
            levels = sorted(set(keys))
            cell = np.array([levels.index(k) for k in keys])
        J = 1 if levels is None else len(levels)

        # Readable cell labels.  Cells are found from the dummy pattern, because
        # that is all a bootstrap resample carries, but the labels are read off
        # the model frame of the original data, which still holds the factor
        # levels -- the base category has no dummy column of its own.
        if J == 1:
            clab = [""]
        else:
            dterms = sorted({info.term_label[j] for j in cells})
            tvars = {tm: term_variables(info, tm) for tm in dterms}
            clab = []
            for jj in range(J):
                r = int(np.flatnonzero(cell == jj)[0])
                bits = []
                for tm in dterms:
                    vs = tvars[tm]
                    if vs and all(v in info.frame.columns for v in vs):
                        # column-wise, so that a mixed-dtype row does not get
                        # upcast and turn an integer level into "0.0"
                        bits.append(
                            ":".join(f"{v}{info.frame[v].iloc[r]}" for v in vs)
                        )
                    else:
                        on = [info.xnames[j] for j in cells
                              if info.term_label[j] == tm and X[r, j] != 0]
                        bits.append(on[0] if on else f"{tm}(base)")
                clab.append(":".join(bits))

        # continuous exogenous regressors usable inside each cell
        wall = np.setdiff1d(exo, disc)
        wc, dropped = [], []
        for jj in range(J):
            r = np.flatnonzero(cell == jj)
            keep = [k for k in wall if np.unique(X[r, k]).size > 1]
            wc.append(np.array(keep, dtype=int))
            if len(keep) < wall.size:
                miss = [info.xnames[k] for k in wall if k not in keep]
                dropped.append(
                    ", ".join(miss) + (f" in {clab[jj]}" if J > 1 else "")
                )
        if dropped and self.verbose:
            warnings.warn(
                "Constant within a category and therefore left out of the copula "
                "terms there, while staying in the model: " + "; ".join(dropped)
                + ".",
                stacklevel=2,
            )

        dP = info.endo_cols.size
        size = np.array([int((cell == jj).sum()) for jj in range(J)])
        need = dP + np.array([w.size for w in wc]) + 2
        tag = ["the sample"] if J == 1 else clab

        bad = np.flatnonzero(size < need)
        if bad.size:
            raise ValueError(
                "Too few observations to estimate the copula structure "
                "separately for: "
                + "; ".join(
                    f"{tag[b]}: {size[b]} observations, {need[b]} needed"
                    for b in bad
                )
                + ". Name fewer variables in 'conditional', merge categories, or "
                "use conditional=False."
            )

        if J > 1 and self.verbose:
            from scipy import stats as _st

            pf = _st.binom.cdf(need - 1, n, size / n)
            ok = float(np.prod(1 - pf))
            if ok < 0.95:
                worst = int(np.argmax(pf))
                warnings.warn(
                    f"Small categories: a bootstrap resample keeps every cell "
                    f"usable only {100 * ok:.0f}% of the time, so roughly "
                    f"{1 / max(ok, 1e-6):.1f} draws are needed per replicate. "
                    f"Tightest: {tag[worst]} with {size[worst]} observations.",
                    stacklevel=2,
                )

        return _JamsLayout(cells=cells, levels=levels, J=J, wc=wc,
                           clab=clab, need=need)

    def __call__(self, X, info, cdf, ties) -> CopulaTerms:
        if self.layout is None:
            self.layout = self._build_layout(info)
        lay = self.layout
        n = X.shape[0]
        P = X[:, info.endo_cols]
        dP = P.shape[1]
        J = lay.J

        if lay.levels is None:
            cell = np.zeros(n, dtype=int)
        else:
            keys = _row_keys(X[:, lay.cells])
            idx = {k: i for i, k in enumerate(lay.levels)}
            cell = np.array([idx.get(k, -1) for k in keys])

        Cstar = np.full((n, dP), np.nan)
        C = np.zeros((n, dP * J))
        names, groups = [], []
        Wstar = None
        col = 0
        for jj in range(J):
            r = np.flatnonzero(cell == jj)
            wj = lay.wc[jj]
            if r.size < lay.need[jj]:
                raise ValueError(
                    f"Category {lay.clab[jj] or 'the sample'} holds only "
                    f"{r.size} observations in this resample, "
                    f"{lay.need[jj]} needed."
                )

            cp = copula_transform_matrix(P[r], cdf, ties, info.endo_names)
            Cstar[r] = cp
            if wj.size:
                cw = copula_transform_matrix(X[np.ix_(r, wj)], cdf, ties,
                                             [info.xnames[k] for k in wj])
                if J == 1:
                    Wstar = cw
                M = np.column_stack([cp, cw])
            else:
                M = cp

            S = np.cov(M, rowvar=False, ddof=1)
            S = np.atleast_2d(S)
            try:
                Si = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                raise ValueError(
                    "The covariance matrix of the copula data is singular in "
                    f"category {lay.clab[jj] or 'the sample'}: two of the "
                    "regressors are collinear there."
                )
            Ck = M @ Si[:, :dP]

            for k in range(dP):
                C[r, col] = Ck[:, k]
                names.append(
                    f"{info.endo_names[k]}_cop"
                    if J == 1
                    else f"{info.endo_names[k]}_cop.{lay.clab[jj]}"
                )
                groups.append((info.endo_names[k], "" if J == 1 else lay.clab[jj]))
                col += 1

        # Wstar is a well-defined single first-stage design only with one cell
        return CopulaTerms(C=C, Cstar=Cstar, names=names,
                           Wstar=Wstar if J == 1 else None, groups=groups)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _row_keys(M: np.ndarray) -> list:
    """A hashable key per row, used to identify the joint categories."""
    return [tuple(np.round(row, 12)) for row in np.atleast_2d(M)]


def _qr_pivot(A: np.ndarray):
    """Column-pivoted QR, returning (Q, R, pivot)."""
    from scipy.linalg import qr

    q, r, p = qr(A, mode="economic", pivoting=True)
    return q, r, p
