"""Fitting engine, bootstrap and the CopregResult object.

Ported from sections 3, 5, 6 and 8 of Copreg_core.R.
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
import pandas as pd
from scipy import stats

from . import diagnostics as dg
from .cdf import CDF_CHOICES, TIES_CHOICES
from .formula import ModelInfo, build_model


# ---------------------------------------------------------------------------
# bootstrap
# ---------------------------------------------------------------------------
def _check_levels(X: np.ndarray, xnames, has_intercept: bool, verbose: bool):
    """Which columns can realistically degenerate in a resample?

    Only dummies and variables with few distinct values; a continuous column
    drawn n times from n distinct values is never constant.  Classifying them
    once up front is what keeps the bootstrap fast.
    """
    n = X.shape[0]
    cols = [j for j in range(X.shape[1]) if not (has_intercept and j == 0)]
    dummy, discrete, thin = [], [], []
    for j in cols:
        v = X[:, j]
        m = None
        if np.all((v == 0) | (v == 1)):
            dummy.append(j)
            m = min(int(v.sum()), n - int(v.sum()))
        elif np.unique(v).size <= 20:
            discrete.append(j)
            _, cnt = np.unique(v, return_counts=True)
            m = int(cnt.min())
        if m is not None:
            # a group of size m survives a resample with probability
            # 1 - (1 - m/n)^n: for m = 1 that is only 63%
            pl = (1 - m / n) ** n
            if pl > 0.01:
                thin.append(f"{xnames[j]}: smallest group {m}, lost in "
                            f"{100 * pl:.0f}% of draws")
    if thin and verbose:
        warnings.warn(
            "Thinly populated column(s). Resamples that lose a level cannot be "
            "used and are redrawn, so the bootstrap gets slow and can fail "
            "outright: " + "; ".join(thin),
            stacklevel=2,
        )
    return np.array(dummy, dtype=int), np.array(discrete, dtype=int)


def _boot_replicate(y, X, info, ctor, cdf, ties, dcols, scols, rng, max_tries):
    """One bootstrap replicate.

    Plain pairs bootstrap: draw row indices with replacement, recompute the
    copula terms on the resampled data, refit.  This is the procedure used in
    every one of the underlying papers.  Resamples in which a factor level
    disappears, or the design otherwise loses rank, are rejected and redrawn.
    """
    n = X.shape[0]
    for a in range(1, max_tries + 1):
        idx = rng.integers(0, n, n)
        Xb = X[idx]

        ok = True
        for j in dcols:
            s = Xb[:, j].sum()
            if s == 0 or s == n:
                ok = False
                break
        if ok:
            for j in scols:
                if np.unique(Xb[:, j]).size < 2:
                    ok = False
                    break
        if not ok:
            continue

        try:
            cb = ctor(Xb, info, cdf, ties)
        except Exception:
            continue

        Ab = np.column_stack([Xb, cb.C])
        if np.linalg.matrix_rank(Ab) < Ab.shape[1]:
            continue
        if np.linalg.matrix_rank(Xb) < Xb.shape[1]:
            continue

        yb = y[idx]
        cf, *_ = np.linalg.lstsq(Ab, yb, rcond=None)
        # the same resample without the copula terms, for the ICON statistic
        ols, *_ = np.linalg.lstsq(Xb, yb, rcond=None)

        beta = cf[: Xb.shape[1]]
        xi = yb - Xb @ beta
        rho = np.array([dg._corr(xi, cb.Cstar[:, k])
                        for k in range(len(info.endo_cols))])
        return np.concatenate([cf, rho, ols, [a]])
    return None


def _bootstrap(y, X, info, ctor, cdf, ties, nboots, seed=None,
               verbose=True, max_tries=200):
    dcols, scols = _check_levels(X, info.xnames, info.has_intercept, verbose)
    rng = np.random.default_rng(seed)

    reps, fail = [], 0
    for _ in range(nboots):
        r = _boot_replicate(y, X, info, ctor, cdf, ties, dcols, scols,
                            rng, max_tries)
        if r is None:
            fail += 1
        else:
            reps.append(r)

    if not reps:
        raise RuntimeError(
            f"Every bootstrap resample was degenerate after {max_tries} "
            "attempts. This usually means a factor level or a cell of the "
            "discrete regressors is too rare. Collapse sparse levels or drop "
            "the variable."
        )
    if fail and verbose:
        warnings.warn(
            f"{fail} of {nboots} bootstrap replicates could not be computed "
            "and were dropped.", stacklevel=2,
        )

    B = np.vstack(reps)
    tries = B[:, -1]
    B = B[:, :-1]
    if tries.mean() > 1.05 and verbose:
        warnings.warn(
            f"{tries.mean():.1f} draws were needed per bootstrap replicate "
            f"({100 * (1 - 1 / tries.mean()):.0f}% of resamples discarded "
            "because a level or cell was missing). The standard errors are "
            "conditional on the draws that survived.",
            stacklevel=2,
        )
    return B


# ---------------------------------------------------------------------------
# result object
# ---------------------------------------------------------------------------
@dataclass
class CopregResult:
    method: str
    cdf: str
    ties: str
    nboots: int
    params: pd.Series
    bse: pd.Series
    vcov: pd.DataFrame
    boot: pd.DataFrame
    ols_params: pd.Series
    ols_bse: pd.Series
    icon: pd.Series
    rho: pd.Series
    rho_se: pd.Series
    fittedvalues: np.ndarray
    resid: np.ndarray                 # xi, structural
    fitted_augmented: np.ndarray
    resid_augmented: np.ndarray       # u
    copula_terms: pd.DataFrame
    copula_data: pd.DataFrame
    copula_groups: list | None
    Wstar: np.ndarray | None
    first_stage_resid: np.ndarray | None
    ols_vcov: pd.DataFrame | None
    endogenous: list
    exogenous: list
    nobs: int
    df_resid: int
    has_intercept: bool
    assumption5: bool
    dhw: bool
    id_condition: str
    diagnostics: dict
    info: ModelInfo = field(repr=False, default=None)
    formula: str = ""

    # -- basic accessors ----------------------------------------------------
    def conf_int(self, level: float = 0.95, kind: str = "normal") -> pd.DataFrame:
        """Confidence intervals: 'normal' from the bootstrap standard errors,
        'percentile' from the bootstrap draws themselves."""
        a = (1 - level) / 2
        if kind == "normal":
            z = stats.norm.ppf(1 - a)
            lo, hi = self.params - z * self.bse, self.params + z * self.bse
        elif kind == "percentile":
            lo = self.boot.quantile(a)
            hi = self.boot.quantile(1 - a)
        else:
            raise ValueError("kind must be 'normal' or 'percentile'.")
        return pd.DataFrame({f"{100 * a:.1f}%": lo, f"{100 * (1 - a):.1f}%": hi})

    def predict(self, newdata=None) -> np.ndarray:
        """Structural model only.  Copula terms never enter prediction: they
        are endogeneity controls, not part of the causal model."""
        if newdata is None:
            return self.fittedvalues
        import patsy

        Xn = np.asarray(
            patsy.build_design_matrices([self.info.design_info], newdata)[0],
            dtype=float,
        )
        beta = self.params[self.info.xnames].values
        return Xn @ beta

    # -- inference tables ---------------------------------------------------
    def summary_frame(self) -> pd.DataFrame:
        z = self.params / self.bse
        return pd.DataFrame({
            "Estimate": self.params,
            "Std. Error": self.bse,
            "z value": z,
            "Pr(>|z|)": 2 * stats.norm.cdf(-np.abs(z)),
        })

    def rho_frame(self) -> pd.DataFrame | None:
        """Endogeneity: rho(P*, xi*), the correlation between the normal score
        of an endogenous regressor and that of the structural error.

        Reported only where the estimator identifies it.
        """
        if self.id_condition != "nonnormality":
            return None
        z = (self.rho / self.rho_se).values
        return pd.DataFrame({
            "Estimate": self.rho.values,
            "Std. Error": self.rho_se.values,
            "z value": z,
            "Pr(>|z|)": 2 * stats.norm.cdf(-np.abs(z)),
        }, index=[f"rho({v}*, xi*)" for v in self.rho.index])

    def dhw_frame(self) -> pd.DataFrame | None:
        """Durbin-Hausman-Wu test of rho = 0.

        Breitung, Mayer & Wied (2024, Corollary 3.2) show that under the null
        the textbook t statistic built from classical OLS standard errors keeps
        a standard normal limit, even though those standard errors are
        inconsistent for the structural coefficients.
        """
        if not self.dhw or self.ols_vcov is None:
            return None
        names = list(self.copula_terms.columns)
        g = self.params[names]
        se0 = pd.Series(np.sqrt(np.diag(self.ols_vcov.values)),
                        index=self.ols_vcov.index)[names]
        t = g / se0
        return pd.DataFrame({
            "Estimate": g, "Std. Error": se0, "t value": t,
            "Pr(>|t|)": 2 * stats.norm.cdf(-np.abs(t)),
        })

    def wald_copula(self) -> dict | None:
        """Bootstrap Wald test of the copula terms' joint significance
        (Liengaard et al. 2025): with several copula terms, testing them one at
        a time is a multiple testing problem."""
        names = list(self.copula_terms.columns)
        if len(names) < 2:
            return None
        return _wald(self.params, self.vcov, names)

    def wald_cells(self) -> dict | None:
        """Whether the copula structure differs across the categories of the
        discrete covariates: gamma^z_k equal in z, for each k."""
        grp = self.copula_groups
        if grp is None or len({c for _, c in grp}) < 2:
            return None
        names = list(self.copula_terms.columns)
        R = []
        for v in dict.fromkeys(e for e, _ in grp):
            j = [i for i, (e, _) in enumerate(grp) if e == v]
            for m in j[1:]:
                r = np.zeros(len(names))
                r[j[0]], r[m] = 1.0, -1.0
                R.append(r)
        return _wald(self.params, self.vcov, names, np.array(R)) if R else None

    def fit_frame(self) -> pd.DataFrame:
        """Both columns use the same residual degrees of freedom: all
        coefficients, the copula terms included, are estimated jointly."""
        y = self.fitted_augmented + self.resid_augmented
        ss = np.sum((y - y.mean()) ** 2) if self.has_intercept else np.sum(y**2)
        nn = self.nobs - int(self.has_intercept)
        df = self.df_resid

        def stat(r):
            r2 = 1 - np.sum(r**2) / ss
            return [np.sqrt(np.sum(r**2) / df), r2, 1 - (1 - r2) * nn / df]

        return pd.DataFrame(
            {"augmented": stat(self.resid_augmented),
             "structural": stat(self.resid)},
            index=["Residual standard error", "R-squared", "Adjusted R-squared"],
        )

    # -- printing -----------------------------------------------------------
    def summary(self) -> str:
        out = [f"\nCopula endogeneity correction: {self.method}",
               f"\nFormula: {self.formula}"]
        out.append("\nCoefficients:\n" + _fmt(self.summary_frame()))

        rf = self.rho_frame()
        if rf is not None:
            out.append(
                "\nEndogeneity: rho(P*, xi*) is the correlation between the "
                "normal score of an\n  endogenous regressor and that of the "
                "structural error, xi* = xi / sigma.\n" + _fmt(rf)
            )
        df = self.dhw_frame()
        if df is not None:
            out.append(
                "\nDurbin-Hausman-Wu test of rho = 0 (BMW 2024, Corollary 3.2),\n"
                "  classical OLS standard errors:\n" + _fmt(df)
            )
        w = self.wald_copula()
        if w is not None:
            out.append(
                f"\nJoint test of the copula terms: chi2({w['df']:.0f}) = "
                f"{w['chisq']:.4f}, p = {w['p']:.4g}"
            )
        wz = self.wald_cells()
        if wz is not None:
            out.append(
                f"Copula structure constant across categories: "
                f"chi2({wz['df']:.0f}) = {wz['chisq']:.4f}, p = {wz['p']:.4g}"
            )
        out.append(f"\nFit, on {self.df_resid} residual degrees of freedom:\n"
                   + _fmt(self.fit_frame()))
        cdf = "-" if self.cdf is None else f'"{self.cdf}"'
        ties = "-" if self.ties is None else f'"{self.ties}"'
        out.append(f"Standard errors from {self.nboots} bootstrap replicates; "
                   f"cdf = {cdf}, ties = {ties}.\n")
        return "\n".join(out)

    def __str__(self) -> str:
        return self.summary()

    # -- validity -----------------------------------------------------------
    def validity(self, level: float = 0.05, power: float = 0.8) -> "Validity":
        return _validity(self, level, power)


def _fmt(df: pd.DataFrame) -> str:
    with pd.option_context("display.float_format", lambda v: f"{v:10.4f}",
                           "display.width", 120):
        return df.to_string()


def _wald(cf: pd.Series, V: pd.DataFrame, names, R=None, tol=1e-8) -> dict:
    """Wald test of R gamma = 0 using the bootstrap covariance matrix.

    Copula terms are frequently close to collinear, so the covariance matrix of
    the contrasts can be near singular.  Inverting it with solve() would give an
    unstable statistic on a degrees-of-freedom count that overstates what the
    data identify, so the Moore-Penrose inverse is used and its effective rank
    reported as the degrees of freedom.
    """
    g = cf[names].values
    if R is None:
        R = np.eye(len(g))
    Rg = R @ g
    RV = R @ V.loc[names, names].values @ R.T
    u, s, vt = np.linalg.svd(RV)
    pos = s > tol * s.max()
    q = int(pos.sum())
    if q == 0:
        return {"chisq": np.nan, "df": 0, "p": np.nan}
    Wi = vt[pos].T @ np.diag(1 / s[pos]) @ u[:, pos].T
    st = float(Rg @ Wi @ Rg)
    return {"chisq": st, "df": q, "p": float(stats.chi2.sf(st, q))}


# ---------------------------------------------------------------------------
# fitting engine
# ---------------------------------------------------------------------------
def copreg_fit(formula, data, ctor, method, cdf, ties, nboots=199,
               subset=None, seed=None, assumption5=True, dhw=False,
               id_condition="nonnormality", verbose=True) -> CopregResult:
    if cdf is not None and cdf not in CDF_CHOICES:
        raise ValueError(f"'cdf' must be one of {CDF_CHOICES}.")
    if ties is not None and ties not in TIES_CHOICES:
        raise ValueError(f"'ties' must be one of {TIES_CHOICES}.")
    if not isinstance(nboots, (int, np.integer)) or nboots < 2:
        raise ValueError("'nboots' must be a single integer >= 2.")

    info = build_model(formula, data, subset=subset, verbose=verbose)
    y, X = info.y, info.X

    cc = ctor(X, info, cdf, ties)
    C, Cstar = cc.C, cc.Cstar
    cnames = cc.names

    A = np.column_stack([X, C])
    if np.linalg.matrix_rank(A) < A.shape[1]:
        # two quite different failures land here, so say which one it is
        if np.linalg.matrix_rank(X) < X.shape[1]:
            raise ValueError(
                "The design matrix is rank deficient before any copula term is "
                "added: the regressors themselves are collinear. A constant "
                "column next to an intercept, or a duplicated regressor, is the "
                "usual cause."
            )
        raise ValueError(
            "The copula terms are perfectly collinear with the regressors, so "
            "the model is not identified. This is what happens when an "
            "endogenous regressor is normally distributed: its copula transform "
            "is then a linear function of itself."
        )

    names = list(info.xnames) + list(cnames)
    cf, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid_a = y - A @ cf
    n, kA = A.shape

    # classical OLS covariance of the augmented regression.  These standard
    # errors are wrong for the structural coefficients -- the copula terms are
    # generated regressors -- but BMW (2024, Corollary 3.2) show the textbook t
    # statistic remains valid for testing rho = 0.
    s2 = np.sum(resid_a**2) / (n - kA)
    try:
        Vc = pd.DataFrame(s2 * np.linalg.inv(A.T @ A), index=names, columns=names)
    except np.linalg.LinAlgError:
        Vc = None

    beta = cf[: X.shape[1]]
    fit_s = X @ beta
    resid_s = y - fit_s
    fit_a = A @ cf

    rho = pd.Series(
        [dg._corr(resid_s, Cstar[:, k]) for k in range(len(info.endo_cols))],
        index=info.endo_names,
    )

    B = _bootstrap(y, X, info, ctor, cdf, ties, nboots, seed=seed,
                   verbose=verbose)
    kp, kx = len(cf), X.shape[1]
    Bc = pd.DataFrame(B[:, :kp], columns=names)
    Br = B[:, kp: kp + len(rho)]
    Bo = B[:, kp + len(rho):]

    V = pd.DataFrame(np.cov(Bc.values, rowvar=False, ddof=1),
                     index=names, columns=names)
    se = pd.Series(np.sqrt(np.diag(V.values)), index=names)
    rho_se = pd.Series(Br.std(axis=0, ddof=1), index=info.endo_names)

    # uncorrected OLS on the same resamples, and the ICON statistic (Qian,
    # Koschmann & Xie 2025, Boundary Condition 1): the inflation of the standard
    # errors caused by adding the copula terms.  Values above 6 flag weak
    # identification or a misspecified dependence model.
    ols_cf = pd.Series(np.linalg.lstsq(X, y, rcond=None)[0], index=info.xnames)
    ols_se = pd.Series(Bo.std(axis=0, ddof=1), index=info.xnames)
    icon = se[info.xnames] / ols_se

    return CopregResult(
        method=method, cdf=cdf, ties=ties, nboots=Bc.shape[0],
        params=pd.Series(cf, index=names), bse=se, vcov=V, boot=Bc,
        ols_params=ols_cf, ols_bse=ols_se, icon=icon,
        rho=rho, rho_se=rho_se,
        fittedvalues=fit_s, resid=resid_s,
        fitted_augmented=fit_a, resid_augmented=resid_a,
        copula_terms=pd.DataFrame(C, columns=cnames),
        copula_data=pd.DataFrame(Cstar, columns=info.endo_names),
        copula_groups=cc.groups, Wstar=cc.Wstar,
        first_stage_resid=cc.resid1, ols_vcov=Vc,
        endogenous=list(info.endo_names),
        exogenous=[v for v in info.xnames if v not in info.endo_names],
        nobs=n, df_resid=n - kA, has_intercept=info.has_intercept,
        assumption5=assumption5, dhw=dhw, id_condition=id_condition,
        diagnostics=dg.model_diagnostics(X, C, Cstar, info,
                                         groups=cc.groups, cnames=cnames),
        info=info, formula=formula,
    )


# ---------------------------------------------------------------------------
# validity
# ---------------------------------------------------------------------------
@dataclass
class Validity:
    method: str
    n: int
    level: float
    power: float
    intercept: bool
    endogenous: list
    nonnormality_of: str
    id_condition: str
    thresholds: dict | None
    step1: pd.DataFrame | None
    step2: dict | None
    step3: pd.DataFrame | None
    step4: dict
    icon: pd.DataFrame
    icon_max: float

    def __str__(self) -> str:
        L = [f"\nValidity check for {self.method}",
             f"n = {self.n}, intercept: {'yes' if self.intercept else 'no'}, "
             f"target power {100 * self.power:.0f}%",
             "Sources: Becker, Proksch & Ringle (2022); Yang, Qian & Xie "
             "(2025);\n         Qian, Koschmann & Xie (2025)"]

        if self.step1 is not None:
            th = self.thresholds
            L.append(f"\n[1] Non-normality of the {self.nonnormality_of}.")
            L.append("    'Yang ok' is KS p < .05; 'Becker ok' is |skewness| >= "
                     f"{th['skewness']}, AD > {th['AD']} or CvM > {th['CvM']}.")
            L.append(_fmt(self.step1))
            if not self.step1["Yang ok"].all():
                L.append("    ! Not sufficiently non-normal on the KS criterion: "
                         + ", ".join(self.step1.index[~self.step1["Yang ok"]]))
        else:
            L.append("\n[1] Identification rests on Assumption 3 of Hu, Qian & "
                     "Xie (2025), not on non-normality of P.")

        if self.step2 is not None:
            L.append("\n[2] Uncorrelatedness of the exogenous regressors with "
                     "the copula term.")
            L.append(_fmt(self.step2["table"]))
            L.append(f"    Joint: R2 = {self.step2['joint.R2']:.4f}, "
                     f"F = {self.step2['joint.F']:.3f}, "
                     f"p = {self.step2['joint.p']:.4g}")
            if self.step2["violated"]:
                L.append("    ! The assumption is rejected. Park & Gupta is then "
                         "biased; use 2sCOPE, IMA or BMW,")
                L.append("      which project this correlation out in a first "
                         "stage (Haschka 2025a).")
        else:
            L.append("\n[2] Not applicable: this estimator projects the "
                     "correlation with W out in its first stage.")

        if self.step3 is not None:
            L.append("\n[3] Exogenous regressors as identifying variation for "
                     "the weakly non-normal P")
            L.append("    (continuous, KS p < .001, F > 10):")
            L.append(_fmt(self.step3))

        s4 = self.step4
        L.append("\n[4] Structural error xi.")
        L.append(f"    skewness {s4['skewness']:.3f}, excess kurtosis "
                 f"{s4['ex.kurtosis']:.3f}, AD {s4['AD']:.3f} "
                 f"(p {s4['AD.p']:.4g}), KS p {s4['KS.p']:.4g}")
        L.append("    Becker et al. require a normal error; Yang et al. and "
                 "Qian et al. permit a")
        L.append("    non-normal one under xi = U + V. The residuals show xi "
                 "rather than U, so their")
        L.append("    shape settles neither question.")

        L.append("\n[5] ICON, the standard error inflation relative to "
                 "uncorrected OLS.")
        L.append("    Above 6 flags weak identification or a misspecified "
                 "dependence model.")
        L.append(_fmt(self.icon))
        L.append(f"    max ICON = {self.icon_max:.3f}"
                 + ("  !" if self.icon_max > 6 else ""))
        return "\n".join(L) + "\n"


def _validity(obj: CopregResult, level: float, power: float) -> Validity:
    X, y = obj.info.X, obj.info.y
    Cs = obj.copula_data.values
    n = X.shape[0]
    nm = obj.endogenous
    K = len(nm)

    drop = {0} if obj.has_intercept else set()
    wc = [j for j in range(X.shape[1])
          if j not in drop and obj.info.xnames[j] not in nm
          and np.unique(X[:, j]).size > 1]
    W = X[:, wc]
    wnames = [obj.info.xnames[j] for j in wc]

    # --- [1] non-normality -------------------------------------------------
    # Which variable has to be non-normal is the estimator's own condition.
    # For BMW it is the first-stage error: their Theorem 2.1 identifies the
    # model if and only if the distribution of e is not normal.
    a3 = obj.id_condition == "assumption3"
    on_resid = not a3 and obj.first_stage_resid is not None
    step1, th = None, None
    if not a3:
        if on_resid:
            M1 = obj.first_stage_resid
            step1, th = dg.nonnormality_table(M1, range(M1.shape[1]), nm, n, power)
        else:
            step1, th = dg.nonnormality_table(
                X, [obj.info.xnames.index(v) for v in nm], nm, n, power)

    # --- [2] the uncorrelatedness assumption -------------------------------
    step2 = None
    if W.shape[1] and obj.assumption5:
        Apg = np.column_stack([X, Cs])
        gpg = np.linalg.lstsq(Apg, y, rcond=None)[0][X.shape[1]:]
        CTT = Cs @ gpg
        r = np.array([dg._corr(W[:, m], CTT) for m in range(W.shape[1])])
        pa = dg.holm(np.array([dg.fisher_z_p(v, n) for v in r]))
        Wd = np.column_stack([np.ones(n), W])
        _, res, rank = dg._ols(Wd, CTT)
        df1, df2 = rank - 1, n - rank
        r2 = 1 - np.sum(res**2) / np.sum((CTT - CTT.mean()) ** 2)
        Fs = (r2 / df1) / ((1 - r2) / df2)
        pj = float(stats.f.sf(Fs, df1, df2))
        step2 = {
            "table": pd.DataFrame({"corr(W, CTT)": np.round(r, 4),
                                   "p (Holm)": pa}, index=wnames),
            "joint.R2": r2, "joint.F": Fs, "joint.p": pj,
            "violated": bool(np.nanmin(pa) < level or pj < level),
        }

    # --- [3] exogenous regressors as identifying variation -----------------
    step3 = None
    weak = [] if step1 is None else list(step1.index[~step1["Yang ok"]])
    if weak and W.shape[1] and obj.Wstar is not None:
        Ws = obj.Wstar
        cont = [np.unique(W[:, j]).size > 20 for j in range(W.shape[1])]
        kw = [dg.ks_normal(W[:, j])[1] for j in range(W.shape[1])]
        D = np.column_stack([np.ones(n), Ws])
        Fm = np.full((Ws.shape[1], len(weak)), np.nan)
        try:
            XtXi = np.linalg.inv(D.T @ D)
            for k, v in enumerate(weak):
                coef, res, rank = dg._ols(D, Cs[:, nm.index(v)])
                se = np.sqrt(np.sum(res**2) / (n - rank) * np.diag(XtXi))
                Fm[:, k] = ((coef / se) ** 2)[1:]
        except np.linalg.LinAlgError:
            pass
        step3 = pd.DataFrame({"continuous": cont, "KS p": kw}, index=wnames)
        for k, v in enumerate(weak):
            step3[f"F: {v}"] = np.round(Fm[:, k], 3) if Fm.shape[0] == len(wnames) \
                else np.nan
        step3["qualifies"] = [
            c and p < 0.001 and np.nanmax(Fm[i]) > 10
            if Fm.shape[0] == len(wnames) else False
            for i, (c, p) in enumerate(zip(cont, kw))
        ]

    # --- [4] error term ----------------------------------------------------
    xi = obj.resid
    A4, ap = dg.ad_test(xi)
    step4 = {"skewness": dg.skewness(xi), "ex.kurtosis": dg.ex_kurtosis(xi),
             "AD": A4, "AD.p": ap, "KS.p": dg.ks_normal(xi)[1]}

    # --- [5] ICON ----------------------------------------------------------
    icon = pd.DataFrame({
        "SE (corrected)": obj.bse[obj.ols_bse.index],
        "SE (uncorrected)": obj.ols_bse,
        "ICON": obj.icon,
    })

    return Validity(
        method=obj.method, n=n, level=level, power=power,
        intercept=obj.has_intercept, endogenous=nm,
        nonnormality_of="first-stage residuals" if on_resid
        else "endogenous regressors",
        id_condition=obj.id_condition, thresholds=th,
        step1=step1, step2=step2, step3=step3, step4=step4,
        icon=icon, icon_max=float(np.nanmax(obj.icon.values)),
    )
