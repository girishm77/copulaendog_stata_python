"""Two-part formula handling.

    y ~ endogenous_1 + endogenous_2 + ... | exogenous_1 + exogenous_2 + ...

Position decides.  A term written before the "|" is endogenous and receives a
copula term; written after it the same term is exogenous and receives none.
A third part, "| discrete", is accepted and used by JAMS.

Everything patsy accepts works: transformations np.log(price), interactions
price:feat, powers I(x**2), categoricals C(store), and "- 1" to drop the
intercept.  Rows with missing values are dropped.

This is the Python counterpart of .copreg_model() in Copreg_core.R.
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

import re
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import patsy


@dataclass
class ModelInfo:
    """Everything the estimators need to know about the design."""

    y: np.ndarray
    X: np.ndarray
    xnames: list
    frame: pd.DataFrame            # rows that survived na-dropping
    endo_cols: np.ndarray          # column indices of the endogenous regressors
    exo_cols: np.ndarray
    endo_names: list
    part: np.ndarray               # 1 endogenous, 2 exogenous, 3 discrete
    order: np.ndarray              # interaction order, 0 for the intercept
    term_label: list               # term label behind each column
    factor_cols: np.ndarray        # columns produced by a categorical variable
    has_intercept: bool
    has_exog: bool
    formula: str
    design_info: object = field(default=None, repr=False)
    n_dropped: int = 0

    @property
    def n(self) -> int:
        return self.X.shape[0]


_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _split_formula(formula: str):
    if "~" not in formula:
        raise ValueError("The formula needs a '~'.")
    lhs, rhs = formula.split("~", 1)
    lhs = lhs.strip()
    if not lhs:
        raise ValueError("The formula must have exactly one dependent variable.")
    parts = [p.strip() for p in rhs.split("|")]
    if len(parts) > 3:
        raise ValueError(
            "The formula must have at most three parts: "
            "endogenous | exogenous | discrete."
        )
    if not parts[0]:
        raise ValueError(
            "No endogenous regressor found in the first part of the formula."
        )
    return lhs, parts


def _term_vars(term, data_cols) -> set:
    """Variable names occurring in a patsy term, restricted to data columns."""
    out = set()
    for f in term.factors:
        code = getattr(f, "code", str(f))
        out |= {m for m in _NAME_RE.findall(code) if m in data_cols}
    return out


def build_model(
    formula: str,
    data: pd.DataFrame,
    subset=None,
    verbose: bool = True,
) -> ModelInfo:
    """Build y, X and the bookkeeping the estimators need."""
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    if subset is not None:
        data = data.loc[np.asarray(subset)]

    lhs, parts = _split_formula(formula)

    # The parts are handed to patsy as written, so that "- 1" or "+ 0" in any
    # of them is interpreted by patsy rather than by a regular expression here.
    # Whether an intercept survived is then read off the design itself.
    cleaned = [p for p in parts if p]
    full = f"{lhs} ~ " + " + ".join(cleaned)

    y_dm, X_dm = patsy.dmatrices(full, data, return_type="dataframe",
                                 NA_action="drop")
    has_intercept = "Intercept" in X_dm.columns
    n_dropped = len(data) - len(X_dm)
    if n_dropped > 0 and verbose:
        warnings.warn(
            f"{n_dropped} observation(s) removed because of missing values.",
            stacklevel=2,
        )

    di = X_dm.design_info
    X = np.asarray(X_dm, dtype=float)
    y = np.asarray(y_dm, dtype=float).ravel()
    xnames = list(X_dm.columns)

    if X.shape[0] <= X.shape[1]:
        raise ValueError(
            f"Not enough complete observations: {X.shape[0]} rows for "
            f"{X.shape[1]} regressors."
        )

    # --- term label and interaction order behind every column ---------------
    term_label = [""] * X.shape[1]
    order = np.zeros(X.shape[1], dtype=int)
    for term, slc in di.term_slices.items():
        name = di.term_names[di.terms.index(term)]
        for j in range(slc.start, slc.stop):
            term_label[j] = name
            order[j] = len(term.factors)

    # --- which formula part does each column come from? ---------------------
    part = np.zeros(X.shape[1], dtype=int)
    part_terms = []
    for q, rhs in enumerate(parts, start=1):
        if not rhs:
            part_terms.append([])
            continue
        dq = patsy.dmatrix(f"{rhs} - 1", data, return_type="dataframe",
                           NA_action="drop").design_info
        labels = set(dq.term_names)
        part_terms.append(sorted(labels))
        for j, lab in enumerate(term_label):
            if lab in labels:
                part[j] = q

    # --- endogenous columns -------------------------------------------------
    # An endogenous term must map to exactly one column carrying its own name.
    # A categorical, or an interaction involving one, expands into several
    # columns and has no single copula term.
    endo_labels = part_terms[0]
    endo_cols, bad = [], []
    for lab in endo_labels:
        j = [k for k, l in enumerate(term_label) if l == lab]
        if len(j) != 1 or xnames[j[0]] != lab:
            bad.append(lab)
        else:
            endo_cols.append(j[0])
    if bad:
        raise ValueError(
            'Every term before the "|" needs its own copula term, so it has to '
            "map to a single numeric column. These do not: "
            + ", ".join(bad)
            + '. A categorical, or an interaction involving one, expands into '
            'several columns. Move it behind the "|" to treat it as exogenous.'
        )
    endo_cols = np.array(sorted(endo_cols), dtype=int)
    endo_names = [xnames[j] for j in endo_cols]
    exo_cols = np.array([j for j in range(X.shape[1]) if j not in set(endo_cols)],
                        dtype=int)

    for j in endo_cols:
        if np.unique(X[:, j]).size < 2:
            raise ValueError(f"Endogenous regressor {xnames[j]!r} is constant.")

    # --- columns produced by a categorical variable -------------------------
    cat_factors = {
        f.name()
        for f, fi in di.factor_infos.items()
        if fi.type == "categorical"
    }
    factor_cols = []
    for term, slc in di.term_slices.items():
        if any(f.name() in cat_factors for f in term.factors):
            factor_cols.extend(range(slc.start, slc.stop))
    factor_cols = np.array(sorted(factor_cols), dtype=int)

    # --- endogenous terms built from the same variable ----------------------
    data_cols = set(map(str, data.columns))
    term_by_label = {
        di.term_names[di.terms.index(t)]: t for t in di.terms
    }
    evars = {
        lab: _term_vars(term_by_label[lab], data_cols)
        for lab in endo_labels
        if lab in term_by_label
    }
    shared = [
        a
        for a in evars
        if any(evars[a] & evars[b] for b in evars if b != a)
    ]
    if shared and verbose:
        warnings.warn(
            "These endogenous terms are built from the same variable(s) and each "
            "gets its own copula term: " + ", ".join(sorted(shared))
            + ". Such terms are strongly collinear with one another, which Qian, "
            "Koschmann & Xie (2025) show inflates the variance of the structural "
            'estimates. Moving the higher-order term behind the "|" treats it as '
            "exogenous and leaves it without a copula term.",
            stacklevel=2,
        )

    # --- ties in the endogenous regressors ----------------------------------
    # What breaks the transformation is a plateau in the CDF: a point mass
    # makes the inverse non-unique (Qian & Xie 2024).
    msgs = []
    for j in endo_cols:
        v = X[:, j]
        _, cnt = np.unique(v, return_counts=True)
        nu, mx = cnt.size, cnt.max() / v.size
        if nu <= 20 or mx > 0.05:
            msgs.append(
                f"{xnames[j]} ({nu} distinct values, largest tie group "
                f"{100 * mx:.1f}%)"
            )
    if msgs and verbose:
        warnings.warn(
            "Endogenous regressor(s) with substantial ties: " + "; ".join(msgs)
            + ". Copula control functions invert an estimated CDF; a point mass "
            "creates a plateau whose inverse is not unique (Qian & Xie 2024).",
            stacklevel=2,
        )

    return ModelInfo(
        y=y,
        X=X,
        xnames=xnames,
        frame=data.loc[X_dm.index],
        endo_cols=endo_cols,
        exo_cols=exo_cols,
        endo_names=endo_names,
        part=part,
        order=order,
        term_label=term_label,
        factor_cols=factor_cols,
        has_intercept=has_intercept,
        has_exog=len(parts) > 1 and bool(parts[1]),
        formula=formula,
        design_info=di,
        n_dropped=n_dropped,
    )


def term_variables(info: ModelInfo, label: str) -> list:
    """Data columns entering the term with this label."""
    di = info.design_info
    for term in di.terms:
        if di.term_names[di.terms.index(term)] == label:
            cols = set(map(str, info.frame.columns))
            return sorted(_term_vars(term, cols))
    return []


def exog_cols(info: ModelInfo, X: np.ndarray | None = None) -> np.ndarray:
    """Columns of the exogenous part(s) usable as W in a first stage.

    A term qualifies when none of the variables in it is endogenous.  That
    keeps an interaction with an endogenous regressor out -- it is not
    exogenous information, whichever side of the "|" it was written on -- while
    letting genuine transformations of the exogenous regressors in, including
    w1:w2 and I(w**2), which is what Breitung, Mayer & Wied (2024) recommend
    under their Assumption A4.
    """
    if X is None:
        X = info.X
    di = info.design_info
    data_cols = {
        f.name() for f in di.factor_infos
    } | {re.sub(r"\W", "", f.name()) for f in di.factor_infos}
    # variables are recovered from the term labels themselves
    label_vars = {}
    for term in di.terms:
        lab = di.term_names[di.terms.index(term)]
        label_vars[lab] = {
            m
            for f in term.factors
            for m in _NAME_RE.findall(getattr(f, "code", str(f)))
        }

    endo_raw = set()
    for lab in {info.term_label[j] for j in info.endo_cols}:
        endo_raw |= label_vars.get(lab, set())
    exo_main = {
        info.term_label[j]
        for j in range(len(info.term_label))
        if info.part[j] >= 2 and info.order[j] == 1
    }
    exo_main_vars = set()
    for lab in exo_main:
        exo_main_vars |= label_vars.get(lab, set())
    endo_vars = endo_raw - exo_main_vars

    keep = []
    for j in range(X.shape[1]):
        lab = info.term_label[j]
        if info.part[j] < 2 or not lab:
            continue
        if label_vars.get(lab, set()) & endo_vars:
            continue
        if np.unique(X[:, j]).size > 1:
            keep.append(j)
    return np.array(keep, dtype=int)
