"""Tests for the Python port.

Run with:  cd python && python3 -m pytest tests -q
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from copulaendog import (  # noqa: E402
    CDF_CHOICES,
    CopReg2sCOPE,
    CopRegBMW,
    CopRegIMA,
    CopRegJAMS,
    CopRegPG,
    cdf_estimate,
    copreg,
    copula_transform,
    sim_endog,
)
from copulaendog.cdf import bw_plugin, bw_silverman, cdf_kde_epan  # noqa: E402

warnings.simplefilter("ignore")

METHODS = ["pg", "2scope", "ima", "bmw", "jams"]


@pytest.fixture(scope="module")
def data():
    df = sim_endog(n=600, seed=11)
    rng = np.random.default_rng(4)
    df["g"] = pd.Categorical(rng.integers(0, 3, len(df)))
    return df


# ---------------------------------------------------------------- CDFs -----
@pytest.mark.parametrize("cdf", [c for c in CDF_CHOICES if c != "kde.cv"])
def test_cdf_in_open_unit_interval(cdf, data):
    u = cdf_estimate(data["P"].to_numpy(), cdf, "max")
    assert np.all(u > 0) and np.all(u < 1)
    assert np.all(np.isfinite(copula_transform(data["P"].to_numpy(), cdf)))


@pytest.mark.parametrize("cdf", [c for c in CDF_CHOICES if c != "kde.cv"])
def test_cdf_is_monotone_in_x(cdf, data):
    x = data["P"].to_numpy()
    u = cdf_estimate(x, cdf, "max")
    o = np.argsort(x)
    assert np.all(np.diff(u[o]) >= -1e-12)


def test_epan_cdf_matches_the_naive_double_sum():
    """The prefix-sum shortcut must agree with the definition."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(300)
    b = bw_silverman(x)
    fast = cdf_kde_epan(x, b)
    u = np.clip((x[:, None] - x[None, :]) / b, -1, 1)
    slow = (0.75 * (u - u**3 / 3) + 0.5).mean(axis=1)
    assert np.allclose(fast, slow, atol=1e-12)


def test_ties_average_differs_only_with_ties():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(200)
    assert np.allclose(cdf_estimate(x, "rank.n1", "max"),
                       cdf_estimate(x, "rank.n1", "average"))
    xt = np.round(x, 1)                      # now there are ties
    assert not np.allclose(cdf_estimate(xt, "rank.n1", "max"),
                           cdf_estimate(xt, "rank.n1", "average"))


def test_plugin_bandwidth_is_positive_and_scales():
    rng = np.random.default_rng(2)
    x = rng.standard_normal(400)
    h1 = bw_plugin(x)
    h2 = bw_plugin(10 * x)
    assert h1 > 0
    assert np.isclose(h2 / h1, 10, rtol=1e-6)   # equivariant under scaling


# ---------------------------------------------------------- estimators -----
@pytest.mark.parametrize("method", METHODS)
def test_every_estimator_beats_ols(method, data):
    """The correction should move alpha towards the truth of 2.0."""
    X = np.column_stack([np.ones(len(data)), data.P, data.x, data.w])
    ols = np.linalg.lstsq(X, data.y, rcond=None)[0][1]
    fit = copreg("y ~ P | x + w", data, method=method, nboots=39, seed=1,
                 verbose=False)
    assert abs(fit.params["P"] - 2.0) < abs(ols - 2.0)
    assert abs(fit.params["P"] - 2.0) < 0.15


@pytest.mark.parametrize("method", METHODS)
def test_result_shapes(method, data):
    fit = copreg("y ~ P | x + w", data, method=method, nboots=39, seed=1,
                 verbose=False)
    k = len(fit.params)
    assert fit.vcov.shape == (k, k)
    assert len(fit.bse) == k
    assert fit.boot.shape[1] == k
    assert fit.nobs == len(data)
    assert np.allclose(fit.resid, data.y - fit.fittedvalues)
    assert fit.conf_int().shape == (k, 2)
    assert fit.conf_int(kind="percentile").shape == (k, 2)
    assert isinstance(fit.summary(), str)
    assert isinstance(str(fit.validity()), str)


def test_seed_makes_the_bootstrap_reproducible(data):
    a = copreg("y ~ P | x + w", data, nboots=25, seed=7, verbose=False)
    b = copreg("y ~ P | x + w", data, nboots=25, seed=7, verbose=False)
    assert np.allclose(a.bse.values, b.bse.values)
    c = copreg("y ~ P | x + w", data, nboots=25, seed=8, verbose=False)
    assert not np.allclose(a.bse.values, c.bse.values)


def test_point_estimates_do_not_depend_on_the_seed(data):
    a = copreg("y ~ P | x + w", data, nboots=25, seed=7, verbose=False)
    b = copreg("y ~ P | x + w", data, nboots=25, seed=8, verbose=False)
    assert np.allclose(a.params.values, b.params.values)


def test_two_stage_variants_are_close_but_not_identical(data):
    a = CopReg2sCOPE("y ~ P | x + w", data, nboots=15, seed=1, verbose=False)
    b = CopRegIMA("y ~ P | x + w", data, nboots=15, seed=1, verbose=False)
    d = abs(a.params["P"] - b.params["P"])
    assert 0 < d < 0.05


def test_bmw_reports_a_dhw_test_and_the_others_do_not(data):
    assert CopRegBMW("y ~ P | x + w", data, nboots=15, seed=1,
                     verbose=False).dhw_frame() is not None
    assert CopRegPG("y ~ P | x + w", data, nboots=15, seed=1,
                    verbose=False).dhw_frame() is None


def test_bmw_tests_the_first_stage_residuals_for_normality(data):
    v = CopRegBMW("y ~ P | x + w", data, nboots=15, seed=1,
                  verbose=False).validity()
    assert v.nonnormality_of == "first-stage residuals"
    v2 = CopRegPG("y ~ P | x + w", data, nboots=15, seed=1,
                  verbose=False).validity()
    assert v2.nonnormality_of == "endogenous regressors"


def test_validity_flags_the_assumption_this_design_violates(data):
    """The data are built with corr(W, P*) != 0, so PG's assumption fails."""
    v = CopRegPG("y ~ P | x + w", data, nboots=15, seed=1,
                 verbose=False).validity()
    assert v.step2 is not None and v.step2["violated"]
    # the two-stage estimators project it out, so the check does not apply
    v2 = CopReg2sCOPE("y ~ P | x + w", data, nboots=15, seed=1,
                      verbose=False).validity()
    assert v2.step2 is None


def test_jams_cells_and_the_wald_tests(data):
    fit = CopRegJAMS("y ~ P | x + w + C(g)", data, nboots=25, seed=1,
                     verbose=False)
    assert len(fit.copula_terms.columns) == 3        # one per category
    assert fit.wald_copula()["df"] == 3
    assert fit.wald_cells()["df"] == 2               # two contrasts
    flat = CopRegJAMS("y ~ P | x + w + C(g)", data, conditional=False,
                      nboots=25, seed=1, verbose=False)
    assert len(flat.copula_terms.columns) == 1
    assert flat.wald_cells() is None


# ------------------------------------------------------------ formulas -----
def test_categorical_endogenous_regressor_is_refused(data):
    with pytest.raises(ValueError, match="own copula term"):
        copreg("y ~ C(g) | x + w", data, nboots=5, verbose=False)


def test_no_exogenous_part_redirects_to_pg(data):
    with pytest.warns(UserWarning, match="identical to Park & Gupta"):
        fit = CopReg2sCOPE("y ~ P", data, nboots=5, verbose=False)
    assert "Park & Gupta" in fit.method


def test_intercept_can_be_dropped(data):
    fit = copreg("y ~ P | x + w - 1", data, nboots=5, verbose=False)
    assert "Intercept" not in fit.params.index
    assert not fit.has_intercept


def test_transformations_and_interactions_are_accepted(data):
    fit = copreg("y ~ P | x + w + x:w", data, method="2scope", nboots=5,
                 verbose=False)
    assert "x:w" in fit.params.index
    fit2 = copreg("y ~ P | np.log(x**2 + 1) + w", data, nboots=5,
                  verbose=False)
    assert any("log" in c for c in fit2.params.index)


def test_normal_endogenous_regressor_is_weakly_identified():
    """With P normal, Phi^-1(F(P)) is nearly affine in P.

    The estimated CDF is not exactly the normal one, so the design does not
    become literally singular -- it becomes very nearly so.  That is the
    failure ICON is built to catch, and it shows up as a huge standard error
    inflation and a copula term with almost no independent variation.
    """
    rng = np.random.default_rng(3)
    n = 400
    v = rng.standard_normal(n)
    e = 0.7 * v + np.sqrt(1 - 0.49) * rng.standard_normal(n)
    df = pd.DataFrame({"P": v, "x": rng.standard_normal(n)})
    df["y"] = 1 + 2 * df.P + df.x + e

    fit = copreg("y ~ P | x", df, cdf="rank.n1", nboots=99, seed=1,
                 verbose=False)
    assert fit.icon.max() > 6                       # weak identification
    assert fit.diagnostics["collinearity"]["omega"].iloc[0] < 0.05
    assert fit.validity().icon_max > 6


def test_predict_excludes_the_copula_terms(data):
    fit = copreg("y ~ P | x + w", data, nboots=5, verbose=False)
    p = fit.predict(data)
    assert np.allclose(p, fit.fittedvalues)
    # the augmented fit is different, which is the point
    assert not np.allclose(p, fit.fitted_augmented)


def test_bad_arguments_are_rejected(data):
    with pytest.raises(ValueError, match="cdf"):
        copreg("y ~ P | x", data, cdf="nonsense", nboots=5, verbose=False)
    with pytest.raises(ValueError, match="ties"):
        copreg("y ~ P | x", data, ties="nonsense", nboots=5, verbose=False)
    with pytest.raises(ValueError, match="nboots"):
        copreg("y ~ P | x", data, nboots=1, verbose=False)
    with pytest.raises(ValueError, match="not implemented"):
        copreg("y ~ P | x", data, method="nonsense", nboots=5, verbose=False)


def test_missing_values_are_dropped(data):
    df = data.copy()
    df.loc[df.index[:10], "x"] = np.nan
    fit = copreg("y ~ P | x + w", df, nboots=5, verbose=False)
    assert fit.nobs == len(df) - 10
