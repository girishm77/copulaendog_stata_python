"""A tour of the Python package.

Run it with the package on the path:

    PYTHONPATH=../ python3 example.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from copulaendog import (  # noqa: E402
    CopRegBMW,
    CopRegPG,
    copreg,
    copula_transform,
    sim_endog,
)

warnings.simplefilter("ignore")
pd.set_option("display.width", 110)


# ---------------------------------------------------------------------------
# 1. the data
# ---------------------------------------------------------------------------
# P = exp(0.8 v + 0.6 w) with corr(v, xi) = 0.6, and
#     y = 1 + 2 P + 1.5 x - 0.5 w + xi
#
# P is monotone in a normal variate, so the Gaussian copula holds exactly,
# while P itself is lognormal and therefore strongly non-normal -- which is
# what identifies the correction.  The 0.6 w term makes corr(W, P*) != 0, so
# the Park & Gupta assumption fails by construction.
df = sim_endog(n=2000, seed=123)
print(df.describe().round(3), "\n")

X = np.column_stack([np.ones(len(df)), df.P, df.x, df.w])
ols = np.linalg.lstsq(X, df.y, rcond=None)[0]
print(f"uncorrected OLS on P: {ols[1]:.4f}   (the truth is 2.0000)\n")


# ---------------------------------------------------------------------------
# 2. every estimator on the same data
# ---------------------------------------------------------------------------
print("=" * 66)
print("all five estimators, y ~ P | x + w")
print("=" * 66)
rows = []
for m in ["pg", "2scope", "ima", "bmw", "jams"]:
    fit = copreg("y ~ P | x + w", df, method=m, nboots=199, seed=1,
                 verbose=False)
    rows.append({
        "method": m,
        "alpha": fit.params["P"],
        "se": fit.bse["P"],
        "rho": fit.rho.iloc[0],
        "max ICON": fit.icon.max(),
    })
print(pd.DataFrame(rows).set_index("method").round(4), "\n")


# ---------------------------------------------------------------------------
# 3. the full output of one fit
# ---------------------------------------------------------------------------
fit = copreg("y ~ P | x + w", df, method="2scope", nboots=199, seed=1,
             verbose=False)
print(fit.summary())

print("percentile confidence intervals:")
print(fit.conf_int(0.95, kind="percentile").round(4), "\n")


# ---------------------------------------------------------------------------
# 4. the validity report
# ---------------------------------------------------------------------------
# Run it on Park & Gupta, which is the estimator whose assumption this data
# violates, and watch it say so.
print(CopRegPG("y ~ P | x + w", df, nboots=199, seed=1,
               verbose=False).validity())


# ---------------------------------------------------------------------------
# 5. BMW reports a Durbin-Hausman-Wu test as well
# ---------------------------------------------------------------------------
bmw = CopRegBMW("y ~ P | x + w", df, nboots=199, seed=1, verbose=False)
print(bmw.summary())


# ---------------------------------------------------------------------------
# 6. formulas: transformations, interactions, categoricals
# ---------------------------------------------------------------------------
rng = np.random.default_rng(7)
df["store"] = pd.Categorical(rng.integers(0, 3, len(df)))
df["y2"] = df.y + 0.4 * df.store.cat.codes

specs = [
    "y2 ~ P | x + w + C(store)",       # a categorical control
    "y2 ~ P | x + w + x:w",            # an interaction among the controls
    "y2 ~ P | np.log(x**2 + 1) + w",   # a transformation
]
for s in specs:
    f = copreg(s, df, method="2scope", nboots=99, seed=1, verbose=False)
    print(f"{s:38s} alpha = {f.params['P']:.4f} ({f.bse['P']:.4f})")
print()


# ---------------------------------------------------------------------------
# 7. JAMS with the copula structure varying across the stores
# ---------------------------------------------------------------------------
jam = copreg("y2 ~ P | x + w + C(store)", df, method="jams", nboots=99,
             seed=1, verbose=False)
print(jam.summary())


# ---------------------------------------------------------------------------
# 8. the copula transformation on its own
# ---------------------------------------------------------------------------
# Useful when the model is not one this package covers: build the term here and
# add it to whatever estimator you are running.
for cdf in ["kde.silverman", "kde.plugin", "rank.n", "ecdf.adj"]:
    c = copula_transform(df.P.to_numpy(), cdf, "max")
    print(f"{cdf:16s} mean {c.mean():7.4f}  sd {c.std(ddof=1):6.4f}  "
          f"corr(P, C) {np.corrcoef(df.P, c)[0, 1]:.4f}")
