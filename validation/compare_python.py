"""Check the Python port against the R reference on validation/simdata.csv.

Bootstrap standard errors are not comparable across languages -- different RNGs
draw different resamples -- so only the point estimates, rho, and the CDF
estimators themselves are compared.  Those are deterministic and must agree to
near machine precision.

Run reference_R.R first.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "python"))

from copulaendog import cdf_estimate, copreg  # noqa: E402

warnings.simplefilter("ignore")

d = pd.read_csv(HERE / "simdata.csv")
d["g"] = pd.Categorical(d["g"])

RUNS = {
    "pg":      ("y ~ P | x + w + C(g)", dict(method="pg")),
    "s2cope":  ("y ~ P | x + w + C(g)", dict(method="2scope")),
    "ima":     ("y ~ P | x + w + C(g)", dict(method="ima")),
    "bmw":     ("y ~ P | x + w + C(g)", dict(method="bmw")),
    "jams":    ("y ~ P | x + w + C(g)", dict(method="jams")),
    "jams_f":  ("y ~ P | x + w + C(g)", dict(method="jams", conditional=False)),
    "pg_ecdf": ("y ~ P | x + w", dict(method="pg", cdf="ecdf.adj")),
    "pg_rank": ("y ~ P | x + w", dict(method="pg", cdf="rank.n")),
    "pg_plug": ("y ~ P | x + w", dict(method="pg", cdf="kde.plugin")),
    "pg_int":  ("y ~ P | x + w + x:w", dict(method="pg")),
}

# R names the columns model.matrix style; map them onto the patsy names so the
# two tables line up term by term.
RENAME = {
    "Intercept": "(Intercept)",
    "C(g)[T.1]": "g1",
    "C(g)[T.2]": "g2",
    "P_cop.C(g)[T.0]": "P_cop.g0",
    "P_cop.C(g)[T.1]": "P_cop.g1",
    "P_cop.C(g)[T.2]": "P_cop.g2",
    "x:w": "x:w",
}


def _clean(name: str) -> str:
    if name in RENAME:
        return RENAME[name]
    if name.startswith("P_cop.C(g)"):
        lvl = name.split("[T.")[-1].rstrip("]")
        return f"P_cop.g{lvl}"
    if name.startswith("P_cop") and "g" in name and "(base)" in name:
        return "P_cop.g0"
    return name


def main() -> int:
    rows = []
    for nm, (formula, kw) in RUNS.items():
        fit = copreg(formula, d, nboots=5, seed=1, verbose=False, **kw)
        for term, est in fit.params.items():
            rows.append({"model": nm, "term": _clean(term), "estimate": est})
        for term, est in fit.rho.items():
            rows.append({"model": nm, "term": f"rho.{term}", "estimate": est})
    py = pd.DataFrame(rows)

    r = pd.read_csv(HERE / "reference_R.csv")
    # R labels JAMS cells by the factor level; align the base cell's name

    m = r.merge(py, on=["model", "term"], how="outer",
                suffixes=("_R", "_py"))
    m["diff"] = (m["estimate_R"] - m["estimate_py"]).abs()

    unmatched = m[m["estimate_R"].isna() | m["estimate_py"].isna()]
    worst = m["diff"].max()

    print("\n=== coefficients and rho: R vs Python ===")
    print(m.sort_values("diff", ascending=False).head(12).to_string(index=False))
    print(f"\nmax |difference| over {m['diff'].notna().sum()} matched terms: "
          f"{worst:.3e}")
    if len(unmatched):
        print("\nterms present in one table only (naming differences, not "
              "numerical ones):")
        print(unmatched[["model", "term"]].to_string(index=False))

    # --- the CDF estimators, column by column ------------------------------
    ru = pd.read_csv(HERE / "reference_R_cdf.csv")
    print("\n=== marginal CDF estimators: max |R - Python| ===")
    cdf_ok = True
    for col in ru.columns:
        pu = cdf_estimate(d["P"].to_numpy(), col, "max")
        dd = np.abs(ru[col].to_numpy() - pu).max()
        cdf_ok &= dd < 1e-9
        print(f"  {col:<15s} {dd:.3e}")

    ok = (worst < 1e-8) and cdf_ok and unmatched.empty
    print("\nPASS" if ok else "\nCHECK THE DIFFERENCES ABOVE")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
