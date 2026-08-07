# How to verify this port

This file exists so that someone who did not write the code — a reviewer, a
referee, or another AI agent — can check every claim it makes without taking
anything on trust. Each section states a claim, gives the command that tests
it, and gives the result to expect.

Nothing here needs the author's machine. It does need R, Python and Stata,
because the whole point is a three-way comparison.

**Read this first.** The central claim is not "the tests pass". It is that
**this port reproduces the R reference implementation to within numerical
noise**. Tests that only compare the port to itself would prove nothing. The
checks below therefore run the original R code and diff against it.

---

## 0. What is being verified

| Claim | Where it is asserted | Section below |
|---|---|---|
| Python matches R to ~1e-12 | README, vignette §Verification | [3](#3-python-vs-r) |
| Stata matches R to ~1e-12 | README, vignette §Verification | [4](#4-stata-vs-r) |
| The CDF estimators match R | README, vignette | [3](#3-python-vs-r), [4](#4-stata-vs-r) |
| The package works on old and new dependency stacks | README §Install | [2](#2-python-test-suite) |
| Both example do-files run cold | README §Try it | [5](#5-the-run-it-and-go-do-files) |
| The published artifacts match the tag | release notes | [6](#6-published-artifacts) |
| Attribution reaches every channel | README §Credit | [7](#7-attribution) |

---

## 1. Prerequisites

```bash
python3 -m pip install numpy scipy pandas patsy pytest
```

```r
install.packages("Formula")
```

Stata 16 or later. The figures quoted here were produced under StataNow/SE
19.5 (Apple Silicon); `c(edition_real)` reports `SE`, though note that
`c(flavor)` reports `IC` for legacy reasons — do not be misled by it.

### Getting the R reference implementation

This is the step that is easy to get wrong. **The R estimator files are not on
the default branch** of the upstream repository — they live on a branch called
`functions`.

```bash
git clone https://github.com/HashtagHaschka/Copula-based-endogeneity-corrections
```

```bash
cd Copula-based-endogeneity-corrections && git fetch origin functions && mkdir -p /tmp/copreg-fn && for f in Copreg_core.R Copreg_pg.R Copreg_2scope.R Copreg_ima.R Copreg_jams.R Copreg_bmw.R; do git show "origin/functions:$f" > "/tmp/copreg-fn/$f"; done
```

`Copreg_core.R` holds the shared machinery and must be sourced first;
`validation/reference_R.R` already does that in the right order.

---

## 2. Python test suite

**Claim.** 40 tests pass, on the current and the previous dependency stack.

```bash
cd python && python3 -m pytest tests -q
```

Expect `40 passed`. The suite is not a smoke test — it asserts substantive
properties, and several tests are written to fail if the port drifts:

- every estimator moves the coefficient *closer to the known truth* than OLS
- the O(n log n) Epanechnikov CDF equals the naive O(n²) double sum to 1e-12
- the plug-in bandwidth is equivariant under rescaling
- `ties()` changes the answer only when there are ties
- a normally distributed endogenous regressor is **refused**, because the
  copula term is then collinear with the regressor and the model is not
  identified
- the validity report flags the assumption the simulated design deliberately
  violates

To check the forward-compatibility claim, run it again against the newest
releases of numpy, scipy and pandas in a throwaway environment:

```bash
python3 -m venv /tmp/newstack && /tmp/newstack/bin/pip install -q --upgrade numpy scipy pandas patsy pytest && /tmp/newstack/bin/python -m pytest python/tests -q
```

---

## 3. Python vs R

**Claim.** Across ten model specifications, all 75 coefficients and ρ values
agree with the R implementation to ~1e-12, and six CDF estimators agree to
machine precision.

```bash
cd validation && COPREG_FN_DIR=/tmp/copreg-fn Rscript reference_R.R
```

```bash
cd validation && python3 compare_python.py
```

Expect a table of per-term differences, then:

```
max |difference| over 75 matched terms: 9.390e-13

=== marginal CDF estimators: max |R - Python| ===
  kde.silverman   2.394e-13
  kde.plugin      1.332e-15
  ecdf.fixed      0.000e+00
  ecdf.adj        1.110e-16
  rank.n          3.331e-16
  rank.n1         5.551e-16

PASS
```

The script exits non-zero and prints `CHECK THE DIFFERENCES ABOVE` if any term
exceeds 1e-8 or fails to match by name.

The ten specifications cover all five estimators, four CDF estimators, an
interaction term, and JAMS both with and without conditioning — chosen so that
every distinct code path in the constructors is exercised, not just the
default one.

`kde.silverman` is the loosest of the six at 2.4e-13, and that is expected: it
is the only estimator here that accumulates a kernel sum over every
observation, so the two languages round differently. The others are closed-form
rank arithmetic and agree exactly or nearly so.

---

## 4. Stata vs R

**Claim.** The same ten specifications agree to 1.34e-12, and the six CDF
estimators Stata implements agree to 1.85e-13.

```bash
cd validation && stata-se -b do stata_selftest.do
```

Then read `stata_selftest.log`. Expect it to end in:

```
ALL CHECKS PASSED
```

and to contain zero lines matching `^r([0-9]+);` — that pattern is how Stata
reports an error, and the do-file wraps risky calls in `capture`, so a silent
failure would otherwise be invisible:

```bash
grep -cE "^r\([0-9]+\);" validation/stata_selftest.log
```

Expect `0`.

The self-test does three things. It runs every estimator plus the validity
report, factor variables and interactions in `exog()`, two endogenous
regressors, `generate()` and all four `predict` statistics. It recovers the
marginal CDFs through `generate()` and compares them against
`reference_R_cdf.csv`. And it asserts every coefficient and ρ against
`reference_R.csv` rather than printing them for a human to eyeball.

It also crosses the `kde.plugin` size gate deliberately, at n = 400, 1499,
1500, 1501 and 2500, because `ce_cdf_gauss()` and `ce_kfe()` switch from an
exact evaluation to a binned one at `ce_exactmax() = 1500`. A failure on one
side of that boundary but not the other points at one of those two functions.

### One trap if you write your own comparison

`import delimited` stores numeric columns as **float** unless you pass
`asdouble`. Float carries about seven significant digits, which alone puts a
~1e-8 floor under any agreement you try to measure — and 1e-8 is the tolerance
being tested. Every import in the self-test passes `asdouble` for this reason.

---

## 5. The run-it-and-go do-files

**Claim.** Downloading one file and running it works, with nothing else
fetched and nothing installed.

The honest test is a cold start in an empty directory, not a run from a clone:

```bash
mkdir /tmp/coldstart && cd /tmp/coldstart && curl -sO https://raw.githubusercontent.com/girishm77/copulaendog_stata_python/main/stata/quickstart.do && stata-se -b do quickstart.do
```

Expect `quickstart.log` to contain `Downloaded copulaendog into …`, no `r(…);`
lines, and a coefficient on `P` near 2 from every estimator against an OLS
estimate near 2.17 — the data are simulated so the truth is known to be
exactly 2.

Repeat with `examples.do` for the full option tour.

---

## 6. Published artifacts

**Claim.** What is on PyPI and attached to the GitHub release is the code at
tag `v0.1.0`.

```bash
pip download --no-deps --no-binary :all: -d /tmp/pv copulaendog && tar xzf /tmp/pv/copulaendog-0.1.0.tar.gz -C /tmp/pv
```

```bash
for f in __init__ cdf constructors core diagnostics estimators formula simulate; do a=$(git show v0.1.0:python/copulaendog/$f.py | shasum -a 256 | cut -c1-16); b=$(shasum -a 256 /tmp/pv/copulaendog-0.1.0/python/copulaendog/$f.py | cut -c1-16); [ "$a" = "$b" ] && echo "match $f.py" || echo "DIFFERS $f.py"; done
```

Expect eight `match` lines.

Note that the **archive checksums are not reproducible**: Python embeds
timestamps in wheels and sdists, so rebuilding from the same source gives a
different sha256. Compare file contents, as above, not archive hashes.

The Stata bundle attached to the release was built with `git show v0.1.0:…`
rather than from the working tree, so it cannot contain anything committed
after the tag. It is byte-identical to the archive submitted to SSC.

---

## 7. Attribution

**Claim.** Rouven E. Haschka and Ashwin Malshe are credited on every channel
this is distributed through, in what that channel actually renders.

| Channel | Check |
|---|---|
| GitHub | `## Credit` is the first section after the summary in `README.md`; also `CITATION.cff`, `LICENSE.ADDITIONAL-TERMS.md`, and a header in every `python/copulaendog/*.py` |
| PyPI page body | the README is the long description — `curl -s https://pypi.org/pypi/copulaendog/json \| grep -c Haschka` |
| PyPI sidebar | `curl -s https://pypi.org/pypi/copulaendog/json \| python3 -c "import json,sys; print(json.load(sys.stdin)['info']['project_urls'])"` — expect labels naming both authors |
| Installed package | `LICENSE` and `LICENSE.ADDITIONAL-TERMS.md` must be present in `dist-info/licenses/`; the GPL section 7(b) term requires the attribution to travel with the code |
| SSC | `net describe copulaendog` renders the `.pkg` description, which carries the full credit |
| Stata help | `help copulaendog`, then the Author and References sections |

Verifying the installed-package case:

```bash
python3 -c "import importlib.metadata as md; print([p.name for p in md.distribution('copulaendog').files if 'LICENSE' in p.name])"
```

Expect both licence files.

---

## 8. What is deliberately not claimed

Being explicit about the boundaries, so nobody has to guess:

- **Three estimators are not ported** — 2sCOPE-np, PANEL, BAYES. The R
  implementation has them; this does not.
- **`kde.cv` is Python-only.** The Stata command rejects it with an error
  rather than silently substituting a different bandwidth.
- **No claim is made about the estimators' statistical properties.** That is
  the business of the papers listed in the README. What is claimed is
  faithfulness of translation, which is what sections 3 and 4 test.
- **Bootstrap standard errors are not compared across languages**, and cannot
  be: R, Python and Stata have different RNGs, so they draw different
  resamples. Only the deterministic quantities — point estimates, ρ, the CDF
  transforms — are diffed. Set a seed and SEs are reproducible *within* a
  language, which section 2 tests.
- **The simulated data are favourable by construction.** They satisfy the
  Gaussian copula exactly and the endogenous regressor is strongly
  non-normal. That is the right design for testing a translation, and the
  wrong design for judging how the estimators behave when their assumptions
  fail.

---

## 9. If a check fails

- **A Python term differs but the CDFs match** — look at the constructor for
  that estimator in `python/copulaendog/constructors.py`. The first stage is
  where the estimators differ from one another.
- **Every Python term differs** — check that `reference_R.R` was run against
  the `functions` branch, not the default branch, and that the R `Formula`
  package is installed.
- **Stata fails only at some sample sizes** — see the `kde.plugin` size gate
  in section 4.
- **Stata agrees only to ~1e-8** — you are almost certainly importing without
  `asdouble`.
- **A Stata factor-variable model errors** — `fvexpand` and `fvrevar` disagree
  about base levels; `copulaendog.ado` builds its indicator columns from
  `_ms_parse_parts` for exactly this reason. See the project log.
