# Project log

A record of how this port was built, what was decided and why, and every
defect found along the way. Written so that a reviewer can tell which parts
are load-bearing and which were judgement calls — and so that the next person
to touch the code knows where the traps are.

For *how to check the claims*, see [VERIFICATION.md](VERIFICATION.md). This
file is the narrative; that one is the runbook.

---

## Provenance

Two upstream repositories, both R:

| Source | Role |
|---|---|
| [HashtagHaschka/Copula-based-endogeneity-corrections](https://github.com/HashtagHaschka/Copula-based-endogeneity-corrections) | The reference implementation. Every estimator, CDF estimator, the bootstrap, the diagnostics and the validity report are ported from it. |
| [ashgreat/endogCopula](https://github.com/ashgreat/endogCopula) | The packaged R version of the same estimators. Shaped the module layout and the documented interface. |

**The estimator code is on the `functions` branch** of the first repository,
not on `main`. `main` holds only the README, `quickstart.R` and `1EXAMPLES`.
This cost time to discover and is the single most likely thing to trip up
anyone trying to reproduce the comparison.

`endogCopula` is itself a port of Haschka's code, with Haschka named as
copyright holder of the reference implementation. So the two sources are not
independent, and agreement between them is not evidence of correctness — only
agreement with the papers would be. What this port claims is faithfulness to
Haschka's code, which is what the verification measures.

### Licence chain

Upstream is GPL-3-or-later **with an additional term under section 7(b)**
requiring that author attribution be preserved in anything conveying the
material, modified versions and larger works included. This port is a derived
work, so it carries the same licence and the same term. That is why the
attribution is not merely in the README but in every source header, in the
`.pkg` description that `ssc describe` renders, and inside the built Python
distribution — see VERIFICATION.md §7.

---

## Scope decisions

**Ported:** the five cross-sectional least-squares estimators — `pg`,
`2scope`, `ima`, `bmw`, `jams` — with seven CDF estimators in Python and six
in Stata, pairs-bootstrap inference, the diagnostics, and the validity report.

**Not ported, deliberately:**

- **2sCOPE-np** (Hu, Qian & Xie 2025) needs a nonparametric conditional CDF
  via the kernel method of Li & Racine, which upstream delegates to R's `np`
  package. Reimplementing that faithfully in Mata and NumPy is a project in
  itself, and it is the only estimator in the family that tolerates discrete
  endogenous regressors — so a half-working version would be worse than none.
- **PANEL** (Haschka 2022) is a different model class with its own likelihood
  and a forward orthogonal deviations transform.
- **BAYES** (Haschka 2025b) replaces bootstrap inference with an MCMC sampler
  in which the marginal CDFs are drawn rather than plugged in.

**`kde.cv` is Python-only.** The cross-validated bandwidth of Li, Li & Racine
(2017) is an O(n²) optimisation inside every bootstrap replicate. Rather than
silently substitute another bandwidth, the Stata command rejects
`cdf(kde.cv)` with an error naming the alternative.

---

## Defects found, and how

Listed because each one is a place the port could still be wrong in some way
not yet tested, and because the *method* that caught each is more reusable
than the fix.

### 1. Python `rho` came back as NaN — pandas index alignment

`rho_frame()` built a DataFrame from Series whose index was the variable name
while passing `index=[f"rho({v}*, xi*)"]`. pandas aligned on the index, found
no overlap, and produced NaN. **Caught by** looking at the printed summary
rather than trusting that the object had been constructed. Fixed by taking
`.values`.

### 2. Mata does not broadcast — `kde.plugin` crashed on every n ≤ 1500

`ce_cdf_gauss()` and `ce_kfe()` subtracted a 1×n row from a k×1 block,
expecting NumPy/R-style broadcasting. Mata's colon operators require one
operand to already have the full shape. Both sites now widen explicitly.

This is the most important entry in this log, for two reasons. It was
**invisible to static review** — the code reads correctly if you carry
NumPy's semantics in your head. And it was reachable only from
`cdf(kde.plugin)` on the exact-evaluation path, so it hit every sample of
1500 observations or fewer and nothing larger. The self-test now crosses that
boundary at n = 400, 1499, 1500, 1501, 2500 precisely so a regression here
cannot hide.

Commit `43b17f9`.

### 3. Factor variables in `exog()` never worked

`fvexpand` drops the base level from the name list while `fvrevar` keeps it,
so the two lists had different lengths and the command exited. Pointing
`fvrevar` at the filtered list is not a fix either: handed specific levels it
*rebases* them and returns an all-zero column for whichever it picks as base.
The kept columns are now built from what `_ms_parse_parts` reports, covering
plain variables, factor levels and interactions.

A related trap: `fvrevar` and `generate` both reset `r()`, so the parse must
be copied into locals before anything else runs.

Commit `43b17f9`.

### 4. The self-test could not complete

Its CDF section called `ce_cdf()` from the Mata prompt, but functions defined
inside an ado file are not visible there — it aborted with `r(3499)` and the
coefficient comparison never ran. It now recovers the marginal CDF through
the public interface (`generate()` plus `normal()`), and asserts rather than
printing for a human to compare.

**A test that cannot fail loudly is worse than no test.** This one printed
numbers for eyeballing; it now exits with a count of failures.

### 5. scipy 1.18 broke `kstest(..., args=)`

`ks_normal()` called `stats.kstest(v, "norm", args=(mean, sd))`. scipy 1.18
reaches `ndtr()` with three positional arguments and raises `TypeError`, so
**every fit died in the diagnostics** on a current scipy. The local scipy was
1.16, which is why the whole test suite passed while the package was broken
for anyone installing fresh.

Fixed by standardising the sample and testing against the standard normal —
the identical test, since the ECDF of (v−m)/s against Φ is the ECDF of v
against Φ((·−m)/s).

**Caught by** installing the built wheel into a clean virtualenv rather than
testing in the development environment. Testing against only the versions you
happen to have pinned is not testing forward compatibility. Commit `b4068d9`.

### 6. The package was not installable, and shipped no licence

`pyproject.toml` sat in `python/` with `readme = "../README.md"`, which
setuptools resolved to nothing: the wheel carried no description and neither
licence file. The missing description is cosmetic; **the missing licence is a
GPL-compliance gap in a redistributed artifact**, and the section 7(b)
attribution term makes it worse.

Moved to the repository root with `package-dir`, which also makes
`pip install git+https://…` work without a subdirectory fragment. Commit
`b4068d9`.

### 7. patsy 1.0.2 imports `packaging` without declaring it

A clean install died at import time with `ModuleNotFoundError: packaging`.
Upstream's bug, but ours to absorb: `packaging` is now declared as a
dependency so users do not have to work around someone else's metadata.

### 8. The `ties()` example demonstrated nothing

`examples.do` claimed to show `ties(max)` differing from `ties(average)`, but
ran under `method(pg)`, whose default `cdf(kde.silverman)` is kernel-based and
never looks at ranks. Both settings printed the same number under a comment
saying they would differ. It now contrasts `rank.n1`, where the setting moves
the estimate, against `kde.silverman`, where it correctly does not.

**Documentation that asserts a behaviour is a test.** This one was failing.

### 9. CRLF churn across machines

Four files round-tripped through a Windows session came back with CRLF line
endings, showing as 1912 changed lines with zero content change.
`.gitattributes` now pins LF.

---

## Judgement calls

**The tag was moved once, then deliberately not moved again.** `v0.1.0` was
first cut at `2da3ea2`, then moved to `8bd6085` when two doc commits landed —
safe, because nothing pointed at it. After the PyPI upload it was **not**
moved again, even though a README commit sits ahead of it, because release
0.1.0 was built from `8bd6085`: moving the tag would make it disagree with the
artifact people download. A tag with dependents is a different object from a
tag without.

**Repository name and command name were kept separate.** The repository is
`copulaendog_stata_python`; the command, the Python package and the ado files
are `copulaendog`. A repository name is read once on a web page, a command
name is typed on every line of every do-file. A Stata command with `python` in
its name would also imply a dependency the Mata implementation does not have.

**Affiliation was narrowed to what could be confirmed.** The help file first
said "Kelley School of Business, Indiana University"; only "Indiana
University" was actually verifiable, so the school was dropped.

**Profile URLs were not guessed.** The authorship note shipped without
LinkedIn and Scholar links in `f179422` and gained them in `5b94df2`, once the
exact addresses were supplied. A wrong profile link on a public page points at
a real other person.

---

## Environment of record

Figures quoted in the README, the vignette and VERIFICATION.md were produced
on:

| | |
|---|---|
| Python | 3.13.9 — numpy 2.3.5, scipy 1.16.3, pandas 2.3.3, patsy 1.0.1 |
| Second stack | numpy 2.5.1, scipy 1.18.0, pandas 3.0.5 |
| R | 4.6.1 with `Formula` |
| Stata | StataNow/SE 19.5, Apple Silicon |

`c(flavor)` reports `IC` on this installation for legacy reasons;
`c(edition_real)` and `about` both report SE. Do not "correct" the docs on the
strength of `c(flavor)`.

---

## Distribution

| Channel | State |
|---|---|
| GitHub | `girishm77/copulaendog_stata_python`, tag and release `v0.1.0` |
| PyPI | live at 0.1.0 — `pip install copulaendog` |
| SSC | submitted to Kit Baum, awaiting hand review |

When SSC accepts, drop the "SSC is still pending" note from `README.md`. For
a revision, send Kit the changed files and bump `Distribution-Date` in
`copulaendog.pkg`.

---

## Where the bodies are buried

If you change one of these, re-read the surrounding comment first.

- **`.copreg_firststage_cols` / `TwoStageConstructor._first_stage_cols`** —
  rank-based CDFs are invariant to monotone transformations, so `w` and
  `I(w^2)` collapse to the same column and the first stage goes singular. The
  redundant columns are dropped **once** and the choice reused, so the design
  does not change between bootstrap replicates.
- **The JAMS cell layout** is derived once from the original design matrix. If
  it were recomputed per resample, cells would be renumbered by order of first
  appearance and the copula columns would silently permute between replicates.
- **`ce_exactmax() = 1500`** in the ado file is the exact/binned switch for
  `kde.plugin`. See defect 2.
- **Bootstrap rejection.** Resamples that lose a factor level, or leave a JAMS
  cell too thin, are redrawn. If that happens often the command says so,
  because the standard errors are then conditional on the draws that survived.
- **`predict` excludes the copula terms** in both languages. They are
  endogeneity controls, not part of the causal model. `xba` needs
  `generate()` because the terms must exist as variables.
