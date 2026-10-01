# R-vs-PyDLNM differential tests

Every test runs the same computation in R (`dlnm`, `mvmeta`/`mixmeta`, via rpy2) and in PyDLNM on identical inputs and
compares numbers. Deterministic quantities must agree to <= 1e-8 relative unless a test states otherwise.

## Running

R must be a build that has `dlnm`, `tsModel` (and `mvmeta`/`mixmeta` for the meta-analysis tests) and that rpy2 can start.
On the development Mac the default R (4.6) has none of them and rpy2 segfaults on it; use the R 4.5 shim from the audit
handoff:

```bash
export AUDIT_DIR="<repo>/audit_handoff_2026-09-30"; source "$AUDIT_DIR/env/r45env.sh"
"$PYDLNM_PY" -m pytest tests -q
```

On a machine where `python -c "import rpy2.robjects as r; r.r('library(dlnm)')"` already works, just run
`python -m pytest tests -q`.

* `PYDLNM_SRC=/path/to/patched/copy` runs the suite against a different source directory (default: the repo root).
* `PYDLNM_XFAIL_OFF=1` runs the known-defect tests as ordinary tests, so their raw failures show.

## Known-defect tests

Tests for defects found in the 2026-09 audit are marked with `known_defect(theme, *finding_ids)` (a strict `xfail`).
The suite is green while the defect exists. When a fix lands, the test XPASSes, strict mode fails the run, and the
marker must be deleted in the same commit as the fix. Theme letters and finding ids refer to
`audit_handoff_2026-09-30/findings/root_cause_themes.md` and `findings_all.json`.

Import order matters: `tests/rhelpers.py` starts R before any PyDLNM module is imported (Q2: several modules overwrite
`os.environ['R_HOME']`).

## Open minor items

Found by the completeness review (a critic pass over uncovered code, R features and user workflows) and left open: none
changes a validated result, and none has a test module yet.

* `attrdl(sim=True)` draws from NumPy's global random generator (no seed argument) and clips negative eigenvalues of a
  non-positive-semidefinite `vcov` to zero (R's `eigen` route gives NaN).
* `find_mmt` builds its confidence interval with the fixed factor 1.96, not `ci_level`.
* `argvar` / `arglag`: R's partial matching of argument names is not supported.
* `compare_centering` / `CenteringManager.compare_centering_strategies` skip a strategy that fails with a warning only.
* `CrossBasis` / `OneBasis` may keep a reference to the caller's exposure array, so editing it in place afterwards changes later results.
* `Rpy2GLMInterface.fit_glm` returns an R object that `getcoef` / `crosspred(model=)` do not accept; pass the interface object.
* `exphist` and the variance handling of `CrossPred` / `crossreduce` are more lenient than R in a few edge cases.
* `MVMeta`: the `converged` flag and the optimum are less reliable for a very ill-conditioned meta-regression design.
* A statsmodels `NegativeBinomial` fit's covariance is conditional on theta, unlike `MASS::glm.nb`'s.
* The embedded R's `.libPaths()` depend on how rpy2 started R (`R_LIBS`); R warnings and convergence messages of `glm` are not passed on by the GLM interfaces.
* Where R prints a message (for example the automatic centering of `crosspred`), PyDLNM issues a Python warning, so `-W error` turns it into an exception.
* No test runs the Europe-2022 chain (cross-basis, `crosspred`, MMT, `crossreduce`, `MVMeta`, BLUP, `attrdl`) as one flow, and its `bs` branch has no test.
