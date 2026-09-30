# R-vs-PyDLNM differential tests

Every test runs the same computation in R (`dlnm`, `mvmeta`/`mixmeta`, via rpy2) and in PyDLNM on identical inputs and
compares numbers. Deterministic quantities must agree to <= 1e-8 relative unless a test states otherwise.

## Running

R must be a build that has `dlnm`, `tsModel` (and `mvmeta`/`mixmeta` for the meta-analysis tests) and that rpy2 can start.
On the development Mac the default R (4.6) has none of them and rpy2 segfaults on it; use the R 4.5 shim from the audit
handoff:

```bash
export AUDIT_DIR="<repo>/audit_handoff_2026-09-29"; source "$AUDIT_DIR/env/r45env.sh"
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
`audit_handoff_2026-09-29/findings/root_cause_themes.md` and `findings_all.json`.

Import order matters: `tests/rhelpers.py` starts R before any PyDLNM module is imported (Q2: several modules overwrite
`os.environ['R_HOME']`).
