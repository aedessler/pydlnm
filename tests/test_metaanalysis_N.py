"""Meta-analysis stage: MVMeta / blup() / MultiLocationDLNM summary versus R mvmeta 1.0.3 (and mixmeta 1.2.0).

Every reference number is computed by R at run time (mvmeta.fit, mvmeta(), blup.mvmeta, mvmeta:::remlprof.fn/.gr,
mixmeta) on the same simulated or England & Wales inputs that PyDLNM receives.

Theme N1a  method other than 'reml' is silently fitted as ML
  mvmeta-est-1    'fixed' / 'mm' / 'vc' / 'REML' / 'ML' / typos all return the ML fit; R has fixed/ml/reml/mm/vc and
                  errors on any other name (case-sensitive)
Theme N1b  input handling that differs from R
  mvmeta-est-4    S as R's vech rows (n, k(k+1)/2) crashes with a broadcast error for k >= 2
  mvmeta-est-3    NaN in y / S / X raises an opaque error; R drops NA rows and NA covariate rows and masks missing
                  outcomes per study
  mvmeta-est-5    under-determined meta-regression (p >= n, singular X'WX) returns -1e10 / garbage, R stops
  mvmeta-est-7    X or S with more rows than y is silently truncated, R stops ("variable lengths differ")
Theme N1c  converged flag
  mvmeta-est-2    converged is False for every ML and ~40 % of REML fits (gtol=1e-8 is below the round-off floor, ML
                  has no analytic gradient) and a spurious UserWarning is issued, although R converges and the
                  estimates agree with a tightly converged R fit
  mvmeta-blup-6   MultiLocationDLNM.get_summary()/fit_meta_analysis() drop loglik / Psi / coefficients when
                  converged is False
Theme N2   agreement with R is optimiser-limited (~1e-5 against R's default control), not machine precision
  mvmeta-est-11, mvmeta-blup-14
                  README claims machine-precision / "Exact match" for the MVMeta stage; measured agreement with R's
                  default optim is 1e-6..1e-5 because R stops at reltol = sqrt(eps). The plain tests below record
                  what IS true: identical objective / gradient / BLUP algebra at identical Psi (1e-10), agreement to
                  <= 1e-6 with R run at reltol = 1e-14, and <= 1e-4 with R's default control.

Status: N1a-N1c are fixed (no test marked as a known defect) and N2 is resolved for the code: the meta_analysis.py
docstring now states the measured precision. The one remaining @known_defect, test_readme_precision_claims_for_the_
mvmeta_stage_match_measurement, checks the README (owned by the project owner, not edited here): it stays xfail until the
README lines that claim machine precision / "Exact match" for the MVMeta stage are qualified. The plain tests guard
behaviour that was already faithful and must keep passing.

Notes for whoever fixes these
  * mvmeta-est-3: R-faithful means matching R's fit, so the "clear ValueError on NaN" first step is not enough to
    turn these tests green; per-study masking of missing outcomes (Psi[~na][:, ~na] in the GLS, the objective, the
    gradient, the IGLS start and blup) and dropping of NA-covariate rows is required.
  * mvmeta-blup-6 (forced-flag test) asserts the root cause: get_summary() must not gate valid results on the flag.
"""
import contextlib
import os
import re
import warnings

import numpy as np
import pytest

from rhelpers import REPO, assert_close, known_defect, max_rel_diff, np2r, r, r2np, rget

NAMES = 'pmm_'          # prefix of every object this module creates in R's global environment

# tight / default optimiser settings handed to mvmeta.control()
R_TIGHT = 'list(maxiter=20000, reltol=1e-14)'
R_DEFAULT = 'list()'

# (n, k, p, tau, seed): simulated meta-analyses with an interior optimum (Psi well inside the PSD cone)
CFGS = [(30, 2, 2, 0.4, 5), (40, 3, 2, 0.3, 11), (30, 2, 1, 0.5, 1), (50, 3, 3, 0.3, 2), (25, 4, 1, 0.3, 7),
        (45, 3, 1, 0.4, 4), (30, 3, 2, 0.5, 6)]
CFG_K5 = (40, 5, 3, 0.3, 21)                     # like the 106-city / England & Wales second stage (k=5, p=3)
CFG_IDS = [f'n{c[0]}k{c[1]}p{c[2]}' for c in CFGS]
# (cfg, method) pairs for the agreement tests; ML with k=5 is skipped (finite-difference gradient makes it ~15 s)
FIT_CASES = [(c, m) for c in CFGS for m in ('reml', 'ml')] + [(CFG_K5, 'reml')]
FIT_IDS = [f'n{c[0]}k{c[1]}p{c[2]}-{m}' for c, m in FIT_CASES]

_R_DEFS = f'''
{NAMES}lists <- function(X, y, Sv) {{
  k <- ncol(y); m <- nrow(y); p <- ncol(X); nay <- is.na(y)
  list(Xlist = lapply(seq(m), function(i) diag(1, k)[!nay[i, ], , drop = FALSE] %x% X[i, , drop = FALSE]),
       ylist = lapply(seq(m), function(i) y[i, ][!nay[i, ]]),
       Slist = lapply(seq(m), function(i) mixmeta:::xpndMat(Sv[i, ])[!nay[i, ], !nay[i, ], drop = FALSE]),
       nalist = lapply(seq(m), function(i) nay[i, ]), k = k, m = m, p = p, nall = sum(!nay))
}}
{NAMES}prof <- function(par, L, what) {{
  f <- get(what, envir = asNamespace("mvmeta"))
  f(par, L$Xlist, L$ylist, L$Slist, L$nalist, L$k, L$m, L$p, L$nall, "unstr", NULL)
}}
{NAMES}formula_fit <- function(y, Sv, X, method, control) {{
  df <- data.frame(id = seq_len(nrow(y))); df$y <- y; df$Sv <- Sv; df$X <- X
  mvmeta(y ~ X - 1, S = Sv, data = df, method = method, control = control)
}}
'''


def require_r(*pkgs):
    """Skip if an R package is missing, else load it. (rhelpers.require_r_packages indexes the invisible result of
    requireNamespace(), which rpy2 3.6 returns as None, so it raises TypeError here; hence this local copy.)"""
    for p in pkgs:
        if not bool(r(f'isTRUE(suppressWarnings(requireNamespace("{p}", quietly=TRUE)))')[0]):
            pytest.skip(f'R package {p} not installed')
        r(f'suppressMessages(library({p}))')


def running_r_home():
    """Home of the R that is actually running. os.environ['R_HOME'] is not reliable: basis.py, improved_glm.py and
    rpy2_glm.py overwrite it with the R 4.6 framework path once R 4.5 is already running, and R then segfaults the
    first time it lazily dlopen()s its LAPACK module (chol / solve / eigen) -- which happens in this module, not in
    the dlnm tests. R.home() just echoes the environment variable, so read the path of a loaded R library instead."""
    p = str(r('as.character(getLoadedDLLs()[["utils"]][["path"]])')[0])
    for _ in range(4):
        p = os.path.dirname(p)
    return p


@contextlib.contextmanager
def correct_r_home():
    """os.environ['R_HOME'] = the running R's home for the duration of the block, previous value restored afterwards
    (so neither the wrong value written by PyDLNM imports nor the correction leaks into other test modules)."""
    before = os.environ.get('R_HOME')
    os.environ['R_HOME'] = running_r_home()
    try:
        yield
    finally:
        if before is None:
            os.environ.pop('R_HOME', None)
        else:
            os.environ['R_HOME'] = before


@pytest.fixture(scope='module', autouse=True)
def _r_env():
    with correct_r_home():
        require_r('mvmeta')
        r(_R_DEFS)
        r('invisible(list(chol(diag(2)), solve(diag(2)), eigen(diag(2)), svd(diag(2)), qr(diag(2))))')   # load LAPACK now


@pytest.fixture(autouse=True)
def _restore_r_home():
    """PyDLNM imports inside the tests (multi_location -> improved_glm) rewrite R_HOME; run each test with the right one."""
    with correct_r_home():
        yield


@pytest.fixture(scope='module', autouse=True)
def _single_threaded_blas():
    """The matrices here are tiny; OpenBLAS worker threads only spin (system time >> user time) and make the module
    several times slower on a busy machine. Numerical results are unaffected."""
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        yield
        return
    with threadpool_limits(limits=1, user_api='blas'):
        yield


# --------------------------------------------------------------------------------------------------------------
# helpers: identical inputs for R and Python
# --------------------------------------------------------------------------------------------------------------
def sim(n, k, p=1, tau=0.3, seed=0, sw=0.2, corr=0.5):
    """y (n,k), S (n,k,k), X (n,p) with an intercept in column 0; deterministic."""
    rng = np.random.default_rng(seed)
    X = np.ones((n, p))
    if p > 1:
        X[:, 1:] = rng.normal(size=(n, p - 1))
    beta = rng.normal(size=(p, k))
    A = rng.normal(size=(k, k))
    Psi = tau ** 2 * (A @ A.T / k + 0.2 * np.eye(k))
    S = np.zeros((n, k, k))
    for i in range(n):
        d = sw * np.exp(rng.normal(scale=0.5, size=k))
        R = np.full((k, k), corr) + (1 - corr) * np.eye(k)
        S[i] = np.outer(d, d) * R
    y = np.array([X[i] @ beta + rng.multivariate_normal(np.zeros(k), Psi + S[i]) for i in range(n)])
    return y, S, X


def vech_rows(S):
    """R's vech-row layout of S: lower triangle of each S_i taken column-wise (mixmeta::vechMat / xpndMat)."""
    n, k, _ = S.shape
    idx = [(a, b) for b in range(k) for a in range(b, k)]
    return np.array([[S[i, a, b] for (a, b) in idx] for i in range(n)])


def r_error(code):
    """None if the R code runs, otherwise the first line of R's error message."""
    try:
        r(code)
    except Exception as e:                           # rpy2 RRuntimeError
        return str(e).strip().splitlines()[0]
    return None


def push(y, S, X):
    np2r(NAMES + 'y', y)
    np2r(NAMES + 'Sv', vech_rows(S))
    np2r(NAMES + 'X', X)


def r_fit(y, S, X, method='reml', control=R_DEFAULT):
    """R mvmeta.fit (the estimation engine of mvmeta()), S passed as vech rows."""
    push(y, S, X)
    r(f'{NAMES}fit <- mvmeta:::mvmeta.fit(as.matrix({NAMES}X), as.matrix({NAMES}y), as.matrix({NAMES}Sv), '
      f'method="{method}", control={control})')
    p, k = X.shape[1], y.shape[1]
    out = dict(coef=np.asarray(r2np(r(f'{NAMES}fit$coefficients'))).reshape(p, k),
               vcov=r2np(r(f'{NAMES}fit$vcov')),
               loglik=float(r2np(r(f'as.numeric({NAMES}fit$logLik)')).ravel()[0]))
    out['psi'] = r2np(r(f'{NAMES}fit$Psi')) if method != 'fixed' else None
    out['converged'] = bool(r(f'{NAMES}fit$converged')[0]) if method in ('ml', 'reml') else None
    return out


def r_blup_fit(y, S, X, method='reml', control=R_DEFAULT):
    """R mvmeta() (formula interface, X includes the intercept) followed by blup(vcov=TRUE); needs k >= 2."""
    push(y, S, X)
    r(f'{NAMES}mv <- {NAMES}formula_fit(as.matrix({NAMES}y), as.matrix({NAMES}Sv), as.matrix({NAMES}X), '
      f'"{method}", {control})')
    return _read_r_blup(y.shape[0], X.shape[1], y.shape[1])


def _read_r_blup(n, p, k):
    r(f'{NAMES}bl <- blup({NAMES}mv, vcov=TRUE)')
    return dict(coef=np.asarray(r2np(r(f'{NAMES}mv$coefficients'))).reshape(p, k),
                vcov=r2np(r(f'{NAMES}mv$vcov')), psi=r2np(r(f'{NAMES}mv$Psi')),
                loglik=float(r2np(r(f'as.numeric(logLik({NAMES}mv))')).ravel()[0]),
                blup=np.array([r2np(r(f'as.numeric({NAMES}bl[[{i + 1}]]$blup)')) for i in range(n)]),
                bvcov=np.array([r2np(r(f'unname({NAMES}bl[[{i + 1}]]$vcov)')) for i in range(n)]))


def py_fit(y, S, X, method='reml', control=None):
    """PyDLNM MVMeta fit; returns (model, list of warning messages)."""
    from meta_analysis import MVMeta
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        m = MVMeta(method=method, control=control).fit(y, S, X)
    return m, [str(x.message) for x in w]


def py_blup(m):
    from meta_analysis import blup
    b = blup(m, vcov=True)
    return np.array([x['blup'] for x in b]), np.array([x['vcov'] for x in b])


def assert_fit_close(m, ref, rtol, what=''):
    """coefficients (p,k), vcov (outcome-major), Psi and loglik of a PyDLNM fit against an R fit."""
    assert_close(m.coefficients, ref['coef'], rtol=rtol, what=f'{what} coefficients')
    assert_close(m.vcov, ref['vcov'], rtol=rtol, what=f'{what} vcov')
    assert_close(m.psi, ref['psi'], rtol=rtol, what=f'{what} Psi')
    assert abs(m.loglik - ref['loglik']) <= rtol * max(1.0, abs(ref['loglik'])), \
        f'{what} loglik: Python {m.loglik!r} vs R {ref["loglik"]!r}'


def py_lists(y, S, X):
    n, k = y.shape
    return [np.kron(np.eye(k), X[i:i + 1]) for i in range(n)], list(y), list(S)


def r_par(L):
    """R's parameter order for a lower-Cholesky factor (column-wise lower triangle)."""
    k = L.shape[0]
    return np.array([L[a, b] for b in range(k) for a in range(b, k)])


# --------------------------------------------------------------------------------------------------------------
# N1a  mvmeta-est-1: estimation method dispatch
# --------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('method', ['foo', 'REML', 'ML', 'Reml'])
@pytest.mark.parametrize('entry', ['MVMeta', 'mvmeta'])
def test_invalid_method_name_raises_like_r(method, entry):
    """R: mvmeta.<method> does not exist -> error (names are case-sensitive: 'REML' and 'ML' are errors too)."""
    import meta_analysis as ma
    y, S, X = sim(30, 2, 2, 0.4, 5)
    push(y, S, X)
    assert r_error(f'mvmeta:::mvmeta.fit(as.matrix({NAMES}X), as.matrix({NAMES}y), as.matrix({NAMES}Sv), '
                   f'method="{method}")') is not None, 'R must reject this method name'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with pytest.raises((ValueError, KeyError)):
            if entry == 'MVMeta':
                ma.MVMeta(method=method).fit(y, S, X)
            else:
                ma.mvmeta(y, S, X, method=method)


def test_fixed_method_is_gls_with_zero_psi_like_r():
    """R mvmeta.fixed: GLS with Psi = 0, no Psi in the result, logLik = -0.5 n log(2 pi) + pdet + pres."""
    y, S, X = sim(30, 2, 2, 0.4, 5)
    ref = r_fit(y, S, X, 'fixed')
    m, _ = py_fit(y, S, X, 'fixed')
    assert_close(m.coefficients, ref['coef'], rtol=1e-9, what='fixed coefficients')
    assert_close(m.vcov, ref['vcov'], rtol=1e-9, what='fixed vcov')
    assert abs(m.loglik - ref['loglik']) <= 1e-9 * abs(ref['loglik']), \
        f'fixed logLik: Python {m.loglik!r} vs R {ref["loglik"]!r}'
    assert m.psi is None or np.allclose(m.psi, 0.0), 'a fixed-effects fit has no between-study covariance'


@pytest.mark.parametrize('method', ['mm', 'vc'])
def test_mm_and_vc_match_r_or_are_refused(method):
    """R implements the method-of-moments ('mm') and variance-components ('vc') estimators; PyDLNM must either
    reproduce R's Psi or say NotImplementedError, never return the ML fit under another name."""
    from meta_analysis import MVMeta
    y, S, X = sim(30, 2, 2, 0.4, 5)
    ref = r_fit(y, S, X, method)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = MVMeta(method=method).fit(y, S, X)
    except NotImplementedError:
        return
    assert_close(m.psi, ref['psi'], rtol=1e-6, what=f'{method} Psi')
    assert_close(m.coefficients, ref['coef'], rtol=1e-6, what=f'{method} coefficients')


# --------------------------------------------------------------------------------------------------------------
# N1b  mvmeta-est-4: S in R's vech-row layout
# --------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('k', [2, 3, 4])
def test_vech_row_S_matches_r(k):
    """R mvmeta accepts S as an (n, k(k+1)/2) matrix of vech rows (xpndMat, column-wise lower triangle)."""
    y, S, X = sim(25, k, 1, 0.3, 7 + k)
    ref = r_fit(y, S, X, 'reml', R_TIGHT)                       # R receives exactly the vech rows
    m, _ = py_fit(y, vech_rows(S), X)
    m3, _ = py_fit(y, S, X)                                     # (n,k,k) input, same numbers
    assert_close(m.coefficients, m3.coefficients, rtol=1e-12, what='vech vs (n,k,k) coefficients')
    assert_close(m.psi, m3.psi, rtol=1e-12, what='vech vs (n,k,k) Psi')
    assert_fit_close(m, ref, rtol=1e-6, what=f'k={k} vech')


@pytest.mark.parametrize('k', [1, 2, 3])
def test_S_layouts_that_work_today_match_r(k):
    """Plain: (n,k,k) array, list of (k,k) matrices and (n,k) within-study variances agree with R
    (R's k-column S means variances only; for k=1 the vech row is the variance)."""
    y, S, X = sim(30, k, 2, 0.4, 3 + k)
    var = np.stack([np.diag(S[i]) for i in range(30)])
    S_diag = np.zeros_like(S)
    for i in range(30):
        S_diag[i] = np.diag(var[i])
    ref_full = r_fit(y, S, X, 'reml', R_TIGHT)
    for name, arg in [('(n,k,k)', S), ('list of (k,k)', [S[i] for i in range(30)])]:
        m, _ = py_fit(y, arg, X)
        assert_fit_close(m, ref_full, rtol=1e-6, what=f'k={k} S {name}')
    # variances only: R gets the same S as vech rows of the diagonal matrices
    ref_diag = r_fit(y, S_diag, X, 'reml', R_TIGHT)
    m, _ = py_fit(y, var, X)
    assert_fit_close(m, ref_diag, rtol=1e-6, what=f'k={k} S (n,k) variances')


def test_S_with_wrong_number_of_columns_raises():
    """R: xpndMat / mkS reject an S whose width is neither k nor k(k+1)/2; Python raises too."""
    from meta_analysis import MVMeta
    y, S, X = sim(30, 3, 1, 0.3, 2)
    bad = np.full((30, 4), 0.05)
    assert r_error('mvmeta(matrix(0, 30, 3) ~ 1, S = matrix(0.05, 30, 4))') is not None
    with pytest.raises(ValueError):
        MVMeta().fit(y, bad, X)


# --------------------------------------------------------------------------------------------------------------
# N1b  mvmeta-est-3: missing values
# --------------------------------------------------------------------------------------------------------------
def _nan_case(kind):
    """(y, S, X, expected number of studies R uses) with NaN inserted the way R's NA handling expects."""
    y, S, X = sim(40, 3, 2, 0.3, 11)
    if kind == 'whole_study':                       # all outcomes and S of study 4 missing: mvmeta() drops the row
        y[4, :] = np.nan
        S[4] = np.nan
        return y, S, X, 39
    if kind == 'covariate':                         # NA covariate: mvmeta() drops the row
        X[3, 1] = np.nan
        return y, S, X, 39
    # 'outcome': partly missing outcomes (S rows/columns of the missing outcome are NaN): all 40 studies used
    for (i, j) in [(3, 1), (7, 1), (12, 1), (5, 0)]:
        y[i, j] = np.nan
        S[i, j, :] = np.nan
        S[i, :, j] = np.nan
    return y, S, X, 40


@pytest.mark.parametrize('method', ['reml', 'ml'])
@pytest.mark.parametrize('kind', ['whole_study', 'covariate', 'outcome'])
def test_missing_values_are_handled_like_r(kind, method):
    y, S, X, n_used = _nan_case(kind)
    push(y, S, X)
    r(f'{NAMES}mv <- {NAMES}formula_fit(as.matrix({NAMES}y), as.matrix({NAMES}Sv), as.matrix({NAMES}X), '
      f'"{method}", {R_TIGHT})')
    assert int(r(f'nrow({NAMES}mv$model)')[0]) == n_used, 'R keeps the studies the case is built around'
    p, k = X.shape[1], y.shape[1]
    ref = dict(coef=np.asarray(r2np(r(f'{NAMES}mv$coefficients'))).reshape(p, k), vcov=r2np(r(f'{NAMES}mv$vcov')),
               psi=r2np(r(f'{NAMES}mv$Psi')), loglik=float(r2np(r(f'as.numeric(logLik({NAMES}mv))')).ravel()[0]))
    m, _ = py_fit(y, S, X, method)
    assert_fit_close(m, ref, rtol=1e-6, what=f'{kind}/{method}')


# --------------------------------------------------------------------------------------------------------------
# N1b  mvmeta-est-5: under-determined meta-regression
# --------------------------------------------------------------------------------------------------------------
UNDERDETERMINED = [(4, 2, 5, 'reml'), (4, 2, 5, 'ml'), (2, 3, 4, 'reml'), (2, 3, 4, 'ml'), (2, 1, 3, 'reml'),
                   (2, 1, 3, 'ml'), (2, 2, 3, 'reml')]


@pytest.mark.parametrize('n,k,p,method', UNDERDETERMINED,
                         ids=[f'n{a}k{b}p{c}-{d}' for a, b, c, d in UNDERDETERMINED])
def test_underdetermined_meta_regression_raises_like_r(n, k, p, method):
    """rank(X) < p (here n < p): R stops in chol(tXWXtot) (REML) / backsolve (ML)."""
    y, S, X = sim(n, k, p, 0.3, 3)
    push(y, S, X)
    assert r_error(f'mvmeta:::mvmeta.fit(as.matrix({NAMES}X), as.matrix({NAMES}y), as.matrix({NAMES}Sv), '
                   f'method="{method}")') is not None, 'R must reject a rank-deficient meta-regression'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with pytest.raises((ValueError, RuntimeError)):
            py_fit(y, S, X, method)


@pytest.mark.parametrize('n,k,p', [(3, 1, 3), (3, 2, 3), (5, 3, 5)])
def test_as_many_studies_as_coefficients_is_valid_and_matches_r(n, k, p):
    """Plain: n == p is estimable (X full rank): R fits it and PyDLNM must keep doing so (a guard 'n > p' would be wrong)."""
    y, S, X = sim(n, k, p, 0.3, 3)
    ref = r_fit(y, S, X, 'reml', R_TIGHT)
    m, _ = py_fit(y, S, X)
    assert abs(m.loglik - ref['loglik']) <= 1e-6 * max(1.0, abs(ref['loglik'])), (m.loglik, ref['loglik'])
    assert_close(m.coefficients, ref['coef'], rtol=1e-5, what='n == p coefficients')


# --------------------------------------------------------------------------------------------------------------
# N1b  mvmeta-est-7: length mismatch between y, X and S
# --------------------------------------------------------------------------------------------------------------
def _mismatch_case(which):
    """(y, S, X) with one argument longer than the 30 studies in y, plus the R code that must fail on the same data."""
    y, S, X = sim(30, 2, 2, 0.4, 5)
    np2r(NAMES + 'y', y)
    np2r(NAMES + 'Sv', vech_rows(S))
    r(f'{NAMES}Sm <- as.matrix({NAMES}Sv)')
    np2r(NAMES + 'x30', X[:, 1])
    if which == 'X_extra':                                        # 35 rows of covariates for 30 studies
        Xb = np.vstack([X, X[:5]])
        np2r(NAMES + 'xb', Xb[:, 1])
        return y, S, Xb, f'mvmeta({NAMES}y ~ {NAMES}xb, S={NAMES}Sm)'
    if which == 'S3_extra':                                       # (35,k,k) S
        Sb = np.concatenate([S, S[:5]])
        np2r(NAMES + 'Sv35', vech_rows(Sb))
        return y, Sb, X, f'mvmeta({NAMES}y ~ {NAMES}x30, S=as.matrix({NAMES}Sv35))'
    if which == 'S2_extra':                                       # (35,k) variances
        var = np.stack([np.diag(S[i]) for i in range(30)])
        Vb = np.vstack([var, var[:5]])
        np2r(NAMES + 'V35', Vb)
        return y, Vb, X, f'mvmeta({NAMES}y ~ {NAMES}x30, S=as.matrix({NAMES}V35))'
    # 'study_dropped': a study was removed from y and S but X was not subset (29 studies, 30 covariate rows)
    np2r(NAMES + 'y29', y[:29])
    r(f'{NAMES}Sm29 <- {NAMES}Sm[1:29, ]')
    return y[:29], S[:29], X, f'mvmeta({NAMES}y29 ~ {NAMES}x30, S={NAMES}Sm29)'


@pytest.mark.parametrize('which', ['X_extra', 'S3_extra', 'S2_extra', 'study_dropped'])
def test_extra_rows_in_X_or_S_raise_like_r(which):
    y, S, X, rcode = _mismatch_case(which)
    assert r_error(rcode) is not None, 'R must reject mismatched lengths'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with pytest.raises(ValueError):
            py_fit(y, S, X)


@pytest.mark.parametrize('which', ['X_short', 'S_short'])
def test_too_few_rows_in_X_or_S_raise(which):
    """Plain: fewer rows than studies already raises (matmul ValueError / IndexError); R raises as well."""
    y, S, X = sim(30, 2, 2, 0.4, 5)
    if which == 'X_short':
        X = X[:25]
    else:
        S = S[:25]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with pytest.raises((ValueError, IndexError)):
            py_fit(y, S, X)


# --------------------------------------------------------------------------------------------------------------
# N1c  mvmeta-est-2: converged flag and spurious warning
# --------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('method', ['reml', 'ml'])
def test_converged_flag_true_on_typical_fits(method):
    """R reports converged=TRUE for these fits and the estimates agree with tightly converged R (see the plain
    tests below), so PyDLNM must not report converged=False."""
    bad = []
    for (n, k, p, tau, seed) in CFGS:
        y, S, X = sim(n, k, p, tau, seed)
        assert r_fit(y, S, X, method)['converged'] is True
        m, _ = py_fit(y, S, X, method)
        if not m.converged:
            bad.append(f'n{n}k{k}p{p}')
    assert not bad, f'converged=False for {bad}'


@pytest.mark.parametrize('method', ['reml', 'ml'])
def test_no_spurious_convergence_warning_on_typical_fits(method):
    bad = {}
    for (n, k, p, tau, seed) in CFGS:
        y, S, X = sim(n, k, p, tau, seed)
        _, msgs = py_fit(y, S, X, method)
        msgs = [x for x in msgs if 'converge' in x.lower()]
        if msgs:
            bad[f'n{n}k{k}p{p}'] = msgs[0][:70]
    assert not bad, f'unexpected warnings: {bad}'


# --------------------------------------------------------------------------------------------------------------
# N1c  mvmeta-blup-6: MultiLocationDLNM summary gated on the converged flag
# --------------------------------------------------------------------------------------------------------------
SUMMARY_KEYS = ('meta_analysis_loglik', 'between_study_variance', 'meta_coefficients')


def _stub_multilocation(seed, n=20, k=3):
    """A MultiLocationDLNM whose first stage is replaced by simulated reduced coefficients / vcov (the second stage
    -- fit_meta_analysis, get_summary -- is the real code). Meta-predictors are avg_temp and temp_range."""
    from multi_location import MultiLocationDLNM
    y, S, X = sim(n, k, 3, 0.3, seed)
    ml = MultiLocationDLNM()
    for i in range(n):
        ml.region_results.append({'reduced': {'coefficients': y[i], 'vcov': S[i]}})
        ml.region_names.append(f'region{i}')
        ml.meta_predictors[f'region{i}'] = {'avg_temp': X[i, 1], 'temp_range': X[i, 2]}
    return ml, y, S, X


@pytest.mark.parametrize('method', ['reml', 'ml'])
def test_get_summary_keeps_meta_results_on_real_fits(method, capsys):
    """A completed fit_meta_analysis() must expose loglik / trace(Psi) / coefficients in get_summary() and print the
    between-study variance, whatever BFGS's line search reported."""
    bad = []
    for seed in (1, 3, 5, 7):
        ml, y, S, X = _stub_multilocation(seed)
        ml.fit_meta_analysis(method=method)
        summ = ml.get_summary()
        printed = 'Between-study variance' in capsys.readouterr().out
        missing = [key for key in SUMMARY_KEYS if key not in summ]
        if missing or not printed:
            bad.append((seed, ml.mv_model.converged, missing, printed))
    assert not bad, f'(seed, converged, missing summary keys, variance printed): {bad}'


def test_get_summary_reports_results_even_if_flag_is_false():
    """Forced flag: a valid fit whose converged flag is False (as scipy reports on flat likelihoods) must still be
    summarised, with meta_analysis_converged=False carried along."""
    ml, y, S, X = _stub_multilocation(5)
    ml.fit_meta_analysis(method='reml')
    ref, _ = py_fit(y, S, X, 'reml')
    ml.mv_model.converged = False
    summ = ml.get_summary()
    assert summ['meta_analysis_converged'] is False
    assert all(key in summ for key in SUMMARY_KEYS), f'missing: {[k for k in SUMMARY_KEYS if k not in summ]}'
    assert summ['meta_analysis_loglik'] == pytest.approx(ref.loglik, rel=1e-12)
    assert summ['between_study_variance'] == pytest.approx(np.trace(ref.psi), rel=1e-12)
    assert_close(summ['meta_coefficients'], ref.coefficients, rtol=1e-12, what='meta_coefficients')


def test_get_summary_when_no_model_fitted():
    """Plain: get_summary() before any meta-analysis reports converged=False and no results."""
    from multi_location import MultiLocationDLNM
    summ = MultiLocationDLNM().get_summary()
    assert summ['meta_analysis_converged'] is False
    assert not any(key in summ for key in SUMMARY_KEYS)
    assert summ['n_regions'] == 0


# --------------------------------------------------------------------------------------------------------------
# N2  plain tests: what agrees with R, and to what level (mvmeta-est-11, mvmeta-blup-14)
# --------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('cfg', CFGS + [CFG_K5], ids=CFG_IDS + ['n40k5p3'])
def test_objective_and_reml_gradient_match_r_at_identical_psi(cfg):
    """Deterministic algebra: negative REML/ML profile log-likelihood and the analytic REML gradient of MVMeta equal
    mvmeta:::remlprof.fn / mlprof.fn / remlprof.gr evaluated at the same Psi (parameters mapped between the two
    orderings: Python row-major lower triangle, R column-wise)."""
    import meta_analysis as ma
    n, k, p, tau, seed = cfg
    y, S, X = sim(n, k, p, tau, seed)
    push(y, S, X)
    r(f'{NAMES}L <- {NAMES}lists(as.matrix({NAMES}X), as.matrix({NAMES}y), as.matrix({NAMES}Sv))')
    Xl, yl, Sl = py_lists(y, S, X)
    rng = np.random.default_rng(1000 + seed)
    ti = np.tril_indices(k)
    order_py = list(zip(*ti))
    order_r = [(a, b) for b in range(k) for a in range(b, k)]
    for rep in range(3):
        L = np.tril(rng.normal(size=(k, k))) * tau
        par_py = L[ti]
        np2r(NAMES + 'par', r_par(L))
        for what, fn in [('remlprof.fn', ma._reml_fn), ('mlprof.fn', ma._ml_fn)]:
            ref = float(r(f'{NAMES}prof({NAMES}par, {NAMES}L, "{what}")')[0])
            assert_close(np.array([-fn(par_py, k, Xl, yl, Sl)]), np.array([ref]), rtol=1e-10,
                         what=f'{what} at rep {rep}')
        g_py = -ma._reml_gr(par_py, k, Xl, yl, Sl)               # gradient of the log-likelihood, Python order
        g_py_in_r = np.array([g_py[order_py.index(ab)] for ab in order_r])
        g_r = r2np(r(f'{NAMES}prof({NAMES}par, {NAMES}L, "remlprof.gr")')).ravel()
        assert_close(g_py_in_r, g_r, rtol=1e-10, what=f'remlprof.gr at rep {rep}')


@pytest.mark.parametrize('cfg', CFGS + [CFG_K5], ids=CFG_IDS + ['n40k5p3'])
def test_blup_formula_matches_r_at_identical_estimates(cfg):
    """blup() is exact algebra: with PyDLNM's own Psi / coefficients / vcov injected into R's mvmeta object, R's
    blup.mvmeta(vcov=TRUE) and PyDLNM's blup() agree to rounding."""
    n, k, p, tau, seed = cfg
    y, S, X = sim(n, k, p, tau, seed)
    m, _ = py_fit(y, S, X)
    push(y, S, X)
    r(f'{NAMES}mv <- {NAMES}formula_fit(as.matrix({NAMES}y), as.matrix({NAMES}Sv), as.matrix({NAMES}X), "reml", list())')
    np2r(NAMES + 'psi', m.psi)
    np2r(NAMES + 'coef', m.coefficients)
    np2r(NAMES + 'vcov', m.vcov)
    r(f'{NAMES}mv$Psi <- {NAMES}psi; {NAMES}mv$coefficients <- {NAMES}coef; {NAMES}mv$vcov <- {NAMES}vcov')
    ref = _read_r_blup(n, p, k)
    b, bv = py_blup(m)
    assert_close(b, ref['blup'], rtol=1e-10, what='BLUP')
    assert_close(bv, ref['bvcov'], rtol=1e-10, what='BLUP vcov')


@pytest.mark.parametrize('cfg,method', FIT_CASES, ids=FIT_IDS)
def test_fit_agrees_with_tightly_converged_r(cfg, method):
    """R at reltol=1e-14: PyDLNM's coefficients, vcov, Psi and loglik agree to <= 1e-6 relative. The estimates are
    correct; the differences to R's DEFAULT control (next test) are R's own stopping error."""
    n, k, p, tau, seed = cfg
    y, S, X = sim(n, k, p, tau, seed)
    ref = r_fit(y, S, X, method, R_TIGHT)
    m, _ = py_fit(y, S, X, method)
    assert_fit_close(m, ref, rtol=1e-6, what=f'{method} vs tight R')
    assert m.loglik >= ref['loglik'] - 1e-8 * abs(ref['loglik']), 'PyDLNM must not stop at a worse optimum'


@pytest.mark.parametrize('cfg,method', FIT_CASES, ids=FIT_IDS)
def test_fit_agreement_with_r_default_control_is_optimizer_limited(cfg, method):
    """Documents the level (mvmeta-est-11, mvmeta-blup-14): against R's DEFAULT optim (reltol = sqrt(eps) on the
    objective) coefficients, vcov, Psi and BLUPs agree to about 1e-6..1e-5 -- not to machine precision. Do NOT
    tighten this bound: R's default fit is itself only that close to the optimum."""
    n, k, p, tau, seed = cfg
    y, S, X = sim(n, k, p, tau, seed)
    ref = r_fit(y, S, X, method, R_DEFAULT)
    m, _ = py_fit(y, S, X, method)
    assert_fit_close(m, ref, rtol=1e-4, what=f'{method} vs default R')
    if k >= 2:
        rb = r_blup_fit(y, S, X, method, R_DEFAULT)
        b, bv = py_blup(m)
        assert_close(b, rb['blup'], rtol=1e-4, what='BLUP vs default R')
        assert_close(bv, rb['bvcov'], rtol=1e-4, what='BLUP vcov vs default R')


def test_reml_loglik_and_coefficient_layout_versus_mixmeta():
    """Conventions (mvmeta-est-11 a/b), plain: PyDLNM follows mvmeta. mixmeta's REML logLik is larger by
    sum(log(diag(chol(sum_i X_i'X_i)))) (ML logLik is identical); mixmeta's coefficient vector and vcov are
    predictor-major, mvmeta's / PyDLNM's outcome-major."""
    require_r('mixmeta')
    n, k, p = 30, 2, 2
    y, S, X = sim(n, k, p, 0.4, 5)
    push(y, S, X)
    Xl, _, _ = py_lists(y, S, X)
    offset = float(np.sum(np.log(np.diag(np.linalg.cholesky(sum(x.T @ x for x in Xl))))))
    for method in ('reml', 'ml'):
        r(f'''{NAMES}df <- data.frame(id = seq_len({n})); {NAMES}df$y <- as.matrix({NAMES}y);
              {NAMES}df$Sv <- as.matrix({NAMES}Sv); {NAMES}df$X <- as.matrix({NAMES}X)
              {NAMES}mx <- suppressWarnings(mixmeta::mixmeta(y ~ X - 1, S = Sv, data = {NAMES}df, method = "{method}",
                                                               control = {R_TIGHT}))''')
        ll_mixmeta = float(r(f'as.numeric(logLik({NAMES}mx))')[0])
        m, _ = py_fit(y, S, X, method)
        shift = offset if method == 'reml' else 0.0
        assert abs((m.loglik + shift) - ll_mixmeta) <= 1e-6, (method, m.loglik + shift, ll_mixmeta)
        # mixmeta is predictor-major: entry (predictor j, outcome l) sits at l + j*k; PyDLNM / mvmeta at l*p + j
        idx_mixmeta = [l + j * k for l in range(k) for j in range(p)]
        vc_mx = r2np(r(f'unname({NAMES}mx$vcov)'))
        assert_close(m.vcov, vc_mx[np.ix_(idx_mixmeta, idx_mixmeta)], rtol=1e-5, what=f'{method} vcov (permuted)')
        coef_mx = r2np(r(f'as.numeric({NAMES}mx$coefficients)')).ravel()
        assert_close(m.coefficients.T.ravel(), coef_mx[idx_mixmeta], rtol=1e-6, what=f'{method} coef order')


@pytest.mark.parametrize('n,k', [(2, 1), (3, 1), (3, 2)])
@pytest.mark.parametrize('method', ['reml', 'ml'])
def test_very_few_studies_match_r(n, k, method):
    """Plain: with 2-3 studies (Psi may sit on the boundary) the log-likelihood still equals R's."""
    y, S, X = sim(n, k, 1, 0.3, 3)
    ref = r_fit(y, S, X, method, R_TIGHT)
    m, _ = py_fit(y, S, X, method)
    assert abs(m.loglik - ref['loglik']) <= 1e-6 * max(1.0, abs(ref['loglik'])), (m.loglik, ref['loglik'])
    assert_close(m.coefficients, ref['coef'], rtol=1e-5, what='coefficients')


def test_default_method_is_reml_and_wrapper_matches_r():
    """Plain: MVMeta() and mvmeta() default to REML (a fix for mvmeta-est-1 must not change that)."""
    import meta_analysis as ma
    y, S, X = sim(30, 2, 2, 0.4, 5)
    assert ma.MVMeta().method == 'reml'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        m = ma.mvmeta(y, S, X)
    assert_fit_close(m, r_fit(y, S, X, 'reml', R_TIGHT), rtol=1e-6, what='mvmeta() default')


def test_one_dimensional_inputs_for_a_single_outcome_match_r():
    """Plain: k=1 with 1-D y and S and no X (intercept only), as R's mvmeta accepts."""
    y, S, X = sim(40, 1, 1, 0.3, 9)
    ref = r_fit(y, S, X, 'reml', R_TIGHT)
    m, _ = py_fit(y[:, 0], S[:, 0, 0], None)
    assert_fit_close(m, ref, rtol=1e-6, what='k=1, 1-D inputs')


def test_fit_leaves_caller_arrays_untouched_and_accepts_dataframes():
    """Plain: fit() does not modify y / S / X, and pandas y / X give the same result as numpy arrays."""
    import pandas as pd
    y, S, X = sim(30, 2, 2, 0.4, 5)
    y0, S0, X0 = y.copy(), S.copy(), X.copy()
    m, _ = py_fit(y, S, X)
    assert np.array_equal(y, y0) and np.array_equal(S, S0) and np.array_equal(X, X0)
    m2, _ = py_fit(pd.DataFrame(y), S, pd.DataFrame(X))
    assert_close(m2.coefficients, m.coefficients, rtol=1e-12, what='DataFrame inputs')
    assert_close(m2.psi, m.psi, rtol=1e-12, what='DataFrame inputs Psi')


# --------------------------------------------------------------------------------------------------------------
# N2  England & Wales second stage (real first-stage estimates computed by R at run time)
# --------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope='module')
def ew_first_stage():
    """Reduced overall-cumulative coefficients / vcov for the 10 England & Wales regions exactly as in
    2015_gasparrini_Lancet_Rcodedata-master/00.prepdata.R + 01.firststage.R, with meta-predictors of 02.secondstage.R."""
    require_r('tsModel')
    csv = REPO / '2015_gasparrini_Lancet_Rcodedata-master' / 'regEngWales.csv'
    r(f'''
    {NAMES}ew <- read.csv("{csv}", row.names = 1); {NAMES}ew$date <- as.Date({NAMES}ew$date)
    {NAMES}regs <- sort(as.character(unique({NAMES}ew$regnames)))
    {NAMES}fs <- lapply({NAMES}regs, function(x) {{
      d <- {NAMES}ew[{NAMES}ew$regnames == x, ]
      cb <- crossbasis(d$tmean, lag = 21, argvar = list(fun = "bs", degree = 2,
                       knots = quantile(d$tmean, c(10, 75, 90)/100, na.rm = TRUE)),
                       arglag = list(knots = logknots(21, 3)))
      mod <- glm(death ~ cb + dow + ns(date, df = 8 * length(unique(year))), d, family = quasipoisson,
                 na.action = "na.exclude")
      red <- crossreduce(cb, mod, cen = mean(d$tmean, na.rm = TRUE))
      list(coef = coef(red), vcov = vcov(red), avg = mean(d$tmean, na.rm = TRUE),
           rng = diff(range(d$tmean, na.rm = TRUE)))
    }})''')
    n = int(r(f'length({NAMES}fs)')[0])
    y = rget(f'do.call(rbind, lapply({NAMES}fs, function(f) unname(f$coef)))')
    S = np.array([rget(f'unname({NAMES}fs[[{i + 1}]]$vcov)') for i in range(n)])
    avg = rget(f'sapply({NAMES}fs, function(f) f$avg)').ravel()
    rng = rget(f'sapply({NAMES}fs, function(f) f$rng)').ravel()
    return y, S, np.column_stack([np.ones(n), avg, rng])


def test_england_wales_second_stage_agrees_with_tightly_converged_r(ew_first_stage):
    """Validated path (REML): 10 regions, k=5 outcomes, meta-regression on avg. temperature and temperature range
    (p=3). Rank-deficient Psi (flat REML surface): still <= 1e-6 against R at reltol=1e-14, BLUPs included."""
    y, S, X = ew_first_stage
    ref = r_fit(y, S, X, 'reml', R_TIGHT)
    m, _ = py_fit(y, S, X, 'reml')
    assert_fit_close(m, ref, rtol=1e-6, what='E&W REML vs tight R')
    rb = r_blup_fit(y, S, X, 'reml', R_TIGHT)
    b, bv = py_blup(m)
    assert_close(b, rb['blup'], rtol=1e-6, what='BLUP')
    assert_close(bv, rb['bvcov'], rtol=1e-6, what='BLUP vcov')


def test_england_wales_second_stage_default_r_level(ew_first_stage):
    """Documents the validated-path level (mvmeta-blup-14): against R's default control the E&W REML second stage
    agrees to about 1e-5 (BLUP 6e-6, Psi 3e-5) -- optimiser-limited, so 'machine precision' is not the right
    description. Do NOT tighten these bounds."""
    y, S, X = ew_first_stage
    ref = r_fit(y, S, X, 'reml', R_DEFAULT)
    m, _ = py_fit(y, S, X, 'reml')
    assert_fit_close(m, ref, rtol=1e-4, what='E&W REML vs default R')
    rb = r_blup_fit(y, S, X, 'reml', R_DEFAULT)
    b, bv = py_blup(m)
    assert_close(b, rb['blup'], rtol=1e-4, what='BLUP')
    assert_close(bv, rb['bvcov'], rtol=1e-4, what='BLUP vcov')


# --------------------------------------------------------------------------------------------------------------
# Added with the fixes: what the N1a / N1b / N1c work made faithful (plain tests, all against R at run time)
# --------------------------------------------------------------------------------------------------------------
def _check_objective_and_gradients(y, S, X, reps=3, tau=0.3, seed=1):
    """-fn equals R's remlprof.fn / mlprof.fn and -gr R's remlprof.gr / mlprof.gr at the same Psi (random lower-Cholesky
    parameters; missing outcomes masked study by study on both sides). Python's row-major parameter order is mapped to
    R's column-wise one explicitly."""
    import meta_analysis as ma
    k = y.shape[1]
    push(y, S, X)
    r(f'{NAMES}L <- {NAMES}lists(as.matrix({NAMES}X), as.matrix({NAMES}y), as.matrix({NAMES}Sv))')
    data = ma._prepare(y, S, X, ma._mvmeta_control(None))
    Xl, yl, Sl, nal = ma._study_lists(data['y'], data['S'], data['X'])
    rng = np.random.default_rng(seed)
    ti = np.tril_indices(k)
    order_py = list(zip(*ti))
    order_r = [(a, b) for b in range(k) for a in range(b, k)]
    for rep in range(reps):
        L = np.tril(rng.normal(size=(k, k))) * tau
        par_py = L[ti]
        np2r(NAMES + 'par', r_par(L))
        for what, fn, gr in [('remlprof', ma._reml_fn, ma._reml_gr), ('mlprof', ma._ml_fn, ma._ml_gr)]:
            ref = float(r(f'{NAMES}prof({NAMES}par, {NAMES}L, "{what}.fn")')[0])
            assert_close(np.array([-fn(par_py, k, Xl, yl, Sl, nal)]), np.array([ref]), rtol=1e-10,
                         what=f'{what}.fn at rep {rep}')
            g_py = -gr(par_py, k, Xl, yl, Sl, nal)                       # gradient of the log-likelihood, Python order
            g_py_in_r = np.array([g_py[order_py.index(ab)] for ab in order_r])
            g_r = r2np(r(f'{NAMES}prof({NAMES}par, {NAMES}L, "{what}.gr")')).ravel()
            assert_close(g_py_in_r, g_r, rtol=1e-10, what=f'{what}.gr at rep {rep}')


@pytest.mark.parametrize('cfg', CFGS + [CFG_K5], ids=CFG_IDS + ['n40k5p3'])
def test_ml_gradient_equals_rs_mlprof_gr_and_reml_gradient_still_does(cfg):
    """N1c: the analytic ML gradient (REML gradient without the tr(inv(X'WX) X_j' W dPsi W X_j) term) equals
    mvmeta:::mlprof.gr (gradchol.ml) at identical Psi, like the REML one equals remlprof.gr."""
    n, k, p, tau, seed = cfg
    y, S, X = sim(n, k, p, tau, seed)
    _check_objective_and_gradients(y, S, X, tau=tau, seed=2000 + seed)


def test_objective_and_gradients_with_missing_outcomes_match_r_at_identical_psi():
    """N1b: the per-study masking of missing outcomes in the GLS, the objective and both gradients is R's."""
    y, S, X, _ = _nan_case('outcome')
    _check_objective_and_gradients(y, S, X)


@pytest.mark.parametrize('method', ['fixed', 'mm', 'vc'])
@pytest.mark.parametrize('cfg', CFGS[:3], ids=CFG_IDS[:3])
def test_fixed_mm_and_vc_are_exact_ports_of_r(cfg, method):
    """N1a: 'fixed' (GLS with Psi = 0, its logLik), 'mm' (method of moments) and 'vc' (variance components) reproduce R's
    mvmeta.fixed / mvmeta.mm / mvmeta.vc to rounding (closed form / identical fixed-point iteration, same default
    reltol); mm / vc have no likelihood (logLik NA in R, NaN here)."""
    y, S, X = sim(*cfg)
    ref = r_fit(y, S, X, method, R_DEFAULT)
    m, msgs = py_fit(y, S, X, method)
    assert not msgs
    assert m.converged is True
    assert_close(m.coefficients, ref['coef'], rtol=1e-10, what=f'{method} coefficients')
    assert_close(m.vcov, ref['vcov'], rtol=1e-10, what=f'{method} vcov')
    if method == 'fixed':
        assert abs(m.loglik - ref['loglik']) <= 1e-10 * abs(ref['loglik'])
        assert np.allclose(m.psi, 0.0)
    else:
        assert_close(m.psi, ref['psi'], rtol=1e-9, what=f'{method} Psi')
        assert np.isnan(m.loglik) and np.isnan(ref['loglik'])


@pytest.mark.parametrize('method', ['reml', 'ml', 'fixed', 'mm', 'vc'])
@pytest.mark.parametrize('kind', ['outcome', 'whole_study', 'covariate'])
def test_missing_data_fits_and_blups_match_r_for_every_method(kind, method):
    """N1b: partly missing outcomes (masked per study), dropped studies / NA covariate rows: coefficients, vcov, Psi and
    the BLUPs (with their variances; R gives the missing outcomes a variance of 1e10, PyDLNM the exact limit) equal
    R's mvmeta() + blup(). blup() of the used studies, in input order, is what R returns (drop_omitted=True)."""
    from meta_analysis import blup
    y, S, X, n_used = _nan_case(kind)
    push(y, S, X)
    r(f'{NAMES}mv <- {NAMES}formula_fit(as.matrix({NAMES}y), as.matrix({NAMES}Sv), as.matrix({NAMES}X), '
      f'"{method}", {R_TIGHT})')
    p, k = X.shape[1], y.shape[1]
    ref = _read_r_blup(n_used, p, k)
    m, _ = py_fit(y, S, X, method)
    assert_close(m.coefficients, ref['coef'], rtol=1e-6, what=f'{kind}/{method} coefficients')
    assert_close(m.vcov, ref['vcov'], rtol=1e-6, what=f'{kind}/{method} vcov')
    if method != 'fixed':
        assert_close(m.psi, ref['psi'], rtol=1e-6, what=f'{kind}/{method} Psi')
    res = blup(m, vcov=True, drop_omitted=True)
    assert len(res) == n_used
    assert_close(np.array([x['blup'] for x in res]), ref['blup'], rtol=1e-6, what='BLUP')
    assert_close(np.array([x['vcov'] for x in res]), ref['bvcov'], rtol=1e-6, what='BLUP vcov')


@pytest.mark.parametrize('kind', ['whole_study', 'covariate'])
def test_blup_list_stays_aligned_with_the_input_rows_when_studies_are_dropped(kind):
    """N1b: dropped studies are reported in m.na_action and get NaN BLUPs in the default (input-aligned) list, so callers
    that index blup() by their own study order (MultiLocationDLNM) cannot be silently misaligned."""
    from meta_analysis import blup
    y, S, X, n_used = _nan_case(kind)
    m, _ = py_fit(y, S, X)
    dropped = 4 if kind == 'whole_study' else 3
    assert m.na_action.tolist() == [dropped] and m.n == n_used and m.n_input == y.shape[0]
    full = blup(m, vcov=True)
    short = blup(m, vcov=True, drop_omitted=True)
    assert len(full) == y.shape[0] and len(short) == n_used
    assert np.isnan(full[dropped]['blup']).all() and np.isnan(full[dropped]['vcov']).all()
    kept = [i for i in range(y.shape[0]) if i != dropped]
    for a, b in zip(short, [full[i] for i in kept]):
        assert np.array_equal(a['blup'], b['blup']) and np.array_equal(a['vcov'], b['vcov'])


def test_vech_rows_with_missing_outcomes_equal_the_full_array():
    """N1b: R's vech-row S (NaN where an outcome is missing) and the (n,k,k) array describe the same model."""
    y, S, X, _ = _nan_case('outcome')
    m_rows, _ = py_fit(y, vech_rows(S), X)
    m_full, _ = py_fit(y, S, X)
    assert_close(m_rows.psi, m_full.psi, rtol=1e-12, what='Psi')
    assert_close(m_rows.coefficients, m_full.coefficients, rtol=1e-12, what='coefficients')


def test_invalid_data_raise_like_r():
    """N1b: infinite values, an S that is missing for an observed outcome (R: "missing pattern in 'y' and S' is not
    consistent"), fewer than two usable studies (R: "less than 2 valid studies after exclusion of missing") and an S
    of the wrong shape are errors with a clear message."""
    y, S, X = sim(30, 2, 2, 0.4, 5)
    yi = y.copy()
    yi[2, 0] = np.inf
    with pytest.raises(ValueError, match='infinite'):
        py_fit(yi, S, X)
    Sn = S.copy()
    Sn[3, 0, 0] = np.nan
    with pytest.raises(ValueError, match='missing pattern'):
        py_fit(y, Sn, X)
    y_one = np.full_like(y, np.nan)
    y_one[0] = 1.0
    with pytest.raises(ValueError, match='less than 2 valid studies'):
        py_fit(y_one, S, X)
    with pytest.raises(ValueError, match="dimensions for 'S'"):
        py_fit(y, S[:, :1, :], X)


def test_refit_after_a_rejected_input_keeps_the_previous_fit():
    """A fit() that is rejected during input validation leaves the fitted model untouched."""
    from meta_analysis import MVMeta
    y, S, X = sim(30, 2, 2, 0.4, 5)
    m = MVMeta().fit(y, S, X)
    psi, coef = m.psi.copy(), m.coefficients.copy()
    with pytest.raises(ValueError):
        m.fit(y, S[:5], X)
    assert np.array_equal(m.psi, psi) and np.array_equal(m.coefficients, coef)


# --------------------------------------------------------------------------------------------------------------
# N2  documentation claim versus measured agreement (mvmeta-est-11, mvmeta-blup-14)
# --------------------------------------------------------------------------------------------------------------
_CLAIM = re.compile(r'machine precision|exact match', re.I)
_MVMETA_STAGE = re.compile(r'mvmeta|full pipeline', re.I)
_QUALIFIED = re.compile(r'toleran|optimi[sz]|reltol|1e-\d|10\^-?\d|e-\d|except|approximate|about|<\s*\d', re.I)


@known_defect('N2', 'mvmeta-est-11', 'mvmeta-blup-14', note="README claims machine precision / Exact match for MVMeta")
def test_readme_precision_claims_for_the_mvmeta_stage_match_measurement():
    """The README may only say 'machine precision' / 'Exact match' about the MVMeta stage if the measured agreement
    with R's default mvmeta backs it (<= 1e-10); otherwise the line has to state the tolerance. Measured: ~1e-6..1e-5."""
    readme = (REPO / 'README.md').read_text(encoding='utf-8').splitlines()
    unqualified = [ln.strip() for ln in readme
                   if _CLAIM.search(ln) and _MVMETA_STAGE.search(ln) and not _QUALIFIED.search(ln)]
    if not unqualified:
        return                                              # claim removed or qualified: nothing to check
    y, S, X = sim(50, 3, 3, 0.3, 2)
    m, _ = py_fit(y, S, X)
    ref = r_fit(y, S, X, 'reml', R_DEFAULT)
    measured = max(max_rel_diff(m.coefficients, ref['coef']), max_rel_diff(m.psi, ref['psi']))
    assert measured <= 1e-10, (f'README claims exactness for the MVMeta stage in {len(unqualified)} line(s), e.g. '
                               f'{unqualified[0]!r}, but Python vs default-R max relative difference is {measured:.1e}')
