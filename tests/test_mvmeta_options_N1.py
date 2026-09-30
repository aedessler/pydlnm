"""MVMeta control options, ill-conditioned S, ownership of the caller's arrays, and the par <-> Psi order
(PyDLNM meta_analysis.py versus R mvmeta 1.0.3 / mixmeta 1.2.0).

Every reference number is computed by R at run time (mvmeta:::mvmeta.fit, mvmeta(), blup.mvmeta, mvmeta:::par2Psi,
mvmeta:::remlprof.fn, mixmeta:::inputcov) on the same simulated inputs that PyDLNM receives.

Theme N1d  MVMeta control options are silently ignored
  mvmeta-est-6, mvmeta-blup-7
                  MVMeta reads control['igls.iter'], ['maxiter'], ['showiter'] and nothing else.
                  * unknown keys are accepted (R: mvmeta.control() -> "unused argument")
                  * igls.iter < 1 is accepted (R: "'igls.iter' in the control list must be positive")
                  * Scor is ignored: for a variance-only (n,k) S, R builds within-study covariances with correlation
                    Scor (mvmeta.fit and blup.mvmeta both call inputcov(sqrt(S), Scor)), which changes Psi, the
                    coefficients and the BLUPs; PyDLNM always returns the Scor=0 fit.  Scor outside [-1, 1] is an
                    R error (inputcov), PyDLNM does not notice.
                  * initPsi, reltol, optim and hessian change nothing (R: initPsi is the optimiser start, reltol /
                    optim the stopping rule, hessian=TRUE adds fit$hessian)
Theme N1e  BFGS aborts early on extremely ill-conditioned S
  mvmeta-est-8    k=5, within-study correlation 0.999, SD ratio 0.03 per outcome, Psi on the boundary (median
                  cond(S_i) ~ 1e15): scipy BFGS stops after ~20 iterations ("precision loss", |g| ~ 1e8) at a log-
                  likelihood 0.4 to 0.8 below the optimum that R's optim BFGS reaches (~175 iterations).
Theme N1f  MVMeta keeps views of the caller's arrays; parameter order of _par2Psi / _Psi2par
  mvmeta-est-9, mvmeta-blup-10
                  fit() stores y / S / X without copying (np.asarray on float64 input), blup() reads them at call
                  time: an in-place edit, or a buffer reused for the next fit, silently changes the BLUPs of a model
                  whose Psi and coefficients did not change.  R's model object has value semantics.
  mvmeta-est-10   _par2Psi documents "column-major like R's lower.tri" but fills the Cholesky factor with
                  np.tril_indices (row-major); the two orders differ for k >= 3.  Numerically harmless for fits (all
                  four sites are consistent), but the parametrisation is not R's.  The fix may be the docstring or
                  the code, so the docstring test below passes either way, while the R-order test asserts the
                  documented R parametrisation (delete it if the docstring is changed instead).

Tests decorated with @known_defect assert the R-faithful (or documented-correct) behaviour and fail today (strict
xfail); the plain tests guard neighbouring behaviour that is already faithful and must keep passing while the fixes land.

Notes for whoever fixes these
  * "Honoured or rejected": for initPsi / reltol / optim / hessian / Scor-as-a-vector the tests accept either an
    observable effect or a clear error that names the option (NotImplementedError / ValueError / TypeError /
    KeyError); only a silent no-op fails.  Scalar Scor on a variance-only S must reproduce R's fit.
  * mvmeta-est-8: the tolerance is relative, loglik >= R_tight - 1e-6 * |R_tight| (about 2e-3 absolute at loglik
    ~1900); the defect is 0.4 to 0.8.  Restarting scipy BFGS from its own exit point gets to about that accuracy.
"""
import contextlib
import os
import re
import warnings

import numpy as np
import pytest

from rhelpers import assert_close, known_defect, max_rel_diff, np2r, r, r2np

NAMES = 'mmo_'          # prefix of every object this module creates in R's global environment

# optimiser settings handed to mvmeta.control()
R_TIGHT = 'list(maxiter=20000, reltol=1e-14)'
R_DEFAULT = 'list()'

REJECTIONS = (NotImplementedError, ValueError, TypeError, KeyError)      # "clear error" for an unsupported option

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
    """Home of the R that is actually running (os.environ['R_HOME'] is rewritten by some PyDLNM imports; R then
    segfaults when it lazily dlopen()s its LAPACK module).  Read the path of a loaded R library instead."""
    p = str(r('as.character(getLoadedDLLs()[["utils"]][["path"]])')[0])
    for _ in range(4):
        p = os.path.dirname(p)
    return p


@contextlib.contextmanager
def correct_r_home():
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
        require_r('mvmeta', 'mixmeta')
        r(_R_DEFS)
        r('invisible(list(chol(diag(2)), solve(diag(2)), eigen(diag(2)), svd(diag(2)), qr(diag(2))))')   # load LAPACK now


@pytest.fixture(autouse=True)
def _restore_r_home():
    with correct_r_home():
        yield


@pytest.fixture(scope='module', autouse=True)
def _single_threaded_blas():
    """The matrices here are tiny; BLAS worker threads only spin.  Numerical results are unaffected."""
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


def sim_var(n, k, p=1, tau=0.3, seed=0, sw=0.2):
    """y (n,k), V (n,k) within-study VARIANCES only (R: a k-column S), X (n,p)."""
    y, S, X = sim(n, k, p, tau, seed, sw)
    return y, np.stack([np.diag(S[i]) for i in range(n)]), X


def sim_illcond(n, k, tau, seed, ratio, corr, base=0.2):
    """Within-study covariances D R D with R = corr*11' + (1-corr)*I and SD ratio `ratio` between consecutive
    outcomes: cond(S_i) grows like ratio^(-2(k-1)) / (1-corr).  Returns y (n,k), S (n,k,k), X = intercept."""
    rng = np.random.default_rng(seed)
    beta = rng.normal(size=k)
    A = rng.normal(size=(k, k))
    Psi = tau ** 2 * (A @ A.T / k + 0.2 * np.eye(k))
    Rm = np.full((k, k), corr) + (1 - corr) * np.eye(k)
    S = np.zeros((n, k, k))
    y = np.zeros((n, k))
    for i in range(n):
        d = base * np.exp(rng.normal(scale=0.3, size=k)) * ratio ** np.arange(k)
        S[i] = np.outer(d, d) * Rm
        y[i] = beta + rng.multivariate_normal(np.zeros(k), Psi + S[i], check_valid='ignore')
    return y, S, np.ones((n, 1))


def vech_rows(S):
    """R's vech-row layout of S: lower triangle of each S_i taken column-wise (mixmeta::vechMat / xpndMat)."""
    n, k, _ = S.shape
    idx = [(a, b) for b in range(k) for a in range(b, k)]
    return np.array([[S[i, a, b] for (a, b) in idx] for i in range(n)])


def unvech_rows(rows, k):
    """Inverse of vech_rows: (n, k(k+1)/2) -> (n,k,k) symmetric."""
    rows = np.asarray(rows, dtype=float)
    S = np.zeros((rows.shape[0], k, k))
    for j, (a, b) in enumerate([(a, b) for b in range(k) for a in range(b, k)]):
        S[:, a, b] = rows[:, j]
        S[:, b, a] = rows[:, j]
    return S


def r_error(code):
    """None if the R code runs, otherwise the first line of R's error message."""
    try:
        r(code)
    except Exception as e:                           # rpy2 RRuntimeError
        return str(e).strip().splitlines()[0]
    return None


def r_ctrl(base=R_TIGHT, **extra):
    """R control list literal: base settings plus extra entries given as R source text."""
    body = base[len('list('):-1]
    more = ', '.join(f'{k}={v}' for k, v in extra.items())
    return 'list(' + ', '.join(s for s in (body, more) if s) + ')'


def r_fit(y, S_rows, X, method='reml', control=R_TIGHT):
    """R mvmeta.fit (the estimation engine of mvmeta()).  S_rows is exactly what R receives: vech rows of the
    within-study covariances (n, k(k+1)/2) or, for a variance-only S, the (n, k) variances."""
    np2r(NAMES + 'y', y)
    np2r(NAMES + 'Sv', S_rows)
    np2r(NAMES + 'X', X)
    r(f'{NAMES}fit <- suppressWarnings(mvmeta:::mvmeta.fit(as.matrix({NAMES}X), as.matrix({NAMES}y), '
      f'as.matrix({NAMES}Sv), method="{method}", control={control}))')
    p, k = X.shape[1], y.shape[1]
    return dict(coef=np.asarray(r2np(r(f'{NAMES}fit$coefficients'))).reshape(p, k),
                vcov=r2np(r(f'{NAMES}fit$vcov')), psi=r2np(r(f'{NAMES}fit$Psi')),
                loglik=float(r2np(r(f'as.numeric({NAMES}fit$logLik)')).ravel()[0]),
                niter=int(r2np(r(f'{NAMES}fit$niter')).ravel()[0]))


def r_blup_fit(y, S_rows, X, method='reml', control=R_TIGHT):
    """R mvmeta() (formula interface, X includes the intercept) followed by blup(vcov=TRUE); needs k >= 2."""
    np2r(NAMES + 'y', y)
    np2r(NAMES + 'Sv', S_rows)
    np2r(NAMES + 'X', X)
    n, p, k = y.shape[0], X.shape[1], y.shape[1]
    r(f'{NAMES}mv <- suppressWarnings({NAMES}formula_fit(as.matrix({NAMES}y), as.matrix({NAMES}Sv), '
      f'as.matrix({NAMES}X), "{method}", {control}))')
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


def fit_or_clear_rejection(y, S, X, control, key):
    """(model, None) if PyDLNM accepts `control`; (None, exception) if it rejects it with an error that names `key`
    (an unsupported option must be refused loudly, never ignored)."""
    try:
        m, _ = py_fit(y, S, X, control=control)
    except REJECTIONS as e:
        assert key.lower() in str(e).lower(), f'control option {key!r} rejected without naming it: {e!r}'
        return None, e
    return m, None


def r_par(L):
    """R's parameter order for a lower-Cholesky factor: lower.tri(L, diag=TRUE) filled column-wise."""
    k = L.shape[0]
    return np.array([L[a, b] for b in range(k) for a in range(b, k)])


def r_par2psi(par, k):
    np2r(NAMES + 'par', np.asarray(par, dtype=float))
    return r2np(r(f'mvmeta:::par2Psi({NAMES}par, {k}, "unstr", NULL)'))


def r_psi2par(Psi):
    """R's initpar for bscov='unstr': vechMat(t(chol(Psi)))."""
    np2r(NAMES + 'Psi', Psi)
    return r2np(r(f'as.numeric(mixmeta::vechMat(t(chol({NAMES}Psi))))')).ravel()


def random_chol_factor(k, seed):
    rng = np.random.default_rng(seed)
    L = np.tril(rng.normal(size=(k, k)))
    L[np.diag_indices(k)] = rng.uniform(0.5, 1.5, size=k)            # positive diagonal: Psi = L L' is positive definite
    return L


# ==============================================================================================================
# N1d  mvmeta-est-6 / mvmeta-blup-7: unknown control keys, igls.iter < 1
# ==============================================================================================================
@known_defect('N1d', 'mvmeta-est-6', 'mvmeta-blup-7', note='control is a free dict; unknown keys are accepted silently')
@pytest.mark.parametrize('key', ['bogus', 'checkPD', 'addSlist'])
def test_unknown_control_key_raises_like_r(key):
    """R: mvmeta.control() has no such argument -> error "unused argument".  (checkPD / addSlist belong to
    mixmeta.control(), not mvmeta.control(), so R refuses them too.)"""
    y, V, X = sim_var(30, 3, 2, 0.3, 5)
    assert r_error(f'mvmeta:::mvmeta.control({key}=1)') is not None, f'R must reject {key!r}'
    np2r(NAMES + 'y', y)
    np2r(NAMES + 'Sv', V)
    np2r(NAMES + 'X', X)
    assert r_error(f'mvmeta:::mvmeta.fit({NAMES}X, {NAMES}y, {NAMES}Sv, method="reml", control=list({key}=1))') is not None
    from meta_analysis import MVMeta
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with pytest.raises(REJECTIONS) as exc:
            MVMeta(control={key: 1}).fit(y, V, X)
    assert key in str(exc.value), f'the error should name the offending key {key!r}: {exc.value!r}'


@known_defect('N1d', 'mvmeta-est-6', note="igls.iter < 1 is accepted; R: \"'igls.iter' in the control list must be positive\"")
@pytest.mark.parametrize('n_iter', [0, -1])
def test_igls_iter_must_be_positive_like_r(n_iter):
    y, V, X = sim_var(30, 3, 2, 0.3, 5)
    assert r_error(f'mvmeta:::mvmeta.control(igls.iter={n_iter})') is not None, 'R must reject igls.iter < 1'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with pytest.raises(REJECTIONS):
            py_fit(y, V, X, control={'igls.iter': n_iter})


# ==============================================================================================================
# N1d  mvmeta-est-6 / mvmeta-blup-7: Scor (within-study correlation for a variance-only S)
# ==============================================================================================================
# (n, k, p, seed), Scor
SCOR_CASES = [((30, 3, 2, 5), 0.6), ((40, 3, 1, 11), 0.5), ((30, 3, 2, 5), -0.2)]
SCOR_IDS = ['n30k3p2-0.6', 'n40k3p1-0.5', 'n30k3p2--0.2']


@known_defect('N1d', 'mvmeta-est-6', 'mvmeta-blup-7', note='Scor is ignored: the (n,k) S is always expanded with np.diag')
@pytest.mark.parametrize('cfg,scor', SCOR_CASES, ids=SCOR_IDS)
def test_scor_with_variance_only_S_matches_r_fit(cfg, scor):
    """R: a k-column S is expanded by inputcov(sqrt(S), Scor): off-diagonals Scor * sqrt(v_i v_j).  Psi, coefficients,
    vcov and logLik all move; PyDLNM must follow (vs R at reltol = 1e-14)."""
    n, k, p, seed = cfg
    y, V, X = sim_var(n, k, p, 0.3, seed)
    r0 = r_fit(y, V, X)
    rs = r_fit(y, V, X, control=r_ctrl(Scor=scor))
    assert max_rel_diff(rs['psi'], r0['psi']) > 1e-2, 'precondition: Scor materially changes the fit in R'
    m, _ = py_fit(y, V, X, control={'Scor': scor})
    assert_fit_close(m, rs, rtol=1e-6, what=f'Scor={scor}')


@known_defect('N1d', 'mvmeta-est-6', 'mvmeta-blup-7', note='Scor is ignored; engine itself is fine with an explicit S')
@pytest.mark.parametrize('scor', [0.6, -0.3])
def test_scor_equals_explicit_within_study_correlation_matrices(scor):
    """Optimiser-independent check of the expansion: Scor with variances must be the same model as passing the
    (n,k,k) covariances that R's inputcov() builds (PyDLNM's engine on explicit S already matches R)."""
    n, k, p = 30, 3, 2
    y, V, X = sim_var(n, k, p, 0.3, 5)
    np2r(NAMES + 'V', V)
    rows = r2np(r(f'mixmeta:::inputcov(sqrt({NAMES}V), {scor!r})'))
    S3 = unvech_rows(rows, k)
    assert np.abs(S3[:, 0, 1]).max() > 0 and np.allclose(S3[:, 1, 1], V[:, 1]), 'precondition: R builds correlated S'
    m_ref, _ = py_fit(y, S3, X)
    m, _ = py_fit(y, V, X, control={'Scor': scor})
    assert_close(m.psi, m_ref.psi, rtol=1e-8, what='Psi')
    assert_close(m.coefficients, m_ref.coefficients, rtol=1e-8, what='coefficients')
    assert_close(m.vcov, m_ref.vcov, rtol=1e-8, what='vcov')
    b, bv = py_blup(m)
    b_ref, bv_ref = py_blup(m_ref)
    assert_close(b, b_ref, rtol=1e-8, what='BLUP')
    assert_close(bv, bv_ref, rtol=1e-8, what='BLUP vcov')


@known_defect('N1d', 'mvmeta-blup-7', 'mvmeta-est-6', note='blup() re-applies Scor in R; PyDLNM BLUPs use the Scor=0 S')
def test_scor_blup_matches_r():
    """blup.mvmeta re-expands S with inputcov(sqrt(S), object$control$Scor), so the BLUPs and their vcov inherit Scor."""
    n, k, p = 30, 3, 2
    y, V, X = sim_var(n, k, p, 0.3, 5)
    rb = r_blup_fit(y, V, X, control=r_ctrl(Scor=0.6))
    rb0 = r_blup_fit(y, V, X)
    assert max_rel_diff(rb['blup'], rb0['blup']) > 1e-3, 'precondition: Scor materially changes the BLUPs in R'
    m, _ = py_fit(y, V, X, control={'Scor': 0.6})
    b, bv = py_blup(m)
    assert_close(b, rb['blup'], rtol=1e-5, what='BLUP')
    assert_close(bv, rb['bvcov'], rtol=1e-5, what='BLUP vcov')


@known_defect('N1d', 'mvmeta-est-6', 'mvmeta-blup-7', note='vector-valued Scor is ignored')
@pytest.mark.parametrize('which', ['per_pair', 'per_study'])
def test_vector_scor_matches_r_or_is_rejected(which):
    """R's inputcov accepts length k(k-1)/2 (one correlation per outcome pair, lower triangle column-wise) or length n
    (one per study).  PyDLNM must follow R or refuse, never return the Scor=0 fit."""
    n, k, p = 30, 3, 2
    y, V, X = sim_var(n, k, p, 0.3, 5)
    vec = np.array([0.2, 0.5, -0.3]) if which == 'per_pair' else np.linspace(-0.4, 0.7, n)
    np2r(NAMES + 'scor', vec)
    r(f'{NAMES}scor <- as.numeric({NAMES}scor)')                     # a plain R vector, as inputcov() expects
    r0 = r_fit(y, V, X)
    rs = r_fit(y, V, X, control=r_ctrl(Scor=f'{NAMES}scor'))
    assert max_rel_diff(rs['psi'], r0['psi']) > 1e-2, 'precondition: Scor materially changes the fit in R'
    m, err = fit_or_clear_rejection(y, V, X, {'Scor': vec}, 'Scor')
    if m is None:
        return
    assert_fit_close(m, rs, rtol=1e-6, what=f'Scor {which}')


@known_defect('N1d', 'mvmeta-est-6', note='|Scor| > 1 is not noticed; R: inputcov "correlations must be between -1 and 1"')
@pytest.mark.parametrize('scor', [1.5, -1.01])
def test_scor_outside_minus_one_one_raises_like_r(scor):
    y, V, X = sim_var(30, 3, 2, 0.3, 5)
    np2r(NAMES + 'y', y)
    np2r(NAMES + 'Sv', V)
    np2r(NAMES + 'X', X)
    assert r_error(f'mvmeta:::mvmeta.fit({NAMES}X, {NAMES}y, {NAMES}Sv, method="reml", control=list(Scor={scor!r}))') \
        is not None, 'R must reject |Scor| > 1 for a variance-only S'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with pytest.raises(REJECTIONS):
            py_fit(y, V, X, control={'Scor': scor})


@pytest.mark.parametrize('cfg', [(30, 3, 2, 5), (40, 3, 1, 11)], ids=['n30k3p2', 'n40k3p1'])
def test_variance_only_S_without_scor_matches_r_zero_correlation(cfg):
    """Plain: the default (Scor = 0, explicit or not) is the diagonal S fit, equal to R's."""
    n, k, p, seed = cfg
    y, V, X = sim_var(n, k, p, 0.3, seed)
    ref = r_fit(y, V, X)
    ref0 = r_fit(y, V, X, control=r_ctrl(Scor=0))
    assert_close(ref0['psi'], ref['psi'], rtol=1e-12, what='R: Scor=0 is the default')
    for control in (None, {}, {'Scor': 0}, {'Scor': 0.0}):
        m, _ = py_fit(y, V, X, control=control)
        assert_fit_close(m, ref, rtol=1e-6, what=f'variance-only S, control={control}')


@pytest.mark.parametrize('scor', [0.5, -0.4])
def test_scor_is_irrelevant_when_a_full_S_is_given(scor):
    """Plain: R only applies Scor when S has k columns (mvmeta.fit: `if (dim(S)[2] == k)`); a full (n,k,k) covariance
    array is used as it is, so the fit does not depend on Scor -- in R and in PyDLNM."""
    y, S, X = sim(30, 3, 2, 0.3, 5)
    rows = vech_rows(S)
    ref = r_fit(y, rows, X)
    ref_s = r_fit(y, rows, X, control=r_ctrl(Scor=scor))
    assert_close(ref_s['psi'], ref['psi'], rtol=1e-12, what='R: Scor has no effect on a full S')
    m, _ = py_fit(y, S, X, control={'Scor': scor})
    assert_fit_close(m, ref_s, rtol=1e-6, what=f'full S with Scor={scor}')


# ==============================================================================================================
# N1d  mvmeta-est-6: initPsi / reltol / optim / hessian must be honoured or rejected
# ==============================================================================================================
@known_defect('N1d', 'mvmeta-est-6', note='initPsi is ignored (R: it is the optimiser start in place of the IGLS value)')
def test_initpsi_is_honoured_or_rejected():
    """One BFGS iteration from R's own optimum stays at the optimum; from the IGLS start it does not.  If initPsi is
    honoured the first must hold; otherwise PyDLNM has to refuse the option."""
    y, V, X = sim_var(30, 3, 2, 0.3, 5)
    ref = r_fit(y, V, X)
    base, _ = py_fit(y, V, X, control={'maxiter': 1})                # IGLS start, a single BFGS iteration
    assert max_rel_diff(base.psi, ref['psi']) > 1e-2, 'precondition: one iteration from the IGLS start is not converged'
    m, err = fit_or_clear_rejection(y, V, X, {'maxiter': 1, 'initPsi': ref['psi']}, 'initPsi')
    if m is None:
        return
    assert max_rel_diff(m.psi, ref['psi']) <= 1e-4, \
        f'initPsi = the optimum was ignored: Psi is {max_rel_diff(m.psi, ref["psi"]):.2e} (relative) from it after one iteration'
    assert m.loglik >= ref['loglik'] - 1e-6 * abs(ref['loglik'])


@known_defect('N1d', 'mvmeta-est-6', note='reltol / optim are ignored (R: they set the optim stopping rule)')
@pytest.mark.parametrize('key,py_control,r_control', [
    ('reltol', {'reltol': 0.1}, 'list(reltol=0.1)'),
    ('optim', {'optim': {'reltol': 0.1}}, 'list(optim=list(reltol=0.1))'),
], ids=['reltol', 'optim'])
def test_stopping_rule_options_are_honoured_or_rejected(key, py_control, r_control):
    """R stops optim after one iteration at reltol = 0.1, a Psi visibly off the converged one.  PyDLNM must show some
    effect of the option or refuse it; bit-identical output to the no-option fit is the silent no-op."""
    y, V, X = sim_var(30, 3, 2, 0.3, 5)
    r_base = r_fit(y, V, X, control=R_DEFAULT)
    r_ctl = r_fit(y, V, X, control=r_control)
    assert max_rel_diff(r_ctl['psi'], r_base['psi']) > 1e-6, f'precondition: {key} changes the fit in R'
    base, _ = py_fit(y, V, X)
    m, err = fit_or_clear_rejection(y, V, X, py_control, key)
    if m is None:
        return
    assert not np.array_equal(m.psi, base.psi), f'control {py_control} has no effect: Psi is bit-identical to the default fit'


@known_defect('N1d', 'mvmeta-est-6', note='hessian=True adds no Hessian (R: fit$hessian, npar x npar)')
def test_hessian_option_returns_a_hessian_or_is_rejected():
    y, V, X = sim_var(30, 3, 2, 0.3, 5)
    r_fit(y, V, X, control=r_ctrl(hessian='TRUE'))                  # pushes the inputs into R and fits with hessian=TRUE
    ref_h = r2np(r(f'if (is.null({NAMES}fit$hessian)) matrix(numeric(0), 0, 0) else {NAMES}fit$hessian'))
    assert ref_h.shape == (6, 6), 'R: fit$hessian is npar x npar (npar = k(k+1)/2)'
    m, err = fit_or_clear_rejection(y, V, X, {'hessian': True}, 'hessian')
    if m is None:
        return
    H = getattr(m, 'hessian', None)
    assert H is not None, 'hessian=True was accepted but no Hessian is returned'
    H = np.asarray(H, dtype=float)
    assert H.shape == ref_h.shape and np.isfinite(H).all()
    assert np.allclose(H, H.T, rtol=1e-4, atol=1e-6 * np.abs(H).max()), 'a Hessian is symmetric'


# --------------------------------------------------------------------------------------------------------------
# N1d plain: options that are honoured today
# --------------------------------------------------------------------------------------------------------------
def test_maxiter_showiter_and_igls_iter_are_honoured(capsys):
    """Plain: the three control keys MVMeta does read keep working (any fix of the validation must keep them)."""
    y, V, X = sim_var(30, 3, 2, 0.3, 5)
    base, _ = py_fit(y, V, X)
    capsys.readouterr()
    # maxiter: one BFGS iteration stops short of the optimum and says so
    m1, msgs = py_fit(y, V, X, control={'maxiter': 1})
    assert not m1.converged and any('iteration' in s.lower() for s in msgs), msgs
    assert max_rel_diff(m1.psi, base.psi) > 1e-3
    # igls.iter: a different start gives a different first iterate (R: the same start value, initpar)
    m2, _ = py_fit(y, V, X, control={'maxiter': 1, 'igls.iter': 1})
    assert max_rel_diff(m2.psi, m1.psi) > 1e-3
    np2r(NAMES + 'y', y)
    np2r(NAMES + 'Sv', V)
    np2r(NAMES + 'X', X)
    assert r_error(f'mvmeta:::mvmeta.control(igls.iter=1, maxiter=1, showiter=TRUE)') is None     # valid in R as well
    # showiter: nothing printed by default, optimiser progress when asked for
    assert capsys.readouterr().out == ''
    py_fit(y, V, X, control={'showiter': True})
    assert capsys.readouterr().out.strip() != '', 'showiter=True should print the optimiser trace'


@pytest.mark.parametrize('cfg', [(30, 2, 2, 0.4, 5), (40, 3, 2, 0.3, 11), (25, 4, 1, 0.3, 7)],
                         ids=['n30k2p2', 'n40k3p2', 'n25k4p1'])
def test_fit_matches_r_with_identical_control(cfg):
    """Plain: REML with the same supported control values on both sides (igls.iter=10, maxiter=500; R additionally
    converged to reltol=1e-14) agrees to <= 1e-6 relative in Psi, coefficients, vcov and logLik."""
    n, k, p, tau, seed = cfg
    y, S, X = sim(n, k, p, tau, seed)
    ref = r_fit(y, vech_rows(S), X, control='list(igls.iter=10, maxiter=500, reltol=1e-14)')
    m, _ = py_fit(y, S, X, control={'igls.iter': 10, 'maxiter': 500})
    assert_fit_close(m, ref, rtol=1e-6, what='REML vs R with identical control')
    assert m.loglik >= ref['loglik'] - 1e-8 * abs(ref['loglik']), 'PyDLNM must not stop at a worse optimum'


# ==============================================================================================================
# N1e  mvmeta-est-8: BFGS abort on extremely ill-conditioned within-study covariances
# ==============================================================================================================
ILL_SEEDS = [7, 11, 501]


@known_defect('N1e', 'mvmeta-est-8', note="scipy BFGS exits with 'precision loss' ~20 iterations from the IGLS start")
@pytest.mark.parametrize('seed', ILL_SEEDS)
def test_extremely_ill_conditioned_S_reaches_r_optimum(seed):
    """k=5, n=40, corr 0.999, SD ratio 0.03, tau=0 (Psi on the boundary): median cond(S_i) ~ 1e15.  R's optim BFGS
    (reltol=1e-14) climbs to the optimum; PyDLNM must reach it too (log-likelihood not below R's beyond 1e-6 relative)."""
    y, S, X = sim_illcond(40, 5, 0.0, seed, 0.03, 0.999)
    assert np.median([np.linalg.cond(S[i]) for i in range(len(S))]) > 1e14, 'precondition: extremely ill-conditioned S'
    ref = r_fit(y, vech_rows(S), X)
    assert ref['niter'] > 50, 'precondition: R needs many iterations (it does not stop at the IGLS start)'
    m, _ = py_fit(y, S, X)
    assert np.isfinite(m.loglik)
    assert m.loglik >= ref['loglik'] - 1e-6 * abs(ref['loglik']), \
        f'PyDLNM logLik {m.loglik:.6f} is {ref["loglik"] - m.loglik:.3f} below the optimum R reaches ({ref["loglik"]:.6f})'


@pytest.mark.parametrize('ratio,corr', [(0.3, 0.99), (0.1, 0.99)], ids=['cond1e6', 'cond1e10'])
def test_ill_conditioned_S_below_1e11_matches_r(ratio, corr):
    """Plain: up to cond(S_i) ~ 1e10 BFGS has no trouble (real first-stage covariances have cond ~ 1e2); a fix of the
    optimiser driver must keep this agreement."""
    y, S, X = sim_illcond(40, 5, 0.0, 7, ratio, corr)
    assert np.median([np.linalg.cond(S[i]) for i in range(len(S))]) > 1e5
    ref = r_fit(y, vech_rows(S), X)
    m, _ = py_fit(y, S, X)
    assert abs(m.loglik - ref['loglik']) <= 1e-8 * abs(ref['loglik']), (m.loglik, ref['loglik'])
    assert_close(m.coefficients, ref['coef'], rtol=1e-5, what='coefficients')


def test_moderately_heterogeneous_ill_conditioned_S_with_interior_psi_matches_r():
    """Plain: Psi well inside the PSD cone (tau=0.3) with strongly correlated S_i (cond ~ 1e4): Python equals R."""
    y, S, X = sim_illcond(40, 4, 0.3, 3, 0.5, 0.99)
    ref = r_fit(y, vech_rows(S), X)
    m, _ = py_fit(y, S, X)
    assert_fit_close(m, ref, rtol=1e-6, what='ill-conditioned S, interior Psi')


# ==============================================================================================================
# N1f  mvmeta-est-9 / mvmeta-blup-10: the fitted model must own its data
# ==============================================================================================================
def _fit(entry, y, S, X):
    import meta_analysis as ma
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return ma.MVMeta().fit(y, S, X) if entry == 'MVMeta' else ma.mvmeta(y, S, X)


def _assert_blup_unchanged(m, before, what):
    b0, v0 = before
    b1, v1 = py_blup(m)
    assert np.array_equal(b1, b0), f'{what}: BLUPs changed by up to {np.abs(b1 - b0).max():.3g} (Psi and coefficients did not)'
    assert np.array_equal(v1, v0), f'{what}: BLUP vcov changed by up to {np.abs(v1 - v0).max():.3g}'


@known_defect('N1f', 'mvmeta-est-9', 'mvmeta-blup-10', note='self.y is the caller array; blup() reads it at call time')
@pytest.mark.parametrize('entry', ['MVMeta', 'mvmeta'])
@pytest.mark.parametrize('edit', ['scale', 'zero'])
def test_inplace_edit_of_y_after_fit_does_not_change_blup(entry, edit):
    """R's model object has value semantics: rescaling or overwriting the caller's y afterwards cannot alter blup()."""
    y, S, X = sim(30, 2, 2, 0.4, 5)
    m = _fit(entry, y, S, X)
    before = py_blup(m)
    psi0, coef0 = m.psi.copy(), m.coefficients.copy()
    if edit == 'scale':
        y *= 2.0
    else:
        y[:] = 0.0
    assert np.array_equal(m.psi, psi0) and np.array_equal(m.coefficients, coef0)
    _assert_blup_unchanged(m, before, f'{entry}: y edited in place ({edit})')


@known_defect('N1f', 'mvmeta-est-9', note='self.S is the caller array; blup() reads it (BLUP and BLUP vcov change)')
def test_inplace_edit_of_S_after_fit_does_not_change_blup():
    y, S, X = sim(30, 2, 2, 0.4, 5)
    m = _fit('MVMeta', y, S, X)
    before = py_blup(m)
    S *= 4.0
    _assert_blup_unchanged(m, before, 'S edited in place')


@known_defect('N1f', 'mvmeta-est-9', note='self.X aliases the caller array (inert for blup today, but the model keeps it)')
def test_inplace_edit_of_X_after_fit_keeps_the_model_design():
    y, S, X = sim(30, 2, 2, 0.4, 5)
    X0 = X.copy()
    m = _fit('MVMeta', y, S, X)
    X *= 3.0
    assert np.array_equal(m.X, X0), 'the fitted model must keep the design matrix it was fitted with'


def test_inplace_edit_of_X_after_fit_does_not_change_blup():
    """Plain: blup() uses the kron-built per-study design, so editing X never changed the BLUPs (keep it that way)."""
    y, S, X = sim(30, 2, 2, 0.4, 5)
    m = _fit('MVMeta', y, S, X)
    before = py_blup(m)
    X *= 3.0
    _assert_blup_unchanged(m, before, 'X edited in place')


@known_defect('N1f', 'mvmeta-est-9', 'mvmeta-blup-10', note='np.asarray(dtype=float) returns the caller buffer')
@pytest.mark.parametrize('name', ['y', 'S', 'X'])
def test_model_does_not_share_memory_with_the_callers_arrays(name):
    y, S, X = sim(30, 2, 2, 0.4, 5)
    m = _fit('MVMeta', y, S, X)
    caller = {'y': y, 'S': S, 'X': X}[name]
    assert not np.shares_memory(getattr(m, name), caller), f'model.{name} is a view of the array passed to fit()'
    if name in ('y', 'S'):                                           # the per-study lists blup() could also read
        per_study = m._ylist if name == 'y' else m._Slist
        assert not any(np.shares_memory(row, caller) for row in per_study), f'model._{name}list is a view of the caller array'


@known_defect('N1f', 'mvmeta-blup-10', 'mvmeta-est-9', note='refitting into one preallocated buffer (common in loops)')
def test_fits_from_one_reused_buffer_stay_independent():
    """Three datasets written one after another into the same preallocated arrays; each model's BLUPs, taken right
    after its own fit, must still be the same when asked for after the loop (and equal those of a fresh fit)."""
    data = [sim(30, 2, 2, 0.4, s) for s in (1, 2, 3)]
    buf_y, buf_S, buf_X = (np.empty_like(a) for a in data[0])
    models, at_fit = [], []
    for y, S, X in data:
        buf_y[:], buf_S[:], buf_X[:] = y, S, X
        m = _fit('MVMeta', buf_y, buf_S, buf_X)
        models.append(m)
        at_fit.append(py_blup(m))
    assert max_rel_diff(at_fit[0][0], at_fit[1][0]) > 1e-2, 'precondition: the datasets differ'
    for i, (m, before) in enumerate(zip(models, at_fit)):
        fresh = _fit('MVMeta', *[a.copy() for a in data[i]])
        assert_close(py_blup(fresh)[0], before[0], rtol=1e-12, what=f'fresh fit {i}')       # deterministic fit
        _assert_blup_unchanged(m, before, f'model {i} after the buffer was reused')


def test_blup_after_fit_matches_r_on_untouched_arrays():
    """Plain: with the caller's arrays left alone, PyDLNM's BLUPs equal R's (tight control) -- the behaviour the
    ownership fix has to keep, for the finding's own dataset shape (k=2, n=30, p=2)."""
    y, S, X = sim(30, 2, 2, 0.4, 5)
    m = _fit('MVMeta', y.copy(), S.copy(), X.copy())
    rb = r_blup_fit(y, vech_rows(S), X)
    b, bv = py_blup(m)
    assert_close(b, rb['blup'], rtol=1e-5, what='BLUP')
    assert_close(bv, rb['bvcov'], rtol=1e-5, what='BLUP vcov')


@pytest.mark.parametrize('kind', ['int_y', 'list_S', 'list_y_int_X'])
def test_non_float64_inputs_are_already_decoupled_from_the_caller(kind):
    """Plain: integer arrays and Python lists were always converted (copied) by fit(); their later edits never reach
    the model.  Float64 arrays are the only aliased case (above)."""
    rng = np.random.default_rng(3)
    y = rng.integers(-5, 6, size=(20, 2))
    S = [np.diag([1.0, 2.0]) * (1 + i % 3) for i in range(20)]
    X = np.ones((20, 1), dtype=int)
    if kind == 'int_y':
        args = (y, np.array(S), X)
        target = y
    elif kind == 'list_S':
        args = (y.astype(float), S, np.ones((20, 1)))
        target = S[0]
    else:
        args = ([list(map(float, row)) for row in y], np.array(S), X)
        target = X
    m = _fit('MVMeta', *args)
    before = py_blup(m)
    if isinstance(target, list):
        target *= 0
    else:
        target[...] = 0
    _assert_blup_unchanged(m, before, f'{kind}: caller edit after fit')


# ==============================================================================================================
# N1f  mvmeta-est-10: parameter order of _par2Psi / _Psi2par
# ==============================================================================================================
@known_defect('N1f', 'mvmeta-est-10',
              note="np.tril_indices fills the Cholesky factor row-major; R's lower.tri(diag=TRUE) fills column-major")
@pytest.mark.parametrize('k', [3, 4, 5])
def test_par_vector_uses_rs_column_major_lower_triangle_order(k):
    """par -> Psi -> par round trip through R: R's par2Psi(par) is PyDLNM's _par2Psi(par), and PyDLNM's _Psi2par(Psi)
    returns R's own initpar vechMat(t(chol(Psi))) -- i.e. parameters could be exchanged with R.  If the decision is to
    only correct the docstring (verifier's preferred fix), this test skips itself once the docstring names the
    row-major order, and test_par_vector_docstring_states_the_order_it_implements carries the guard."""
    import meta_analysis as ma
    doc = (ma._par2Psi.__doc__ or '').lower()
    if ('row-major' in doc or 'row major' in doc) and not np.allclose(ma._par2Psi(np.arange(1.0, 7.0), 3),
                                                                      r_par2psi(np.arange(1.0, 7.0), 3)):
        pytest.skip("finding resolved by the docstring-only fix: the row-major order is documented as PyDLNM's own")
    L = random_chol_factor(k, seed=40 + k)
    par_r = r_par(L)
    psi_r = r_par2psi(par_r, k)
    np.testing.assert_allclose(psi_r, L @ L.T, rtol=1e-13, atol=1e-13)          # R documents: lower.tri filled column-wise
    np.testing.assert_allclose(r_psi2par(psi_r), par_r, rtol=1e-10, atol=1e-12)
    assert_close(ma._par2Psi(par_r, k), psi_r, rtol=1e-12, what=f'k={k}: _par2Psi(par) vs R par2Psi(par)')
    assert_close(ma._Psi2par(psi_r), par_r, rtol=1e-10, what=f'k={k}: _Psi2par(Psi) vs R vechMat(t(chol(Psi)))')
    np.testing.assert_allclose(ma._Psi2par(ma._par2Psi(par_r, k)), par_r, rtol=1e-10, atol=1e-12)
    # the literal example of the finding: par = 1..6, k = 3
    p6 = np.arange(1.0, 7.0)
    assert_close(ma._par2Psi(p6, 3), r_par2psi(p6, 3), rtol=1e-12, what='par = 1..6, k = 3')


@known_defect('N1f', 'mvmeta-est-10', note='docstring says column-major like R, the code is row-major')
def test_par_vector_docstring_states_the_order_it_implements():
    """Passes after EITHER fix: the docstring of _par2Psi may claim R's column-major order only if the function does
    that, and otherwise has to name the order it really uses (row-major, np.tril_indices)."""
    import meta_analysis as ma
    doc = (ma._par2Psi.__doc__ or '').lower()
    k = 3
    par = np.arange(1.0, 7.0)
    actual = ma._par2Psi(par, k)
    L_col = np.zeros((k, k))
    for j, (a, b) in enumerate([(a, b) for b in range(k) for a in range(b, k)]):
        L_col[a, b] = par[j]
    L_row = np.zeros((k, k))
    L_row[np.tril_indices(k)] = par
    is_r_order = np.allclose(actual, r_par2psi(par, k), rtol=1e-12, atol=0) and np.allclose(actual, L_col @ L_col.T)
    is_row_order = np.allclose(actual, L_row @ L_row.T)
    assert is_r_order != is_row_order, 'the two orders must be distinguishable for k=3'
    says_row = 'row-major' in doc or 'row major' in doc
    says_col = 'column-major' in doc or 'column major' in doc
    if says_col and not says_row:
        assert is_r_order, "docstring claims R's column-major lower-triangle order but _par2Psi fills the factor row-major"
    else:
        assert says_row, 'docstring names no parameter order'
        assert is_r_order or is_row_order


@pytest.mark.parametrize('k', [1, 2])
def test_par_vector_order_agrees_with_r_for_one_and_two_outcomes(k):
    """Plain: for k = 1, 2 the row- and column-major orders coincide, so PyDLNM's parameters are R's."""
    import meta_analysis as ma
    L = random_chol_factor(k, seed=50 + k)
    par_r = r_par(L)
    assert_close(ma._par2Psi(par_r, k), r_par2psi(par_r, k), rtol=1e-12, what=f'k={k} _par2Psi')
    assert_close(ma._Psi2par(L @ L.T), r_psi2par(L @ L.T), rtol=1e-10, what=f'k={k} _Psi2par')


@pytest.mark.parametrize('k', [1, 2, 3, 4, 5])
def test_par_psi_round_trips_inside_pydlnm(k):
    """Plain: _Psi2par(_par2Psi(par)) == par for a factor with positive diagonal, and _par2Psi(_Psi2par(Psi)) == Psi."""
    import meta_analysis as ma
    L = random_chol_factor(k, seed=60 + k)
    par = ma._Psi2par(L @ L.T)
    np.testing.assert_allclose(ma._par2Psi(par, k), L @ L.T, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(ma._Psi2par(ma._par2Psi(par, k)), par, rtol=1e-10, atol=1e-12)
    assert par.shape == (k * (k + 1) // 2,)
    rng = np.random.default_rng(70 + k)
    A = rng.normal(size=(k, k))
    Psi = A @ A.T + 0.1 * np.eye(k)
    np.testing.assert_allclose(ma._par2Psi(ma._Psi2par(Psi), k), Psi, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize('cfg', [(25, 3, 1, 0.3, 4), (30, 4, 2, 0.3, 9)], ids=['k3', 'k4'])
def test_reml_objective_equals_r_and_gradient_matches_finite_differences(cfg):
    """Plain guard for the ordering fix (all four np.tril_indices sites must change together): (a) the REML
    log-likelihood at the same Psi equals R's remlprof.fn, whatever parameter order each side uses; (b) the analytic
    gradient is the gradient of the objective in PyDLNM's own parameters (central differences)."""
    import meta_analysis as ma
    n, k, p, tau, seed = cfg
    y, S, X = sim(n, k, p, tau, seed)
    np2r(NAMES + 'y', y)
    np2r(NAMES + 'Sv', vech_rows(S))
    np2r(NAMES + 'X', X)
    r(f'{NAMES}L <- {NAMES}lists(as.matrix({NAMES}X), as.matrix({NAMES}y), as.matrix({NAMES}Sv))')
    Xl = [np.kron(np.eye(k), X[i:i + 1]) for i in range(n)]
    yl, Sl = list(y), list(S)
    for rep in range(2):
        L = random_chol_factor(k, seed=80 + 10 * k + rep) * tau
        Psi = L @ L.T
        par_py = ma._Psi2par(Psi)
        np2r(NAMES + 'par', r_psi2par(Psi))
        ref = float(r(f'{NAMES}prof({NAMES}par, {NAMES}L, "remlprof.fn")')[0])
        assert_close(np.array([-ma._reml_fn(par_py, k, Xl, yl, Sl)]), np.array([ref]), rtol=1e-10,
                     what=f'k={k} remlprof.fn at the same Psi')
        g = ma._reml_gr(par_py, k, Xl, yl, Sl)
        h = 1e-6
        g_fd = np.array([(ma._reml_fn(par_py + h * e, k, Xl, yl, Sl) - ma._reml_fn(par_py - h * e, k, Xl, yl, Sl)) / (2 * h)
                         for e in np.eye(len(par_py))])
        assert np.abs(g - g_fd).max() <= 1e-5 * np.abs(g_fd).max(), (g, g_fd)
