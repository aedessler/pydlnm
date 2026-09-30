"""Gap mixmeta_control_names_rejected: MVMeta(control=) versus the control list of R's mixmeta() (Europe 2022 call).

The published Europe-2022 script (europe_summer_2022_heat-main/code.R) fits its second stage with

    mixmeta(COEF ~ TEMP_AVG + TEMP_IQR, VCOV_LIST, data = ..., method = "reml",
            control = list(showiter = TRUE, igls.inititer = 10))

and then blup(fit, vcov = TRUE).  mixmeta.control() is NOT mvmeta.control():

    mixmeta.control:  optim showiter maxiter initPsi Psifix Scor addSlist inputna inputvar loglik.iter igls.inititer
                      hessian vc.adj reltol checkPD set.negeigen
    mvmeta.control:   optim showiter maxiter initPsi Psifix Psicor Scor inputna inputvar igls.iter hessian vc.adj
                      reltol set.negeigen

mixmeta-only: addSlist, loglik.iter, igls.inititer, checkPD.   mvmeta-only: Psicor, igls.iter.
PyDLNM's MVMeta implements mvmeta.control only (meta_analysis._CONTROL_NAMES) and raises TypeError "unused argument in
control" for every mixmeta-only key, so the control list of the published script cannot be passed.  The existing test
tests/test_mvmeta_options_N1.py::test_unknown_control_key_raises_like_r asserts that 'checkPD' / 'addSlist' are refused
because "R refuses them too"; that is true of mvmeta() and false of mixmeta(), the function the Europe analysis (and the
README's Europe validation) uses.

What this module checks (every reference is computed by R at run time, on the same simulated Europe-like second stage:
14 regions, k = 4 spline coefficients, meta-predictors TEMP_AVG and TEMP_IQR, per-region 4 x 4 covariances):

  Plain tests (pass today, guard the neighbourhood)
    * R premise: the two control vocabularies above, igls.inititer <= 0 accepted by mixmeta (mvmeta refuses
      igls.iter < 1), mixmeta's checkPD = TRUE refuses an indefinite S_i that the default accepts.
    * "Numbers are unaffected once the mixmeta-only key is avoided": MVMeta(control = {'igls.iter': 10}) equals
      mixmeta(..., control = list(igls.inititer = 10)) for coefficients, vcov, Psi, logLik, BLUPs and BLUP vcovs
      (tight R stopping rule: 1e-6, R's default stopping rule: 1e-4, the measured optimiser-limited precision).
    * showiter = TRUE (the published value) does not change the fit; mixmeta's default Scor = NULL equals Scor = 0.
    * a mildly indefinite S_i is accepted by default in R and in PyDLNM, with the same fit.
  Known-defect tests (fail today, all one root cause: mixmeta-only control names raise TypeError)
    For each mixmeta-only option R accepts the call; PyDLNM must either honour it (fit equal to mixmeta's) or refuse it
    with a NotImplementedError that names the option.  TypeError "unused argument" (what mvmeta.control would say) and
    ValueError "igls.iter must be positive" (mvmeta's rule, not mixmeta's) are both wrong for a mixmeta call.
    * igls.inititer = 10 with the published showiter = TRUE and R's default stopping rule; tight; other values;
      igls.inititer <= 0 (mixmeta: 0 IGLS iterations, start from diag(0.001); NOT an error)
    * loglik.iter in hybrid / newton / igls / rigls, checkPD = TRUE / FALSE, addSlist (S given through control)
    * the mixmeta-only key accompanied by every estimation method (fixed / mm / vc are deterministic: 1e-8)
    * checkPD = TRUE on an indefinite S_i: R stops, PyDLNM must not silently fit

Tolerances: deterministic quantities (fixed / mm / vc) 1e-8; fitted REML / ML optimum against R run with
control = list(maxiter = 20000, reltol = 1e-14) 1e-6 (README: the optimum agrees to <= 1e-6); against R's default
control 1e-4.  The observed differences are 1e-9 (tight) and 1e-5 (default).
"""
import contextlib
import os
import warnings

import numpy as np
import pytest

from rhelpers import assert_close, known_defect, max_rel_diff, np2r, r, r2np

G = 'gmc_'                      # prefix of every object this module creates in R's global environment

R_TIGHT = 'maxiter = 20000, reltol = 1e-14'
PY_TIGHT = {'maxiter': 20000, 'reltol': 1e-14}
PUBLISHED_R = 'list(showiter = TRUE, igls.inititer = 10)'           # europe_summer_2022_heat-main/code.R, line 217
PUBLISHED_PY = {'showiter': True, 'igls.inititer': 10}

TOL_DET, TOL_TIGHT, TOL_DEFAULT = 1e-8, 1e-6, 1e-4

NOTE = 'MVMeta accepts mvmeta.control names only; mixmeta.control has igls.inititer / loglik.iter / checkPD / addSlist'
DEFECT = known_defect('GAP', 'mixmeta_control_names_rejected', note=NOTE)


# --------------------------------------------------------------------------------------------------------------
# R environment (same precautions as the other meta-analysis modules)
# --------------------------------------------------------------------------------------------------------------
def require_r(*pkgs):
    for p in pkgs:
        if not bool(r(f'isTRUE(suppressWarnings(requireNamespace("{p}", quietly=TRUE)))')[0]):
            pytest.skip(f'R package {p} not installed')
        r(f'suppressMessages(library({p}))')


def running_r_home():
    """Home of the R that is actually running (PyDLNM imports rewrite os.environ['R_HOME'])."""
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
        r('invisible(list(chol(diag(2)), solve(diag(2)), eigen(diag(2)), svd(diag(2)), qr(diag(2))))')   # load LAPACK now


@pytest.fixture(autouse=True)
def _restore_r_home():
    with correct_r_home():
        yield


@pytest.fixture(scope='module', autouse=True)
def _single_threaded_blas():
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        yield
        return
    with threadpool_limits(limits=1, user_api='blas'):
        yield


def r_error(code):
    """None if the R code runs, otherwise the first line of R's error message."""
    try:
        r(code)
    except Exception as e:                           # rpy2 RRuntimeError
        return str(e).strip().splitlines()[0]
    return None


# --------------------------------------------------------------------------------------------------------------
# Europe-like second stage: identical inputs for R and Python
# --------------------------------------------------------------------------------------------------------------
def europe_stage2(n=14, k=4, seed=2022, indefinite=False):
    """Reduced coefficients y (n, k), their covariances S (n, k, k) and the meta-predictors TEMP_AVG, TEMP_IQR.

    The design is what the formula ~ TEMP_AVG + TEMP_IQR gives: intercept, average temperature, temperature IQR."""
    rng = np.random.default_rng(seed)
    avg = rng.normal(15.0, 4.0, n)
    iqr = rng.uniform(6.0, 12.0, n)
    X = np.column_stack([np.ones(n), avg, iqr])
    beta = np.vstack([rng.normal(0.0, 0.5, k), rng.normal(0.0, 0.02, k), rng.normal(0.0, 0.03, k)])     # (p, k)
    A = rng.normal(size=(k, k))
    Psi = 0.05 ** 2 * (A @ A.T / k + 0.3 * np.eye(k))
    S = np.zeros((n, k, k))
    for i in range(n):
        d = 0.08 * np.exp(rng.normal(scale=0.4, size=k))
        corr = 0.6 ** np.abs(np.subtract.outer(np.arange(k), np.arange(k)))        # spline coefficients: neighbours correlate
        S[i] = np.outer(d, d) * corr
    y = np.array([X[i] @ beta + rng.multivariate_normal(np.zeros(k), Psi + S[i]) for i in range(n)])
    if indefinite:                                       # one S_i with a tiny negative eigenvalue (S_i + Psi is still PD)
        w, V = np.linalg.eigh(S[3])
        w[0] = -2e-5
        S[3] = V @ np.diag(w) @ V.T
    return dict(y=y, S=S, avg=avg, iqr=iqr, X=X, n=n, k=k, p=3)


def push(d):
    """Put the inputs into R's global environment exactly as the published script holds them."""
    n = d['n']
    np2r(G + 'COEF', d['y'])
    np2r(G + 'S3', d['S'])
    np2r(G + 'TEMP_AVG', d['avg'])
    np2r(G + 'TEMP_IQR', d['iqr'])
    r(f'{G}VCOV <- lapply(seq_len({n}), function(i) {G}S3[i, , ])')
    r(f'{G}df <- data.frame(vREG = seq_len({n}))')


def r_mixmeta(d, control, method='reml', with_blup=True):
    """R mixmeta() called like the Europe script; `control` is R source text of the control list."""
    push(d)
    n, k, p = d['n'], d['k'], d['p']
    r(f'{G}fit <- suppressWarnings(mixmeta::mixmeta({G}COEF ~ {G}TEMP_AVG + {G}TEMP_IQR, {G}VCOV, data = {G}df, '
      f'control = {control}, method = "{method}"))')
    ref = dict(coef=r2np(r(f'as.numeric({G}fit$coefficients)')).ravel(), vcov=r2np(r(f'unname({G}fit$vcov)')),
               psi=None if method == 'fixed' else r2np(r(f'unname({G}fit$Psi)')),
               loglik=float(r2np(r(f'as.numeric(logLik({G}fit))')).ravel()[0]) if method in ('ml', 'reml') else None)
    if with_blup:
        r(f'{G}bl <- mixmeta::blup({G}fit, vcov = TRUE)')
        ref['blup'] = np.array([r2np(r(f'as.numeric({G}bl[[{i + 1}]]$blup)')).ravel() for i in range(n)])
        ref['bvcov'] = np.array([r2np(r(f'unname({G}bl[[{i + 1}]]$vcov)')) for i in range(n)])
    return ref


def py_mixmeta_like(d, control, method='reml'):
    """PyDLNM MVMeta called like the Europe script.  Returns the fitted model (warnings silenced)."""
    from meta_analysis import MVMeta
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return MVMeta(method=method, control=control).fit(d['y'], d['S'], d['X'])


def fit_or_documented_refusal(d, control, key, method='reml'):
    """The fitted model if PyDLNM honours `control`; None if it refuses with a NotImplementedError that names `key`
    (an unsupported mixmeta option must be refused loudly).  TypeError "unused argument" / ValueError propagate: R does
    not raise for this call, so neither may PyDLNM."""
    try:
        return py_mixmeta_like(d, control, method)
    except NotImplementedError as e:
        assert key in str(e), f'NotImplementedError does not name the option {key!r}: {e}'
        return None


def as_mixmeta(m, d, method):
    """A PyDLNM fit and its BLUPs in mixmeta's layout and logLik convention: coefficient vector and vcov predictor-major
    (entry of predictor j, outcome l at l + j*k; PyDLNM / mvmeta outcome-major, l*p + j); REML logLik plus mixmeta's
    constant sum(log(diag(chol(sum_i X_i'X_i)))) = k * sum(log(diag(chol(X'X))))."""
    from meta_analysis import blup
    k, p = d['k'], d['p']
    to_py = [l + j * k for l in range(k) for j in range(p)]            # PyDLNM position -> mixmeta position
    from_py = np.argsort(to_py)
    out = dict(coef=m.coefficients.T.ravel()[from_py], vcov=m.vcov[np.ix_(from_py, from_py)],
               psi=m.psi if method != 'fixed' else None)
    if method in ('ml', 'reml'):
        shift = k * float(np.sum(np.log(np.diag(np.linalg.cholesky(d['X'].T @ d['X']))))) if method == 'reml' else 0.0
        out['loglik'] = m.loglik + shift
    b = blup(m, vcov=True)
    out['blup'] = np.array([x['blup'] for x in b])
    out['bvcov'] = np.array([x['vcov'] for x in b])
    return out


def assert_matches_mixmeta(m, ref, d, method, rtol, what):
    """Coefficients, vcov, Psi, logLik, BLUPs and BLUP vcovs of a PyDLNM fit against mixmeta's."""
    out = as_mixmeta(m, d, method)
    for key in ('coef', 'vcov', 'psi', 'blup', 'bvcov'):
        if ref.get(key) is not None:
            assert_close(out[key], ref[key], rtol=rtol, what=f'{what}: {key}')
    if ref.get('loglik') is not None:
        assert abs(out['loglik'] - ref['loglik']) <= rtol * max(1.0, abs(ref['loglik'])), \
            f'{what}: logLik Python {out["loglik"]!r} vs mixmeta {ref["loglik"]!r}'


def dists(a, b):
    """max relative difference per quantity between two fits held in mixmeta's layout."""
    return {key: max_rel_diff(a[key], b[key]) for key in ('coef', 'vcov', 'psi', 'blup', 'bvcov')
            if a.get(key) is not None and b.get(key) is not None}


def assert_within_rs_stopping_error(m, d, ref_default, ref_tight, what):
    """Against R's DEFAULT stopping rule the agreement is limited by R itself (optim stops at reltol = 1.5e-8 of the
    objective, leaving R's estimates up to ~1e-4 short of the optimum on this flat objective).  Pinned: (a) PyDLNM is at
    the optimum (R with the tight rule, 1e-6), and (b) its distance to R's default fit is no larger than R's own
    distance to that optimum (plus 1e-6) or 1e-4, whichever is larger."""
    out = as_mixmeta(m, d, 'reml')
    own = dists(ref_default, ref_tight)
    got = dists(out, ref_default)
    far = dists(out, ref_tight)
    for key in own:
        assert far[key] <= TOL_TIGHT, f'{what}: {key} is {far[key]:.2e} from the optimum'
        assert got[key] <= max(TOL_DEFAULT, own[key] + 1e-6), \
            f'{what}: {key} differs {got[key]:.2e} from R default, R default itself is {own[key]:.2e} off the optimum'


# ==============================================================================================================
# R premise: the two control vocabularies
# ==============================================================================================================
MIXMETA_ONLY = ['igls.inititer', 'loglik.iter', 'checkPD', 'addSlist']
MVMETA_ONLY = ['igls.iter', 'Psicor']


def test_r_mixmeta_and_mvmeta_control_vocabularies_differ():
    """The premise of this module, checked in R: each of the four mixmeta-only names is accepted by mixmeta.control and
    refused by mvmeta.control ("unused argument"); igls.iter / Psicor are the other way round; the published control list
    is a valid mixmeta control and an invalid mvmeta control."""
    mixmeta_names = set(str(x) for x in r('names(formals(mixmeta:::mixmeta.control))'))
    mvmeta_names = set(str(x) for x in r('names(formals(mvmeta:::mvmeta.control))'))
    assert set(MIXMETA_ONLY) <= mixmeta_names and not set(MIXMETA_ONLY) & mvmeta_names
    assert set(MVMETA_ONLY) <= mvmeta_names and not set(MVMETA_ONLY) & mixmeta_names
    for key, val in zip(MIXMETA_ONLY, ['10', '"newton"', 'TRUE', 'NULL']):
        assert r_error(f'mixmeta:::mixmeta.control({key} = {val})') is None, f'mixmeta.control must accept {key}'
        assert r_error(f'mvmeta:::mvmeta.control({key} = {val})') is not None, f'mvmeta.control must refuse {key}'
    for key in MVMETA_ONLY:
        assert r_error(f'mvmeta:::mvmeta.control({key} = 1)') is None
        assert r_error(f'mixmeta:::mixmeta.control({key} = 1)') is not None, f'mixmeta.control must refuse {key}'
    assert r_error(f'do.call(mixmeta:::mixmeta.control, {PUBLISHED_R})') is None
    assert r_error(f'do.call(mvmeta:::mvmeta.control, {PUBLISHED_R})') is not None
    # igls.inititer <= 0 is legal in mixmeta (it means "no IGLS iterations"); mvmeta stops for igls.iter < 1
    assert r_error('mixmeta:::mixmeta.control(igls.inititer = 0)') is None
    assert float(r('mixmeta:::mixmeta.control(igls.inititer = -3)$igls.inititer')[0]) == 0.0
    assert r_error('mvmeta:::mvmeta.control(igls.iter = 0)') is not None


def test_r_mixmeta_accepts_published_control_and_blup_like_europe():
    """The published call itself runs in R (formula with meta-predictors, list of covariances, blup(vcov = TRUE))."""
    d = europe_stage2()
    ref = r_mixmeta(d, PUBLISHED_R)
    assert ref['coef'].shape == (d['k'] * d['p'],) and ref['blup'].shape == (d['n'], d['k'])
    assert ref['bvcov'].shape == (d['n'], d['k'], d['k'])
    assert np.all(np.isfinite(ref['blup'])) and np.all(np.linalg.eigvalsh(ref['psi']) > 0)


# ==============================================================================================================
# Plain: numbers are unaffected once the mixmeta-only key is avoided
# ==============================================================================================================
@pytest.mark.parametrize('seed', [2022, 7])        # 2022: Psi interior; 7: Psi on the boundary (rank-deficient)
def test_europe_stage2_without_mixmeta_keys_matches_mixmeta_tight(seed):
    """MVMeta(control={'igls.iter': 10, tight}) == mixmeta(control=list(showiter=FALSE, igls.inititer=10, tight)):
    coefficients, vcov, Psi, REML logLik (mixmeta constant), BLUPs and BLUP vcovs to 1e-6."""
    d = europe_stage2(seed=seed)
    ref = r_mixmeta(d, f'list(showiter = FALSE, igls.inititer = 10, {R_TIGHT})')
    m = py_mixmeta_like(d, dict(PY_TIGHT, **{'igls.iter': 10}))
    assert_matches_mixmeta(m, ref, d, 'reml', TOL_TIGHT, f'seed {seed}')


def test_europe_stage2_without_mixmeta_keys_matches_mixmeta_default_stopping():
    """Same against R's default stopping rule (the published control list minus the trace).  R's own stopping error is up
    to ~3e-4 here (Psi is weakly identified), so the bound is R's distance to the optimum, see
    assert_within_rs_stopping_error."""
    d = europe_stage2()
    ref_default = r_mixmeta(d, 'list(showiter = FALSE, igls.inititer = 10)')
    ref_tight = r_mixmeta(d, f'list(showiter = FALSE, igls.inititer = 10, {R_TIGHT})')
    m = py_mixmeta_like(d, {'igls.iter': 10})
    assert_within_rs_stopping_error(m, d, ref_default, ref_tight, 'default stopping rule')


@pytest.mark.parametrize('method', ['fixed', 'mm', 'vc'])
def test_deterministic_methods_match_mixmeta_to_rounding(method):
    """fixed / mm / vc have no optimiser: mixmeta and PyDLNM agree to 1e-8 (measured ~1e-15) on the Europe-like
    second stage, with the mixmeta-only key left out on the PyDLNM side."""
    d = europe_stage2()
    ref = r_mixmeta(d, 'list(igls.inititer = 10)', method=method)
    m = py_mixmeta_like(d, {'igls.iter': 10}, method=method)
    assert_matches_mixmeta(m, ref, d, method, TOL_DET, method)


def test_showiter_true_is_the_published_value_and_does_not_change_the_fit(capsys):
    """The published control has showiter = TRUE.  PyDLNM accepts it, prints a trace (as R does) and returns the same
    fit as with showiter = FALSE; R's trace option does not change the optimum either."""
    d = europe_stage2()
    quiet = py_mixmeta_like(d, dict(PY_TIGHT, showiter=False, **{'igls.iter': 10}))
    capsys.readouterr()
    loud = py_mixmeta_like(d, dict(PY_TIGHT, showiter=True, **{'igls.iter': 10}))
    assert capsys.readouterr().out.strip(), 'showiter = TRUE should print the iteration trace (R does)'
    for name in ('coefficients', 'vcov', 'psi'):
        assert_close(getattr(loud, name), getattr(quiet, name), rtol=1e-12, what=f'showiter changes {name}')
    assert abs(loud.loglik - quiet.loglik) <= 1e-12 * max(1.0, abs(quiet.loglik))
    ref_quiet = r_mixmeta(d, f'list(showiter = FALSE, igls.inititer = 10, {R_TIGHT})', with_blup=False)
    ref_loud = r_mixmeta(d, f'list(showiter = TRUE, igls.inititer = 10, {R_TIGHT})', with_blup=False)
    assert_close(ref_loud['psi'], ref_quiet['psi'], rtol=1e-12, what='R: showiter changes Psi')


def test_default_scor_of_mixmeta_equals_scor_zero_for_variance_only_S():
    """mixmeta.control has Scor = NULL (mvmeta.control: 0).  With a variance-only S (n x k), mixmeta's inputcov(NULL)
    builds zero within-study correlations, so PyDLNM's default (Scor = 0) and an explicit Scor = None agree with it."""
    d = europe_stage2()
    V = np.stack([np.diag(s) for s in d['S']])
    np2r(G + 'V', V)
    np2r(G + 'COEF', d['y'])
    np2r(G + 'TEMP_AVG', d['avg'])
    np2r(G + 'TEMP_IQR', d['iqr'])
    r(f'{G}df <- data.frame(vREG = seq_len({d["n"]}))')
    r(f'{G}fv <- suppressWarnings(mixmeta::mixmeta({G}COEF ~ {G}TEMP_AVG + {G}TEMP_IQR, {G}V, data = {G}df, '
      f'control = list({R_TIGHT}), method = "reml"))')
    ref = dict(coef=r2np(r(f'as.numeric({G}fv$coefficients)')).ravel(), vcov=r2np(r(f'unname({G}fv$vcov)')),
               psi=r2np(r(f'unname({G}fv$Psi)')), loglik=float(r(f'as.numeric(logLik({G}fv))')[0]))
    from meta_analysis import MVMeta
    for control in (dict(PY_TIGHT), dict(PY_TIGHT, Scor=None), dict(PY_TIGHT, Scor=0)):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = MVMeta(method='reml', control=control).fit(d['y'], V, d['X'])
        assert_matches_mixmeta(m, ref, d, 'reml', TOL_TIGHT, f'variance-only S, control {sorted(control)}')


def test_mildly_indefinite_S_is_accepted_by_default_like_r():
    """mixmeta does not check S_i by default (checkPD = NULL): an S_i with a tiny negative eigenvalue is fitted, and
    PyDLNM gives the same fit.  (checkPD = TRUE is the known-defect test below.)"""
    d = europe_stage2(indefinite=True)
    assert np.linalg.eigvalsh(d['S'][3]).min() < 0
    ref = r_mixmeta(d, f'list({R_TIGHT})')
    m = py_mixmeta_like(d, dict(PY_TIGHT))
    assert_matches_mixmeta(m, ref, d, 'reml', TOL_TIGHT, 'indefinite S_3')


# ==============================================================================================================
# Known defect: mixmeta-only control names are rejected (TypeError "unused argument")
# ==============================================================================================================
@DEFECT
def test_published_europe_control_list_is_usable():
    """The control list of the published script, verbatim on both sides (showiter = TRUE, igls.inititer = 10, R's default
    stopping rule): PyDLNM must honour it (the fit is at mixmeta's optimum, BLUPs included, and as close to R's default
    fit as R's own stopping error allows) or refuse it with a NotImplementedError naming 'igls.inititer'.
    Today: TypeError "unused argument in control: 'igls.inititer'"."""
    d = europe_stage2()
    ref_default = r_mixmeta(d, PUBLISHED_R)
    ref_tight = r_mixmeta(d, f'list(showiter = TRUE, igls.inititer = 10, {R_TIGHT})')
    m = fit_or_documented_refusal(d, dict(PUBLISHED_PY), 'igls.inititer')
    if m is not None:
        assert_within_rs_stopping_error(m, d, ref_default, ref_tight, 'published control list')


@DEFECT
@pytest.mark.parametrize('n_init', [3, 10, 25])
def test_igls_inititer_matches_mixmeta_tight(n_init):
    """igls.inititer = n is accepted and the fit equals mixmeta's (tight stopping rule, 1e-6): coefficients, vcov, Psi,
    logLik, BLUPs, BLUP vcovs.  The IGLS start length only changes where the optimiser starts, never the optimum."""
    d = europe_stage2()
    ref = r_mixmeta(d, f'list(showiter = FALSE, igls.inititer = {n_init}, {R_TIGHT})')
    m = fit_or_documented_refusal(d, dict(PY_TIGHT, showiter=False, **{'igls.inititer': n_init}), 'igls.inititer')
    if m is not None:
        assert_matches_mixmeta(m, ref, d, 'reml', TOL_TIGHT, f'igls.inititer={n_init}')


@DEFECT
@pytest.mark.parametrize('n_init', [0, -3])
def test_igls_inititer_nonpositive_is_accepted_like_r(n_init):
    """mixmeta.control: `if (igls.inititer <= 0L) igls.inititer <- 0` -- no IGLS iterations, the optimiser starts from
    diag(0.001).  It is not an error (mvmeta's igls.iter < 1 is), so a blind alias onto igls.iter ("'igls.iter' must be
    positive") would be wrong too.  The fit equals mixmeta's at the optimum."""
    d = europe_stage2()
    ref = r_mixmeta(d, f'list(igls.inititer = {n_init}, {R_TIGHT})')
    m = fit_or_documented_refusal(d, dict(PY_TIGHT, **{'igls.inititer': n_init}), 'igls.inititer')
    if m is not None:
        assert_matches_mixmeta(m, ref, d, 'reml', TOL_TIGHT, f'igls.inititer={n_init}')


@DEFECT
@pytest.mark.parametrize('loglik_iter', ['hybrid', 'newton', 'igls', 'rigls'])
def test_loglik_iter_is_honoured_or_refused_by_name(loglik_iter):
    """loglik.iter picks mixmeta's optimisation route (RIGLS start then BFGS, BFGS only, pure (R)IGLS).  All routes end
    at the same REML optimum (R with the tight stopping rule: 1e-6), so PyDLNM may honour the option by ignoring the route,
    or refuse it with a NotImplementedError naming 'loglik.iter'.  Today: TypeError "unused argument"."""
    d = europe_stage2()
    with warnings.catch_warnings():                                   # R: "'rigls' used instead than 'igls'"
        warnings.simplefilter('ignore')
        ref = r_mixmeta(d, f'list(loglik.iter = "{loglik_iter}", {R_TIGHT})')
    m = fit_or_documented_refusal(d, dict(PY_TIGHT, **{'loglik.iter': loglik_iter}), 'loglik.iter')
    if m is not None:
        assert_matches_mixmeta(m, ref, d, 'reml', TOL_TIGHT, f'loglik.iter={loglik_iter}')


@DEFECT
@pytest.mark.parametrize('check_pd', [True, False])
def test_checkPD_flag_on_positive_definite_S(check_pd):
    """checkPD = TRUE / FALSE on positive-definite S_i changes nothing in R.  PyDLNM: same fit (1e-6) or a
    NotImplementedError naming 'checkPD'.  (tests/test_mvmeta_options_N1.py pins a refusal of checkPD on the premise 'R
    refuses it too': true for mvmeta(), false for mixmeta().)"""
    d = europe_stage2()
    flag = 'TRUE' if check_pd else 'FALSE'
    ref = r_mixmeta(d, f'list(checkPD = {flag}, {R_TIGHT})')
    m = fit_or_documented_refusal(d, dict(PY_TIGHT, checkPD=check_pd), 'checkPD')
    if m is not None:
        assert_matches_mixmeta(m, ref, d, 'reml', TOL_TIGHT, f'checkPD={flag}')


@DEFECT
def test_checkPD_true_refuses_an_indefinite_S_like_r():
    """With checkPD = TRUE mixmeta stops on an S_i that has a negative eigenvalue ("Problems with positive-definiteness in
    'S'") although the default accepts it (test above).  PyDLNM must not silently fit: a ValueError about positive
    definiteness or a NotImplementedError naming 'checkPD'.  Today: TypeError "unused argument in control: 'checkPD'"."""
    d = europe_stage2(indefinite=True)
    push(d)
    err = r_error(f'mixmeta::mixmeta({G}COEF ~ {G}TEMP_AVG + {G}TEMP_IQR, {G}VCOV, data = {G}df, '
                  f'control = list(checkPD = TRUE), method = "reml")')
    assert err is not None and 'positive-definiteness' in err, f'R must stop with a positive-definiteness error: {err}'
    try:
        py_mixmeta_like(d, dict(PY_TIGHT, checkPD=True))
    except NotImplementedError as e:
        assert 'checkPD' in str(e)
    except ValueError as e:                                           # numpy LinAlgError is a ValueError
        assert 'positive' in str(e).lower(), f'ValueError for the wrong reason: {e}'
    else:
        pytest.fail('checkPD = TRUE on an indefinite S_i was silently fitted (R: error)')


@DEFECT
def test_addSlist_supplies_S_through_control_like_r():
    """mixmeta(y ~ x) without S and control = list(addSlist = list of k x k matrices) equals the fit with S = that list.
    PyDLNM: same fit, or a NotImplementedError naming 'addSlist'.  (With S also given R stops: "'addSlist' only allowed
    without 'S'"; PyDLNM must not silently ignore either of them.)"""
    d = europe_stage2()
    push(d)
    n = d['n']
    ref = r_mixmeta(d, f'list({R_TIGHT})', with_blup=False)
    r(f'{G}fa <- suppressWarnings(mixmeta::mixmeta({G}COEF ~ {G}TEMP_AVG + {G}TEMP_IQR, data = {G}df, '
      f'control = list(addSlist = {G}VCOV, {R_TIGHT}), method = "reml"))')
    via_control = dict(coef=r2np(r(f'as.numeric({G}fa$coefficients)')).ravel(), vcov=r2np(r(f'unname({G}fa$vcov)')),
                       psi=r2np(r(f'unname({G}fa$Psi)')), loglik=float(r(f'as.numeric(logLik({G}fa))')[0]))
    assert_close(via_control['psi'], ref['psi'], rtol=1e-8, what='R: addSlist fit vs S fit')
    both = r_error(f'mixmeta::mixmeta({G}COEF ~ {G}TEMP_AVG + {G}TEMP_IQR, {G}VCOV, data = {G}df, '
                   f'control = list(addSlist = {G}VCOV), method = "reml")')
    assert both is not None and 'addSlist' in both
    from meta_analysis import MVMeta
    addS = [s for s in d['S']]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = MVMeta(method='reml', control=dict(PY_TIGHT, addSlist=addS)).fit(d['y'], None, d['X'])
    except NotImplementedError as e:
        assert 'addSlist' in str(e)
        return
    assert_matches_mixmeta(m, via_control, d, 'reml', TOL_TIGHT, 'addSlist')
    with pytest.raises((ValueError, NotImplementedError)):            # R: error when both S and addSlist are given
        MVMeta(method='reml', control=dict(PY_TIGHT, addSlist=addS)).fit(d['y'], d['S'], d['X'])


@DEFECT
@pytest.mark.parametrize('method', ['fixed', 'mm', 'vc', 'ml', 'reml'])
def test_mixmeta_only_key_is_accepted_with_every_method(method):
    """control = list(igls.inititer = 10) next to every estimation method: R's mixmeta() accepts it (for fixed / mm / vc it
    is unused).  PyDLNM: same fit (fixed / mm / vc are deterministic: 1e-8; ml / reml with the tight rule: 1e-6) or a
    NotImplementedError naming 'igls.inititer'."""
    d = europe_stage2()
    tight = method in ('ml', 'reml')
    ref = r_mixmeta(d, f'list(igls.inititer = 10{", " + R_TIGHT if tight else ""})', method=method)
    control = dict(PY_TIGHT, **{'igls.inititer': 10}) if tight else {'igls.inititer': 10}
    m = fit_or_documented_refusal(d, control, 'igls.inititer', method=method)
    if m is not None:
        assert_matches_mixmeta(m, ref, d, method, TOL_TIGHT if tight else TOL_DET, f'{method} + igls.inititer')
