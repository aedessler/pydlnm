"""coef/vcov/at validation of crosspred() and crossreduce() (theme H2) versus R dlnm 2.4.10.

R checks its inputs before it predicts: crosspred() stops with 'coef/vcov not consistent with basis matrix' when
length(coef) != ncol(basis), dim(vcov)[1] != length(coef) or any(is.na(coef)) / any(is.na(vcov)) (crossreduce() has the
same check with the message 'coef/vcov do not consistent ...'); mkat() accepts a scalar `at` and a MATRIX `at` whose rows
are exposure histories (ncol = diff(lag)+1, bylag must be 1).  PyDLNM has no such checks.  Every reference below is
computed by R at test run time (rpy2); nothing is copied from Python output.

Theme H2
  crosspred-core-13, crosspred-grid-10
                        NaN in coef / vcov is accepted and propagated (also on the reduced-coefficient route and through
                        model=); a coef / vcov LARGER than the basis is silently trimmed to its first ncol(basis)
                        entries (a coef with a leading intercept gives wrong numbers without any error); a scalar `at`
                        (R accepts at=25) crashes with an rpy2 RRuntimeError ("'dims' cannot be of length 0").
                        NOT asserted (not divergences, verifier): 1-D vcov and non-numeric ci_level make R stop as
                        well (plain tests pin that Python stops too); Inf is accepted by R (is.na(Inf) is FALSE) and so
                        is deliberately not tested; the reduced-coefficient route (len(coef) == variable-basis df) is a
                        documented Python extension and must keep working.
  crosspred-core-14, crosspred-grid-12
                        matrix-valued `at` (exposure histories): PyDLNM treats the 2-D array as a vector of predvar
                        without error, so each row is evaluated as a CONSTANT history equal to its lag-0 value.  The
                        R-faithful behaviour (allfit / allse / matfit / cumfit ... equal to R) is asserted;
                        test_matrix_at_never_silently_wrong accepts the minimum fix (NotImplementedError / ValueError for
                        ndim > 1) and the ncol / bylag checks accept either exception, so a partial fix XPASSes only
                        the tests it satisfies.
  crossreduce-12        a model exposing only .params (no covariance) gets a fabricated vcov = 1e-6 * I
                        (model_info type 'dummy') instead of stopping like R's getvcov().
  crossreduce-13        NaN coef / vcov accepted (R stops); a size mismatch is rejected only by an unfriendly numpy matmul
                        ValueError (R: 'coef/vcov do not consistent with basis matrix'); a (p, 1) column coef returns a
                        (k, 1) reduced coefficient where R returns a plain vector.
                        NOT asserted as a defect: non-symmetric vcov (R accepts it too; a plain test pins that a fix adds
                        no symmetry check).

Model-route tests use a duck-typed fit with numpy .params / .cov_params() and design names in .model.exog_names
('cbv1.l1', ..., 'x1', ...; cross-basis block first), so that a positional and a by-name (R-like) selection of the
cross-basis block both satisfy them.

Tests decorated with @known_defect assert the R-faithful behaviour and fail today (strict xfail); the plain tests guard
neighbouring behaviour that is already faithful (valid inputs, reduced route, model route with trailing covariate
terms, shorter / undersized inputs, length-one `at`) and must keep passing while the fixes land.
"""
import contextlib
import io
import os
import re
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from rhelpers import assert_close, chicago, known_defect, max_rel_diff, np2r, r, rget


def _warm_up_r_lapack():
    """Load R's lazily loaded LAPACK module while os.environ['R_HOME'] still points at the R that is embedded.
    Workaround for audit finding Q2 (basis.py and the *_glm modules overwrite R_HOME; the first La_*() call afterwards
    dlopens the wrong R's modules/lapack and segfaults).  Remove once Q2 is fixed."""
    os.environ['R_HOME'] = os.path.dirname(str(r('.Library')[0]))      # the R_HOME R itself started with
    r('invisible(chol2inv(chol(diag(2) + 1))); invisible(solve(diag(2)))')


_warm_up_r_lapack()

LAG = 6                                        # L: a cross-basis has L+1 lag columns; short keeps calls in ms
AT = np.arange(-5.0, 30.0 + 1e-9, 5.0)         # 8 prediction points inside the Chicago temperature range
CEN = 15.0
R_MSG = r'coef/vcov.*consistent with basis matrix'     # crosspred: "not consistent", crossreduce: "do not consistent"
STOPS = (ValueError, NotImplementedError)


# --------------------------------------------------------------------------------------------------------------
# helpers: identical inputs for R and Python
# --------------------------------------------------------------------------------------------------------------
def _rand_coef_vcov(n, seed, scale=0.05):
    """Deterministic coefficient vector and SPD covariance matrix."""
    rng = np.random.default_rng(seed)
    coef = rng.normal(0.0, scale, n)
    a = rng.normal(0.0, 1.0, (n, n))
    vcov = a @ a.T * scale ** 2 / n * 0.05 + np.eye(n) * 1e-5
    return coef, (vcov + vcov.T) / 2


@pytest.fixture(scope='module')
def case():
    """Cross-basis (bs deg 2 x ns, lag 0..6, explicit knots AND boundary knots), its R onebasis (reduced route),
    a one-dimensional ns basis, and fixed coef / vcov for each, in both R (globals h2_*) and PyDLNM."""
    from basis import CrossBasis, OneBasis
    temp = chicago()['temp']
    np2r('h2_temp', temp)
    r(f'h2_kv <- quantile(h2_temp, c(.25, .75)); h2_bk <- range(h2_temp); h2_kn <- logknots({LAG}, nk=2)')
    kv, bk, kn = rget('h2_kv'), rget('h2_bk'), rget('h2_kn')
    r(f'h2_cb <- crossbasis(h2_temp, lag={LAG}, argvar=list(fun="bs", degree=2, knots=h2_kv, Boundary.knots=h2_bk), '
      f'arglag=list(fun="ns", knots=h2_kn, Boundary.knots=c(0, {LAG})))')
    r('h2_ob <- do.call("onebasis", c(list(x=h2_temp), attr(h2_cb, "argvar")))')       # the reduced route's reference
    r('h2_one <- onebasis(h2_temp, fun="ns", knots=h2_kv, Boundary.knots=h2_bk)')
    cb = _quiet(CrossBasis, temp, lag=LAG, argvar={'fun': 'bs', 'degree': 2, 'knots': kv, 'Boundary_knots': bk},
                arglag={'fun': 'ns', 'knots': kn, 'Boundary_knots': np.array([0.0, float(LAG)])})
    one = _quiet(OneBasis, temp, fun='ns', knots=kv, Boundary_knots=bk)
    assert_close(np.asarray(cb.basis), rget('unclass(h2_cb)'), rtol=1e-12, what='cross-basis (set-up sanity)')
    assert_close(np.asarray(one.basis), rget('unclass(h2_one)'), rtol=1e-12, what='onebasis (set-up sanity)')
    p, n_one, n_var = cb.shape[1], one.shape[1], int(rget('ncol(h2_ob)')[0])
    coef, vcov = _rand_coef_vcov(p, 1)
    ocoef, ovcov = _rand_coef_vcov(n_one, 2)
    rcoef, rvcov = _rand_coef_vcov(n_var, 3)
    for name, arr in (('h2_cf', coef), ('h2_vc', vcov), ('h2_ocf', ocoef), ('h2_ovc', ovcov), ('h2_rcf', rcoef),
                      ('h2_rvc', rvcov), ('h2_at', AT)):
        np2r(name, arr)
    return SimpleNamespace(cb=cb, one=one, p=p, n_one=n_one, n_var=n_var, coef=coef, vcov=vcov, ocoef=ocoef,
                           ovcov=ovcov, rcoef=rcoef, rvcov=rvcov)


def _quiet(fn, *a, **kw):
    """Call fn with PyDLNM's diagnostic print()s and warnings silenced (the reduced route prints: other finding)."""
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return fn(*a, **kw)


def _r_error(code):
    """Message of the R error raised by `code`, or None when R evaluates it without error."""
    try:
        r(code)
    except Exception as exc:                                # rpy2 RRuntimeError
        return ' '.join(str(exc).split())
    return None


def _summary(res):
    """Short description of what Python returned (for failure messages)."""
    if hasattr(res, 'allfit'):
        return (f'allfit[:3]={np.round(np.asarray(res.allfit, dtype=float)[:3], 5).tolist()}, '
                f'coefficients.shape={np.shape(res.coefficients)}, vcov.shape={np.shape(res.vcov)}, '
                f'NaN allfit/allse={int(np.isnan(res.allfit).sum())}/{int(np.isnan(res.allse).sum())}')
    return (f'coef.shape={np.shape(res.coef)}, vcov.shape={np.shape(res.vcov)}, NaN coef/vcov='
            f'{int(np.isnan(res.coef).sum())}/{int(np.isnan(res.vcov).sum())}, model_info={res.model_info}')


def assert_stops_like_r(py_call, r_code, what, pattern=R_MSG):
    """R evaluates `r_code` and stops with `pattern`; `py_call` must raise ValueError carrying the same message."""
    r_msg = _r_error(r_code)
    assert r_msg is not None and re.search(pattern, r_msg), f'{what}: R reference did not stop as expected: {r_msg!r}'
    try:
        res = _quiet(py_call)
    except ValueError as exc:
        assert re.search(pattern, str(exc)), (f'{what}: Python raised ValueError({str(exc)[:100]!r}); '
                                              f'R stops with {_short(r_msg)!r}')
        return
    except Exception as exc:                                # noqa: BLE001
        pytest.fail(f'{what}: Python raised {type(exc).__name__} ({str(exc)[:100]}), expected ValueError like R: '
                    f'{_short(r_msg)!r}')
    pytest.fail(f'{what}: Python returned without error ({_summary(res)}); R stops with {_short(r_msg)!r}')


def _disagreements(pp, fields, r_obj='h2_pR', rtol=1e-10):
    """One message per field of `pp` that differs from the R object `r_obj` (empty list: all agree)."""
    bad = []
    for f in fields:
        py = getattr(pp, f, None)
        if py is None:
            bad.append(f'{f}: missing in Python')
            continue
        try:
            assert_close(np.atleast_1d(np.asarray(py, dtype=float)), rget(f'{r_obj}${f}'), rtol=rtol, what=f)
        except AssertionError as exc:
            bad.append(str(exc))
    return bad


def _compare(pp, fields, r_obj='h2_pR', rtol=1e-10):
    """All `fields` of `pp` against the R object `r_obj`; one assertion listing every disagreement."""
    bad = _disagreements(pp, fields, r_obj, rtol)
    assert not bad, ' | '.join(bad)


def _short(r_message):
    """The message proper of an rpy2 error string ('Error in f(...) : message')."""
    return r_message.rsplit(': ', 1)[-1].strip()


def _fields(kind, link=True):
    fit = {'all': ['allfit', 'allse'], 'mat': ['matfit', 'matse'], 'cum': ['cumfit', 'cumse']}[kind]
    if link:
        return fit + [f'{kind}RR{s}' for s in ('fit', 'low', 'high')]
    return fit + [f'{kind}low', f'{kind}high']


ALL_FIELDS = _fields('mat') + _fields('all')


def _push_candidate(coef, vcov):
    """Push candidate (possibly malformed) coef / vcov into R as h2_bc / h2_bv (2-D arrays become matrices)."""
    np2r('h2_bc', np.asarray(coef, dtype=float))
    np2r('h2_bv', np.asarray(vcov, dtype=float))


# malformed coef / vcov derived from the valid pair of a basis with p columns: name -> (coef, vcov)
def _invalid_values(coef, vcov):
    p = len(coef)
    c_nan = coef.copy()
    c_nan[min(3, p - 1)] = np.nan
    v_nan = vcov.copy()
    v_nan[0, min(2, p - 1)] = v_nan[min(2, p - 1), 0] = np.nan                   # off-diagonal, symmetric
    v_nan_diag = vcov.copy()
    v_nan_diag[p - 1, p - 1] = np.nan
    return {'nan_coef': (c_nan, vcov), 'nan_vcov_offdiag': (coef, v_nan), 'nan_vcov_diag': (coef, v_nan_diag)}


def _oversized(coef, vcov):
    p = len(coef)
    v_trail = np.eye(p + 1)
    v_trail[:p, :p] = vcov
    v_lead = np.eye(p + 1)
    v_lead[1:, 1:] = vcov
    return {'vcov_larger': (coef, v_trail),                                      # coef p, vcov (p+1)^2
            'coef_larger_trailing': (np.r_[coef, 1.0], v_trail),                 # extra LAST element (was dropped)
            'coef_larger_leading_intercept': (np.r_[0.7, coef], v_lead),         # extra FIRST element (wrong numbers)
            'coef_larger_vcov_same': (np.r_[coef, 1.0], vcov)}                   # coef p+1, vcov p x p


def _undersized(coef, vcov):
    p = len(coef)
    return {'vcov_smaller': (coef, vcov[:-1, :-1]),                              # coef p, vcov (p-1)^2
            'coef_shorter': (coef[:-1], vcov[:-1, :-1]),                         # coef p-1, vcov (p-1)^2
            'coef_shorter_vcov_full': (coef[:-1], vcov)}                         # coef p-1, vcov p x p


KINDS = ['cb', 'one']                          # basis kinds with a full-coefficient crosspred path
R_BASIS = {'cb': 'h2_cb', 'one': 'h2_one'}


def _pair(case, kind):
    return (case.coef, case.vcov) if kind == 'cb' else (case.ocoef, case.ovcov)


def _basis(case, kind):
    return case.cb if kind == 'cb' else case.one


def _crosspred(case, kind, coef, vcov, **kw):
    from prediction import crosspred
    kw = dict(dict(model_link='log', at=AT, cen=CEN), **kw)
    return crosspred(_basis(case, kind), coef=coef, vcov=vcov, **kw)


def _r_crosspred_code(kind, coef='h2_bc', vcov='h2_bv', at='h2_at', extra='', basis=None):
    return (f'crosspred({basis or R_BASIS[kind]}, coef={coef}, vcov={vcov}, model.link="log", at={at}, '
            f'cen={CEN!r}{extra})')


# ==============================================================================================================
# crosspred-core-13, crosspred-grid-10: NaN accepted, oversized coef / vcov silently trimmed
# ==============================================================================================================
CASE_NAMES = list({**_invalid_values(np.zeros(16), np.eye(16)), **_oversized(np.zeros(16), np.eye(16))})


@pytest.mark.parametrize('kind', KINDS)
@pytest.mark.parametrize('name', CASE_NAMES)
def test_crosspred_malformed_coef_vcov_stop_like_r(case, kind, name):
    """R: stop('coef/vcov not consistent with basis matrix') on NaN coef / vcov and on any coef / vcov whose size is
    not ncol(basis).  PyDLNM returns a prediction: NaN-filled for NaN input, silently trimmed to the first ncol(basis)
    entries for oversized input (a coef with a leading intercept is shifted by one position: wrong numbers)."""
    coef, vcov = _pair(case, kind)
    bad = {**_invalid_values(coef, vcov), **_oversized(coef, vcov)}[name]
    _push_candidate(*bad)
    assert_stops_like_r(lambda: _crosspred(case, kind, *bad), 'suppressWarnings(' + _r_crosspred_code(kind) + ')',
                        f'{kind}/{name}')


@pytest.mark.parametrize('name', list(_invalid_values(np.zeros(4), np.eye(4))))
def test_crosspred_reduced_route_nan_stops_like_r(case, name):
    """The reduced route (len(coef) == variable-basis df of the CrossBasis) is R's crosspred(onebasis, coef, vcov):
    NaN in coef / vcov stops there.  PyDLNM propagates the NaN into allfit / allse."""
    bad = _invalid_values(case.rcoef, case.rvcov)[name]
    _push_candidate(*bad)
    assert_stops_like_r(lambda: _crosspred(case, 'cb', *bad),
                        'suppressWarnings(' + _r_crosspred_code('cb', basis='h2_ob') + ')', f'reduced/{name}')


def _duck_model(case, coef, vcov):
    """A fitted-model look-alike: numpy .params, .cov_params() and the design column names (.model.exog_names, as a
    statsmodels results object has them): the cross-basis block first ('cbv1.l1', ...), then covariate terms.
    Named, so that both a positional and a by-name (R-like) selection of the cross-basis block work."""
    names = [f'cb{c}' for c in case.cb.colnames] + [f'x{i}' for i in range(1, len(coef) - case.p + 1)]

    class _DuckFit:
        params = np.asarray(coef, dtype=float)
        model = SimpleNamespace(exog_names=names)

        def cov_params(self):
            return np.asarray(vcov, dtype=float)
    return _DuckFit()


def _push_r_model(coef, vcov, cb_name='h2_cb', cls='h2mod'):
    """R S3 model object `h2_mod` whose coefficients are named like a glm's (cross-basis block first, then x1..)."""
    np2r('h2_mcoef', np.asarray(coef, dtype=float))
    np2r('h2_mvcov', np.asarray(vcov, dtype=float))
    r(f'''
    h2_mnm <- c(paste0("{cb_name}", colnames({cb_name})), paste0("x", seq_len(length(h2_mcoef) - ncol({cb_name}))))
    names(h2_mcoef) <- h2_mnm
    dimnames(h2_mvcov) <- list(h2_mnm, h2_mnm)
    h2_mod <- structure(list(coefficients=h2_mcoef, vcov=h2_mvcov), class="{cls}")
    coef.{cls} <- function(object, ...) object$coefficients
    vcov.{cls} <- function(object, ...) object$vcov
    ''')


def _with_extra_terms(case, n_extra=3, nan_at=None):
    """Cross-basis block followed by n_extra covariate terms (as in a glm with day-of-week / trend terms)."""
    p = case.p
    coef, vcov = _rand_coef_vcov(p + n_extra, 11)
    coef[:p], vcov[:p, :p] = case.coef, case.vcov
    vcov[:p, p:] = vcov[p:, :p] = 0.0
    if nan_at is not None:
        coef = coef.copy()
        coef[nan_at] = np.nan
    return coef, vcov


@known_defect('H2', 'crosspred-core-13', 'crosspred-grid-10',
              note='model= route: an aliased (NaN) cross-basis coefficient is accepted (R stops)')
def test_crosspred_model_route_nan_stops_like_r(case):
    """R: an NA coefficient inside the cross-basis block of a fitted model (aliased term) -> crosspred stops."""
    coef, vcov = _with_extra_terms(case, nan_at=2)
    _push_r_model(coef, vcov)
    assert_stops_like_r(lambda: _crosspred_model(case, coef, vcov),
                        'suppressWarnings(crosspred(h2_cb, model=h2_mod, model.link="log", at=h2_at, cen=15))',
                        'model route, NaN coef')


def _crosspred_model(case, coef, vcov, **kw):
    from prediction import crosspred
    return crosspred(case.cb, model=_duck_model(case, coef, vcov), model_link='log', at=AT, cen=CEN, **kw)


# --------------------------------------------------------------------------------------------------------------
# plain: valid input unchanged, sizes that Python already rejects, inputs that stop in R and in Python
# --------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('form', ['ndarray', 'lists'])
def test_valid_full_inputs_match_r(case, form):
    """BASELINE: valid full-length coef / vcov (ndarray or nested Python lists) give R's lag-specific, overall and
    cumulative fields, and the stored coefficients / vcov are the inputs."""
    coef, vcov = case.coef, case.vcov
    r(f'h2_pR <- {_r_crosspred_code("cb", coef="h2_cf", vcov="h2_vc", extra=", cumul=TRUE")}')
    py_coef, py_vcov = (coef, vcov) if form == 'ndarray' else (coef.tolist(), vcov.tolist())
    pp = _quiet(_crosspred, case, 'cb', py_coef, py_vcov, cumul=True)
    _compare(pp, ALL_FIELDS + _fields('cum'))
    assert_close(np.ravel(pp.coefficients), coef, rtol=1e-15, what='coefficients')
    assert_close(np.asarray(pp.vcov), vcov, rtol=1e-15, what='vcov')


def test_valid_onebasis_inputs_match_r(case):
    """BASELINE: valid full-length coef / vcov of a one-dimensional basis."""
    np2r('h2_bc', case.ocoef)
    r(f'h2_pR <- {_r_crosspred_code("one", coef="h2_ocf", vcov="h2_ovc")}')
    pp = _quiet(_crosspred, case, 'one', case.ocoef, case.ovcov)
    _compare(pp, _fields('all'))


def test_reduced_route_overall_fields_match_r_onebasis(case):
    """BASELINE (documented Python extension, must keep working): len(coef) == variable-basis df of the CrossBasis is
    R's crosspred(onebasis, coef, vcov) for the overall fields."""
    r(f'h2_pR <- {_r_crosspred_code("cb", coef="h2_rcf", vcov="h2_rvc", basis="h2_ob")}')
    pp = _quiet(_crosspred, case, 'cb', case.rcoef, case.rvcov)
    _compare(pp, _fields('all'))


@pytest.mark.parametrize('with_callable', [True, False], ids=['cov_params()', 'cov_params-attribute'])
def test_model_route_takes_the_crossbasis_block_of_a_larger_model_like_r(case, with_callable):
    """BASELINE: a fitted model has more parameters than the cross-basis (covariates after the cross-basis block):
    crosspred(model=) must still pick the cross-basis block, as R does by name.  Guards the fix of the coef= / vcov=
    validation against being applied to the whole model parameter vector."""
    coef, vcov = _with_extra_terms(case)
    _push_r_model(coef, vcov)
    r('h2_pR <- crosspred(h2_cb, model=h2_mod, model.link="log", at=h2_at, cen=15)')
    if with_callable:
        pp = _quiet(_crosspred_model, case, coef, vcov)
    else:
        from prediction import crosspred

        names = _duck_model(case, coef, vcov).model.exog_names

        class _AttrFit:
            params, cov_params, model = coef, vcov, SimpleNamespace(exog_names=names)
        pp = _quiet(crosspred, case.cb, model=_AttrFit(), model_link='log', at=AT, cen=CEN)
    _compare(pp, ['coefficients', 'vcov'] + ALL_FIELDS)


@pytest.mark.parametrize('kind', KINDS)
@pytest.mark.parametrize('name', ['vcov_smaller', 'coef_shorter', 'coef_shorter_vcov_full'])
def test_crosspred_undersized_inputs_raise_value_error(case, kind, name):
    """BASELINE: coef / vcov smaller than the basis (and not the reduced-coefficient length): R stops, PyDLNM raises
    ValueError as well (message differs today: other finding, not asserted)."""
    coef, vcov = _pair(case, kind)
    bad = _undersized(coef, vcov)[name]
    _push_candidate(*bad)
    assert _r_error('suppressWarnings(' + _r_crosspred_code(kind) + ')') is not None, 'R reference did not stop'
    with pytest.raises(ValueError):
        _quiet(_crosspred, case, kind, *bad)


@pytest.mark.parametrize('name', ['reduced_vcov_smaller', 'reduced_vcov_larger', 'reduced_vcov_full_size'])
def test_reduced_route_size_mismatch_raises_value_error(case, name):
    """BASELINE: reduced coef with a vcov of another size: R's crosspred(onebasis) stops; PyDLNM raises ValueError."""
    k = case.n_var
    vcov = {'reduced_vcov_smaller': case.rvcov[:-1, :-1],
            'reduced_vcov_larger': np.pad(case.rvcov, ((0, 1), (0, 1)), constant_values=0.0) + np.eye(k + 1) * 1e-3,
            'reduced_vcov_full_size': case.vcov}[name]
    _push_candidate(case.rcoef, vcov)
    assert _r_error('suppressWarnings(' + _r_crosspred_code('cb', basis='h2_ob') + ')') is not None
    with pytest.raises(ValueError):
        _quiet(_crosspred, case, 'cb', case.rcoef, vcov)


def test_crosspred_one_dimensional_vcov_stops_like_r(case):
    """BASELINE (verifier: not a divergence): R stops on a 1-D vcov (with a cryptic message); PyDLNM raises too."""
    np2r('h2_bc', case.coef)
    np2r('h2_bv', case.vcov[:, 0])
    assert _r_error('suppressWarnings(' + _r_crosspred_code('cb') + ')') is not None
    with pytest.raises((ValueError, IndexError, TypeError)):
        _quiet(_crosspred, case, 'cb', case.coef, case.vcov[:, 0])


CI_LEVELS = [('"a"', 'a'), ('NULL', None), ('1', 1.0), ('0', 0.0), ('1.5', 1.5), ('-0.1', -0.1), ('TRUE', True)]


@pytest.mark.parametrize('rci,ci', CI_LEVELS, ids=[repr(c) for _, c in CI_LEVELS])
def test_crosspred_invalid_ci_level_stops_like_r(case, rci, ci):
    """BASELINE (verifier: not a divergence): R stops unless 0 < ci.level < 1 is numeric; PyDLNM raises as well."""
    code = _r_crosspred_code('cb', coef='h2_cf', vcov='h2_vc', extra=f', ci.level={rci}')
    assert _r_error(f'suppressWarnings({code})') is not None, 'R reference did not stop'
    with pytest.raises((ValueError, TypeError)):
        _quiet(_crosspred, case, 'cb', case.coef, case.vcov, ci_level=ci)


def test_crosspred_needs_model_or_coef_and_vcov_like_r(case):
    """BASELINE: R: "At least 'model' or 'coef'-'vcov' must be provided"."""
    from prediction import crosspred
    assert _r_error('crosspred(h2_cb, coef=h2_cf, model.link="log", at=h2_at, cen=15)') is not None
    with pytest.raises(ValueError):
        _quiet(crosspred, case.cb, coef=case.coef, model_link='log', at=AT, cen=CEN)


# ==============================================================================================================
# crosspred-core-13: scalar `at`
# ==============================================================================================================
SCALARS = [25.0, 25, np.float64(25.0), np.array(25.0)]
SCALAR_IDS = ['float', 'int', 'np.float64', '0d-array']


@pytest.mark.parametrize('kind', KINDS)
@pytest.mark.parametrize('at', SCALARS, ids=SCALAR_IDS)
def test_scalar_at_matches_r(case, kind, at):
    """R: at=25 is a length-one vector (mkat: sort(unique(at))).  PyDLNM crashes with an obscure RRuntimeError."""
    coef, vcov = _pair(case, kind)
    np2r('h2_bc', coef)
    np2r('h2_bv', vcov)
    r(f'h2_pR <- {_r_crosspred_code(kind, at="25")}')
    pp = _quiet(_crosspred, case, kind, coef, vcov, at=at)
    assert np.asarray(pp.predvar, dtype=float).tolist() == [25.0]
    _compare(pp, _fields('mat') + _fields('all') if kind == 'cb' else _fields('all'))


@pytest.mark.parametrize('kind', KINDS)
@pytest.mark.parametrize('at', [[25.0], (25.0,), np.array([25.0])], ids=['list', 'tuple', 'array1'])
def test_one_element_at_matches_r(case, kind, at):
    """BASELINE: a length-one `at` already works and equals R."""
    coef, vcov = _pair(case, kind)
    np2r('h2_bc', coef)
    np2r('h2_bv', vcov)
    r(f'h2_pR <- {_r_crosspred_code(kind, at="25")}')
    pp = _quiet(_crosspred, case, kind, coef, vcov, at=at)
    _compare(pp, _fields('mat') + _fields('all') if kind == 'cb' else _fields('all'))


# ==============================================================================================================
# crosspred-core-14, crosspred-grid-12: matrix-valued `at` (one exposure history per row)
# ==============================================================================================================
def _histories(kind):
    """Matrices of exposure histories, nrow x (LAG+1) columns (lag 0 first), inside the Chicago temperature range."""
    t = np.linspace(0.0, 1.0, LAG + 1)
    if kind == 'varying':                                       # falling, constant, rising, hump
        return np.vstack([20.0 - 10.0 * t, np.full(LAG + 1, 18.0), 5.0 + 20.0 * t, 12.0 + 14.0 * np.sin(np.pi * t)])
    if kind == 'mixed':                                         # constant, constant, trending (crosspred-grid-12)
        return np.vstack([np.full(LAG + 1, 10.0), np.full(LAG + 1, 15.0), np.linspace(5.0, 20.0, LAG + 1)])
    if kind == 'one_history':
        return np.clip(np.random.default_rng(5).normal(15.0, 6.0, (1, LAG + 1)), -20.0, 30.0)
    if kind == 'many':                                          # 12 random histories
        return np.clip(np.random.default_rng(6).normal(14.0, 7.0, (12, LAG + 1)), -20.0, 30.0)
    raise KeyError(kind)


HISTORIES = ['varying', 'mixed', 'one_history', 'many']
MATRIX_DEFECT = ('H2', 'crosspred-core-14', 'crosspred-grid-12')


def _r_matrix_crosspred(hist, lag=None, cumul=False, bylag=None):
    np2r('h2_atm', hist)                                        # a 2-D array becomes an R matrix, rows = histories
    extra = f', cumul={"TRUE" if cumul else "FALSE"}'
    extra += '' if lag is None else f', lag=c({lag[0]}, {lag[1]})'
    extra += '' if bylag is None else f', bylag={bylag!r}'
    return _r_crosspred_code('cb', coef='h2_cf', vcov='h2_vc', at='h2_atm', extra=extra)


@pytest.mark.parametrize('cumul', [False, True], ids=['nocumul', 'cumul'])
@pytest.mark.parametrize('hist', HISTORIES)
def test_matrix_at_matches_r(case, hist, cumul):
    """R: rows of `at` are exposure histories (column j = exposure at lag j); predvar is the row index, and matfit[i, j]
    is the effect of history i at lag j.  PyDLNM evaluates each row as a constant history equal to its lag-0 value."""
    H = _histories(hist)
    r(f'h2_pR <- {_r_matrix_crosspred(H, cumul=cumul)}')
    pp = _quiet(_crosspred, case, 'cb', case.coef, case.vcov, at=H, cumul=cumul)
    bad = []
    if np.asarray(pp.predvar).shape != (H.shape[0],):
        bad.append(f'predvar shape {np.shape(pp.predvar)} (R: one entry per row, {H.shape[0]})')
    bad += _disagreements(pp, ALL_FIELDS + (_fields('cum') if cumul else []))
    assert not bad, ' | '.join(bad)


@pytest.mark.parametrize('lag', [(0, 3), (2, 5)], ids=['lag0to3', 'lag2to5'])
def test_matrix_at_with_lag_subperiod_matches_r(case, lag):
    """R: ncol(at) must equal diff(lag)+1 of the requested lag sub-period; columns are the lags lag[0]..lag[1]."""
    H = _histories('varying')[:, :lag[1] - lag[0] + 1]
    r(f'h2_pR <- {_r_matrix_crosspred(H, lag=lag)}')
    pp = _quiet(_crosspred, case, 'cb', case.coef, case.vcov, at=H, lag=list(lag))
    _compare(pp, ALL_FIELDS)


def test_matrix_at_never_silently_wrong(case):
    """Minimum acceptable fix: either implement the R semantics or refuse (NotImplementedError / ValueError for a
    2-D `at`).  Today a prediction that matches no R output is returned without any error or warning."""
    H = _histories('varying')
    r(f'h2_pR <- {_r_matrix_crosspred(H)}')
    try:
        pp = _quiet(_crosspred, case, 'cb', case.coef, case.vcov, at=H)
    except STOPS:
        return
    ref = rget('h2_pR$allfit')
    got = np.asarray(pp.allfit, dtype=float)
    d = max_rel_diff(got, ref) if got.shape == ref.shape else float('inf')
    assert d <= 1e-10, (f'matrix `at` accepted but allfit {np.round(got, 5).tolist()} (shape {got.shape}) != '
                        f'R {np.round(ref, 5).tolist()}; predvar shape {np.shape(pp.predvar)}')


@pytest.mark.parametrize('ncol,lag', [(LAG, None), (LAG + 2, None), (1, None), (LAG + 1, (0, 3))],
                         ids=['too_few', 'too_many', 'single_column', 'sublag_mismatch'])
def test_matrix_at_wrong_ncol_stops_like_r(case, ncol, lag):
    """R: stop("matrix in 'at' must have ncol=diff(lag)+1").  PyDLNM accepts a matrix of any width."""
    base = _histories('varying')
    H = np.hstack([base, base])[:, :ncol]
    msg = _r_error(_r_matrix_crosspred(H, lag=lag))
    assert msg is not None and 'ncol=diff(lag)+1' in msg, f'R reference: {msg!r}'
    try:
        pp = _quiet(_crosspred, case, 'cb', case.coef, case.vcov, at=H, **({} if lag is None else {'lag': list(lag)}))
    except STOPS:
        return
    pytest.fail(f'matrix `at` with {ncol} columns (lag={lag}) accepted (predvar shape {np.shape(pp.predvar)}); '
                f'R stops: {_short(msg)!r}')


@pytest.mark.parametrize('bylag', [0.5, 2.0])
def test_matrix_at_with_bylag_not_one_stops_like_r(case, bylag):
    """R: stop("'bylag!=1 not allowed with 'at' in matrix form").  PyDLNM silently predicts on a bylag grid."""
    H = _histories('varying')
    msg = _r_error(_r_matrix_crosspred(H, bylag=bylag))
    assert msg is not None and 'bylag!=1' in msg, f'R reference: {msg!r}'
    try:
        pp = _quiet(_crosspred, case, 'cb', case.coef, case.vcov, at=H, bylag=bylag)
    except STOPS:
        return
    pytest.fail(f'matrix `at` with bylag={bylag} accepted (matfit shape {np.shape(pp.matfit)}); R stops: {_short(msg)!r}')


def test_vector_at_unchanged_next_to_matrix_support(case):
    """BASELINE: a plain 1-D `at` (the validated route) still equals R, with the lag-specific and overall fields."""
    r(f'h2_pR <- {_r_crosspred_code("cb", coef="h2_cf", vcov="h2_vc", extra=", cumul=TRUE")}')
    pp = _quiet(_crosspred, case, 'cb', case.coef, case.vcov, cumul=True)
    assert np.asarray(pp.predvar).shape == (len(AT),)
    _compare(pp, ALL_FIELDS + _fields('cum'))


# ==============================================================================================================
# crossreduce-13: NaN accepted, unfriendly size errors, column coefficients
# ==============================================================================================================
def _crossreduce(case, coef, vcov, **kw):
    from crossreduce import crossreduce
    return crossreduce(case.cb, coef=coef, vcov=vcov, cen=CEN, **kw)


def _r_crossreduce_code(coef='h2_bc', vcov='h2_bv'):
    return f'crossreduce(h2_cb, coef={coef}, vcov={vcov}, model.link="log", cen={CEN!r})'


@pytest.mark.parametrize('name', list(_invalid_values(np.zeros(16), np.eye(16))))
def test_crossreduce_nan_coef_vcov_stop_like_r(case, name):
    """R: any(is.na(coef)) || any(is.na(vcov)) -> stop('coef/vcov do not consistent with basis matrix').  PyDLNM
    returns NaN reduced coefficients / vcov (an aliased cross-basis coefficient from a GLM ends up here)."""
    bad = _invalid_values(case.coef, case.vcov)[name]
    _push_candidate(*bad)
    assert_stops_like_r(lambda: _crossreduce(case, *bad), 'suppressWarnings(' + _r_crossreduce_code() + ')', name)


def test_crossreduce_model_route_nan_stops_like_r(case):
    from crossreduce import crossreduce
    coef, vcov = _with_extra_terms(case, nan_at=2)
    _push_r_model(coef, vcov)
    assert_stops_like_r(lambda: crossreduce(case.cb, model=_duck_model(case, coef, vcov), cen=CEN),
                        'suppressWarnings(crossreduce(h2_cb, model=h2_mod, model.link="log", cen=15))',
                        'model route, NaN coef')


SIZE_CASES = {**_oversized(np.zeros(16), np.eye(16)), **_undersized(np.zeros(16), np.eye(16))}


@pytest.mark.parametrize('name', list(SIZE_CASES))
def test_crossreduce_size_mismatch_reports_r_message(case, name):
    """R: length(coef) != ncol(basis) or dim(vcov) != length(coef) -> 'coef/vcov do not consistent with basis
    matrix'.  PyDLNM raises a ValueError from numpy ('matmul: Input operand 1 has a mismatch ...')."""
    bad = {**_oversized(case.coef, case.vcov), **_undersized(case.coef, case.vcov)}[name]
    _push_candidate(*bad)
    assert_stops_like_r(lambda: _crossreduce(case, *bad), 'suppressWarnings(' + _r_crossreduce_code() + ')', name)


@pytest.mark.parametrize('name', list(SIZE_CASES))
def test_crossreduce_size_mismatch_raises_value_error(case, name):
    """BASELINE: R stops; PyDLNM raises ValueError (today from numpy matmul) and never returns a result."""
    bad = {**_oversized(case.coef, case.vcov), **_undersized(case.coef, case.vcov)}[name]
    _push_candidate(*bad)
    assert _r_error('suppressWarnings(' + _r_crossreduce_code() + ')') is not None, 'R reference did not stop'
    with pytest.raises(ValueError):
        _quiet(_crossreduce, case, *bad)


def test_crossreduce_column_coef_returns_plain_vector_like_r(case):
    """R: `coef` given as a p x 1 matrix is accepted and the reduced coefficients are a plain numeric vector.
    PyDLNM returns them with shape (k, 1)."""
    np2r('h2_bc', case.coef.reshape(-1, 1))
    np2r('h2_bv', case.vcov)
    r(f'h2_red <- {_r_crossreduce_code()}')
    assert bool(r('is.null(dim(h2_red$coefficients))')[0]), 'R reference is expected to be a plain vector'
    red = _quiet(_crossreduce, case, case.coef.reshape(-1, 1), case.vcov)
    assert np.shape(red.coef) == rget('unname(h2_red$coefficients)').shape, f'reduced coef shape {np.shape(red.coef)}'
    assert_close(red.coef, rget('unname(h2_red$coefficients)'), rtol=1e-12, what='reduced coef')
    assert_close(red.vcov, rget('unname(h2_red$vcov)'), rtol=1e-12, what='reduced vcov')


@pytest.mark.parametrize('form', ['ndarray', 'lists'])
def test_crossreduce_valid_inputs_match_r(case, form):
    """BASELINE: valid coef / vcov (ndarray or nested lists) reduce to R's overall coefficients and vcov."""
    np2r('h2_bc', case.coef)
    np2r('h2_bv', case.vcov)
    r(f'h2_red <- {_r_crossreduce_code()}')
    coef, vcov = (case.coef, case.vcov) if form == 'ndarray' else (case.coef.tolist(), case.vcov.tolist())
    red = _quiet(_crossreduce, case, coef, vcov)
    assert np.shape(red.coef) == (case.n_var,)
    assert_close(red.coef, rget('unname(h2_red$coefficients)'), rtol=1e-12, what='reduced coef')
    assert_close(red.vcov, rget('unname(h2_red$vcov)'), rtol=1e-12, what='reduced vcov')


def test_crossreduce_non_symmetric_vcov_accepted_like_r(case):
    """BASELINE (verifier: part of crossreduce-13 refuted): R checks only length / dim / NA, so a non-symmetric vcov is
    accepted by R and by PyDLNM alike, with identical reduced vcov.  The fix must not add a symmetry check."""
    vcov = case.vcov + np.triu(np.full_like(case.vcov, 1e-4), 1)
    np2r('h2_bc', case.coef)
    np2r('h2_bv', vcov)
    r(f'h2_red <- {_r_crossreduce_code()}')
    red = _quiet(_crossreduce, case, case.coef, vcov)
    assert_close(red.vcov, rget('unname(h2_red$vcov)'), rtol=1e-12, what='reduced vcov (non-symmetric input)')


def test_crossreduce_model_route_takes_the_crossbasis_block_like_r(case):
    """BASELINE: crossreduce(model=) of a model with covariate terms after the cross-basis block picks the cross-basis
    block (R: by name) -- must survive the coef/vcov validation fix."""
    from crossreduce import crossreduce
    coef, vcov = _with_extra_terms(case)
    _push_r_model(coef, vcov)
    r('h2_red <- crossreduce(h2_cb, model=h2_mod, model.link="log", cen=15)')
    red = _quiet(crossreduce, case.cb, model=_duck_model(case, coef, vcov), cen=CEN)
    assert_close(red.coef, rget('unname(h2_red$coefficients)'), rtol=1e-12, what='reduced coef')
    assert_close(red.vcov, rget('unname(h2_red$vcov)'), rtol=1e-12, what='reduced vcov')


# ==============================================================================================================
# crossreduce-12: model without a covariance matrix
# ==============================================================================================================
def _push_r_model_without_vcov(coef):
    np2r('h2_ncoef', np.asarray(coef, dtype=float))
    r('''
    names(h2_ncoef) <- paste0("h2_cb", colnames(h2_cb))
    h2_nomod <- structure(list(coefficients=h2_ncoef), class="h2nocov")
    coef.h2nocov <- function(object, ...) object$coefficients
    ''')


def test_crossreduce_model_without_vcov_stops_like_r(case):
    """R: getvcov() stops ('methods for coef() and vcov() must exist ...') for a model that has coefficients but no
    covariance.  PyDLNM returns reduced standard errors built from a fabricated 1e-6 * I vcov, without a warning."""
    from crossreduce import crossreduce

    class _OnlyParams:
        params = case.coef.copy()
    _push_r_model_without_vcov(case.coef)
    msg = _r_error('crossreduce(h2_cb, model=h2_nomod, model.link="log", cen=15)')
    assert msg is not None and re.search(r'coef\(\) and vcov\(\) must exist', msg), f'R reference: {msg!r}'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            red = crossreduce(case.cb, model=_OnlyParams(), cen=CEN)
        except (ValueError, AttributeError):
            return
    pytest.fail(f'crossreduce returned without error for a model with no vcov: model_info={red.model_info}, '
                f'reduced vcov diagonal={np.round(np.diag(red.vcov), 9).tolist()} (an artefact of vcov = 1e-6 * I); '
                f'R stops')


def test_crossreduce_model_with_coef_attribute_only_raises(case):
    """BASELINE: an sklearn-like model (coef_ but no covariance) is already refused, like R."""
    from crossreduce import crossreduce

    class _CoefOnly:
        coef_ = np.zeros(case.p)
    with pytest.raises(ValueError):
        crossreduce(case.cb, model=_CoefOnly(), cen=CEN)


@pytest.mark.parametrize('with_callable', [True, False], ids=['cov_params()', 'cov_params-attribute'])
def test_crossreduce_model_with_params_and_cov_matches_r_coef_vcov_route(case, with_callable):
    """BASELINE: a model that does expose params AND a covariance (method or plain attribute) reduces exactly like the
    coef= / vcov= route, i.e. like R -- must keep working when the params-only placeholder is removed."""
    from crossreduce import crossreduce
    np2r('h2_bc', case.coef)
    np2r('h2_bv', case.vcov)
    r(f'h2_red <- {_r_crossreduce_code()}')
    if with_callable:
        model = _duck_model(case, case.coef, case.vcov)
    else:
        class _AttrModel:
            params, cov_params = case.coef.copy(), case.vcov.copy()
            model = SimpleNamespace(exog_names=[f'cb{c}' for c in case.cb.colnames])
        model = _AttrModel()
    red = _quiet(crossreduce, case.cb, model=model, cen=CEN)
    assert_close(red.coef, rget('unname(h2_red$coefficients)'), rtol=1e-12, what='reduced coef')
    assert_close(red.vcov, rget('unname(h2_red$vcov)'), rtol=1e-12, what='reduced vcov')


def test_crossreduce_needs_model_or_coef_and_vcov_like_r(case):
    """BASELINE: R stops with "At least 'model' or 'coef'-'vcov' must be provided"."""
    from crossreduce import crossreduce
    assert _r_error('crossreduce(h2_cb, coef=h2_cf, model.link="log", cen=15)') is not None
    with pytest.raises(ValueError):
        _quiet(crossreduce, case.cb, coef=case.coef, cen=CEN)
