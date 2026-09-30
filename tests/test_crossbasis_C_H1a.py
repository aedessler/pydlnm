"""CrossBasis lag dimension: default / single-lag / negative-lag handling (theme C) and lag utilities (theme H1a).

Every test computes the reference in R (dlnm 2.4.10, at run time) and compares PyDLNM on identical inputs.

Theme C  -- R crossbasis() replaces arglag by strata(df=1, intercept=TRUE) when arglag is empty or diff(lag)==0,
            and shifts by tsModel::Lag(), which also supports negative lags.
  crossbasis-1, basis-cont-10, basis-discrete-2   default (empty) arglag is ns/logknots in PyDLNM, not strata(df=1)
  crossbasis-2, attr-point-10                     lag=0 / [k,k] crashes (ns on one lag value); lin/strata arglag
                                                  is silently rebuilt as ns and crashes in the time-series builder
  crossbasis-7                                    negative lags give an all-NaN cross-basis
Theme H1a -- utils.mklag / utils.seqlag versus dlnm:::mklag / dlnm:::seqlag
  crossbasis-10                                   mklag does not round (R: round(lag[1:2])); NaN lags are not rejected
  crossbasis-11                                   seqlag overshoots lag[2] when `by` does not divide the range
                                                  (crosspred(bylag=...) then predicts beyond the lag range)

Tests decorated with @known_defect assert the R-faithful behaviour and fail today (strict xfail); the plain tests
guard neighbouring behaviour that is already faithful and must keep passing while the fixes land.
"""
import contextlib
import copy
import io
import itertools

import numpy as np
import pytest

from rhelpers import assert_close, chicago, known_defect, np2r, r, rget


# --------------------------------------------------------------------------------------------------------------
# helpers: identical inputs for R and Python
# --------------------------------------------------------------------------------------------------------------
_pushed = itertools.count()


def _series(n, start=0):
    """A slice of R's chicagoNMMAPS temperature (no missing values)."""
    return chicago()['temp'][start:start + n].copy()


def _rlit(v):
    """Python value -> R literal.  Arrays are pushed into the R workspace (no decimal round trip)."""
    if isinstance(v, (bool, np.bool_)):
        return 'TRUE' if v else 'FALSE'
    if isinstance(v, str):
        return f'"{v}"'
    if isinstance(v, (int, float, np.integer, np.floating)):
        return repr(float(v))
    name = f'.cbt_pushed{next(_pushed)}'
    np2r(name, np.atleast_1d(np.asarray(v, dtype=float)))
    return name


def _rlist(d):
    """dict -> R list(); PyDLNM spells R's `Boundary.knots` as `Boundary_knots`."""
    return 'list(' + ', '.join(f'{"Boundary.knots" if k == "Boundary_knots" else k}={_rlit(v)}'
                               for k, v in d.items()) + ')'


def _rlag(lag):
    return f'c({", ".join(repr(float(v)) for v in lag)})' if isinstance(lag, (list, tuple)) else repr(float(lag))


def _rvec(v):
    """numeric scalar / sequence -> R literal (NaN aware)."""
    vals = np.atleast_1d(np.asarray(v, dtype=float))
    lit = ', '.join('NaN' if np.isnan(t) else repr(float(t)) for t in vals)
    return f'c({lit})' if (np.ndim(v) > 0 or len(vals) != 1) else lit


def rquantile(x, probs):
    np2r('cbt_xq', x)
    return rget(f'quantile(cbt_xq, c({", ".join(repr(float(p)) for p in probs)}))')


def r_crossbasis(x, lag, argvar, arglag=None):
    """R crossbasis(); arglag=None omits the argument (R default), {} passes list()."""
    np2r('cbt_x', x)
    call = f'crossbasis(cbt_x, lag={_rlag(lag)}, argvar={_rlist(argvar)}'
    if arglag is not None:
        call += f', arglag={_rlist(arglag)}'
    r(f'cbt_cb <- {call})')
    return dict(mat=rget('unclass(cbt_cb)'), df=rget('attr(cbt_cb,"df")'), lag=rget('attr(cbt_cb,"lag")'),
                range=rget('attr(cbt_cb,"range")'), colnames=list(r('colnames(cbt_cb)')))


def py_crossbasis(x, lag, argvar, arglag=None):
    from basis import CrossBasis
    kw = {} if arglag is None else {'arglag': copy.deepcopy(arglag)}   # PyDLNM mutates dicts; not the topic here
    return CrossBasis(x, lag=lag, argvar=copy.deepcopy(argvar), **kw)


def assert_crossbasis_matches_r(x, lag, argvar, arglag=None, rtol=1e-10, attrs=True):
    """R crossbasis() versus CrossBasis: values (shape and NaN pattern included), then df and column names."""
    ref = r_crossbasis(x, lag, argvar, arglag)
    cb = py_crossbasis(x, lag, argvar, arglag)
    what = f'crossbasis lag={lag} arglag={arglag}'
    assert_close(np.asarray(cb.basis), ref['mat'], rtol=rtol, what=what)
    if attrs:
        assert tuple(int(d) for d in cb.df) == tuple(int(d) for d in ref['df']), f'{what}: df {cb.df} vs R {ref["df"]}'
        assert list(cb.colnames) == ref['colnames'], f'{what}: column names differ'
    return cb, ref


def r_mklag(lag):
    return rget(f'dlnm:::mklag({_rvec(lag)})')


def r_seqlag(lag, by):
    return rget(f'dlnm:::seqlag({_rvec(lag)}, by={float(by)!r})')


def r_fails(code):
    try:
        r(code)
    except Exception:            # rpy2 RRuntimeError
        return True
    return False


# argvar specifications shared by several tests
def _bs2_knots(x):
    return {'fun': 'bs', 'degree': 2, 'knots': rquantile(x, [.10, .75, .90])}


BS2_DF5 = {'fun': 'bs', 'degree': 2, 'df': 5}
NS_DF3 = {'fun': 'ns', 'df': 3}
NS_DF4 = {'fun': 'ns', 'df': 4}


# ==============================================================================================================
# Theme C: default arglag = strata(df=1, intercept=TRUE)                      crossbasis-1, basis-cont-10, basis-discrete-2
# ==============================================================================================================
@known_defect('C', 'crossbasis-1', note='empty arglag builds ns/logknots (4 lag columns) instead of one strata column')
@pytest.mark.parametrize('lag', [5, 21])
@pytest.mark.parametrize('argvar', [BS2_DF5, NS_DF4], ids=['bs2df5', 'nsdf4'])
def test_default_arglag_is_one_unconstrained_lag_column(lag, argvar):
    x = _series(400)
    cb, ref = assert_crossbasis_matches_r(x, lag, argvar, arglag=None)
    assert cb.df[1] == 1 and int(ref['df'][1]) == 1


@known_defect('C', 'basis-cont-10', note='empty arglag / {} and lag 0 / [0,0]: ns lag basis instead of strata df=1')
@pytest.mark.parametrize('lag', [1, 5, 0, [0, 0]], ids=['lag1', 'lag5', 'lag0', 'lag00'])
@pytest.mark.parametrize('arglag', [None, {}], ids=['omitted', 'emptydict'])
def test_default_arglag_with_ns_knots_at_median(lag, arglag):
    x = _series(800)
    argvar = {'fun': 'ns', 'knots': rquantile(x, [.5])}      # ns with one knot: 2 var columns
    cb, ref = assert_crossbasis_matches_r(x, lag, argvar, arglag=arglag)
    assert ref['mat'].shape == (800, 2) and cb.shape == (800, 2)


@known_defect('C', 'basis-discrete-2', note='default arglag for [0,1] and offset [2,6] lag ranges')
@pytest.mark.parametrize('lag', [[0, 1], [2, 6]], ids=['0to1', '2to6'])
def test_default_arglag_short_and_offset_lag_ranges(lag):
    assert_crossbasis_matches_r(_series(300), lag, NS_DF3, arglag=None)


@known_defect('C', 'basis-discrete-2', 'crossbasis-1', 'attr-point-10',
              note='explicit arglag strata df=1: R adds intercept=TRUE and gets one column of ones')
def test_explicit_strata_df1_lag_basis_is_one_column():
    x = _series(400)
    cb, ref = assert_crossbasis_matches_r(x, 5, NS_DF3, arglag={'fun': 'strata', 'df': 1})
    assert ref['mat'].shape == (400, 3) and cb.shape == (400, 3)


# ==============================================================================================================
# Theme C: single-lag cross-bases (lag=0, [k,k])                              crossbasis-2, attr-point-10, basis-discrete-2
# ==============================================================================================================
@known_defect('C', 'crossbasis-2', 'attr-point-10', 'basis-cont-10', 'basis-discrete-2',
              note='ns(df=4) on a single lag value: all interior knots match left boundary knot')
@pytest.mark.parametrize('lag', [0, [0, 0], [1, 1], [3, 3]], ids=['lag0', 'lag00', 'lag11', 'lag33'])
def test_single_lag_crossbasis_matches_r(lag):
    x = _series(300)
    cb, ref = assert_crossbasis_matches_r(x, lag, _bs2_knots(x), arglag={'fun': 'ns', 'df': 3})
    assert ref['mat'].shape == (300, 5)          # R: 5 var columns x 1 lag column, whatever arglag says
    k = int(lag[1]) if isinstance(lag, list) else 0
    assert int(np.isnan(np.asarray(cb.basis)).any(axis=1).sum()) == k


@known_defect('C', 'attr-point-10', 'crossbasis-2', 'basis-discrete-2',
              note='lag=0 with an explicit strata arglag (R ignores arglag when diff(lag)==0)')
@pytest.mark.parametrize('arglag', [{'fun': 'strata', 'df': 1}, {'fun': 'strata', 'breaks': np.array([1., 3.])},
                                    None], ids=['strata_df1', 'strata_breaks', 'omitted'])
def test_single_lag_ignores_user_arglag(arglag):
    x = _series(300, start=1000)
    cb, ref = assert_crossbasis_matches_r(x, 0, _bs2_knots(x), arglag=arglag)
    assert ref['mat'].shape == (300, 5)


# ==============================================================================================================
# Theme C: lin / strata lag functions must not be rebuilt as ns                                     attr-point-10
# ==============================================================================================================
@known_defect('C', 'attr-point-10', note='arglag lin is re-created as ns in the time-series builder: cannot broadcast')
def test_lin_lag_basis_matches_r():
    x = _series(300, start=1000)
    cb, ref = assert_crossbasis_matches_r(x, 5, _bs2_knots(x), arglag={'fun': 'lin'})
    assert ref['mat'].shape == (300, 10)         # intercept + slope over the lag


@known_defect('C', 'attr-point-10', note='arglag strata with breaks is re-created as ns: cannot broadcast')
def test_strata_breaks_lag_basis_matches_r():
    x = _series(300, start=1000)
    cb, ref = assert_crossbasis_matches_r(x, 6, _bs2_knots(x),
                                          arglag={'fun': 'strata', 'breaks': np.array([1., 3.])})
    assert ref['mat'].shape == (300, 15)


# ==============================================================================================================
# Theme C: negative lags (tsModel::Lag shifts forward, NaN only in the last |lag| rows)                crossbasis-7
# ==============================================================================================================
@known_defect('C', 'crossbasis-7', note='negative-lag columns stay NaN: every row of the cross-basis is NaN')
@pytest.mark.parametrize('lag, n_nan', [(-3, 3), ([-2, 2], 4), ([-3, -1], 3)], ids=['m3', 'm2to2', 'm3tom1'])
def test_negative_lag_crossbasis_matches_r(lag, n_nan):
    x = _series(300)
    cb, ref = assert_crossbasis_matches_r(x, lag, _bs2_knots(x), arglag={'fun': 'ns', 'df': 3})
    assert int(np.isnan(ref['mat']).any(axis=1).sum()) == n_nan
    assert int(np.isnan(np.asarray(cb.basis)).any(axis=1).sum()) == n_nan


@known_defect('C', 'crossbasis-7', note='negative lags with an integer lag basis')
def test_negative_lag_integer_lag_basis_matches_r():
    x = _series(200, start=500)
    assert_crossbasis_matches_r(x, [-2, 1], NS_DF3, arglag={'fun': 'integer'})


# ==============================================================================================================
# Theme C, plain tests: neighbouring behaviour that is already faithful
# ==============================================================================================================
@pytest.mark.parametrize('lag', [5, 10, [2, 7], [0, 3]], ids=['5', '10', '2to7', '0to3'])
@pytest.mark.parametrize('arglag', [{'fun': 'ns', 'df': 3}, {'fun': 'ns', 'df': 4, 'intercept': False}],
                         ids=['ns3', 'ns4_nointercept'])
def test_explicit_ns_lag_basis_matches_r(lag, arglag):
    x = _series(300)
    assert_crossbasis_matches_r(x, lag, _bs2_knots(x), arglag=arglag)


@pytest.mark.parametrize('lag', [[0, 3], [1, 4], 6], ids=['0to3', '1to4', '6'])
def test_integer_lag_basis_matches_r(lag):
    x = _series(250, start=200)
    assert_crossbasis_matches_r(x, lag, NS_DF3, arglag={'fun': 'integer'})


def test_ns_logknots_lag_basis_with_offset_lag_range_matches_r():
    """bs(deg 2) x ns(logknots) on lag [2,21]: exercises utils.logknots (non-zero minimum lag) and the NaN head."""
    from utils import logknots
    x = _series(300)
    kl = rget('logknots(c(2,21), nk=3)')
    np.testing.assert_allclose(logknots([2, 21], nk=3), kl, rtol=1e-12)
    cb, ref = assert_crossbasis_matches_r(x, [2, 21], _bs2_knots(x), arglag={'fun': 'ns', 'knots': kl})
    assert int(np.isnan(np.asarray(cb.basis)).any(axis=1).sum()) == 21


def test_crossbasis_series_with_interior_nan_matches_r():
    """A missing exposure makes the rows that look back to it NaN, in R and in Python alike."""
    x = _series(300)
    x[[100, 180]] = np.nan
    cb, ref = assert_crossbasis_matches_r(x, [0, 4], NS_DF4, arglag={'fun': 'ns', 'df': 3})
    assert int(np.isnan(ref['mat']).any(axis=1).sum()) == 4 + 2 * 5     # head + 5 rows after each interior NaN


@pytest.mark.parametrize('kind', ['pandas_series', 'float32', 'int64'])
def test_crossbasis_input_types_match_r(kind):
    import pandas as pd
    base = np.round(_series(300)).astype(np.int64)
    x = {'pandas_series': pd.Series(base.astype(float)), 'float32': base.astype(np.float32), 'int64': base}[kind]
    ref = r_crossbasis(base.astype(float), 6, NS_DF3, {'fun': 'ns', 'df': 3})
    cb = py_crossbasis(x, 6, NS_DF3, {'fun': 'ns', 'df': 3})
    assert_close(np.asarray(cb.basis), ref['mat'], rtol=1e-10, what=f'crossbasis input {kind}')


def test_crossbasis_attributes_match_r():
    x = _series(300)
    cb, ref = assert_crossbasis_matches_r(x, [1, 6], _bs2_knots(x), arglag={'fun': 'ns', 'df': 3})
    np.testing.assert_array_equal(np.asarray(cb.lag, dtype=float), ref['lag'])
    np.testing.assert_allclose(np.asarray(cb.range, dtype=float), ref['range'], rtol=0, atol=0)


# ==============================================================================================================
# Theme H1a: mklag rounding                                                                            crossbasis-10
# ==============================================================================================================
NONINT_LAGS = [2.5, 3.5, 0.4, 0.6, 5.6, -2.5, [0.4, 5.4], [0.5, 2.5], [1.5, 3.5], [2.6, 7.4]]


@known_defect('H1a', 'crossbasis-10', note='mklag returns the unrounded values; R returns round(lag[1:2])')
@pytest.mark.parametrize('lag', NONINT_LAGS, ids=[str(v) for v in NONINT_LAGS])
def test_mklag_rounds_noninteger_lags(lag):
    from utils import mklag
    np.testing.assert_array_equal(np.asarray(mklag(lag), dtype=float), r_mklag(lag))


@known_defect('H1a', 'crossbasis-10', note='mklag accepts NaN lags; R stops with "missing value where TRUE/FALSE needed"')
@pytest.mark.parametrize('lag', [[np.nan, 3.0], np.nan], ids=['pair', 'scalar'])
def test_mklag_rejects_missing_lag(lag):
    from utils import mklag
    assert r_fails(f'dlnm:::mklag({_rvec(lag)})')
    with pytest.raises((ValueError, TypeError)):
        mklag(lag)


@known_defect('H1a', 'crossbasis-10', note='non-integer lag range: Python lag set has one extra lag / shifted lags')
@pytest.mark.parametrize('lag, knot', [(2.5, 1.2), ([0.4, 5.4], 2.5), ([1.5, 6.5], 4.0)],
                         ids=['2.5', '0.4to5.4', '1.5to6.5'])
def test_crossbasis_noninteger_lag_uses_rounded_lag_range(lag, knot):
    """The ns lag knot is interior to both the rounded (R) and the unrounded range, so R does not error."""
    x = _series(300)
    cb, ref = assert_crossbasis_matches_r(x, lag, NS_DF3, arglag={'fun': 'ns', 'knots': np.array([knot])})
    np.testing.assert_array_equal(np.asarray(cb.lag, dtype=float), ref['lag'])


@known_defect('H1a', 'crossbasis-10', note='logknots inherits the unrounded mklag: knots placed on the wrong range')
@pytest.mark.parametrize('x', [[0, 1.5], 2.5, [0.4, 5.4]], ids=['0to1.5', '2.5', '0.4to5.4'])
def test_logknots_noninteger_range_is_rounded_first(x):
    from utils import logknots
    ref = rget(f'logknots({_rvec(x)}, nk=3)')
    np.testing.assert_allclose(np.asarray(logknots(x, nk=3), dtype=float), ref, rtol=1e-12)


# Theme H1a, plain: mklag already faithful for integer-valued input
INT_LAGS = [0, 1, 5, -1, -3, 1000.0, [0, 5], [2, 7], [3, 3], [-2, 2], [-3, -1], [0, 21], [5.0, 8.0]]


@pytest.mark.parametrize('lag', INT_LAGS, ids=[str(v) for v in INT_LAGS])
def test_mklag_integer_valued_lags_match_r(lag):
    from utils import mklag
    np.testing.assert_array_equal(np.asarray(mklag(lag), dtype=float), r_mklag(lag))


@pytest.mark.parametrize('lag', [[5, 2], [1, 2, 3], []], ids=['decreasing', 'length3', 'empty'])
def test_mklag_invalid_lags_raise_like_r(lag):
    from utils import mklag
    assert r_fails('dlnm:::mklag(numeric(0))' if len(lag) == 0 else f'dlnm:::mklag({_rvec(lag)})')
    with pytest.raises((ValueError, TypeError)):
        mklag(lag)


@pytest.mark.parametrize('x, nk', [(21, 3), ([2, 21], 3), ([0, 10], 2), ([1, 8], 4), (100, 5), ([-10, 0], 3)],
                         ids=str)
def test_logknots_integer_range_matches_r(x, nk):
    from utils import logknots
    ref = rget(f'logknots({_rvec(x)}, nk={nk})')
    np.testing.assert_allclose(np.asarray(logknots(x, nk=nk), dtype=float), ref, rtol=1e-12)


# ==============================================================================================================
# Theme H1a: seqlag overshoot                                                                          crossbasis-11
# ==============================================================================================================
OVERSHOOT = [([0, 5], 2), ([0, 5], 0.3), ([0, 3], 0.7), ([0, 6], 4), ([2, 9], 3), ([0, 10], 3), ([0, 1], 0.4)]


@known_defect('H1a', 'crossbasis-11', note='np.arange(from, to+by, by) includes a value above lag[2]; R seq() never does')
@pytest.mark.parametrize('lag, by', OVERSHOOT, ids=[f'{a}by{b}' for a, b in OVERSHOOT])
def test_seqlag_never_exceeds_upper_lag(lag, by):
    from utils import seqlag
    ref = r_seqlag(lag, by)
    py = np.asarray(seqlag(lag, by=by), dtype=float)
    assert ref.max() <= lag[1] + 1e-12                       # sanity: this is what R does
    assert_close(py, ref, rtol=1e-12, what=f'seqlag({lag}, by={by})')


def _crosspred_bylag_setup():
    """Cross-basis (lag 6) with fixed random coefficients, in both R and PyDLNM.

    The lag basis has explicit knots AND Boundary.knots, so that crosspred() at a lag grid that does not reach lag 6
    still evaluates the very same lag basis in R and Python (PyDLNM does not record resolved knots: other finding)."""
    x = _series(600)
    argvar = {'fun': 'ns', 'knots': rquantile(x, [.25, .75])}
    arglag = {'fun': 'ns', 'knots': np.array([3.0]), 'Boundary_knots': np.array([0.0, 6.0])}
    ref = r_crossbasis(x, 6, argvar, arglag)
    cb = py_crossbasis(x, 6, argvar, arglag)
    nc = ref['mat'].shape[1]
    rng = np.random.default_rng(1)
    coef = rng.normal(0, .02, nc)
    vcov = np.diag(rng.uniform(1e-4, 3e-4, nc))
    np2r('cbt_cf', coef)
    np2r('cbt_V', vcov)
    at = np.arange(-8, 30.01, 2.0)
    np2r('cbt_at', at)
    return cb, coef, vcov, at


def _crosspred_r_py(cb, coef, vcov, at, lag, bylag):
    from prediction import crosspred
    lag_r = '' if lag is None else f', lag={_rlag(lag)}'
    r(f'cbt_cp <- crosspred(cbt_cb, coef=cbt_cf, vcov=cbt_V, model.link="log", at=cbt_at, cen=15, bylag={float(bylag)!r}{lag_r})')
    ref = dict(matfit=rget('cbt_cp$matfit'), matse=rget('cbt_cp$matse'), allfit=rget('cbt_cp$allfit'),
               colnames=list(r('colnames(cbt_cp$matfit)')))
    with contextlib.redirect_stdout(io.StringIO()):
        cp = crosspred(cb, coef=coef, vcov=vcov, model_link='log', at=at, cen=15, bylag=bylag,
                       **({} if lag is None else {'lag': lag}))
    return cp, ref


@known_defect('H1a', 'crossbasis-11', note='crosspred(bylag) predicts at lags beyond the cross-basis lag range')
@pytest.mark.parametrize('lag, bylag', [(None, 4), ([0, 5], 2)], ids=['lag6_by4', 'sub0to5_by2'])
def test_crosspred_bylag_not_dividing_lag_range_matches_r(lag, bylag):
    cb, coef, vcov, at = _crosspred_bylag_setup()
    cp, ref = _crosspred_r_py(cb, coef, vcov, at, lag, bylag)
    assert list(cp.lag_names) == ref['colnames'], f'lag columns: Python {list(cp.lag_names)} vs R {ref["colnames"]}'
    assert_close(np.asarray(cp.matfit), ref['matfit'], rtol=1e-8, what='matfit')
    assert_close(np.asarray(cp.matse), ref['matse'], rtol=1e-8, what='matse')
    assert_close(np.asarray(cp.allfit), ref['allfit'], rtol=1e-8, what='allfit')


# Theme H1a, plain: seqlag / bylag that divide the lag range are already faithful
DIVIDING = [([0, 5], 1), ([0, 5], 0.5), ([2, 7], 1), ([0, 21], 1), ([0, 10], 0.25), ([-3, 0], 1), ([0, 10], 0.1),
            ([0, 1], 0.1), ([0, 100], 0.1), ([0, 6], 0.2), ([0, 5.5], 0.5), ([1, 1], 1)]


@pytest.mark.parametrize('lag, by', DIVIDING, ids=[f'{a}by{b}' for a, b in DIVIDING])
def test_seqlag_dividing_by_matches_r(lag, by):
    from utils import seqlag
    assert_close(np.asarray(seqlag(lag, by=by), dtype=float), r_seqlag(lag, by), rtol=1e-12,
                 what=f'seqlag({lag}, by={by})')


@pytest.mark.parametrize('bylag', [1, 2, 0.5, 0.3])
def test_crosspred_bylag_dividing_lag_range_matches_r(bylag):
    cb, coef, vcov, at = _crosspred_bylag_setup()
    cp, ref = _crosspred_r_py(cb, coef, vcov, at, None, bylag)
    assert_close(np.asarray(cp.matfit), ref['matfit'], rtol=1e-8, what='matfit')
    assert_close(np.asarray(cp.matse), ref['matse'], rtol=1e-8, what='matse')
    assert_close(np.asarray(cp.allfit), ref['allfit'], rtol=1e-8, what='allfit')
