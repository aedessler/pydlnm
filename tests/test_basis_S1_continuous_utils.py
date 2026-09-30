"""Theme S1 (other basis functions), continuous bases and utils group.

R-vs-PyDLNM differential tests for the continuous one-dimensional bases (dlnm onebasis() with ns/bs/lin/poly and a
user-defined function), the R-splines wrappers in enhanced_splines.py, and the knot / exposure-history helpers of
utils.py (equalknots(), logknots(), exphist()).  The reference is always computed by R at test time (dlnm 2.4.10 and
splines), never copied from Python output.  Where R itself raises, the faithful behaviour is that PyDLNM raises too.

Findings covered (all fixed: every test is an ordinary test that asserts the R-faithful behaviour)

  basis-cont-3     ns/bs with neither df nor knots: R has no interior knots (1 / `degree` columns); Python used to hard-code
                   df=4 (OneBasis, enhanced_splines, CrossBasis with a bare argvar)
  basis-cont-6     PolynomialBasis: NaN/Inf/all-NaN x (R: NaN rows, scale ignores NaN), float degree, degree=0,
                   constant x=0 (R: 0/0 = NaN)
  basis-cont-7,    equalknots: equally spaced VALUES along range(x) (not quantiles), nk / intercept arguments,
  crossbasis-13    df=1 default, R's knot counts (bs df-degree-intercept, strata df-intercept), errors when no knots
  basis-cont-8     user-defined callable: crosspred(OneBasis) replays 'range', `cen` is passed to a function that
                   declares it, attributes returned by the function are kept
  basis-cont-12    ns at a single x (and a single non-missing x): R defaults Boundary.knots to x * c(7, 9) / 8
  basis-cont-13    scalar knots (float / np.float64 / 0-d) for ns/bs (OneBasis, enhanced_splines, CrossBasis)
  basis-cont-14    OneBasis.summary() reports the number of columns (R: df = ncol), not the requested df
  basis-cont-15    enhanced_splines: attributes carry R's knots / boundary knots; smooth_spline_basis does not
                   silently ignore lambda_smooth (UserWarning)
  crossbasis-12    exphist: default lag c(0, length(exp) - 1), `times` are 1-based indices of any length (rounded,
                   fill outside the series), and it is not quadratic in n
  crossbasis-14    logknots: bare call (df=1) has no knots and errors in R, NaN in a range vector, non-integer lag
                   range is rounded by mklag

Also guarded: ns/bs with explicit df, knots, degree, intercept and Boundary.knots (also NaN, tiny and constant x, knot
container types, integer / list x), lin / poly without NaN, `cen` stored as an attribute, bs at a single x, ns at 2..5
points, custom callables that accept **kwargs, crosspred(OneBasis) for lin / poly, enhanced bs/ns numerics, exphist with
default times, logknots for nk / df / fun / degree / intercept grids, equalknots for uniformly spaced x.
"""
import contextlib
import io
import itertools
import time
import warnings

import numpy as np
import pytest

from rhelpers import assert_close, chicago, max_rel_diff, np2r, r, rget

import rpy2.rinterface_lib.embedded as _emb

RRuntimeError = _emb.RRuntimeError

THEME = 'S1'
P = 'tS1c_'                         # prefix of every R global created here (other modules share the R session)
_R_NAMES = {'Boundary_knots': 'Boundary.knots', 'boundary_knots': 'Boundary.knots'}    # python spelling -> R spelling


# ---------------------------------------------------------------------------------------------------------------------
# helpers: run the same call in R and in PyDLNM on identical inputs
# ---------------------------------------------------------------------------------------------------------------------
def _rval(name, v):
    """R expression for a python argument.  Non-integer numbers and arrays are pushed to R as doubles (no decimal
    round trip); scalars are always sent as length-1 vectors."""
    if v is None:
        return 'NULL'
    if isinstance(v, (bool, np.bool_)):
        return 'TRUE' if v else 'FALSE'
    if isinstance(v, str):
        return '"%s"' % v
    if np.ndim(v) == 0 and float(v).is_integer() and abs(float(v)) < 1e9:
        return str(int(v))
    np2r(P + 'a_' + name, np.atleast_1d(np.asarray(v, dtype=float)))
    return P + 'a_' + name


def _rargs(kw):
    return ''.join(f', {_R_NAMES.get(k, k)}={_rval(k, v)}' for k, v in kw.items())


def _last_line(e):
    lines = [s.strip() for s in str(e).strip().splitlines() if s.strip()]
    return lines[-1] if lines else type(e).__name__


def _try_r(fn):
    """(value, None) or (None, message) when R raises."""
    try:
        return fn(), None
    except RRuntimeError as e:
        return None, _last_line(e)


def _try_py(fn):
    """(value, None) or (None, message) when PyDLNM raises anything."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return np.asarray(fn(), dtype=float), None
    except Exception as e:                                                       # noqa: BLE001
        return None, f'{type(e).__name__}: {_last_line(e)}'


def compare_with_r(r_fn, py_fn, what, rtol=1e-8):
    """None if PyDLNM reproduces R - the same array (shape, NaN pattern, values), or an error wherever R errors - else
    a message that says how they differ."""
    ref, rerr = _try_r(r_fn)
    py, perr = _try_py(py_fn)
    if ref is None:
        return None if py is None else f'{what}: R raises ({rerr}) but PyDLNM returns {py.tolist()}'
    if py is None:
        return f'{what}: R returns shape {ref.shape}, PyDLNM raises {perr}'
    if py.shape != ref.shape:
        return f'{what}: shape PyDLNM {py.shape} vs R {ref.shape}'
    if not np.array_equal(np.isnan(py), np.isnan(ref)):
        return f'{what}: NaN pattern differs (PyDLNM {int(np.isnan(py).sum())} NaN cells, R {int(np.isnan(ref).sum())})'
    d = max_rel_diff(py, ref)
    return None if d <= rtol else f'{what}: max relative diff {d:.3e} > {rtol:.0e}'


def assert_matches_r(r_fn, py_fn, what, rtol=1e-8):
    msg = compare_with_r(r_fn, py_fn, what, rtol)
    assert msg is None, msg


def _fmt(v):
    """Short, stable text for a test id / message (arrays are abbreviated)."""
    if isinstance(v, (list, tuple, np.ndarray)) and np.ndim(v) > 0:
        a = np.asarray(v, dtype=float).ravel()
        return '[' + ','.join(f'{t:g}' for t in a[:4]) + (',..' if a.size > 4 else '') + f']#{a.size}'
    if isinstance(v, (float, np.floating)):
        return f'{float(v):g}'
    return str(v)


def _ids(kw):
    return ','.join(f'{k}={_fmt(v)}' for k, v in kw.items()) or 'defaults'


def _sid(v):
    """Test id without blanks (so that `pytest -rf` lines stay one token per test)."""
    return _ids(v) if isinstance(v, dict) else ''.join(str(v).split())


# ---- onebasis / crossbasis / splines -------------------------------------------------------------------------------
def r_onebasis(x, fun, **kw):
    """R: onebasis(x, fun, ...) as a plain matrix (the object stays in R as tS1c_b; read attributes with r_attr)."""
    np2r(P + 'x', x)
    r(f'{P}b <- suppressWarnings(onebasis({P}x, fun="{fun}"{_rargs(kw)}))')
    return rget(f'matrix(as.numeric({P}b), nrow=nrow({P}b))')


def r_attr(name):
    """numeric attribute of the last R onebasis (empty array if NULL)."""
    return rget(f'as.numeric(attr({P}b, "{name}"))')


def py_onebasis(x, fun, **kw):
    from basis import OneBasis
    return OneBasis(x, fun=fun, **kw)


def onebasis_vs_r(x, fun, rtol=1e-8, **kw):
    return compare_with_r(lambda: r_onebasis(x, fun, **kw), lambda: py_onebasis(x, fun, **kw).basis,
                          f'onebasis({fun}, {_ids(kw)})', rtol)


def assert_onebasis_matches_r(x, fun, rtol=1e-8, **kw):
    msg = onebasis_vs_r(x, fun, rtol=rtol, **kw)
    assert msg is None, msg


def r_splines(fun, x, **kw):
    """R: splines::ns / splines::bs called directly."""
    np2r(P + 'x', x)
    r(f'{P}s <- suppressWarnings(splines::{fun}({P}x{_rargs(kw)}))')
    return rget(f'matrix(as.numeric({P}s), nrow=length({P}x))')


def r_crossbasis(x, lag, argvar, arglag):
    np2r(P + 'cx', x)
    rl = lambda pre, d: 'list(' + ', '.join(f'{_R_NAMES.get(k, k)}={_rval(pre + k, v)}' for k, v in d.items()) + ')'
    r(f'{P}cb <- suppressWarnings(crossbasis({P}cx, lag={lag}, argvar={rl("v_", argvar)}, arglag={rl("l_", arglag)}))')
    return rget(f'matrix(as.numeric({P}cb), nrow=nrow({P}cb))')


def py_crossbasis(x, lag, argvar, arglag):
    from basis import CrossBasis
    # fresh dicts: CrossBasis keeps and mutates the caller's dicts (separate finding)
    return CrossBasis(x, lag=lag, argvar=dict(argvar), arglag=dict(arglag)).basis


def assert_crossbasis_matches_r(x, lag, argvar, arglag, rtol=1e-8):
    assert_matches_r(lambda: r_crossbasis(x, lag, argvar, arglag), lambda: py_crossbasis(x, lag, argvar, arglag),
                     f'crossbasis(lag={lag}, argvar={_ids(argvar)}, arglag={_ids(arglag)})', rtol)


# ---- utils ---------------------------------------------------------------------------------------------------------
def r_numeric(fname, *args, **kw):
    """R: as.numeric(fname(args, kw)); R errors raise RRuntimeError."""
    parts = [_rval(f'p{i}', a) for i, a in enumerate(args)] + [f'{_R_NAMES.get(k, k)}={_rval(k, v)}'
                                                              for k, v in kw.items()]
    return rget(f'as.numeric({fname}({", ".join(parts)}))')


def r_exphist(exposure, **kw):
    """R: exphist(exp, times, lag, fill) as a matrix."""
    np2r(P + 'exp', exposure)
    parts = ''.join(f', {k}={_rval("h" + k, v)}' for k, v in kw.items())
    return rget(f'unclass(exphist({P}exp{parts}))')


def _x_normal(n=200, seed=1, mu=15.0, sd=8.0):
    return np.random.default_rng(seed).normal(mu, sd, n)


def _x_gamma(n=400, seed=2):
    return np.random.default_rng(seed).gamma(4, 4, n)


def _temp(n, start=0):
    return chicago()['temp'][start:start + n].copy()


def _exposure(n=25, seed=0, nan_at=()):
    e = np.random.default_rng(seed).normal(size=n)
    e[list(nan_at)] = np.nan
    return e


# =====================================================================================================================
# ns / bs through OneBasis: plain (already faithful)
# =====================================================================================================================
_KN3 = np.quantile(_x_normal(), [.25, .5, .75])
_BK_WIDE = np.array([-20.0, 45.0])
_BK_NARROW = np.quantile(_x_normal(), [.05, .95])            # some of x falls outside these boundary knots


@pytest.mark.parametrize('fun', ['ns', 'bs'])
@pytest.mark.parametrize('kw', [
    {'df': 1}, {'df': 3}, {'df': 4}, {'df': 5}, {'df': 8}, {'df': 5, 'intercept': True},
    {'knots': _KN3[:1]}, {'knots': _KN3[:2]}, {'knots': _KN3}, {'knots': _KN3, 'intercept': True},
    {'df': 5, 'knots': _KN3},                                      # R ignores df when knots are given
    {'df': 4, 'Boundary_knots': _BK_WIDE}, {'knots': _KN3, 'Boundary_knots': _BK_WIDE},
    {'df': 4, 'Boundary_knots': _BK_NARROW}, {'knots': _KN3, 'Boundary_knots': _BK_NARROW},
    {'knots': _KN3, 'Boundary_knots': _BK_WIDE[::-1]},
], ids=_ids)
def test_spline_with_df_or_knots_matches_r(fun, kw):
    """ns / bs with an explicit df or explicit knots: intercept, df+knots, wider / narrower / reversed Boundary.knots."""
    assert_onebasis_matches_r(_x_normal(), fun, rtol=1e-10, **kw)


@pytest.mark.parametrize('degree', [1, 2, 3, 4])
@pytest.mark.parametrize('kw', [{'df': 6}, {'knots': _KN3}, {'df': 6, 'intercept': True}], ids=_ids)
def test_bs_degree_matches_r(degree, kw):
    assert_onebasis_matches_r(_x_normal(), 'bs', rtol=1e-10, degree=degree, **kw)


@pytest.mark.parametrize('fun, kw', [
    ('ns', {'df': 4}), ('ns', {'knots': _KN3}), ('bs', {'df': 5}), ('bs', {'knots': _KN3, 'degree': 2}),
    ('lin', {}), ('lin', {'intercept': True}),
], ids=lambda v: v if isinstance(v, str) else _ids(v))
@pytest.mark.parametrize('where', [(0, 5, 50), (0, 119)], ids=['interior', 'ends'])
def test_nan_in_x_gives_nan_rows_like_r(fun, kw, where):
    """Missing exposure days: NaN rows for ns / bs / lin (also when the NaN sit at the extremes of the series)."""
    x = _x_normal(120)
    x[list(where)] = np.nan
    assert_onebasis_matches_r(x, fun, rtol=1e-10, **kw)


@pytest.mark.parametrize('n', [2, 3, 4, 5])
@pytest.mark.parametrize('fun, kw', [('ns', {'df': 3}), ('ns', {'knots': [5.0]}), ('bs', {'df': 4}),
                                     ('bs', {'knots': [5.0], 'degree': 2})], ids=lambda v: v if isinstance(v, str) else _ids(v))
def test_tiny_series_matches_r(n, fun, kw):
    """2 to 5 observations (df-derived knots coincide with data values)."""
    x = np.array([3.0, 8.0, 1.0, 9.5, 6.0])[:n]
    assert_onebasis_matches_r(x, fun, rtol=1e-10, **kw)


@pytest.mark.parametrize('fun, kw', [('bs', {'df': 4}), ('bs', {'df': 6, 'degree': 2}), ('bs', {'knots': [5.2]})],
                         ids=lambda v: v if isinstance(v, str) else _ids(v))
@pytest.mark.parametrize('xval', [5.0, -3.0])
def test_bs_at_a_single_x_matches_r(fun, kw, xval):
    """bs at one value (as crosspred does for the centring basis) is already the same as R."""
    assert_onebasis_matches_r([xval], fun, rtol=1e-10, **kw)


def test_ns_at_x_zero_behaves_like_r():
    """x = 0 alone: R's Boundary.knots = x * c(7, 9) / 8 is degenerate (0, 0); whatever R does (error), PyDLNM does."""
    assert_onebasis_matches_r([0.0], 'ns', df=3)


@pytest.mark.parametrize('kind', ['list', 'tuple', 'series', 'empty', 'unsorted', 'duplicate', 'at-boundary'])
@pytest.mark.parametrize('fun', ['ns', 'bs'])
def test_knot_container_types_match_r(fun, kind):
    """Knots given as list / tuple / pandas Series / empty / unsorted / duplicated / on the boundary: all as in R."""
    import pandas as pd
    x = np.round(_x_normal(200, seed=6), 1)
    kq = np.quantile(x, [.3, .6])
    py_knots = {'list': list(kq), 'tuple': tuple(kq), 'series': pd.Series(kq), 'empty': np.array([]),
                'unsorted': kq[::-1], 'duplicate': [12.0, 12.0], 'at-boundary': [x.min()]}[kind]
    r_knots = {'list': kq, 'tuple': kq, 'series': kq, 'empty': np.array([]), 'unsorted': kq[::-1],
               'duplicate': np.array([12.0, 12.0]), 'at-boundary': np.array([x.min()])}[kind]
    r_fn = lambda: r_onebasis(x, fun, knots=r_knots)
    assert_matches_r(r_fn, lambda: py_onebasis(x, fun, knots=py_knots).basis, f'{fun} knots as {kind}', rtol=1e-10)


@pytest.mark.parametrize('fun, kw', [('ns', {'df': 3}), ('ns', {'knots': [7.0]}), ('bs', {'df': 4}), ('lin', {}),
                                     ('lin', {'intercept': True})], ids=lambda v: v if isinstance(v, str) else _ids(v))
def test_constant_x_matches_r(fun, kw):
    """A constant exposure: R's ns() stops (degenerate boundary knots) and so must PyDLNM; bs and lin agree."""
    assert_onebasis_matches_r(np.full(20, 7.0), fun, rtol=1e-12, **kw)


@pytest.mark.parametrize('kw', [{'df': 5}, {'df': 5, 'intercept': True}], ids=_ids)
def test_integer_typed_and_list_x_match_r(kw):
    xi = np.arange(0, 22)
    ref = r_onebasis(xi.astype(float), 'ns', **kw)
    assert_close(py_onebasis(xi, 'ns', **kw).basis, ref, rtol=1e-10, what='integer-typed x')
    assert_close(py_onebasis(list(xi.astype(float)), 'ns', **kw).basis, ref, rtol=1e-10, what='list x')


# =====================================================================================================================
# basis-cont-3: ns / bs with neither df nor knots (R: knots = NULL, i.e. no interior knots)
# =====================================================================================================================
_NEITHER = [('ns', {}), ('ns', {'intercept': True}), ('ns', {'Boundary_knots': _BK_WIDE}),
            ('bs', {}), ('bs', {'degree': 1}), ('bs', {'degree': 2}), ('bs', {'degree': 2, 'intercept': True})]


@pytest.mark.parametrize('fun, kw', _NEITHER, ids=lambda v: v if isinstance(v, str) else _ids(v))
def test_onebasis_neither_df_nor_knots_matches_r(fun, kw):
    """R: onebasis(x, 'ns') has 1 column, bs has `degree` columns (one more with intercept); PyDLNM gives 4."""
    assert_onebasis_matches_r(_x_normal(300, seed=3), fun, rtol=1e-10, **kw)


def test_bs_cubic_with_intercept_and_no_df_matches_r():
    """Coincidence that must survive the default-df fix: cubic bs with intercept and no interior knots has 4 columns."""
    assert_onebasis_matches_r(_x_normal(300, seed=3), 'bs', rtol=1e-10, degree=3, intercept=True)


@pytest.mark.parametrize('fun, kw', [('ns', {}), ('ns', {'intercept': True}), ('bs', {}), ('bs', {'degree': 2}),
                                     ('bs', {'degree': 1, 'intercept': True}),
                                     ('ns', {'boundary_knots': (-20.0, 45.0)})],
                         ids=lambda v: v if isinstance(v, str) else _ids(v))
def test_enhanced_neither_df_nor_knots_matches_r(fun, kw):
    """ns_enhanced / bs_enhanced with neither df nor knots are splines::ns / splines::bs with their own defaults."""
    import enhanced_splines as es
    x = _x_normal(250, seed=9)
    pyf = es.ns_enhanced if fun == 'ns' else es.bs_enhanced
    assert_matches_r(lambda: r_splines(fun, x, **kw), lambda: pyf(x, **kw)[0], f'{fun}_enhanced({_ids(kw)})', 1e-10)


@pytest.mark.parametrize('argvar', [{}, {'fun': 'ns'}, {'fun': 'bs'}, {'fun': 'bs', 'degree': 2}], ids=_ids)
def test_crossbasis_bare_var_basis_matches_r(argvar):
    """argvar without df / knots (the crosspred docstring example is {'fun': 'bs'}): R gives 1 (ns) or `degree` (bs)
    exposure columns; PyDLNM builds 4 (ns) or 3 (bs) and the bs case raises IndexError."""
    assert_crossbasis_matches_r(_temp(400), 4, argvar, {'fun': 'ns', 'knots': np.array([1.0, 3.0])}, rtol=1e-10)


# =====================================================================================================================
# basis-cont-12: ns at a single x
# =====================================================================================================================
@pytest.mark.parametrize('x, kw', [
    ([5.0], {'df': 3}), ([5.0], {'df': 2, 'intercept': True}), ([5.0], {'df': 1}), ([5.0], {'knots': [5.2]}),
    ([-3.0], {'df': 3}),
    ([5.0, np.nan], {'df': 3}),          # one non-missing value: R's ns() also applies x * c(7, 9) / 8
], ids=_sid)
def test_ns_single_point_matches_r(x, kw):
    """R: ns(5, df=3) is a 1 x 3 matrix (Boundary.knots defaults to 5 * c(7, 9) / 8); PyDLNM passes Boundary.knots =
    (5, 5) to R, which raises (or returns NaN)."""
    assert_onebasis_matches_r(x, 'ns', rtol=1e-10, **kw)


# =====================================================================================================================
# basis-cont-13: scalar knots
# =====================================================================================================================
@pytest.mark.parametrize('kind', ['float', 'np.float64', 'quantile', 'zero-d'])
@pytest.mark.parametrize('fun', ['ns', 'bs'])
def test_scalar_knot_matches_r(fun, kind):
    """A single interior knot given as a scalar (knots=np.quantile(x, .5) is an np.float64) is valid in R."""
    x = _x_normal(150, seed=5)
    med = np.quantile(x, .5)
    knot = {'float': float(med), 'np.float64': np.float64(med), 'quantile': np.quantile(x, .5),
            'zero-d': np.array(med)}[kind]
    ref = lambda: r_onebasis(x, fun, knots=np.atleast_1d(med))
    assert_matches_r(ref, lambda: py_onebasis(x, fun, knots=knot).basis, f'{fun} scalar knot ({kind})', 1e-10)


@pytest.mark.parametrize('fun', ['ns', 'bs'])
def test_enhanced_scalar_knot_matches_r(fun):
    import enhanced_splines as es
    x = _x_normal(150, seed=5)
    med = float(np.quantile(x, .5))
    pyf = es.ns_enhanced if fun == 'ns' else es.bs_enhanced
    assert_matches_r(lambda: r_splines(fun, x, knots=np.atleast_1d(med)), lambda: pyf(x, knots=np.float64(med))[0],
                     f'{fun}_enhanced scalar knot', 1e-10)


@pytest.mark.parametrize('argvar, arglag', [
    ({'fun': 'ns', 'knots': np.float64(10.0)}, {'fun': 'ns', 'knots': np.array([1.0, 3.0])}),
    ({'fun': 'bs', 'knots': np.float64(10.0), 'degree': 2}, {'fun': 'ns', 'knots': np.array([1.0, 3.0])}),
    ({'fun': 'ns', 'knots': np.array([5.0, 15.0])}, {'fun': 'ns', 'knots': np.float64(2.0)}),
], ids=lambda d: _ids(d))
def test_crossbasis_scalar_knot_matches_r(argvar, arglag):
    assert_crossbasis_matches_r(_temp(300), 4, argvar, arglag, rtol=1e-10)


# =====================================================================================================================
# lin / poly
# =====================================================================================================================
@pytest.mark.parametrize('degree', [1, 2, 3, 5])
@pytest.mark.parametrize('intercept', [False, True])
@pytest.mark.parametrize('scale', [None, 30.0], ids=['auto-scale', 'scale=30'])
def test_poly_without_nan_matches_r(degree, intercept, scale):
    kw = {'degree': degree, 'intercept': intercept}
    if scale is not None:
        kw['scale'] = scale
    assert_onebasis_matches_r(_x_normal(200, seed=7), 'poly', rtol=1e-12, **kw)


def test_poly_scale_attribute_matches_r():
    x = _x_normal(150, seed=8)
    r_onebasis(x, 'poly', degree=3)
    ob = py_onebasis(x, 'poly', degree=3)
    assert_close(np.atleast_1d(ob.attributes['scale']), r_attr('scale'), rtol=1e-14, what='poly scale attribute')


@pytest.mark.parametrize('intercept', [False, True])
def test_lin_matches_r(intercept):
    assert_onebasis_matches_r(_x_normal(100, seed=9), 'lin', rtol=1e-14, intercept=intercept)


# =====================================================================================================================
# basis-cont-6: PolynomialBasis vs R poly()
# =====================================================================================================================
def _with_nan(n=150, seed=11, at=(0, 7, 60)):
    x = _x_normal(n, seed=seed)
    x[list(at)] = np.nan
    return x


@pytest.mark.parametrize('kw', [
    {'degree': 2}, {'degree': 3, 'intercept': True}, {'degree': 2, 'scale': 30.0},
    {'degree': 1, 'intercept': True, 'scale': 30.0},
], ids=_ids)
def test_poly_nan_in_x_gives_nan_rows_like_r(kw):
    """R: NaN rows, and scale = max(abs(x), na.rm=TRUE) ignores them.  PyDLNM raises even when scale is supplied."""
    assert_onebasis_matches_r(_with_nan(), 'poly', rtol=1e-12, **kw)


def test_poly_scale_attribute_ignores_nan():
    x = _with_nan()
    r_onebasis(x, 'poly', degree=2)
    py = py_onebasis(x, 'poly', degree=2)
    assert_close(np.atleast_1d(py.attributes['scale']), r_attr('scale'), rtol=1e-14,
                 what='poly scale attribute with NaN in x')


@pytest.mark.parametrize('kind', ['all-NaN', 'Inf'])
def test_poly_all_nan_and_inf_x_match_r(kind):
    x = np.full(8, np.nan) if kind == 'all-NaN' else _x_normal(50, seed=12)
    if kind == 'Inf':
        x[3] = np.inf
    assert_onebasis_matches_r(x, 'poly', degree=2)


@pytest.mark.parametrize('degree', [2.0, np.float64(3.0)], ids=['float', 'np.float64'])
@pytest.mark.parametrize('intercept', [False, True])
def test_poly_float_degree_matches_r(degree, intercept):
    """R's numeric degree is always a double."""
    x = _x_normal(100, seed=13)
    ref = lambda: r_onebasis(x, 'poly', degree=float(degree), intercept=intercept)
    assert_matches_r(ref, lambda: py_onebasis(x, 'poly', degree=degree, intercept=intercept).basis,
                     f'poly degree={degree!r}', 1e-12)


def test_poly_degree_zero_matches_r():
    """outer(x/scale, (1-intercept):degree) with degree 0 and no intercept is the sequence 1:0, i.e. columns x and 1."""
    assert_onebasis_matches_r(_x_normal(60, seed=14), 'poly', degree=0)


def test_poly_degree_zero_with_intercept_matches_r():
    assert_onebasis_matches_r(_x_normal(60, seed=14), 'poly', degree=0, intercept=True)


@pytest.mark.parametrize('intercept', [False, True])
def test_poly_constant_zero_x_matches_r(intercept):
    """Deliberate Python deviation (scale 0 -> 1); R-faithful is NaN (with the intercept column equal to 1)."""
    assert_onebasis_matches_r(np.zeros(12), 'poly', degree=2, intercept=intercept)


# =====================================================================================================================
# basis-cont-8: user-defined function
# =====================================================================================================================
def _myquad(x, k=2):
    """(x - 20) / 10 and its k-th power; no **kwargs, so unexpected keywords must not be passed in."""
    z = (np.asarray(x, dtype=float) - 20.0) / 10.0
    return np.column_stack([z, z ** k])


def _myquad_kw(x, k=2, **kwargs):
    return _myquad(x, k)


_R_MYQUAD = f'{P}myquad <- function(x, k=2) {{ z <- (x-20)/10; cbind(z, z^k) }}'


def _r_crosspred_onebasis(fun_name, temp, at, coef, cen, **kw):
    """R: crosspred(onebasis(temp, fun_name, ...), coef, vcov = 1e-4 I, log link, at, cen) -> (allfit, allse)."""
    np2r(P + 'temp', temp)
    np2r(P + 'at', at)
    np2r(P + 'coef', coef)
    r(f'{P}V <- diag({len(coef)}) * 1e-4')
    r(f'{P}ob <- onebasis({P}temp, fun="{fun_name}"{_rargs(kw)})')
    cen_arg = ', cen=FALSE' if cen is None else f', cen={cen}'      # cen=NULL would be R's automatic mid-range centring
    r(f'{P}pr <- crosspred({P}ob, coef={P}coef, vcov={P}V, model.link="log", at={P}at{cen_arg})')
    return rget(f'{P}pr$allfit'), rget(f'{P}pr$allse')


def _py_crosspred_onebasis(ob, at, coef, cen):
    from prediction import crosspred
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        p = crosspred(ob, coef=coef, vcov=np.eye(len(coef)) * 1e-4, model_link='log', at=at,
                      cen=False if cen is None else cen)   # None = no centering (R: cen=FALSE)
    return np.asarray(p.allfit, dtype=float).ravel(), np.asarray(p.allse, dtype=float).ravel()


def _custom_setup():
    temp = _temp(600)
    at = np.round(np.linspace(np.quantile(temp, .05), np.quantile(temp, .95), 9), 3)
    r(_R_MYQUAD)
    return temp, at, np.array([0.03, -0.02])


def test_custom_callable_matrix_matches_r():
    """OneBasis with a user-defined function evaluates it on x (R: onebasis(x, fun='name', k=2))."""
    temp, _, _ = _custom_setup()
    assert_matches_r(lambda: r_onebasis(temp, f'{P}myquad', k=2), lambda: py_onebasis(temp, _myquad, k=2).basis,
                     'custom basis matrix', 1e-13)


def test_crosspred_custom_callable_with_kwargs_matches_r():
    """A callable that accepts **kwargs survives crosspred's replay of the stored attributes (already faithful)."""
    temp, at, coef = _custom_setup()
    ref_fit, ref_se = _r_crosspred_onebasis(f'{P}myquad', temp, at, coef, 20.0, k=2)
    fit, se = _py_crosspred_onebasis(py_onebasis(temp, _myquad_kw, k=2), at, coef, 20.0)
    assert_close(fit, ref_fit, rtol=1e-10, what='allfit')
    assert_close(se, ref_se, rtol=1e-10, what='allse')


@pytest.mark.parametrize('cen', [20.0, None], ids=['cen=20', 'uncentred'])
def test_crosspred_custom_callable_without_kwargs_matches_r(cen):
    """R's mkXpred() passes only the stored attributes that are formals of the function; PyDLNM passes range=... and
    raises TypeError for a function without **kwargs."""
    temp, at, coef = _custom_setup()
    ref_fit, ref_se = _r_crosspred_onebasis(f'{P}myquad', temp, at, coef, cen, k=2)
    fit, se = _py_crosspred_onebasis(py_onebasis(temp, _myquad, k=2), at, coef, cen)
    assert_close(fit, ref_fit, rtol=1e-10, what='allfit')
    assert_close(se, ref_se, rtol=1e-10, what='allse')


def _cenfun(x, cen=None):
    return (np.asarray(x, dtype=float) - (0.0 if cen is None else cen)).reshape(-1, 1)


def test_custom_callable_receives_cen_like_r():
    """R: checkonebasis keeps `cen` in the arguments when the function declares a `cen` formal."""
    x = _temp(20)
    r(f'{P}cenfun <- function(x, cen=NULL) {{ z <- x - if(is.null(cen)) 0 else cen; matrix(z, ncol=1) }}')
    assert_matches_r(lambda: r_onebasis(x, f'{P}cenfun', cen=20.0),
                     lambda: py_onebasis(x, _cenfun, cen=20.0).basis, 'custom basis with cen', 1e-13)


class _Tagged(np.ndarray):
    """ndarray subclass whose instance carries extra attributes (the Python analogue of R attributes())."""


def _attrfun(x, s=3):
    m = np.asarray(x, dtype=float).reshape(-1, 1).view(_Tagged)
    m.scale = s * 2
    return m


def test_custom_callable_attributes_are_kept_like_r():
    """R: attributes(basis) of the returned matrix are copied into the onebasis object (here 'scale')."""
    x = _temp(20)
    r(f'{P}attrfun <- function(x, s=3) {{ m <- matrix(x, ncol=1); attr(m, "scale") <- s*2; m }}')
    r_onebasis(x, f'{P}attrfun', s=5)
    ob = py_onebasis(x, _attrfun, s=5)
    assert 'scale' in ob.attributes, f"attribute 'scale' returned by the function is lost; kept: {sorted(ob.attributes)}"
    assert_close(np.atleast_1d(ob.attributes['scale']), r_attr('scale'), rtol=1e-14, what="attributes['scale']")


@pytest.mark.parametrize('fun, kw', [('ns', {'df': 4}), ('bs', {'knots': _KN3}), ('poly', {'degree': 2})],
                         ids=lambda v: v if isinstance(v, str) else _ids(v))
def test_cen_is_stored_and_leaves_the_basis_unchanged_like_r(fun, kw):
    """onebasis(..., cen=15) keeps `cen` as an attribute; centring is applied at prediction time, not in the basis."""
    x = _x_normal(150, seed=18)
    plain = r_onebasis(x, fun, **kw)
    with_cen = r_onebasis(x, fun, cen=15.0, **kw)
    assert_close(with_cen, plain, rtol=1e-14, what='R basis with and without cen')
    ob = py_onebasis(x, fun, cen=15.0, **kw)
    assert_close(ob.basis, with_cen, rtol=1e-10, what=f'{fun} basis with cen')
    assert_close(np.atleast_1d(ob.attributes['cen']), r_attr('cen'), rtol=1e-14, what='cen attribute')


@pytest.mark.parametrize('cen', [20.0, None], ids=['cen=20', 'uncentred'])
@pytest.mark.parametrize('fun, kw', [('lin', {}), ('poly', {'degree': 2}), ('poly', {'degree': 3, 'scale': 25.0})],
                         ids=lambda v: v if isinstance(v, str) else _ids(v))
def test_crosspred_onebasis_lin_and_poly_match_r(fun, kw, cen):
    """crosspred() on a lin / poly OneBasis (poly keeps its scale in the attributes) reproduces R's fit and s.e."""
    temp = _temp(600)
    at = np.round(np.linspace(np.quantile(temp, .05), np.quantile(temp, .95), 9), 3)
    ncol = r_onebasis(temp, fun, **kw).shape[1]
    coef = np.random.default_rng(19).normal(0, .05, ncol)
    ref_fit, ref_se = _r_crosspred_onebasis(fun, temp, at, coef, cen, **kw)
    fit, se = _py_crosspred_onebasis(py_onebasis(temp, fun, **kw), at, coef, cen)
    assert_close(fit, ref_fit, rtol=1e-10, what='allfit')
    assert_close(se, ref_se, rtol=1e-10, what='allse')


# =====================================================================================================================
# basis-cont-14: summary reports df = number of columns
# =====================================================================================================================
def _summary_df(text):
    """The integer after 'df:' / 'Degrees of freedom:' in a summary printout, or None."""
    import re
    m = re.search(r'(?:degrees of freedom|\bdf)\s*:\s*(\d+)', text, flags=re.I)
    return int(m.group(1)) if m else None


def _r_summary_df(x, fun, **kw):
    r_onebasis(x, fun, **kw)
    return _summary_df('\n'.join(r(f'capture.output(summary({P}b))')))


_KX = _x_normal(250, seed=15)
_KQ = np.quantile(_KX, [.2, .4, .6, .8])


@pytest.mark.parametrize('fun, kw', [
    ('ns', {'knots': _KQ[:2]}), ('ns', {'knots': _KQ[:1]}), ('ns', {'df': 5, 'knots': _KQ[:3]}),
    ('bs', {'knots': _KQ}), ('bs', {'knots': _KQ[:3], 'degree': 2}), ('bs', {'df': 8, 'knots': _KQ[:1]}),
], ids=lambda v: v if isinstance(v, str) else _ids(v))
def test_onebasis_summary_df_is_number_of_columns(fun, kw):
    """summary.onebasis prints df = ncol(object); with knots the stored df attribute is stale (default 4)."""
    ref = _r_summary_df(_KX, fun, **kw)
    ob = py_onebasis(_KX, fun, **kw)
    assert ref == ob.shape[1], 'test setup: R df must equal the number of columns'
    assert _summary_df(ob.summary()) == ref, f'summary says {_summary_df(ob.summary())}, R and ncol say {ref}'


@pytest.mark.parametrize('fun, kw', [('ns', {'df': 5}), ('ns', {'df': 3}), ('bs', {'df': 6, 'degree': 2}),
                                     ('bs', {'df': 5})], ids=lambda v: v if isinstance(v, str) else _ids(v))
def test_onebasis_summary_df_with_df_argument_matches_r(fun, kw):
    ob = py_onebasis(_KX, fun, **kw)
    assert _summary_df(ob.summary()) == _r_summary_df(_KX, fun, **kw)


# =====================================================================================================================
# enhanced_splines.py
# =====================================================================================================================
_ENH_CFG = [
    ('ns', {'df': 5}), ('ns', {'df': 4, 'intercept': True}), ('ns', {'knots': _KQ[:3]}),
    ('ns', {'knots': _KQ[:3], 'boundary_knots': (-20.0, 45.0)}), ('ns', {'df': 4, 'boundary_knots': (3.0, 25.0)}),
    ('ns', {'df': 5, 'knots': _KQ[:3]}),
    ('bs', {'df': 5}), ('bs', {'df': 6, 'degree': 2, 'intercept': True}), ('bs', {'knots': _KQ[:3], 'degree': 2}),
    ('bs', {'knots': _KQ[:3], 'boundary_knots': (-20.0, 45.0)}), ('bs', {'df': 5, 'boundary_knots': (3.0, 25.0)}),
    ('bs', {'df': 5, 'knots': _KQ[:3]}),
]


@pytest.mark.parametrize('fun, kw', _ENH_CFG, ids=lambda v: v if isinstance(v, str) else _ids(v))
@pytest.mark.parametrize('nan_at', [(), (3, 40)], ids=['no-NaN', 'NaN'])
def test_enhanced_splines_match_r(fun, kw, nan_at):
    """bs_enhanced / ns_enhanced are splines::bs / splines::ns (df, knots, degree, intercept, boundary, NaN)."""
    import enhanced_splines as es
    x = _x_normal(250, seed=9)
    x[list(nan_at)] = np.nan
    pyf = es.ns_enhanced if fun == 'ns' else es.bs_enhanced
    assert_matches_r(lambda: r_splines(fun, x, **kw), lambda: pyf(x, **kw)[0], f'{fun}_enhanced({_ids(kw)})', 1e-10)


@pytest.mark.parametrize('fun', ['ns', 'bs'])
def test_enhanced_class_wrappers_match_functions(fun):
    import enhanced_splines as es
    x = _x_normal(120, seed=10)
    if fun == 'ns':
        cls, f, kw = es.EnhancedNaturalSplineBasis, es.ns_enhanced, {'knots': _KQ[:2]}
    else:
        cls, f, kw = es.EnhancedBSplineBasis, es.bs_enhanced, {'df': 6, 'degree': 2}
    assert_close(cls(**kw)(x), f(x, **kw)[0], rtol=1e-14, what=f'Enhanced {fun} class')


@pytest.mark.parametrize('fun, kw', [
    ('ns', {'df': 5}), ('ns', {'knots': _KQ[:3]}), ('ns', {'df': 4, 'boundary_knots': (-20.0, 45.0)}),
    ('bs', {'df': 6, 'degree': 2}), ('bs', {'knots': _KQ[:2], 'boundary_knots': (-20.0, 45.0)}),
], ids=lambda v: v if isinstance(v, str) else _ids(v))
def test_enhanced_attributes_report_r_knots_and_boundary(fun, kw):
    """R's ns/bs return the interior knots (derived from df) and the Boundary.knots they used; the attribute dict of
    ns_enhanced / bs_enhanced is documented to carry them but only has fun / intercept / n_basis."""
    import enhanced_splines as es
    x = _x_normal(250, seed=9)
    r_splines(fun, x, **kw)
    ref_knots, ref_bk = rget(f'as.numeric(attr({P}s, "knots"))'), rget(f'as.numeric(attr({P}s, "Boundary.knots"))')
    pyf = es.ns_enhanced if fun == 'ns' else es.bs_enhanced
    basis, attrs = pyf(x, **kw)
    assert 'knots' in attrs, f'attributes lack the interior knots: {sorted(attrs)}'
    assert_close(attrs['knots'], ref_knots, rtol=1e-12, what='attributes knots')
    assert 'boundary_knots' in attrs, f'attributes lack the boundary knots: {sorted(attrs)}'
    assert_close(np.asarray(attrs['boundary_knots'], dtype=float), ref_bk, rtol=1e-12, what='attributes boundary_knots')
    assert attrs.get('n_basis', basis.shape[1]) == basis.shape[1]


def test_smooth_spline_basis_lambda_is_not_silently_ignored():
    """smooth_spline_basis is documented as a smoothing-spline basis controlled by lambda_smooth.  Either the basis
    depends on lambda_smooth, or the function must say (UserWarning) that the value is ignored."""
    import enhanced_splines as es
    if not hasattr(es, 'smooth_spline_basis'):
        return                                   # removing the misleading function is one valid fix
    x = _x_normal(200, seed=16)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        b1, _ = es.smooth_spline_basis(x, lambda_smooth=1.0, df=5)
        b100, _ = es.smooth_spline_basis(x, lambda_smooth=100.0, df=5)
    differs = b1.shape != b100.shape or not np.array_equal(b1, b100)
    warned = any(issubclass(w.category, UserWarning) for w in caught)
    assert differs or warned, 'lambda_smooth=1 and lambda_smooth=100 give the identical basis and no warning is issued'


def test_validate_spline_against_r_reports_the_r_basis():
    """validate_spline_against_r returns the basis of splines::bs / ns and its diagnostics (rank, columns)."""
    import enhanced_splines as es
    if not hasattr(es, 'validate_spline_against_r'):
        pytest.skip('validate_spline_against_r was removed')
    x = _x_normal(150, seed=17)
    for fun, kw in [('bs', {'df': 5}), ('ns', {'df': 4})]:
        out = es.validate_spline_against_r(x, fun, **kw)
        ref = r_splines(fun, x, **kw)
        assert_close(out['basis_matrix'], ref, rtol=1e-10, what=f'{fun} basis_matrix')
        assert out['diagnostics']['n_basis'] == ref.shape[1]
        assert out['diagnostics']['rank'] == int(rget(f'qr({P}s)$rank')[0])


# =====================================================================================================================
# utils.equalknots   (basis-cont-7, crossbasis-13)
# =====================================================================================================================
def _equalknots_r_vs_py(x, what, **kw):
    from utils import equalknots
    return compare_with_r(lambda: r_numeric('equalknots', x, **kw), lambda: equalknots(x, **kw), what, rtol=1e-12)


@pytest.mark.parametrize('df', [2, 3, 4, 5, 6])
def test_equalknots_uniform_x_ns_matches_r(df):
    """For uniformly spaced x, equally spaced quantiles are equally spaced values: the ns knot count df-1 agrees."""
    x = np.linspace(0.0, 10.0, 101)
    msg = _equalknots_r_vs_py(x, f'equalknots(uniform, ns, df={df})', fun='ns', df=df)
    assert msg is None, msg


@pytest.mark.parametrize('intercept', [False, True])
@pytest.mark.parametrize('fun', ['ns', 'bs', 'strata'])
def test_equalknots_df_grid_matches_r(fun, intercept):
    """fun / df / degree / intercept grid on a skewed sample: R's knots (or its 'no knots' error) in every cell.
    R counts ns df-1-intercept, bs df-degree-intercept, strata df-intercept knots; PyDLNM ignores intercept and counts
    bs df-degree-1, strata df-1 (and returns [] where R raises)."""
    x = _x_gamma()
    bad = []
    for degree in ((2, 3) if fun == 'bs' else (3,)):
        for df in range(1, 9):
            msg = _equalknots_r_vs_py(x, f'equalknots({fun}, df={df}, degree={degree}, intercept={intercept})',
                                      fun=fun, df=df, degree=degree, intercept=intercept)
            if msg:
                bad.append(msg)
    assert not bad, f'{len(bad)} configurations differ from R:\n' + '\n'.join(bad[:6])


@pytest.mark.parametrize('nk', [1, 2, 3, 4])
def test_equalknots_nk_matches_r(nk):
    msg = _equalknots_r_vs_py(_x_gamma(), f'equalknots(nk={nk})', nk=nk)
    assert msg is None, msg


def test_equalknots_use_the_range_and_ignore_nan():
    """R: range(x, na.rm=TRUE) and equal spacing along it, whatever the distribution of x."""
    x = _x_gamma(300, seed=4)
    x[[5, 77, 120]] = np.nan
    msg = _equalknots_r_vs_py(x, 'equalknots(gamma with NaN, ns, df=4)', fun='ns', df=4)
    assert msg is None, msg


@pytest.mark.parametrize('kw', [
    {}, {'fun': 'ns', 'df': 1}, {'fun': 'ns', 'df': 2, 'intercept': True}, {'fun': 'bs', 'df': 3},
    {'fun': 'bs', 'df': 4, 'intercept': True}, {'fun': 'bs', 'df': 2, 'degree': 1, 'intercept': True},
    {'fun': 'strata', 'df': 1, 'intercept': True}, {'nk': 0}, {'fun': 'spline', 'df': 4},
], ids=_ids)
def test_equalknots_without_knots_raises_like_r(kw):
    """R stops with 'choice of arguments defines no knots' (df defaults to 1; an unknown fun fails match.arg)."""
    from utils import equalknots
    x = _x_gamma()
    _, rerr = _try_r(lambda: r_numeric('equalknots', x, **kw))
    assert rerr is not None, 'test setup: R must raise for this call'
    with pytest.raises(ValueError):
        equalknots(x, **kw)


# =====================================================================================================================
# utils.logknots
# =====================================================================================================================
def _logknots_r_vs_py(x, what, **kw):
    from utils import logknots
    return compare_with_r(lambda: r_numeric('logknots', x, **kw), lambda: logknots(x, **kw), what, rtol=1e-12)


_LAG_RANGES = [21, [0, 21], [2, 21], [0, 10], [3, 30], 100, [1, 8], 7, [5, 5], -10, [-10, 0], [0, 2], 2, 1]


@pytest.mark.parametrize('lag', _LAG_RANGES, ids=_sid)
def test_logknots_nk_matches_r(lag):
    """nk given: R's formula, incl. non-zero minimum lag, negative ranges, scalar form, and the errors for a null range."""
    bad = [m for nk in (1, 2, 3, 4, 5)
           if (m := _logknots_r_vs_py(np.asarray(lag, dtype=float), f'logknots({lag}, nk={nk})', nk=nk))]
    assert not bad, '\n'.join(bad)


@pytest.mark.parametrize('fun', ['ns', 'bs', 'strata'])
@pytest.mark.parametrize('intercept', [False, True])
def test_logknots_df_grid_matches_r(fun, intercept):
    """df / degree given: R's knot counts, and Python raises wherever R does (df too small for the function)."""
    bad = []
    for degree, df in itertools.product((2, 3), range(1, 9)):
        m = _logknots_r_vs_py(np.array([21.0]), f'logknots(21, {fun}, df={df}, degree={degree}, intercept={intercept})',
                              fun=fun, df=df, degree=degree, intercept=intercept)
        if m:
            bad.append(m)
    assert not bad, '\n'.join(bad)


@pytest.mark.parametrize('x', [[0, 1, 2, 3, 7, 10], [5, 1, 2, 9, 10]], ids=_sid)
def test_logknots_range_of_a_vector_matches_r(x):
    """length >= 3: the range of the vector (not a lag range), unsorted input included."""
    msg = _logknots_r_vs_py(np.asarray(x, dtype=float), f'logknots({x}, nk=3)', nk=3)
    assert msg is None, msg


@pytest.mark.parametrize('kw', [{}, {'intercept': False}, {'fun': 'bs'}, {'fun': 'strata'}], ids=_ids)
@pytest.mark.parametrize('x', [21, [0, 21]], ids=_sid)
def test_logknots_bare_call_raises_like_r(x, kw):
    """R: df defaults to 1, which defines no knots ('choice of arguments defines no knots')."""
    from utils import logknots
    _, rerr = _try_r(lambda: r_numeric('logknots', np.asarray(x, dtype=float), **kw))
    assert rerr is not None, 'test setup: R must raise for this call'
    with pytest.raises(ValueError):
        logknots(x, **kw)


@pytest.mark.parametrize('x', [[0, 1, 2, np.nan, 10], [np.nan, 0, 4, 9, 10], [3, 6, 9, 30, np.nan, np.nan]], ids=_sid)
def test_logknots_nan_in_range_vector_matches_r(x):
    """R: range(x, na.rm=TRUE) for a vector of length >= 3."""
    msg = _logknots_r_vs_py(np.asarray(x, dtype=float), f'logknots({x}, nk=3)', nk=3)
    assert msg is None, msg


@pytest.mark.parametrize('x', [[0.3, 10.7], [0.5, 10.5], [2.2, 20.6], 21.4, 20.8], ids=_sid)
def test_logknots_non_integer_lag_range_is_rounded_like_r(x):
    """R: mklag() rounds a lag range to integers before the knots are placed."""
    msg = _logknots_r_vs_py(np.asarray(x, dtype=float), f'logknots({x}, nk=3)', nk=3)
    assert msg is None, msg


# =====================================================================================================================
# utils.exphist   (crossbasis-12)
# =====================================================================================================================
def _exphist_r_vs_py(exposure, what, rtol=1e-12, **kw):
    from utils import exphist
    return compare_with_r(lambda: r_exphist(exposure, **kw), lambda: exphist(exposure, **kw), what, rtol)


_EXP_LAGS = [[0, 3], [0, 10], [2, 6], [1, 5], [4, 12], [0, 24], [3, 24], [-2, 3], [-3, -1], 5, -3]
_EXP_FILLS = [0.0, np.nan, 99.0]


@pytest.mark.parametrize('lag', _EXP_LAGS, ids=_sid)
def test_exphist_default_times_matches_r(lag):
    """times omitted (histories at every observation): lags with a minimum > 0, negative lags, lags as long as the
    series, fill 0 / NaN / 99, NaN inside the exposure - all already the same as R."""
    e = _exposure(nan_at=(4, 9))
    bad = [m for fill in _EXP_FILLS if (m := _exphist_r_vs_py(e, f'exphist(lag={lag}, fill={fill})', lag=lag, fill=fill))]
    assert not bad, '\n'.join(bad)


@pytest.mark.parametrize('fill', [0.0, 99.0])
def test_exphist_default_lag_matches_r(fill):
    """R: lag defaults to c(0, length(exp) - 1), a 25 x 25 matrix here; PyDLNM returns 25 x 2."""
    msg = _exphist_r_vs_py(_exposure(), f'exphist(default lag, fill={fill})', fill=fill)
    assert msg is None, msg


_TIMES = {'subset': [5, 10, 20], 'first-and-last': [1, 25], 'beyond-the-end': [25, 30, 40],
          'before-the-start': [-3, 0, 2], 'rounded': [3.4, 10.6, 17.5], 'unsorted-repeated': [12, 4, 4, 20]}


@pytest.mark.parametrize('lag', [[0, 4], [2, 6], [-2, 3]], ids=_sid)
@pytest.mark.parametrize('name', list(_TIMES))
def test_exphist_times_are_indices_like_r(name, lag):
    """R evaluates the history at each entry of `times` (rounded; positions outside 1..n are filled); PyDLNM raises
    'times and exposure must have the same length'."""
    e = _exposure(nan_at=(4, 9))
    bad = [m for fill in _EXP_FILLS
           if (m := _exphist_r_vs_py(e, f'exphist(times={_TIMES[name]}, lag={lag}, fill={fill})',
                                     times=np.asarray(_TIMES[name], dtype=float), lag=lag, fill=fill))]
    assert not bad, '\n'.join(bad)


def test_exphist_times_of_the_same_length_are_indices_like_r():
    """times = 10..34 has the length of the exposure, so PyDLNM accepts it and returns a matrix of the right shape but
    with the wrong values (it treats times as timestamps of the observations, R as positions in the series)."""
    e = _exposure()
    msg = _exphist_r_vs_py(e, 'exphist(times=10..34, lag=[0,4])', times=np.arange(10.0, 35.0), lag=[0, 4])
    assert msg is None, msg


def test_exphist_runs_in_linear_time():
    """A 22-year daily series (n=8000), lag 0..21: R needs ~30 ms and a vectorised port a few ms, the quadratic loops
    (every (time, lag) pair scans all times) several seconds.  The result must still equal R's."""
    from utils import exphist
    e = np.random.default_rng(1).normal(size=8000)
    t0 = time.perf_counter()
    out = exphist(e, lag=[0, 21])
    elapsed = time.perf_counter() - t0
    assert_close(out, r_exphist(e, lag=[0, 21]), rtol=1e-12, what='exphist n=8000')
    assert elapsed < 0.25, f'exphist(n=8000, lag=[0, 21]) took {elapsed:.2f} s (limit 0.25 s)'
