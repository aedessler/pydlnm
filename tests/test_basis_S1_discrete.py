"""Theme S1 (other basis functions), discrete / strata / threshold / integer group.

R-vs-PyDLNM differential tests for the non-spline one-dimensional bases (dlnm strata(), thr(), poly(), integer(), ps(),
cr()) and the default-argument behaviour of ns()/bs() reached through OneBasis and CrossBasis. The reference is always
computed by R at test time (onebasis() / crossbasis() of dlnm 2.4.10), never copied from Python output.

Known defects (strict xfail, @known_defect('S1', ...); each test asserts the R-faithful behaviour)

  basis-discrete-4   StrataBasis: df counts columns (df-intercept quantile breaks), df=1+intercept is one all-ones
                     column, ref=0 semantics, invalid ref is an error
  crossbasis-18      same StrataBasis defect seen through argvar/arglag 'strata' (plus R's old 'knots' -> 'breaks'
                     alias, NaN rows, CrossBasis routing of a strata var/lag basis)
  basis-discrete-5   StrataBasis: NaN exposures get a real stratum instead of NA rows
  basis-discrete-15  StrataBasis breaks: sorted/uniquified, scalar breaks, df attribute, df=0
  basis-discrete-6   ThresholdBasis: sorted thr.value, default side 'd' for several thresholds, h/l use the minimum,
                     d uses min and max, resolved thr.value/side attributes and their round trip
  basis-discrete-9   ns/bs default df=NULL (1 / degree columns), CrossBasis(argvar={'fun':'bs'}) crash
  basis-discrete-14  OneBasis has no 'ps', 'cr' (and 'integer') function
  crossbasis-20      argvar fun='integer' (OneBasis and CrossBasis)
  basis-discrete-16, crossbasis-19
                     PolynomialBasis crashes on NaN input (R: NaN rows, scale ignores NaN)

Plain (unmarked) tests guard the behaviour that is already faithful and must stay so while the fixes land: strata with
explicit breaks / intercept, single thresholds, poly without NaN, ns/bs with explicit df or knots, integer lag basis,
error behaviour for unknown functions.
"""
import numpy as np
import pytest

from rhelpers import assert_close, chicago, known_defect, max_rel_diff, np2r, r, rget

import rpy2.rinterface_lib.embedded as _emb

RRuntimeError = _emb.RRuntimeError

THEME = 'S1'


# ---------------------------------------------------------------------------------------------------------------------
# helpers: run the same call in R (dlnm) and in PyDLNM
# ---------------------------------------------------------------------------------------------------------------------
_R_NAMES = {'thr_value': 'thr.value', 'Boundary_knots': 'Boundary.knots'}    # python spelling -> R spelling


def _rval(name, v):
    """R expression for a python argument (arrays are pushed to R as doubles, so no literal rounding)."""
    if v is None:
        return 'NULL'
    if isinstance(v, (bool, np.bool_)):
        return 'TRUE' if v else 'FALSE'
    if isinstance(v, str):
        return '"%s"' % v
    if np.ndim(v) == 0:
        return repr(v.item() if hasattr(v, 'item') else v)
    np2r('.arg_' + name, v)
    return '.arg_' + name


def _rargs(prefix, kw):
    return ''.join(f', {_R_NAMES.get(k, k)}={_rval(prefix + k, v)}' for k, v in kw.items())


def _rlist(prefix, d):
    return 'list(' + ', '.join(f'{_R_NAMES.get(k, k)}={_rval(prefix + k, v)}' for k, v in d.items()) + ')'


def r_onebasis(x, fun, **kw):
    """R: onebasis(x, fun, ...) -> plain matrix (attributes stay in R as `.b`, read them with r_attr)."""
    np2r('.x', x)
    r(f'.b <- suppressWarnings(onebasis(.x, fun="{fun}"{_rargs("ob_", kw)}))')
    return rget('matrix(as.numeric(.b), nrow=nrow(.b))')


def r_attr(name):
    """numeric attribute of the last R onebasis (empty array if NULL)."""
    return rget(f'as.numeric(attr(.b, "{name}"))')


def r_attr_str(name):
    return str(r(f'as.character(attr(.b, "{name}"))')[0])


def py_onebasis(x, fun, **kw):
    from basis import OneBasis
    return OneBasis(x, fun=fun, **kw)


def onebasis_vs_r(x, fun, rtol=1e-8, **kw):
    """None if PyDLNM reproduces R's onebasis(x, fun, **kw) - same matrix, or an error wherever R errors - else a
    message that says how it differs."""
    try:
        ref, rerr = r_onebasis(x, fun, **kw), None
    except RRuntimeError as e:
        ref, rerr = None, str(e).strip().splitlines()[-1]
    try:
        py, perr = np.asarray(py_onebasis(x, fun, **kw).basis, dtype=float), None
    except Exception as e:                                                       # noqa: BLE001
        py, perr = None, f'{type(e).__name__}: {e}'
    tag = f'onebasis({fun}, {kw})'
    if ref is None:
        return None if py is None else f'{tag}: R raises ({rerr}) but PyDLNM returns shape {py.shape}'
    if py is None:
        return f'{tag}: R returns shape {ref.shape}, PyDLNM raises {perr}'
    if py.shape != ref.shape:
        return f'{tag}: shape PyDLNM {py.shape} vs R {ref.shape}'
    if not np.array_equal(np.isnan(py), np.isnan(ref)):
        return f'{tag}: NaN pattern differs (PyDLNM {int(np.isnan(py).sum())} NaN cells, R {int(np.isnan(ref).sum())})'
    d = max_rel_diff(py, ref)
    return None if d <= rtol else f'{tag}: max relative diff {d:.3e} > {rtol:.0e}'


def assert_onebasis_matches_r(x, fun, rtol=1e-8, **kw):
    msg = onebasis_vs_r(x, fun, rtol=rtol, **kw)
    assert msg is None, msg


def r_crossbasis(x, lag, argvar, arglag):
    np2r('.x', x)
    r(f'.cb <- suppressWarnings(crossbasis(.x, lag={_rval("cb_lag", lag)}, argvar={_rlist("cbv_", argvar)}, '
      f'arglag={_rlist("cbl_", arglag)}))')
    return rget('matrix(as.numeric(.cb), nrow=nrow(.cb))')


def py_crossbasis(x, lag, argvar, arglag):
    from basis import CrossBasis
    # fresh dicts: CrossBasis keeps and mutates the caller's dicts (separate finding)
    return CrossBasis(x, lag=lag, argvar=dict(argvar), arglag=dict(arglag))


def crossbasis_vs_r(x, lag, argvar, arglag, rtol=1e-8):
    """None if CrossBasis reproduces R's crossbasis() (matrix incl. NaN pattern), else a message."""
    ref = r_crossbasis(x, lag, argvar, arglag)
    tag = f'crossbasis(lag={lag}, argvar={argvar}, arglag={arglag})'
    try:
        py = np.asarray(py_crossbasis(x, lag, argvar, arglag).basis, dtype=float)
    except Exception as e:                                                       # noqa: BLE001
        return f'{tag}: R returns shape {ref.shape}, PyDLNM raises {type(e).__name__}: {e}'
    if py.shape != ref.shape:
        return f'{tag}: shape PyDLNM {py.shape} vs R {ref.shape}'
    if not np.array_equal(np.isnan(py), np.isnan(ref)):
        return f'{tag}: NaN pattern differs'
    d = max_rel_diff(py, ref)
    return None if d <= rtol else f'{tag}: max relative diff {d:.3e} > {rtol:.0e}'


def assert_crossbasis_matches_r(x, lag, argvar, arglag, rtol=1e-8):
    msg = crossbasis_vs_r(x, lag, argvar, arglag, rtol=rtol)
    assert msg is None, msg


def _need_r_package(pkg):
    """Skip unless R package `pkg` is installed (rhelpers.require_r_packages breaks on rpy2 3.6: invisible -> None)."""
    if not bool(r(f'isTRUE(suppressWarnings(requireNamespace("{pkg}", quietly=TRUE)))')[0]):
        pytest.skip(f'R package {pkg} not installed')


def _ids(kw):
    return ','.join(f'{k}={np.asarray(v).tolist() if isinstance(v, (list, np.ndarray)) else v}' for k, v in kw.items())


def _x_normal(n=200, seed=1, mu=15.0, sd=8.0):
    return np.random.default_rng(seed).normal(mu, sd, n)


def _x_uniform(n=200, seed=4, hi=30.0):
    return np.random.default_rng(seed).uniform(0.0, hi, n)


def _x_categorical(n=200, seed=1, nlev=5):
    return np.random.default_rng(seed).integers(0, nlev, n).astype(float)


def _temp(n):
    return chicago()['temp'][:n]


# =====================================================================================================================
# STRATA (dlnm strata.R)
# =====================================================================================================================
# ---- plain: configurations that are already faithful today -----------------------------------------------------------
@pytest.mark.parametrize('breaks', [[10.0], [10.0, 20.0], [5.0, 15.0, 25.0]], ids=str)
@pytest.mark.parametrize('intercept', [False, True])
def test_strata_explicit_breaks_match_r(breaks, intercept):
    """Explicit sorted breaks with a valid ref >= 1 (each ref value, with and without intercept) are exact."""
    x = _x_normal()
    for ref in range(1, len(breaks) + 2):
        assert_onebasis_matches_r(x, 'strata', breaks=np.array(breaks), ref=ref, intercept=intercept)


@pytest.mark.parametrize('df', [2, 3, 4, 5])
def test_strata_df_with_intercept_matches_r(df):
    """df with intercept=TRUE: df-1 quantile breaks in both implementations (the lag-basis convention)."""
    x = _x_normal()
    for ref in range(1, df + 1):
        assert_onebasis_matches_r(x, 'strata', df=df, ref=ref, intercept=True)


@pytest.mark.parametrize('ref', [1, 2])
def test_strata_default_df1_matches_r(ref):
    """df=1, intercept=FALSE: one break at the median, two strata, ref dropped."""
    assert_onebasis_matches_r(_x_normal(), 'strata', df=1, ref=ref, intercept=False)


@pytest.mark.parametrize('xval', [4.0, 10.0, 15.0, 20.0, 27.0])
def test_strata_at_a_single_x_matches_r(xval):
    """Strata evaluated at one value (what crosspred does for the centring basis) picks the same stratum as R's cut()."""
    for ref in (1, 2, 3):
        assert_onebasis_matches_r(np.array([xval]), 'strata', breaks=np.array([10.0, 20.0]), ref=ref)


def test_strata_without_missing_values_matches_r():
    """Control for the NaN tests below: same data, no NaN."""
    x = _x_uniform(60, seed=7, hi=10.0)
    assert_onebasis_matches_r(x, 'strata', breaks=np.array([3.0, 7.0]))


# ---- basis-discrete-4 / crossbasis-18: df, intercept and ref semantics -----------------------------------------------
@pytest.mark.parametrize('df,ref', [(2, 1), (3, 1), (3, 2), (4, 3), (5, 1)])
def test_strata_df_is_the_number_of_columns(df, ref):
    """R: intercept=FALSE gives df columns (df quantile breaks, df+1 strata, minus the reference)."""
    x = _x_normal()
    assert_onebasis_matches_r(x, 'strata', df=df, ref=ref, intercept=False)
    assert py_onebasis(x, 'strata', df=df, ref=ref, intercept=False).shape[1] == df


@pytest.mark.parametrize('df,ref', [(1, 1), (1, 0), (0, 0)])
def test_strata_intercept_without_breaks_is_a_single_column_of_ones(df, ref):
    """R special case df-intercept <= 0 without breaks: one all-ones column, ref/intercept handling is skipped."""
    x = _x_normal()
    assert_onebasis_matches_r(x, 'strata', df=df, ref=ref, intercept=True)
    b = np.asarray(py_onebasis(x, 'strata', df=df, ref=ref, intercept=True).basis)
    assert b.shape == (len(x), 1) and np.all(b == 1.0)


@pytest.mark.parametrize('kw', [dict(df=3, intercept=True), dict(df=4, intercept=False), dict(df=3, intercept=False),
                                dict(breaks=np.array([10.0, 20.0]), intercept=True),
                                dict(breaks=np.array([10.0, 20.0]), intercept=False),
                                dict(breaks=np.array([5.0, 15.0, 25.0]), intercept=True)], ids=_ids)
def test_strata_ref_zero(kw):
    """R: ref=0 with intercept=TRUE keeps every stratum and adds no intercept column; with intercept=FALSE it is
    reset to ref=1."""
    assert_onebasis_matches_r(_x_normal(), 'strata', ref=0, **kw)


@pytest.mark.parametrize('kw', [dict(breaks=np.array([10.0]), ref=3), dict(breaks=np.array([10.0, 20.0]), ref=4),
                                dict(df=1, intercept=True, ref=2), dict(df=2, intercept=True, ref=3)], ids=_ids)
def test_strata_invalid_ref_is_an_error(kw):
    """R stops with "wrong value in 'ref' argument" whenever ref is not in 0..number of strata."""
    x = _x_normal()
    with pytest.raises(RRuntimeError):
        r_onebasis(x, 'strata', **kw)                    # R itself refuses (reference behaviour)
    with pytest.raises(ValueError):
        py_onebasis(x, 'strata', **kw)


@pytest.mark.parametrize('kw', [dict(df=3), dict(df=5), dict(df=3, ref=2)], ids=_ids)
def test_strata_default_breaks_and_df_attributes(kw):
    """The resolved breaks/df attributes (which crosspred re-uses) are R's."""
    x = _x_normal()
    r_onebasis(x, 'strata', **kw)
    ob = py_onebasis(x, 'strata', **kw)
    assert_close(np.atleast_1d(ob.attributes['breaks']).astype(float), r_attr('breaks'), rtol=1e-8,
                 what='breaks attribute')
    assert int(ob.attributes['df']) == int(r_attr('df')[0]), \
        f"df attribute: PyDLNM {ob.attributes['df']} vs R {int(r_attr('df')[0])}"


def test_strata_full_df_ref_intercept_grid_matches_r():
    """Every (df, ref, intercept) combination with default breaks, R errors included."""
    x = _x_normal()
    bad = []
    for df in range(0, 6):
        for ref in range(0, 4):
            for intercept in (False, True):
                msg = onebasis_vs_r(x, 'strata', df=df, ref=ref, intercept=intercept)
                if msg:
                    bad.append(msg)
    assert not bad, f'{len(bad)} of 48 configurations differ from R, e.g.\n' + '\n'.join(bad[:6])


@known_defect(THEME, 'crossbasis-18', 'basis-discrete-8',
              note="R's old 'knots' argument is renamed 'breaks' for strata (with a warning)")
@pytest.mark.parametrize('knots', [[10.0, 20.0], [5.0, 15.0, 25.0]], ids=str)
def test_strata_old_knots_argument_is_an_alias_of_breaks(knots):
    """R checkonebasis(): onebasis(x, 'strata', knots=k) is onebasis(x, 'strata', breaks=k); PyDLNM ignored it."""
    x = _x_normal()
    ref = r_onebasis(x, 'strata', knots=np.array(knots))
    assert ref.shape[1] == len(knots)                    # R used the knots as breaks
    py = np.asarray(py_onebasis(x, 'strata', knots=np.array(knots)).basis)
    assert_close(py, ref, rtol=1e-8, what="strata(knots=...) as breaks")


# ---- basis-discrete-5: NaN exposures --------------------------------------------------------------------------------
def _x_with_nan():
    x = _x_uniform(60, seed=7, hi=10.0)
    x[[0, 17, 41, 59]] = np.nan                          # including both ends of the series
    return x


@pytest.mark.parametrize('kw', [dict(breaks=np.array([3.0, 7.0])), dict(breaks=np.array([3.0, 7.0]), ref=2),
                                dict(breaks=np.array([3.0, 7.0]), intercept=True), dict(df=1),
                                dict(df=3, intercept=True)], ids=_ids)
def test_strata_nan_exposure_gives_na_rows(kw):
    """R: cut() gives NA, so the strata columns of a missing exposure are NA (an intercept column stays 1)."""
    x = _x_with_nan()
    assert_onebasis_matches_r(x, 'strata', **kw)
    b = np.asarray(py_onebasis(x, 'strata', **kw).basis)
    assert np.isnan(b[np.isnan(x)][:, -1]).all()


# ---- basis-discrete-15: breaks handling -----------------------------------------------------------------------------
@pytest.mark.parametrize('breaks,ref', [([15.0, 5.0], 1), ([15.0, 5.0], 2), ([15.0, 5.0], 3), ([15.0, 5.0, 25.0], 1),
                                        ([5.0, 5.0, 15.0], 1), ([25.0, 5.0, 15.0, 5.0], 2), (15.0, 1), (15.0, 2)],
                         ids=lambda v: str(v))
def test_strata_breaks_are_sorted_unique_and_may_be_scalar(breaks, ref):
    """R: breaks <- sort(unique(breaks)): reversed, non-monotone and duplicated cut-points work, a scalar is one break."""
    x = _x_uniform()
    b = breaks if np.ndim(breaks) == 0 else np.array(breaks)
    assert_onebasis_matches_r(x, 'strata', breaks=b, ref=ref)


@pytest.mark.parametrize('kw', [dict(breaks=np.array([3.0, 7.0])), dict(breaks=np.array([3.0, 7.0]), intercept=True),
                                dict(breaks=np.array([5.0, 15.0, 25.0])), dict(breaks=np.array([15.0, 5.0, 5.0]))],
                         ids=_ids)
def test_strata_df_attribute_follows_breaks(kw):
    """R: df <- length(breaks) + intercept and breaks <- sort(unique(breaks)) are stored as attributes."""
    x = _x_uniform()
    r_onebasis(x, 'strata', **kw)
    ob = py_onebasis(x, 'strata', **kw)
    assert int(ob.attributes['df']) == int(r_attr('df')[0]), \
        f"df attribute: PyDLNM {ob.attributes['df']} vs R {int(r_attr('df')[0])}"
    assert_close(np.atleast_1d(ob.attributes['breaks']).astype(float), r_attr('breaks'), rtol=0, what='breaks attribute')


def test_strata_df_zero_is_one_column_of_ones():
    """R: df=0 without breaks leaves breaks NULL: one all-ones column (ref is not applied)."""
    x = _x_normal()
    assert_onebasis_matches_r(x, 'strata', df=0)


# =====================================================================================================================
# THRESHOLD (dlnm thr.R)
# =====================================================================================================================
def _x_thr():
    return _x_normal(120, seed=3)


# ---- plain: single threshold ----------------------------------------------------------------------------------------
@pytest.mark.parametrize('thr', [10.0, [15.0], None], ids=lambda t: str(t))
@pytest.mark.parametrize('side', ['h', 'l', 'd'])
@pytest.mark.parametrize('intercept', [False, True])
def test_thr_single_threshold_matches_r(thr, side, intercept):
    """One threshold (given, as a length-1 vector, or the default median), every side, with/without intercept."""
    kw = dict(side=side, intercept=intercept)
    if thr is not None:
        kw['thr_value'] = np.atleast_1d(thr).astype(float) if isinstance(thr, list) else thr
    assert_onebasis_matches_r(_x_thr(), 'thr', **kw)


@pytest.mark.parametrize('side', ['h', 'l', 'd'])
def test_thr_x_exactly_on_the_threshold_matches_r(side):
    assert_onebasis_matches_r(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), 'thr', thr_value=3.0, side=side)


def test_thr_missing_values_with_explicit_threshold_match_r():
    x = _x_thr()
    x[[2, 50]] = np.nan
    assert_onebasis_matches_r(x, 'thr', thr_value=12.0, side='d')


def test_thr_invalid_side_raises_in_both():
    x = _x_thr()
    with pytest.raises(RRuntimeError):
        r_onebasis(x, 'thr', thr_value=10.0, side='x')
    with pytest.raises(ValueError):
        py_onebasis(x, 'thr', thr_value=10.0, side='x')


# ---- basis-discrete-6: several thresholds, default side -------------------------------------------------------------
_THR_SETS = [[10.0, 20.0], [20.0, 10.0], [5.0, 12.0, 25.0], [25.0, 5.0, 12.0]]


@known_defect(THEME, 'basis-discrete-6', note="side defaults to 'd' when thr.value has several elements")
@pytest.mark.parametrize('thr', _THR_SETS, ids=str)
@pytest.mark.parametrize('intercept', [False, True])
def test_thr_default_side_is_double_for_several_thresholds(thr, intercept):
    """R: side <- ifelse(length(thr.value) > 1, 'd', 'h'); thr.value is sorted and min/max are the hinges."""
    x = _x_thr()
    assert_onebasis_matches_r(x, 'thr', thr_value=np.array(thr), intercept=intercept)
    assert py_onebasis(x, 'thr', thr_value=np.array(thr), intercept=intercept).shape[1] == 2 + intercept


@known_defect(THEME, 'basis-discrete-6', note='side h/l with a vector: one column at min(thr.value), not one per value')
@pytest.mark.parametrize('side', ['h', 'l'])
@pytest.mark.parametrize('thr', _THR_SETS, ids=str)
def test_thr_hl_side_uses_only_the_minimum_threshold(thr, side):
    """R: for side 'h'/'l' only min(thr.value) is used, giving a single column."""
    x = _x_thr()
    assert_onebasis_matches_r(x, 'thr', thr_value=np.array(thr), side=side)
    assert py_onebasis(x, 'thr', thr_value=np.array(thr), side=side).shape[1] == 1


@known_defect(THEME, 'basis-discrete-6', note="side 'd' with >2 or unsorted thresholds: R uses min and max")
@pytest.mark.parametrize('thr', [[20.0, 10.0], [5.0, 12.0, 25.0], [25.0, 5.0, 12.0], [12.0, 25.0, 5.0, 18.0]], ids=str)
@pytest.mark.parametrize('intercept', [False, True])
def test_thr_double_side_uses_min_and_max_threshold(thr, intercept):
    """R: thr.value[c(1, length)] of the sorted vector, so the lower hinge is at min and the upper at max."""
    assert_onebasis_matches_r(_x_thr(), 'thr', thr_value=np.array(thr), side='d', intercept=intercept)


@known_defect(THEME, 'basis-discrete-6', note="thr.value/side attributes hold R's resolved values")
@pytest.mark.parametrize('kw', [dict(thr_value=np.array([20.0, 10.0])),
                                dict(thr_value=np.array([5.0, 12.0, 25.0]), side='d'),
                                dict(thr_value=np.array([25.0, 5.0, 12.0]), side='h'),
                                dict(thr_value=np.array([25.0, 5.0, 12.0]), side='l')], ids=_ids)
def test_thr_resolved_attributes_match_r(kw):
    """R stores the sorted, reduced thr.value and the resolved side; crosspred rebuilds the basis from them."""
    x = _x_thr()
    r_onebasis(x, 'thr', **kw)
    ob = py_onebasis(x, 'thr', **kw)
    assert ob.attributes['side'] == r_attr_str('side'), \
        f"side attribute: PyDLNM {ob.attributes['side']!r} vs R {r_attr_str('side')!r}"
    assert_close(np.atleast_1d(ob.attributes['thr.value']).astype(float), r_attr('thr.value'), rtol=0,
                 what='thr.value attribute')


@known_defect(THEME, 'basis-discrete-6', note='rebuilding from the stored attributes at new x must reproduce R')
@pytest.mark.parametrize('kw', [dict(thr_value=np.array([20.0, 10.0])),
                                dict(thr_value=np.array([25.0, 5.0, 12.0]), side='d'),
                                dict(thr_value=np.array([25.0, 5.0, 12.0]), side='h', intercept=True)], ids=_ids)
def test_thr_attributes_round_trip_to_new_x(kw):
    """What crosspred does: build at x, read the attributes, rebuild at a new grid; R does the same with its attributes
    and the result must be the basis of R's onebasis() at the new grid."""
    x, xnew = _x_thr(), np.linspace(-5.0, 35.0, 41)
    r_onebasis(x, 'thr', **kw)
    r_thr, r_side = r_attr('thr.value'), r_attr_str('side')
    ref = r_onebasis(xnew, 'thr', thr_value=r_thr, side=r_side, intercept=kw.get('intercept', False))
    at = py_onebasis(x, 'thr', **kw).attributes
    py = np.asarray(py_onebasis(xnew, 'thr', thr_value=at['thr.value'], side=at['side'],
                                intercept=kw.get('intercept', False)).basis)
    assert_close(py, ref, rtol=1e-8, what='rebuilt threshold basis')


@known_defect(THEME, 'basis-discrete-6', 'basis-discrete-8',
              note="needs the multi-threshold default side and R's 'thr.value' spelling (addon patch)")
def test_thr_dotted_thr_value_argument_is_honoured():
    """R-named argument round trip: OneBasis(x, 'thr', **{'thr.value': v}) is R's onebasis(x, 'thr', thr.value=v),
    which for two thresholds means side 'd' (R's default); today the argument is swallowed by **kwargs."""
    x = _x_thr()
    v = np.array([10.0, 20.0])
    ref = r_onebasis(x, 'thr', thr_value=v)
    py = np.asarray(py_onebasis(x, 'thr', **{'thr.value': v}).basis)
    assert_close(py, ref, rtol=1e-8, what="thr.value= spelling")


# =====================================================================================================================
# POLY (dlnm poly.R)
# =====================================================================================================================
_X_POLY_NAN = np.array([1.0, 2.0, np.nan, 4.0, 5.0, -6.0, 7.0])


def _x_poly_nan_random():
    x = _x_normal(80, seed=11, mu=2.0, sd=10.0)
    x[[3, 40]] = np.nan
    return x


@pytest.mark.parametrize('degree', [1, 2, 3])
@pytest.mark.parametrize('intercept', [False, True])
@pytest.mark.parametrize('scale', [None, 12.5])
def test_poly_without_nan_matches_r(degree, intercept, scale):
    kw = dict(degree=degree, intercept=intercept)
    if scale is not None:
        kw['scale'] = scale
    x = _x_normal(80, seed=11, mu=2.0, sd=10.0)
    assert_onebasis_matches_r(x, 'poly', rtol=1e-12, **kw)


@known_defect(THEME, 'basis-discrete-16', 'crossbasis-19',
              note='sklearn PolynomialFeatures rejects NaN; np.max(abs(x)) is NaN')
@pytest.mark.parametrize('degree', [1, 2, 3])
@pytest.mark.parametrize('intercept', [False, True])
@pytest.mark.parametrize('xname', ['small', 'random'])
def test_poly_nan_gives_nan_rows_like_r(degree, intercept, xname):
    """R: scale = max(abs(x), na.rm=TRUE), outer(x/scale, ...) - NaN rows only (intercept column stays 1)."""
    x = _X_POLY_NAN if xname == 'small' else _x_poly_nan_random()
    assert_onebasis_matches_r(x, 'poly', rtol=1e-12, degree=degree, intercept=intercept)


@known_defect(THEME, 'basis-discrete-16', 'crossbasis-19', note='scale attribute ignores NaN in R')
def test_poly_scale_attribute_ignores_nan():
    x = _x_poly_nan_random()
    r_onebasis(x, 'poly', degree=2)
    ob = py_onebasis(x, 'poly', degree=2)
    assert_close(np.atleast_1d(ob.attributes['scale']).astype(float), r_attr('scale'), rtol=1e-14, what='scale')


@known_defect(THEME, 'basis-discrete-16', 'crossbasis-19', note='explicit scale with NaN input')
def test_poly_nan_with_user_scale():
    assert_onebasis_matches_r(_x_poly_nan_random(), 'poly', rtol=1e-12, degree=3, scale=20.0)


def _cb_poly_x():
    x = _temp(300).copy()
    x[[40, 41]] = np.nan
    return x


@known_defect(THEME, 'crossbasis-19', 'basis-discrete-16', note='CrossBasis(argvar poly) with NaN exposure raises')
def test_crossbasis_poly_with_nan_exposure_has_r_nan_pattern():
    """R: crossbasis() passes NaN exposures through (rows that see a NaN in the lag window are NaN). Only the
    NaN pattern is compared here; the values need the CrossBasis dispatch fix (next test)."""
    x = _cb_poly_x()
    argvar, arglag = dict(fun='poly', degree=2), dict(fun='ns', df=3)
    ref = r_crossbasis(x, 3, argvar, arglag)
    py = np.asarray(py_crossbasis(x, 3, argvar, arglag).basis, dtype=float)
    assert py.shape == ref.shape
    assert np.array_equal(np.isnan(py), np.isnan(ref))


@known_defect(THEME, 'crossbasis-19', 'basis-discrete-16', 'basis-discrete-1',
              note='needs the NaN fix and the CrossBasis dispatch fix (argvar fun is replaced by bs today)')
def test_crossbasis_poly_with_nan_exposure_matches_r():
    assert_crossbasis_matches_r(_cb_poly_x(), 3, dict(fun='poly', degree=2), dict(fun='ns', df=3))


# =====================================================================================================================
# DEFAULT df of ns / bs (basis-discrete-9)
# =====================================================================================================================
@pytest.mark.parametrize('kw', [dict(fun='ns', df=1), dict(fun='ns', df=4), dict(fun='bs', df=3),
                                dict(fun='bs', df=5, degree=2), dict(fun='ns', df=5, intercept=True)], ids=_ids)
def test_spline_with_explicit_df_matches_r(kw):
    kw = dict(kw)
    fun = kw.pop('fun')
    assert_onebasis_matches_r(_temp(400), fun, rtol=1e-10, **kw)


def test_spline_with_explicit_knots_matches_r():
    x = _temp(400)
    kv = np.quantile(x, [0.10, 0.75, 0.90])
    assert_onebasis_matches_r(x, 'ns', rtol=1e-10, knots=kv)
    assert_onebasis_matches_r(x, 'bs', rtol=1e-10, knots=kv, degree=2)


@known_defect(THEME, 'basis-discrete-9', 'basis-discrete-1', 'crossbasis-4',
              note='SplineBasis/BSplineBasis default df=4; R default df=NULL')
@pytest.mark.parametrize('spec', [dict(fun='ns'), dict(fun='bs'), dict(fun='bs', degree=2),
                                  dict(fun='ns', intercept=True)], ids=_ids)
def test_spline_default_df_is_null_like_r(spec):
    """R: ns() without df/knots has 1 column, bs() has `degree` columns (no interior knots)."""
    spec = dict(spec)
    fun = spec.pop('fun')
    assert_onebasis_matches_r(_temp(400), fun, rtol=1e-10, **spec)


@known_defect(THEME, 'basis-discrete-9', 'basis-discrete-1', 'crossbasis-4',
              note='OneBasis(x) with no fun: ns with df=NULL (one column)')
def test_onebasis_default_function_is_ns_with_one_column():
    x = _temp(300)
    np2r('.x', x)
    r('.b <- onebasis(.x)')
    ref = rget('matrix(as.numeric(.b), nrow=nrow(.b))')
    from basis import OneBasis
    assert_close(np.asarray(OneBasis(x).basis), ref, rtol=1e-10, what='onebasis(x)')


_CB_DEFAULT_DF = [
    (dict(fun='bs'), dict(fun='ns', df=4), 21),                    # the prediction.py docstring var basis
    (dict(), dict(fun='ns', df=4), 6),                             # argvar={}: R default fun is ns
    (dict(fun='bs', degree=2), dict(fun='ns', df=3), 5),
    (dict(fun='ns'), dict(fun='ns', df=4), 6),                     # ns without df: 1 var column (Python: 4)
    (dict(fun='ns'), dict(fun='integer'), 3),                      # ... and with an integer lag basis
]


@known_defect(THEME, 'basis-discrete-9', 'basis-discrete-1', 'crossbasis-4',
              note='IndexError / wrong column count: time-series path has its own df defaults')
@pytest.mark.parametrize('argvar,arglag,lag', _CB_DEFAULT_DF, ids=lambda v: str(v))
def test_crossbasis_var_basis_without_df_matches_r(argvar, arglag, lag):
    """crossbasis(temp, lag, argvar=list(fun='bs')) with an explicit lag basis (the default-arglag issue is separate)."""
    assert_crossbasis_matches_r(_temp(1500), lag, argvar, arglag, rtol=1e-10)


def test_crossbasis_docstring_example_runs():
    """prediction.py docstring: CrossBasis(temp, lag=21, argvar={'fun': 'bs'}). Only 'it builds' is asserted here
    (its column count also depends on the default-arglag finding)."""
    from basis import CrossBasis
    temp = chicago()['temp']
    cb = CrossBasis(temp, lag=21, argvar={'fun': 'bs'})
    assert cb.shape[0] == len(temp)
    assert np.isnan(np.asarray(cb.basis)[:21]).all()


@pytest.mark.parametrize('argvar,arglag,lag', [
    (dict(fun='ns', df=4), dict(fun='ns', df=3), 6),
    (dict(fun='bs', df=5, degree=2), dict(fun='ns', df=4), 8),
    (dict(fun='ns', knots=np.array([5.0, 15.0, 25.0])), dict(fun='integer'), 4)], ids=lambda v: str(v))
def test_crossbasis_with_explicit_df_or_knots_matches_r(argvar, arglag, lag):
    """Validated-path style CrossBasis (explicit df/knots, ns or integer lag) is exact and must stay so."""
    assert_crossbasis_matches_r(_temp(1500), lag, argvar, arglag, rtol=1e-10)


# =====================================================================================================================
# neighbouring behaviour that is already faithful (plain tests)
# =====================================================================================================================
@pytest.mark.parametrize('intercept', [False, True])
def test_lin_matches_r(intercept):
    assert_onebasis_matches_r(_x_normal(), 'lin', rtol=1e-14, intercept=intercept)


@pytest.mark.parametrize('fun,kw', [('lin', {}), ('ns', dict(df=3)), ('bs', dict(df=4, degree=2)),
                                    ('ns', dict(knots=np.array([5.0, 12.0, 20.0])))], ids=lambda v: str(v))
def test_missing_values_propagate_as_nan_rows_like_r(fun, kw):
    """lin/ns/bs (explicit df or knots): NaN input gives NaN rows, the spline is fitted to the finite values only."""
    x = _x_normal(150, seed=9)
    x[[0, 60, 149]] = np.nan
    assert_onebasis_matches_r(x, fun, rtol=1e-10, **kw)


@pytest.mark.parametrize('kind', ['list', 'int64', 'series'])
@pytest.mark.parametrize('fun,kw', [('lin', {}), ('poly', dict(degree=2)), ('strata', dict(breaks=np.array([10.0, 20.0]))),
                                    ('thr', dict(thr_value=12.0, side='d'))], ids=lambda v: str(v))
def test_onebasis_input_types_match_r(fun, kw, kind):
    """list, integer ndarray and pandas Series (with a non-default index) are the same numbers for R and PyDLNM."""
    import pandas as pd
    base = np.round(_x_normal(80, seed=6)).astype(np.int64)
    x = {'list': base.astype(float).tolist(), 'int64': base,
         'series': pd.Series(base.astype(float), index=[f'd{i}' for i in range(len(base))])}[kind]
    ref = r_onebasis(base.astype(float), fun, **kw)
    py = np.asarray(py_onebasis(x, fun, **kw).basis)
    assert_close(py, ref, rtol=1e-12, what=f'{fun} from {kind}')


@pytest.mark.parametrize('fun,kw', [('strata', dict(breaks=np.array([10.0, 20.0]))), ('thr', dict(thr_value=12.0)),
                                    ('poly', dict(degree=2))], ids=lambda v: str(v))
def test_onebasis_does_not_modify_its_input(fun, kw):
    x = _x_uniform(60, seed=7, hi=30.0)
    before = x.copy()
    py_onebasis(x, fun, **kw)
    assert np.array_equal(x, before)


@pytest.mark.parametrize('fun,kw', [('lin', {}), ('strata', dict(breaks=np.array([10.0, 20.0]))),
                                    ('strata', dict(breaks=np.array([10.0, 20.0]), ref=2)),
                                    ('poly', dict(degree=2))], ids=lambda v: str(v))
def test_crosspred_on_onebasis_matches_r(fun, kw):
    """The attributes stored by these OneBasis functions are enough for crosspred to rebuild the basis on a new grid
    and at the centring value (allfit/allse against R with identical coef/vcov)."""
    from prediction import crosspred
    x, at = _temp(500), np.arange(-5.0, 30.0, 2.5)
    ob = py_onebasis(x, fun, **kw)
    ncol = ob.shape[1]
    rng = np.random.default_rng(3)
    coef = rng.normal(0.0, 0.1, ncol)
    A = rng.normal(0.0, 0.05, (ncol, ncol))
    vcov = A @ A.T + 1e-3 * np.eye(ncol)
    np2r('.coef', coef)
    np2r('.vcov', vcov)
    np2r('.at', at)
    r_onebasis(x, fun, **kw)
    r('.cp <- suppressWarnings(crosspred(.b, coef=.coef, vcov=.vcov, at=.at, cen=15))')
    cp = crosspred(ob, coef=coef, vcov=vcov, at=at, cen=15.0)
    assert_close(np.ravel(cp.allfit), rget('as.numeric(.cp$allfit)'), rtol=1e-10, what='allfit')
    assert_close(np.ravel(cp.allse), rget('as.numeric(.cp$allse)'), rtol=1e-10, what='allse')


# =====================================================================================================================
# ps / cr / integer through OneBasis (basis-discrete-14, crossbasis-20)
# =====================================================================================================================
def _attr_matrix(ob, name):
    v = ob.attributes.get(name)
    return None if v is None else np.asarray(v, dtype=float)


@known_defect(THEME, 'basis-discrete-14', note="OneBasis has no 'ps' (P-spline) function")
@pytest.mark.parametrize('kw', [dict(), dict(df=6), dict(df=8, degree=2), dict(df=6, intercept=True), dict(df=6, diff=1),
                                dict(df=6, fx=True)], ids=_ids)
def test_onebasis_ps_matches_r(kw):
    """R ps(): equally spaced B-spline knots via splineDesign, difference penalty S in the attributes."""
    x = _x_normal(150, seed=2)
    assert_onebasis_matches_r(x, 'ps', rtol=1e-10, **kw)
    ob = py_onebasis(x, 'ps', **kw)
    if kw.get('fx'):
        assert ob.attributes.get('S') is None
    else:
        assert_close(np.ravel(_attr_matrix(ob, 'S')), r_attr('S'), rtol=1e-10, what='penalty matrix S')
    assert_close(_attr_matrix(ob, 'knots'), r_attr('knots'), rtol=1e-10, what='knots attribute')


@known_defect(THEME, 'basis-discrete-14', note="OneBasis has no 'cr' (cubic regression spline) function")
@pytest.mark.parametrize('kw', [dict(), dict(df=6), dict(df=6, intercept=True), dict(df=5, fx=True)], ids=_ids)
def test_onebasis_cr_matches_r(kw):
    """R cr(): mgcv smooth.construct.cr.smooth.spec at quantile knots, penalty S in the attributes."""
    _need_r_package('mgcv')
    x = _x_normal(150, seed=2)
    assert_onebasis_matches_r(x, 'cr', rtol=1e-8, **kw)
    ob = py_onebasis(x, 'cr', **kw)
    if kw.get('fx'):
        assert ob.attributes.get('S') is None
    else:
        assert_close(np.ravel(_attr_matrix(ob, 'S')), r_attr('S'), rtol=1e-8, what='penalty matrix S')


@known_defect(THEME, 'basis-discrete-14', note="OneBasis has no 'ps'/'cr': missing values, knot range, few distinct x")
@pytest.mark.parametrize('fun,kw,nan', [('ps', dict(df=6), True), ('ps', dict(df=6, knots=np.array([-10.0, 40.0])), False),
                                        ('cr', dict(df=6), True), ('cr', dict(df=6, intercept=True), True)], ids=str)
def test_onebasis_ps_cr_missing_values_and_knots_match_r(fun, kw, nan):
    """NaN rows are re-inserted after the fit (R: nax/nmat); ps accepts a two-element range as `knots`."""
    if fun == 'cr':
        _need_r_package('mgcv')
    x = _x_normal(150, seed=2)
    if nan:
        x[[5, 77]] = np.nan
    assert_onebasis_matches_r(x, fun, rtol=1e-8, **kw)


@known_defect(THEME, 'basis-discrete-14', note="OneBasis has no 'cr': fewer distinct x than knots (rows added for mgcv)")
def test_onebasis_cr_with_few_distinct_values_matches_r():
    _need_r_package('mgcv')
    x = np.random.default_rng(8).integers(0, 5, 120).astype(float)              # 5 distinct values < 7 knots
    assert_onebasis_matches_r(x, 'cr', rtol=1e-8, df=6)


@known_defect(THEME, 'basis-discrete-14', 'crossbasis-20', 'basis-discrete-10',
              note="OneBasis has no 'integer' function")
@pytest.mark.parametrize('kw', [dict(), dict(intercept=True), dict(values=np.arange(0.0, 8.0))], ids=_ids)
def test_onebasis_integer_matches_r(kw):
    """R integer(): one indicator column per level (sorted distinct values or `values`), first dropped unless intercept."""
    assert_onebasis_matches_r(_x_categorical(), 'integer', **kw)


@known_defect(THEME, 'basis-discrete-14', 'crossbasis-20', 'basis-discrete-10',
              note="integer basis: NaN rows and single level")
def test_onebasis_integer_missing_values_and_single_level():
    x = _x_categorical(60, seed=5, nlev=4)
    x[[3, 30]] = np.nan
    assert_onebasis_matches_r(x, 'integer')
    assert_onebasis_matches_r(np.full(20, 3.0), 'integer')                # one level: column kept, intercept forced


@pytest.mark.parametrize('fun', ['hthr', 'lthr', 'dthr', 'foo'])
def test_onebasis_unknown_function_names_raise_in_both(fun):
    """R (dlnm 2.4.10) has no such function any more and errors; PyDLNM raises ValueError (no silent fallback)."""
    x = _x_normal(50)
    with pytest.raises(RRuntimeError):
        r_onebasis(x, fun)
    with pytest.raises(ValueError):
        py_onebasis(x, fun)


# =====================================================================================================================
# CrossBasis with a strata / integer var or lag basis (crossbasis-18, crossbasis-20)
# =====================================================================================================================
@pytest.mark.parametrize('argvar,arglag,lag', [
    (dict(fun='strata', breaks=np.array([10.0, 20.0])), dict(fun='ns', df=3), 6),
    (dict(fun='ns', df=3), dict(fun='strata', breaks=np.array([1.0, 5.0, 10.0])), 15),
    (dict(fun='ns', df=3), dict(fun='strata', df=3), 15)], ids=lambda v: str(v))
def test_crossbasis_strata_var_or_lag_basis_matches_r(argvar, arglag, lag):
    """R builds crossbasis() from strata() columns of onebasis(); fixed with theme B (CrossBasis uses the marginal
    bases instead of recomputing bs/ns)."""
    assert_crossbasis_matches_r(_temp(400), lag, argvar, arglag)


def test_crossbasis_strata_df_var_basis_matches_r():
    assert_crossbasis_matches_r(_temp(400), 6, dict(fun='strata', df=3), dict(fun='ns', df=3))


@known_defect(THEME, 'crossbasis-20', note="argvar fun='integer' is unknown to OneBasis / the time-series path")
@pytest.mark.parametrize('argvar,arglag,lag', [
    (dict(fun='integer'), dict(fun='integer'), 3),
    (dict(fun='integer'), dict(fun='ns', df=3), 3),
    (dict(fun='integer', intercept=True), dict(fun='ns', df=3), 3)], ids=lambda v: str(v))
def test_crossbasis_integer_var_basis_matches_r(argvar, arglag, lag):
    """R: one indicator column per distinct exposure value (categorical exposure), crossed with the lag basis."""
    assert_crossbasis_matches_r(_x_categorical(200, seed=1, nlev=5), lag, argvar, arglag)


@pytest.mark.parametrize('lag', [3, [1, 4]])
def test_crossbasis_integer_lag_basis_matches_r(lag):
    """Integer lag basis (the Europe validation configuration) with an ns var basis: exact today."""
    assert_crossbasis_matches_r(_temp(500), lag, dict(fun='ns', knots=np.array([5.0, 15.0, 25.0])),
                                dict(fun='integer'), rtol=1e-10)
