"""GAP masked_arrays_mask_ignored: numpy masked arrays lose their mask at every `np.asarray(x, dtype=float)` entry point.

Numpy masked arrays (what netCDF4 / xarray `.to_masked_array()` / `np.ma.masked_where` / `np.ma.masked_equal` return for
gridded climate exposures and for outcomes with missing cells) carry the missing cells in a separate boolean mask; the
raw `.data` under the mask is a fill value (9.96921e36, -9999, 1e20) or an arbitrary genuine-looking number.
`np.asarray(x, dtype=float)` returns that raw data and silently discards the mask, so the masked cells become REAL data:
the range, the knots, the poly scale, the whole basis, the lag windows, the prediction grid, the attributable numbers,
the meta-analysis and the GLM all change, with no NaN and no warning.

R has no masked arrays; the R meaning of a masked cell is NA.  Every test therefore builds the same data NA-coded,
computes the reference in R (dlnm crossbasis / onebasis / crosspred / crossreduce, Gasparrini's attrdl.R, mvmeta(),
glm(na.action = na.exclude), tsModel::Lag, dlnm::exphist, equalknots) at run time, and demands that PyDLNM fed the
MASKED array gives that result.  Where R itself stops on the NA-coded input (NA coefficients) the masked array must be
rejected; a PyDLNM that refuses a masked array with an error that names the mask (TypeError / ValueError) is also
accepted for the tests that need a value, so that the two possible fixes (fill the mask with NaN, or refuse masked
arrays) both satisfy them.  The fill variants exercise three situations:
  netcdf     fill value 9.96921e36 under the mask (netCDF4 default): range, knots and scale explode
  minus9999  fill value -9999 under the mask
  kept       the genuine value stays under the mask (`np.ma.masked_array(data, mask)`): range/knots unchanged, but the
             masked rows are used as if observed instead of being NaN

Plain (unmarked) tests guard what is faithful today and must survive the fix: a masked array without masked entries is
the plain array, `np.ma.masked_invalid` (NaN retained under the mask) equals the NaN-coded series, and PyDLNM never
modifies the caller's masked array.

Entry points covered: OneBasis, CrossBasis (series and lag matrix), crosspred (`at` vector / matrix, coef, vcov, basis
from a masked series), crossreduce (`at`, coef), attrdl (x, cases, coef; both directions), attr_heat_cold,
attr_by_percentiles, MVMeta.fit (y, S, X), Rpy2GLMInterface.fit_glm / rpy2_glm.as_vector (response, covariates, weights,
offset), utils.lagmatrix / exphist / equalknots and centering.find_mmt_blup.

Status: fixed.  Every entry point reads user data through utils.asfloat (masked cells and nullable pandas values
-> NaN, everything else np.asarray(dtype=float)); the tests below used to be strict xfails (@DEFECT) and now pass.
"""
import contextlib
import copy
import io
import itertools
import os
import warnings

import numpy as np
import pandas as pd
import pytest
import rpy2.robjects as ro

from rhelpers import REPO, assert_close, chicago, np2r, r, r2np, rget

PFX = 'gmm_'                 # prefix of every object this module creates in R's global environment
ATTRDL_R = REPO / '2015_gasparrini_Lancet_Rcodedata-master' / 'attrdl.R'

pytestmark = pytest.mark.filterwarnings('ignore')


# --------------------------------------------------------------------------------------------------------------
# environment: PyDLNM imports rewrite R_HOME; R then segfaults when it lazily loads LAPACK.  Run every test with the
# home of the R that is actually running (same guard as the other modules).
# --------------------------------------------------------------------------------------------------------------
def _running_r_home():
    p = str(r('as.character(getLoadedDLLs()[["utils"]][["path"]])')[0])
    for _ in range(4):
        p = os.path.dirname(p)
    return p


@contextlib.contextmanager
def _correct_r_home():
    before = os.environ.get('R_HOME')
    os.environ['R_HOME'] = _running_r_home()
    try:
        yield
    finally:
        if before is None:
            os.environ.pop('R_HOME', None)
        else:
            os.environ['R_HOME'] = before


@pytest.fixture(autouse=True)
def _r_home_guard():
    with _correct_r_home():
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


def require_r(*pkgs):
    """Skip if an R package is missing, else load it (rhelpers.require_r_packages breaks on rpy2 3.6)."""
    for p in pkgs:
        if not bool(r(f'isTRUE(suppressWarnings(requireNamespace("{p}", quietly=TRUE)))')[0]):
            pytest.skip(f'R package {p} not installed')
        r(f'suppressMessages(library({p}))')


# --------------------------------------------------------------------------------------------------------------
# masked / NaN-coded data builders (identical cells masked in both)
# --------------------------------------------------------------------------------------------------------------
NETCDF_FILL = 9.96921e36
FILLS = {'netcdf': NETCDF_FILL, 'minus9999': -9999.0, 'kept': None}     # None: the genuine value stays under the mask
REJECTED = 'rejected'


def mask_at(shape, positions):
    m = np.zeros(shape, dtype=bool)
    m[tuple(np.array(positions).T)] = True
    return m


def masked_like(data, mask, fill):
    """np.ma.masked_array of `data` with `mask` masked; under the mask the `fill` value (None: the real value)."""
    data = np.array(data, dtype=float)
    raw = data.copy() if fill is None else np.where(mask, fill, data)
    return np.ma.masked_array(raw, mask=np.array(mask, dtype=bool), copy=True)


def nan_coded(data, mask):
    return np.where(mask, np.nan, np.array(data, dtype=float))


def accept_or_reject(f):
    """f(), or REJECTED if PyDLNM refuses the masked input with an error that names the mask."""
    try:
        return f()
    except (TypeError, ValueError) as exc:
        if 'mask' in str(exc).lower():
            return REJECTED
        raise


def quiet(f, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with contextlib.redirect_stdout(io.StringIO()):
            return f(*args, **kwargs)


def r_error(code):
    """None if the R code runs, else the first line of R's error message."""
    try:
        r(code)
    except Exception as e:                                          # rpy2 RRuntimeError
        return str(e).strip().splitlines()[0]
    return None


_pushed = itertools.count()


def _rarg(v):
    """Python value -> R literal; arrays and lists are pushed into the R workspace (no decimal round trip)."""
    if isinstance(v, (bool, np.bool_)):
        return 'TRUE' if v else 'FALSE'
    if isinstance(v, str):
        return f'"{v}"'
    if isinstance(v, (int, float, np.integer, np.floating)):
        return repr(float(v))
    name = f'{PFX}pushed{next(_pushed)}'
    np2r(name, np.atleast_1d(np.asarray(v, dtype=float)))
    return name


_RNAME = {'Boundary_knots': 'Boundary.knots', 'thr_value': 'thr.value'}


def rargs(d):
    return ', '.join(f'{_RNAME.get(k, k)}={_rarg(v)}' for k, v in d.items())


def rlist(d):
    return f'list({rargs(d)})'


def rlag(lag):
    return f'c({", ".join(repr(float(v)) for v in lag)})' if isinstance(lag, (list, tuple)) else repr(float(lag))


def series(n, positions, start=0):
    """Chicago temperature / deaths (first n days from `start`) and the mask of `positions`."""
    ch = chicago()
    return ch['temp'][start:start + n].copy(), ch['death'][start:start + n].copy(), mask_at(n, [[p] for p in positions])


def rquantile(x_na, probs):
    np2r(PFX + 'xq', x_na)
    return rget(f'quantile({PFX}xq, c({", ".join(repr(float(p)) for p in probs)}), na.rm=TRUE)')


# ==============================================================================================================
# 1. OneBasis
# ==============================================================================================================
ONEBASIS_SPECS = {
    'lin': ('lin', {}),
    'ns_df4': ('ns', {'df': 4}),
    'bs_df5_deg3': ('bs', {'df': 5, 'degree': 3}),
    'poly3': ('poly', {'degree': 3}),
    'thr15': ('thr', {'thr_value': 15.0}),
    'strata_breaks': ('strata', {'breaks': [5.0, 15.0, 22.0]}),
}


def r_onebasis(x_na, fun, spec):
    np2r(PFX + 'x', x_na)
    r(f'{PFX}ob <- onebasis({PFX}x, fun="{fun}"' + (', ' + rargs(spec) if spec else '') + ')')
    out = dict(mat=r2np(r(f'unclass({PFX}ob)')), range=rget(f'attr({PFX}ob, "range")'))
    if bool(r(f'!is.null(attr({PFX}ob, "knots"))')[0]):
        out['knots'] = rget(f'as.numeric(attr({PFX}ob, "knots"))')
    return out


def assert_onebasis_matches(ob, ref, what):
    assert_close(np.asarray(ob.basis), ref['mat'], rtol=1e-10, what=f'{what}: basis')
    assert_close(np.asarray(ob.range), ref['range'], rtol=1e-12, what=f'{what}: range')
    if 'knots' in ref and ob.attributes.get('knots') is not None:
        assert_close(np.atleast_1d(ob.attributes['knots']), ref['knots'], rtol=1e-10, what=f'{what}: knots')


@pytest.mark.parametrize('fill', list(FILLS))
@pytest.mark.parametrize('name', list(ONEBASIS_SPECS))
def test_onebasis_masked_x_equals_r_na(name, fill):
    """R onebasis() of the NA-coded series: NaN rows, range and the data-driven knots ignore the masked cells."""
    from basis import OneBasis
    fun, spec = ONEBASIS_SPECS[name]
    x, _, mask = series(300, [10, 120, 250])
    ref = r_onebasis(nan_coded(x, mask), fun, spec)
    ob = accept_or_reject(lambda: OneBasis(masked_like(x, mask, FILLS[fill]), fun, **copy.deepcopy(spec)))
    if ob is REJECTED:
        return
    assert_onebasis_matches(ob, ref, f'{name}/{fill}')


@pytest.mark.parametrize('fill', ['netcdf', 'kept'])
def test_onebasis_masked_matrix_is_flattened_column_wise_with_na(fill):
    """R: x <- as.vector(x) flattens a matrix column by column; masked cells are NA."""
    from basis import OneBasis
    x, _, _ = series(400, [])
    x2 = x.reshape(200, 2, order='F')
    mask = mask_at(x2.shape, [[5, 0], [60, 1], [150, 1]])
    ref = r_onebasis(nan_coded(x2, mask).ravel(order='F'), 'ns', {'df': 3})
    ob = accept_or_reject(lambda: OneBasis(masked_like(x2, mask, FILLS[fill]), 'ns', df=3))
    if ob is REJECTED:
        return
    assert_onebasis_matches(ob, ref, f'matrix/{fill}')


# ==============================================================================================================
# 2. CrossBasis
# ==============================================================================================================
def _cb_specs(x_na):
    k = rquantile(x_na, [.10, .75, .90])
    return {
        'ns4_ns3_lag5': dict(lag=5, argvar={'fun': 'ns', 'df': 4}, arglag={'fun': 'ns', 'df': 3}),
        'bs2_knots_poly_lag7': dict(lag=7, argvar={'fun': 'bs', 'degree': 2, 'knots': k},
                                    arglag={'fun': 'poly', 'degree': 3}),
        'lin_ns_lag2_6': dict(lag=[2, 6], argvar={'fun': 'lin'}, arglag={'fun': 'ns', 'df': 2}),
        'thr_ns_lag4': dict(lag=4, argvar={'fun': 'thr', 'thr_value': 15.0}, arglag={'fun': 'ns', 'df': 2}),
        'poly3_poly2_lag3': dict(lag=3, argvar={'fun': 'poly', 'degree': 3}, arglag={'fun': 'poly', 'degree': 2}),
    }


CB_SPEC_NAMES = ['ns4_ns3_lag5', 'bs2_knots_poly_lag7', 'lin_ns_lag2_6', 'thr_ns_lag4', 'poly3_poly2_lag3']


def r_crossbasis(x_na, spec, name=PFX + 'cb'):
    np2r(PFX + 'cx', x_na)
    r(f'{name} <- crossbasis({PFX}cx, lag={rlag(spec["lag"])}, argvar={rlist(spec["argvar"])}, '
      f'arglag={rlist(spec["arglag"])})')
    out = dict(mat=r2np(r(f'unclass({name})')), df=rget(f'attr({name}, "df")'), range=rget(f'attr({name}, "range")'),
               lag=rget(f'attr({name}, "lag")'))
    if bool(r(f'!is.null(attr({name}, "argvar")$knots)')[0]):
        out['knots'] = rget(f'as.numeric(attr({name}, "argvar")$knots)')
    return out


def py_crossbasis(x, spec):
    from basis import CrossBasis
    return quiet(CrossBasis, x, lag=copy.deepcopy(spec['lag']), argvar=copy.deepcopy(spec['argvar']),
                 arglag=copy.deepcopy(spec['arglag']))


def assert_crossbasis_matches(cb, ref, what):
    assert_close(np.asarray(cb.basis), ref['mat'], rtol=1e-10, what=f'{what}: cross-basis')
    assert_close(np.asarray(cb.range, dtype=float), ref['range'], rtol=1e-12, what=f'{what}: range')
    assert tuple(int(d) for d in cb.df) == tuple(int(d) for d in ref['df']), f'{what}: df {cb.df} vs R {ref["df"]}'
    if 'knots' in ref and cb.argvar.get('knots') is not None:
        assert_close(np.atleast_1d(cb.argvar['knots']), ref['knots'], rtol=1e-10, what=f'{what}: resolved knots')


@pytest.mark.parametrize('fill', list(FILLS))
@pytest.mark.parametrize('name', CB_SPEC_NAMES)
def test_crossbasis_masked_series_equals_r_na(name, fill):
    """crossbasis() of the NA-coded series: the masked days and the lag windows that contain them are NaN rows, the
    range and the data-driven knots come from the observed days only."""
    x, _, mask = series(400, [50, 120, 300])
    x_na = nan_coded(x, mask)
    spec = _cb_specs(x_na)[name]
    ref = r_crossbasis(x_na, spec)
    cb = accept_or_reject(lambda: py_crossbasis(masked_like(x, mask, FILLS[fill]), spec))
    if cb is REJECTED:
        return
    assert_crossbasis_matches(cb, ref, f'{name}/{fill}')


@pytest.mark.parametrize('fill', ['netcdf', 'minus9999', 'kept'])
def test_crossbasis_masked_lag_matrix_equals_r_na(fill):
    """x given as a matrix of exposure histories (n x (lag+1)), as for gridded / pre-lagged exposures."""
    from utils import lagmatrix
    x, _, _ = series(300, [])
    hist = lagmatrix(x, np.arange(4))
    mask = mask_at(hist.shape, [[50, 1], [51, 1], [100, 3], [200, 0]])
    hist_na = nan_coded(hist, mask)
    spec = dict(lag=3, argvar={'fun': 'ns', 'df': 3}, arglag={'fun': 'ns', 'df': 2})
    ref = r_crossbasis(hist_na, spec)
    cb = accept_or_reject(lambda: py_crossbasis(masked_like(hist, mask, FILLS[fill]), spec))
    if cb is REJECTED:
        return
    assert_crossbasis_matches(cb, ref, f'lag matrix/{fill}')


# ==============================================================================================================
# 3. crosspred / crossreduce
# ==============================================================================================================
CEN = 15.0
AT_GRID = np.array([-5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 28.0])


@pytest.fixture(scope='module')
def pcase():
    """A cross-basis on a NaN-coded series built in R and PyDLNM, with fixed coefficients on both sides."""
    with _correct_r_home():
        x, _, mask = series(500, [40, 210, 333])
        x_na = nan_coded(x, mask)
        spec = dict(lag=4, argvar={'fun': 'bs', 'degree': 2, 'knots': rquantile(x_na, [.25, .5, .75])},
                    arglag={'fun': 'ns', 'df': 3})
        r_crossbasis(x_na, spec, name=PFX + 'pcb')
        cb = py_crossbasis(x_na, spec)
        k = cb.shape[1]
        coef = np.random.default_rng(3).normal(0, .03, k)
        vcov = np.eye(k) * 1e-4
        np2r(PFX + 'pcoef', coef)
        np2r(PFX + 'pvcov', vcov)
        r(f'{PFX}pcoef <- as.numeric({PFX}pcoef)')
        return dict(x=x, mask=mask, x_na=x_na, spec=spec, cb=cb, coef=coef, vcov=vcov, k=k)


def r_crosspred(at_na=None, cen=CEN, basis=PFX + 'pcb', coef=PFX + 'pcoef'):
    at_arg = ''
    if at_na is not None:
        np2r(PFX + 'pat', at_na)
        at_arg = f', at={PFX}pat'
    r(f'{PFX}cp <- crosspred({basis}, coef={coef}, vcov={PFX}pvcov, model.link="log"{at_arg}, cen={cen!r})')
    return {k: r2np(r(f'{PFX}cp${k}')) for k in ('predvar', 'allfit', 'allse', 'matfit', 'matse')}


def assert_pred_matches(pred, ref, what, rtol=1e-8):
    assert_close(np.asarray(pred.predvar, dtype=float), ref['predvar'], rtol=rtol, what=f'{what}: predvar')
    for k in ('allfit', 'allse', 'matfit', 'matse'):
        assert_close(np.asarray(getattr(pred, k)), ref[k], rtol=rtol, what=f'{what}: {k}')


@pytest.mark.parametrize('fill', ['netcdf', 'minus9999', 'kept'])
@pytest.mark.parametrize('kind', ['cross', 'one'])
def test_crosspred_masked_at_vector_drops_the_cells_like_r_na(pcase, kind, fill):
    """R mkat(): at <- sort(unique(at)) drops NA.  PyDLNM uses the fill value as an exposure to predict at."""
    from basis import OneBasis
    from prediction import crosspred
    at = AT_GRID.copy()
    mask = mask_at(at.shape, [[2], [5]])
    at_na, at_m = nan_coded(at, mask), masked_like(at, mask, FILLS[fill])
    if kind == 'cross':
        ref = r_crosspred(at_na)
        basis, coef, vcov = pcase['cb'], pcase['coef'], pcase['vcov']
    else:
        ob = OneBasis(pcase['x_na'], 'ns', df=4)
        np2r(PFX + 'ox', pcase['x_na'])
        r(f'{PFX}ob1 <- onebasis({PFX}ox, fun="ns", df=4)')
        np2r(PFX + 'pcoef1', pcase['coef'][:4])
        r(f'{PFX}pcoef1 <- as.numeric({PFX}pcoef1)')
        np2r(PFX + 'pvcov1', np.eye(4) * 1e-4)
        r(f'{PFX}cp <- crosspred({PFX}ob1, coef={PFX}pcoef1, vcov={PFX}pvcov1, model.link="log", at={_rarg(at_na)}, '
          f'cen={CEN!r})')
        ref = {k: r2np(r(f'{PFX}cp${k}')) for k in ('predvar', 'allfit', 'allse', 'matfit', 'matse')}
        basis, coef, vcov = ob, pcase['coef'][:4], np.eye(4) * 1e-4
    pred = accept_or_reject(lambda: quiet(crosspred, basis, coef=coef, vcov=vcov, model_link='log', at=at_m, cen=CEN))
    if pred is REJECTED:
        return
    assert_pred_matches(pred, ref, f'at vector/{kind}/{fill}')


@pytest.mark.parametrize('fill', ['minus9999', 'kept'])
def test_crosspred_masked_at_matrix_gives_nan_rows_like_r_na(pcase, fill):
    """`at` as a matrix of exposure histories (n x n_lags): R keeps an NA cell as NA in matfit / allfit."""
    from prediction import crosspred
    n_lag = 5
    at = np.tile(np.array([0.0, 5.0, 10.0, 15.0, 20.0, 25.0]).reshape(-1, 1), (1, n_lag))
    mask = mask_at(at.shape, [[2, 1], [4, 3]])
    ref = r_crosspred(nan_coded(at, mask))
    pred = accept_or_reject(lambda: quiet(crosspred, pcase['cb'], coef=pcase['coef'], vcov=pcase['vcov'],
                                          model_link='log', at=masked_like(at, mask, FILLS[fill]), cen=CEN))
    if pred is REJECTED:
        return
    assert_pred_matches(pred, ref, f'at matrix/{fill}')


def test_crosspred_default_grid_of_a_basis_built_from_a_masked_fill_series(pcase):
    """End to end: CrossBasis(masked series) -> crosspred(): the default grid spans the observed range (R: range of
    the non-missing days), so predvar, allfit and matfit equal R's for the NA-coded series.  The netCDF fill value
    under the mask stretches the range (and the default prediction grid) to 9.96921e36."""
    _default_grid_case(pcase, 'netcdf')


def test_crosspred_default_grid_with_genuine_values_under_the_mask_is_unaffected(pcase):
    """Guard: crosspred reads only the range / resolved arguments of the basis, so genuine values under the mask
    (none extreme) give R's grid and predictions today."""
    _default_grid_case(pcase, 'kept')


def _default_grid_case(pcase, fill):
    from prediction import crosspred
    ref = r_crosspred()
    x_m = masked_like(pcase['x'], pcase['mask'], FILLS[fill])
    pred = accept_or_reject(lambda: quiet(crosspred, py_crossbasis(x_m, pcase['spec']), coef=pcase['coef'],
                                          vcov=pcase['vcov'], model_link='log', cen=CEN))
    if pred is REJECTED:
        return
    assert_pred_matches(pred, ref, f'default grid/{fill}')


@pytest.mark.parametrize('which', ['coef', 'vcov'])
def test_crosspred_masked_coefficient_or_vcov_entry_is_rejected_like_na(pcase, which):
    """R: any(is.na(coef)) || any(is.na(vcov)) -> error 'coef/vcov not consistent'.  A masked coefficient carries a
    perfectly valid-looking number under the mask, so PyDLNM predicts with it."""
    from prediction import crosspred
    k = pcase['k']
    coef, vcov = pcase['coef'].copy(), pcase['vcov'].copy()
    if which == 'coef':
        bad_r = f'crosspred({PFX}pcb, coef=c({PFX}pcoef[-{k}], NA), vcov={PFX}pvcov, model.link="log", cen=15)'
        coef_m = np.ma.masked_array(coef, mask=mask_at(k, [[k - 1]]))
        call = lambda: quiet(crosspred, pcase['cb'], coef=coef_m, vcov=vcov, model_link='log', cen=CEN)
    else:
        bad_r = (f'crosspred({PFX}pcb, coef={PFX}pcoef, vcov=replace({PFX}pvcov, 1, NA), model.link="log", cen=15)')
        vcov_m = np.ma.masked_array(vcov, mask=mask_at(vcov.shape, [[0, 0]]))
        call = lambda: quiet(crosspred, pcase['cb'], coef=coef, vcov=vcov_m, model_link='log', cen=CEN)
    assert r_error(bad_r) is not None, 'R must stop on an NA coefficient / vcov entry'
    with pytest.raises((ValueError, TypeError)):
        call()


@pytest.mark.parametrize('fill', ['netcdf', 'kept'])
def test_crossreduce_masked_at_equals_r_na(pcase, fill):
    """crossreduce(at=<vector with masked cells>): R's mkat drops NA; fit / se / reduced coefficients follow."""
    from crossreduce import crossreduce
    at = AT_GRID.copy()
    mask = mask_at(at.shape, [[2], [5]])
    np2r(PFX + 'rat', nan_coded(at, mask))
    r(f'{PFX}cr <- crossreduce({PFX}pcb, coef={PFX}pcoef, vcov={PFX}pvcov, model.link="log", type="overall", '
      f'at={PFX}rat, cen={CEN!r})')
    ref = {k: r2np(r(f'{PFX}cr${k}')) for k in ('fit', 'se', 'coefficients')}
    res = accept_or_reject(lambda: quiet(crossreduce, pcase['cb'], coef=pcase['coef'], vcov=pcase['vcov'],
                                         model_link='log', type='overall', cen=CEN,
                                         at=masked_like(at, mask, FILLS[fill])))
    if res is REJECTED:
        return
    assert_close(np.asarray(res.fit), ref['fit'], rtol=1e-8, what='crossreduce fit')
    assert_close(np.asarray(res.se), ref['se'], rtol=1e-8, what='crossreduce se')
    assert_close(np.asarray(res.coef), ref['coefficients'], rtol=1e-8, what='crossreduce coef')


def test_crossreduce_masked_coefficient_is_rejected_like_na(pcase):
    from crossreduce import crossreduce
    k = pcase['k']
    assert r_error(f'crossreduce({PFX}pcb, coef=c({PFX}pcoef[-{k}], NA), vcov={PFX}pvcov, model.link="log", cen=15)') \
        is not None
    coef_m = np.ma.masked_array(pcase['coef'], mask=mask_at(k, [[k - 1]]))
    with pytest.raises((ValueError, TypeError)):
        quiet(crossreduce, pcase['cb'], coef=coef_m, vcov=pcase['vcov'], model_link='log', cen=CEN)


# ==============================================================================================================
# 4. attrdl and the attr_* wrappers
# ==============================================================================================================
class ImprovedGLMInterface:
    """Stand-in for a fitted PyDLNM GLM wrapper (same class name => getcoef / getvcov / getlink work, link 'log')."""

    def __init__(self, coef, vcov):
        self.cb_coef = np.asarray(coef, dtype=float)
        self.cb_vcov = np.asarray(vcov, dtype=float)


def _load_r_attrdl():
    require_r('tsModel')
    if not bool(r(f'exists("{PFX}env")')[0]):
        ro.globalenv[PFX + 'src'] = str(ATTRDL_R)
        r(f'{PFX}env <- new.env(); sys.source({PFX}src, envir={PFX}env)')


@pytest.fixture(scope='module')
def acase():
    """Series with missing days (NaN-coded), its cross-basis (R and PyDLNM) and fixed coefficients."""
    with _correct_r_home():
        x, cases, xmask = series(600, [100, 250, 251, 420])
        cmask = mask_at(600, [[30], [150], [380]])
        x_na, cases_na = nan_coded(x, xmask), nan_coded(cases, cmask)
        spec = dict(lag=4, argvar={'fun': 'ns', 'df': 3}, arglag={'fun': 'ns', 'df': 3})
        r_crossbasis(x_na, spec, name=PFX + 'acb')
        cb = py_crossbasis(x_na, spec)
        k = cb.shape[1]
        coef = np.random.default_rng(7).normal(0, .04, k)
        vcov = np.eye(k) * 1e-4
        np2r(PFX + 'acoef', coef)
        np2r(PFX + 'avcov', vcov)
        r(f'{PFX}acoef <- as.numeric({PFX}acoef)')
        return dict(x=x, cases=cases, xmask=xmask, cmask=cmask, x_na=x_na, cases_na=cases_na, spec=spec, cb=cb,
                    coef=coef, vcov=vcov, k=k)


def r_attrdl(ac, x_na, cases_na, type='an', dir='forw', tot=True, rng=None, coef=None):
    _load_r_attrdl()
    np2r(PFX + 'ax', x_na)
    np2r(PFX + 'ac', cases_na)
    r(f'{PFX}ax <- as.numeric({PFX}ax); {PFX}ac <- as.numeric({PFX}ac)')
    args = [f'{PFX}ax', f'{PFX}acb', f'{PFX}ac', f'coef={coef or PFX + "acoef"}', f'vcov={PFX}avcov',
            f'type="{type}"', f'dir="{dir}"', f'tot={"TRUE" if tot else "FALSE"}', f'cen={CEN!r}']
    if rng is not None:
        np2r(PFX + 'arng', np.array(rng, dtype=float))          # pushed, not printed: R's parser can be 1 ulp off
        args.append(f'range=as.numeric({PFX}arng)')
    return np.atleast_1d(rget(f'{PFX}env$attrdl(' + ', '.join(args) + ')')).astype(float)


def py_attrdl(ac, x, cases, tot=True, **kw):
    import attribution
    return quiet(attribution.attrdl, x, ac['cb'], cases, coef=ac['coef'], vcov=ac['vcov'], cen=CEN, tot=tot, **kw)


def _attr_out(res, type, tot):
    return np.atleast_1d(np.asarray(res[type + '_total'] if tot else res[type], dtype=float))


@pytest.mark.parametrize('fill', ['netcdf', 'kept'])
@pytest.mark.parametrize('tot', [True, False], ids=['total', 'per-obs'])
@pytest.mark.parametrize('dir', ['forw', 'back'])
@pytest.mark.parametrize('which', ['x', 'cases', 'both'])
def test_attrdl_masked_exposure_and_cases_equal_r_na(acase, which, dir, tot, fill):
    """attrdl(x, basis, cases): a masked exposure day makes every window that contains it NA in R; a masked case count
    is NA in the (forward moving average of the) cases.  PyDLNM attributes real numbers to them."""
    ac = acase
    x_arg = masked_like(ac['x'], ac['xmask'], FILLS[fill]) if which in ('x', 'both') else ac['x']
    c_arg = masked_like(ac['cases'], ac['cmask'], FILLS[fill]) if which in ('cases', 'both') else ac['cases']
    x_ref = ac['x_na'] if which in ('x', 'both') else ac['x']
    c_ref = ac['cases_na'] if which in ('cases', 'both') else ac['cases']
    ref = r_attrdl(ac, x_ref, c_ref, type='an', dir=dir, tot=tot)
    res = accept_or_reject(lambda: py_attrdl(ac, x_arg, c_arg, tot=tot, type='an', dir=dir))
    if res is REJECTED:
        return
    assert_close(_attr_out(res, 'an', tot), ref, rtol=1e-8, what=f'attrdl an {which}/{dir}/tot={tot}/{fill}')


@pytest.mark.parametrize('dir', ['forw', 'back'])
def test_attrdl_af_with_masked_exposure_and_cases_equals_r_na(acase, dir):
    ac = acase
    ref = r_attrdl(ac, ac['x_na'], ac['cases_na'], type='af', dir=dir, tot=True)
    res = accept_or_reject(lambda: py_attrdl(
        ac, masked_like(ac['x'], ac['xmask'], NETCDF_FILL), masked_like(ac['cases'], ac['cmask'], None),
        type='af', dir=dir))
    if res is REJECTED:
        return
    assert_close(_attr_out(res, 'af', True), ref, rtol=1e-8, what=f'attrdl af total {dir}')


def test_attrdl_with_range_and_masked_exposure_equals_r_na(acase):
    """range=(lo, hi): exposure outside it is set to the centering value; a masked day must stay NA, not be 'outside'
    (the fill value 9.96921e36 is outside every range and would silently become the null-risk exposure)."""
    ac = acase
    rng = (round(float(np.nanquantile(ac['x_na'], .6)) + 0.0123, 4), 100.0)       # not equal to an observed value
    ref = r_attrdl(ac, ac['x_na'], ac['cases'], type='an', dir='forw', tot=False, rng=rng)
    res = accept_or_reject(lambda: py_attrdl(ac, masked_like(ac['x'], ac['xmask'], NETCDF_FILL), ac['cases'],
                                             tot=False, type='an', dir='forw', range=rng))
    if res is REJECTED:
        return
    assert_close(_attr_out(res, 'an', False), ref, rtol=1e-8, what='attrdl range + masked exposure')


def test_attrdl_masked_coefficient_gives_na_like_r(acase):
    """R attrdl with an NA coefficient returns NA; a masked coefficient (valid number under the mask) gives a number."""
    ac = acase
    k = ac['k']
    coef_na = ac['coef'].copy()
    coef_na[2] = np.nan
    np2r(PFX + 'acoef_na', coef_na)
    r(f'{PFX}acoef_na <- as.numeric({PFX}acoef_na)')
    ref = r_attrdl(ac, ac['x'], ac['cases'], coef=PFX + 'acoef_na')
    assert np.isnan(ref).all(), 'R: an NA coefficient gives an NA attributable number'
    coef_m = np.ma.masked_array(ac['coef'], mask=mask_at(k, [[2]]))
    import attribution
    res = accept_or_reject(lambda: quiet(attribution.attrdl, ac['x'], ac['cb'], ac['cases'], coef=coef_m,
                                         vcov=ac['vcov'], cen=CEN, type='an', dir='forw'))
    if res is REJECTED:
        return
    assert np.isnan(_attr_out(res, 'an', True)).all(), f'masked coefficient gave {_attr_out(res, "an", True)}'


@pytest.mark.parametrize('split', ['cen', 'percentile'])
def test_attr_heat_cold_masked_exposure_and_cases_equal_r_na(acase, split):
    """attr_heat_cold on masked input: the thresholds (percentiles of the OBSERVED exposure) and both totals equal R's
    attrdl with range = (-Inf, threshold) / (threshold, Inf)."""
    import attribution
    ac = acase
    if split == 'cen':
        lo = hi = CEN
    else:
        lo, hi = (float(v) for v in rquantile(ac['x_na'], [.025, .975]))
    cold = r_attrdl(ac, ac['x_na'], ac['cases_na'], type='an', dir='forw', tot=True, rng=(-np.inf, lo))
    heat = r_attrdl(ac, ac['x_na'], ac['cases_na'], type='an', dir='forw', tot=True, rng=(hi, np.inf))
    res = accept_or_reject(lambda: quiet(
        attribution.attr_heat_cold, masked_like(ac['x'], ac['xmask'], NETCDF_FILL), ac['cb'],
        masked_like(ac['cases'], ac['cmask'], None), coef=ac['coef'], vcov=ac['vcov'], cen=CEN, split=split))
    if res is REJECTED:
        return
    assert_close(np.array([res['cold']['threshold'], res['heat']['threshold']]), np.array([lo, hi]), rtol=1e-10,
                 what='thresholds')
    assert_close(np.array([res['summary']['cold_an_total']]), cold, rtol=1e-8, what='cold AN total')
    assert_close(np.array([res['summary']['heat_an_total']]), heat, rtol=1e-8, what='heat AN total')


def test_attr_by_percentiles_masked_exposure_equals_r_na(acase):
    """Bins [p_low, p_high) of the observed exposure: the thresholds are R's quantile(na.rm=TRUE), the totals are R
    attrdl over the bin (x == the upper threshold excluded, here by subtracting 1e-9)."""
    import attribution
    ac = acase
    bins = [(1, 5), (90, 95), (99, 100)]
    res = accept_or_reject(lambda: quiet(
        attribution.attr_by_percentiles, masked_like(ac['x'], ac['xmask'], NETCDF_FILL), ac['cb'], ac['cases'],
        coef=ac['coef'], vcov=ac['vcov'], percentile_ranges=bins, cen=CEN))
    if res is REJECTED:
        return
    for lo_p, hi_p in bins:
        lo, hi = (float(v) for v in rquantile(ac['x_na'], [lo_p / 100, hi_p / 100]))
        upper = hi if hi_p >= 100 else hi - 1e-9
        ref = r_attrdl(ac, ac['x_na'], ac['cases'], type='an', dir='forw', tot=True, rng=(lo, upper))
        entry = res[f'pct_{lo_p}_{hi_p}']
        assert_close(np.asarray(entry['thresholds'], dtype=float), np.array([lo, hi]), rtol=1e-10,
                     what=f'thresholds {lo_p}-{hi_p}')
        assert_close(np.array([entry['results']['an_total']]), ref, rtol=1e-8, what=f'AN total bin {lo_p}-{hi_p}')


# ==============================================================================================================
# 5. MVMeta.fit
# ==============================================================================================================
_R_DEFS = f'''
{PFX}formula_fit <- function(y, Sv, X, method) {{
  df <- data.frame(id = seq_len(nrow(y))); df$y <- y; df$Sv <- Sv; df$X <- X
  mvmeta(y ~ X - 1, S = Sv, data = df, method = method,
         control = list(maxiter = 20000, reltol = 1e-14))
}}
'''


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
    n, k, _ = S.shape
    idx = [(a, b) for b in range(k) for a in range(b, k)]
    return np.array([[S[i, a, b] for (a, b) in idx] for i in range(n)])


MISSING_OUTCOMES = [(3, 1), (7, 1), (12, 1), (5, 0)]


def r_mvmeta(y_na, S_na, X_na, method='reml'):
    """R mvmeta() (formula interface, X includes the intercept) on the NA-coded data."""
    require_r('mvmeta')
    r(_R_DEFS)
    np2r(PFX + 'y', y_na)
    np2r(PFX + 'Sv', vech_rows(S_na))
    np2r(PFX + 'X', X_na)
    n, p, k = y_na.shape[0], X_na.shape[1], y_na.shape[1]
    r(f'{PFX}mv <- suppressWarnings({PFX}formula_fit(as.matrix({PFX}y), as.matrix({PFX}Sv), as.matrix({PFX}X), '
      f'"{method}"))')
    return dict(coef=np.asarray(r2np(r(f'{PFX}mv$coefficients'))).reshape(p, k), vcov=r2np(r(f'{PFX}mv$vcov')),
                psi=r2np(r(f'{PFX}mv$Psi')), loglik=float(r2np(r(f'as.numeric(logLik({PFX}mv))')).ravel()[0]),
                n_used=int(r(f'nrow({PFX}mv$model)')[0]))


def py_mvmeta(y, S, X, method='reml'):
    from meta_analysis import MVMeta
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return MVMeta(method=method).fit(y, S, X)


def assert_mvmeta_matches(m, ref, what, rtol=1e-6):
    assert m.n == ref['n_used'], f'{what}: {m.n} studies used, R uses {ref["n_used"]}'
    assert_close(m.coefficients, ref['coef'], rtol=rtol, what=f'{what}: coefficients')
    assert_close(m.vcov, ref['vcov'], rtol=rtol, what=f'{what}: vcov')
    assert_close(m.psi, ref['psi'], rtol=rtol, what=f'{what}: Psi')
    assert abs(m.loglik - ref['loglik']) <= rtol * max(1.0, abs(ref['loglik'])), \
        f'{what}: loglik Python {m.loglik!r} vs R {ref["loglik"]!r}'


def _mv_masks(S):
    """y mask (n,k) and the S mask (n,k,k): the rows / columns of the missing outcomes (as R's NA handling expects)."""
    y_mask = mask_at((40, 3), MISSING_OUTCOMES)
    s_mask = np.zeros(S.shape, dtype=bool)
    for (i, j) in MISSING_OUTCOMES:
        s_mask[i, j, :] = True
        s_mask[i, :, j] = True
    return y_mask, s_mask


@pytest.mark.parametrize('fill', ['minus9999', 'kept'])
@pytest.mark.parametrize('with_S', [False, True], ids=['y-only', 'y-and-S'])
def test_mvmeta_masked_outcomes_are_missing_like_na(with_S, fill):
    """A masked entry of y is a missing outcome (R: NA in y; studies keep their other outcomes); with S masked on the
    same rows/columns (R's convention for an NA outcome) the fit is the same.  PyDLNM uses the value under the mask
    as an observed outcome (no NaN in mv.y, na_action None)."""
    y, S, X = sim(40, 3, 2, 0.3, 11)
    y_mask, s_mask = _mv_masks(S)
    S_na = np.where(s_mask, np.nan, S) if with_S else S
    ref = r_mvmeta(nan_coded(y, y_mask), S_na, X)
    y_m = masked_like(y, y_mask, FILLS[fill])
    S_m = masked_like(S, s_mask, FILLS[fill]) if with_S else S
    m = accept_or_reject(lambda: py_mvmeta(y_m, S_m, X))
    if m is REJECTED:
        return
    assert np.isnan(m.y).sum() == len(MISSING_OUTCOMES), 'the masked outcomes must be missing in the fitted model'
    assert_mvmeta_matches(m, ref, f'masked outcomes/{fill}/with_S={with_S}')


def test_mvmeta_masked_covariate_drops_the_study_like_na():
    """R mvmeta(): a study with an NA covariate is dropped (na.omit).  A masked covariate keeps the study, with the
    value under the mask as its covariate."""
    y, S, X = sim(40, 3, 2, 0.3, 11)
    x_mask = mask_at(X.shape, [[3, 1]])
    ref = r_mvmeta(y, S, nan_coded(X, x_mask))
    assert ref['n_used'] == 39
    m = accept_or_reject(lambda: py_mvmeta(y, S, masked_like(X, x_mask, None)))
    if m is REJECTED:
        return
    assert m.na_action is not None and m.na_action.tolist() == [3]
    assert_mvmeta_matches(m, ref, 'masked covariate')


def test_mvmeta_masked_variance_of_an_observed_outcome_is_rejected_like_na():
    """R: an NA in S for an observed outcome -> error ('missing pattern in y and S is not consistent').  A masked
    S entry keeps its number under the mask and is used as the study's variance."""
    y, S, X = sim(40, 3, 2, 0.3, 11)
    s_mask = np.zeros(S.shape, dtype=bool)
    s_mask[4, 0, 0] = True
    S_na = np.where(s_mask, np.nan, S)
    require_r('mvmeta')
    r(_R_DEFS)
    np2r(PFX + 'y', y)
    np2r(PFX + 'Sv', vech_rows(S_na))
    np2r(PFX + 'X', X)
    assert r_error(f'suppressWarnings({PFX}formula_fit(as.matrix({PFX}y), as.matrix({PFX}Sv), as.matrix({PFX}X), '
                   f'"reml"))') is not None, 'R must stop on an NA variance of an observed outcome'
    with pytest.raises((ValueError, TypeError)):
        py_mvmeta(y, masked_like(S, s_mask, None), X)


# ==============================================================================================================
# 6. GLM interface: rpy2_glm.as_vector and Rpy2GLMInterface.fit_glm
# ==============================================================================================================
@pytest.mark.parametrize('shape', ['vector', 'column'])
@pytest.mark.parametrize('fill', list(FILLS))
def test_as_vector_masked_cells_are_nan(shape, fill):
    """as_vector(values, n, what) is how y / weights / offset enter R's glm(): masked cells must reach R as NA."""
    from rpy2_glm import as_vector
    _, death, mask = series(200, [20, 77])
    arr = masked_like(death, mask, FILLS[fill])
    if shape == 'column':
        arr = arr.reshape(-1, 1)
    np2r(PFX + 'v', nan_coded(death, mask))
    ref = rget(f'as.numeric({PFX}v)')
    out = accept_or_reject(lambda: as_vector(arr, 200, 'y'))
    if out is REJECTED:
        return
    assert_close(out, ref, rtol=0, what=f'as_vector {shape}/{fill}')


def _glm_setup():
    x, death, _ = series(600, [])
    spec = dict(lag=3, argvar={'fun': 'ns', 'df': 3}, arglag={'fun': 'ns', 'df': 2})
    r_crossbasis(x, spec, name=PFX + 'gcb')
    cb = py_crossbasis(x, spec)
    t = np.arange(600, dtype=float)
    other = np.column_stack([np.sin(2 * np.pi * t / 365.25), np.cos(2 * np.pi * t / 365.25)])
    weights = 1.0 + (t % 7) / 7.0
    offset = 0.1 * np.sin(t / 50.0)
    return x, death, cb, other, weights, offset


def r_glm_block(y, other, weights, offset):
    """cross-basis block of coef / vcov of glm(y ~ cb + other [, weights, offset], quasipoisson, na.exclude)."""
    d = {'y': y, 'o1': other[:, 0], 'o2': other[:, 1], 'w': weights, 'off': offset}
    for k, v in d.items():
        np2r(PFX + 'd_' + k, v)
    r(f'{PFX}gd <- data.frame(' + ', '.join(f'{k}=as.numeric({PFX}d_{k})' for k in d) + ')')
    r(f'{PFX}gm <- glm(y ~ {PFX}gcb + o1 + o2, data={PFX}gd, family=quasipoisson, weights=w, offset=off, '
      f'na.action=na.exclude)')
    r(f'{PFX}gi <- grep("{PFX}gcb", names(coef({PFX}gm)))')
    return dict(coef=rget(f'unname(coef({PFX}gm)[{PFX}gi])'), vcov=rget(f'unname(vcov({PFX}gm)[{PFX}gi, {PFX}gi])'),
                fitted=rget(f'as.numeric(fitted({PFX}gm))'), nobs=int(r(f'nobs({PFX}gm)')[0]))


@pytest.mark.parametrize('which', ['response', 'covariate', 'weights', 'offset'])
def test_fit_glm_masked_inputs_are_excluded_like_na(which):
    """R glm(na.action = na.exclude) excludes every row with an NA response, covariate, weight or offset and pads the
    fitted values with NA.  PyDLNM fits the rows with the value that happens to sit under the mask."""
    from rpy2_glm import Rpy2GLMInterface
    _, death, cb, other, weights, offset = _glm_setup()
    n = len(death)
    m_rows = {'response': [[30], [200], [201]], 'covariate': [[80, 0], [300, 1]], 'weights': [[120]],
              'offset': [[450]]}[which]
    data = {'response': death, 'covariate': other, 'weights': weights, 'offset': offset}
    mask = mask_at(data[which].shape, m_rows)
    na = {k: v.copy() for k, v in data.items()}
    na[which] = nan_coded(data[which], mask)
    ref = r_glm_block(na['response'], na['covariate'], na['weights'], na['offset'])
    mk = {k: v for k, v in data.items()}
    mk[which] = masked_like(data[which], mask, None)

    def fit():
        ifc = Rpy2GLMInterface(cb)
        quiet(ifc.fit_glm, mk['response'], family='quasipoisson', other_vars=mk['covariate'],
              weights=mk['weights'], offset=mk['offset'])
        return ifc

    ifc = accept_or_reject(fit)
    if ifc is REJECTED:
        return
    coef, vcov = ifc.get_crossbasis_coefficients()
    fitted = ifc.fitted()
    assert_close(fitted, ref['fitted'], rtol=1e-8, what=f'{which}: fitted values (NaN rows)')
    assert_close(coef, ref['coef'], rtol=1e-8, what=f'{which}: cross-basis coefficients')
    assert_close(vcov, ref['vcov'], rtol=1e-8, what=f'{which}: cross-basis vcov')


# ==============================================================================================================
# 7. utilities: lagmatrix, exphist, equalknots, find_mmt_blup
# ==============================================================================================================
@pytest.mark.parametrize('fill', ['netcdf', 'kept'])
def test_lagmatrix_masked_equals_tsmodel_lag_of_na(fill):
    """tsModel::Lag(x, 0:3) of the NA-coded series (PyDLNM utils.lagmatrix is its port)."""
    require_r('tsModel')
    from utils import lagmatrix
    x, _, mask = series(200, [40, 120])
    np2r(PFX + 'lx', nan_coded(x, mask))
    ref = rget(f'unclass(tsModel::Lag(as.numeric({PFX}lx), 0:3))')
    out = accept_or_reject(lambda: lagmatrix(masked_like(x, mask, FILLS[fill]), np.arange(4)))
    if out is REJECTED:
        return
    assert_close(out, ref, rtol=1e-12, what='lagmatrix')


@pytest.mark.parametrize('fill', ['netcdf', 'kept'])
def test_exphist_masked_equals_r_na(fill):
    """dlnm::exphist of the NA-coded profile: the exposure history keeps the NA cells."""
    from utils import exphist
    x, _, mask = series(120, [30, 80])
    np2r(PFX + 'hx', nan_coded(x, mask))
    ref = rget(f'unclass(exphist(as.numeric({PFX}hx), lag=3))')
    out = accept_or_reject(lambda: exphist(masked_like(x, mask, FILLS[fill]), lag=3))
    if out is REJECTED:
        return
    assert_close(out, ref, rtol=1e-12, what='exphist')


@pytest.mark.parametrize('fun', ['ns', 'bs', 'strata'])
def test_equalknots_masked_fill_value_equals_r_na(fun):
    """dlnm::equalknots of the NA-coded series: equally spaced knots over the OBSERVED range (the netCDF fill value
    under the mask stretches the range to 9.96921e36)."""
    _equalknots_case(fun, 'netcdf')


@pytest.mark.parametrize('fun', ['ns', 'bs', 'strata'])
def test_equalknots_masked_genuine_values_keep_the_range(fun):
    """Guard: equalknots only uses min / max, so genuine values under the mask (none of them extreme) are harmless."""
    _equalknots_case(fun, 'kept')


def _equalknots_case(fun, fill):
    from utils import equalknots
    x, _, mask = series(300, [10, 120, 250])
    np2r(PFX + 'ex', nan_coded(x, mask))
    ref = rget(f'as.numeric(equalknots(as.numeric({PFX}ex), nk=3, fun="{fun}"))')
    out = accept_or_reject(lambda: equalknots(masked_like(x, mask, FILLS[fill]), nk=3, fun=fun))
    if out is REJECTED:
        return
    assert_close(np.atleast_1d(out), ref, rtol=1e-10, what=f'equalknots {fun}')


@pytest.mark.parametrize('fill', ['netcdf', 'kept'])
def test_find_mmt_blup_masked_series_equals_r_recipe_on_observed_days(fill):
    """Lancet 02.secondstage.R: predvar <- quantile(x, 1:99/100); bvar <- onebasis(predvar, 'bs', knots = quantile(x,
    c(10,75,90)/100), degree = 2, Boundary.knots = range(x)); minperccity <- which.min(bvar %*% blup), all on the
    observed days (na.rm)."""
    from centering import find_mmt_blup
    x, _, mask = series(800, [100, 400, 650])
    np2r(PFX + 'mx', nan_coded(x, mask))
    blup = np.array([-0.30, 0.10, 0.20, 0.40, 0.50])
    np2r(PFX + 'mb', blup)
    r(f'{PFX}mxx <- as.numeric({PFX}mx); {PFX}mxx <- {PFX}mxx[!is.na({PFX}mxx)]')
    r(f'{PFX}predvar <- quantile({PFX}mxx, 1:99/100)')
    r(f'{PFX}bvar <- onebasis({PFX}predvar, fun="bs", knots=quantile({PFX}mxx, c(10,75,90)/100), degree=2, '
      f'Boundary.knots=range({PFX}mxx))')
    r(f'{PFX}risk <- as.numeric({PFX}bvar %*% as.numeric({PFX}mb))')
    ref_mmt = float(r2np(r(f'unname({PFX}predvar[which.min({PFX}risk)])')).ravel()[0])
    ref_pct = int(r(f'which.min({PFX}risk)')[0])
    res = accept_or_reject(lambda: find_mmt_blup(masked_like(x, mask, FILLS[fill]), blup, fun='bs', degree=2))
    if res is REJECTED:
        return
    assert res['percentile'] == ref_pct, f'MMT percentile {res["percentile"]} vs R {ref_pct}'
    assert_close(np.array([res['mmt']]), np.array([ref_mmt]), rtol=1e-10, what='MMT')
    assert_close(np.asarray(res['predvar']), rget(f'as.numeric({PFX}predvar)'), rtol=1e-10, what='predvar grid')


# ==============================================================================================================
# 8. guards: behaviour that is faithful today and must survive the fix
# ==============================================================================================================
def test_masked_array_without_masked_entries_is_the_plain_array():
    """np.ma.masked_array(x) (nomask), mask=False and an all-False mask array: no cell is missing -> plain result."""
    from basis import OneBasis
    x, _, _ = series(300, [])
    spec = dict(lag=5, argvar={'fun': 'ns', 'df': 4}, arglag={'fun': 'ns', 'df': 3})
    ref = r_crossbasis(x, spec)
    ref_ob = r_onebasis(x, 'ns', {'df': 4})
    variants = {'nomask': np.ma.masked_array(x), 'mask=False': np.ma.masked_array(x, mask=False),
                'all-False array': np.ma.masked_array(x, mask=np.zeros(x.shape, dtype=bool))}
    for label, xm in variants.items():
        assert_crossbasis_matches(py_crossbasis(xm, spec), ref, label)
        assert_onebasis_matches(OneBasis(xm, 'ns', df=4), ref_ob, label)


def test_masked_invalid_keeps_the_nan_under_the_mask_and_equals_the_nan_coded_series():
    """np.ma.masked_invalid(x) leaves the NaN in .data, so it is already equal to the NaN-coded series today."""
    x, _, mask = series(400, [50, 120, 300])
    x_na = nan_coded(x, mask)
    spec = _cb_specs(x_na)['ns4_ns3_lag5']
    ref = r_crossbasis(x_na, spec)
    xm = np.ma.masked_invalid(x_na)
    assert xm.mask.sum() == 3
    assert_crossbasis_matches(py_crossbasis(xm, spec), ref, 'masked_invalid')


def test_fully_observed_masked_array_reaches_attrdl_and_crosspred_like_the_plain_one(acase):
    """A masked array whose mask is empty gives R's numbers in attrdl, crosspred and MVMeta (no cell is missing)."""
    from prediction import crosspred
    ac = acase
    full_x = np.ma.masked_array(ac['x'])
    full_c = np.ma.masked_array(ac['cases'])
    ref = r_attrdl(ac, ac['x'], ac['cases'], type='an', dir='forw', tot=True)
    res = py_attrdl(ac, full_x, full_c, type='an', dir='forw')
    assert_close(_attr_out(res, 'an', True), ref, rtol=1e-8, what='attrdl on an all-observed masked array')
    k = ac['k']
    at = np.ma.masked_array(np.array([0.0, 10.0, 20.0]))
    np2r(PFX + 'gat', np.array([0.0, 10.0, 20.0]))
    r(f'{PFX}gp <- crosspred({PFX}acb, coef={PFX}acoef, vcov={PFX}avcov, model.link="log", at={PFX}gat, cen=15)')
    pred = quiet(crosspred, ac['cb'], coef=np.ma.masked_array(ac['coef']), vcov=np.ma.masked_array(ac['vcov']),
                 model_link='log', at=at, cen=CEN)
    assert_close(np.asarray(pred.allfit), rget(f'{PFX}gp$allfit'), rtol=1e-8, what='crosspred allfit')
    assert k == len(pred.coefficients)


@pytest.mark.parametrize('fill', ['netcdf', 'kept'])
def test_masked_input_is_not_modified_by_the_entry_points(acase, fill):
    """Whatever the fix does with the mask (np.ma.filled copies), the caller's masked array keeps its data and mask."""
    from basis import CrossBasis, OneBasis
    import attribution
    ac = acase
    x_m = masked_like(ac['x'], ac['xmask'], FILLS[fill])
    c_m = masked_like(ac['cases'], ac['cmask'], FILLS[fill])
    data_before, mask_before = x_m.data.copy(), x_m.mask.copy()
    cdata_before, cmask_before = c_m.data.copy(), c_m.mask.copy()
    accept_or_reject(lambda: quiet(CrossBasis, x_m, lag=4, argvar={'fun': 'ns', 'df': 3}, arglag={'fun': 'ns', 'df': 3}))
    accept_or_reject(lambda: OneBasis(x_m, 'ns', df=3))
    accept_or_reject(lambda: py_attrdl(ac, x_m, c_m, type='an', dir='forw'))
    assert np.array_equal(x_m.data, data_before) and np.array_equal(x_m.mask, mask_before)
    assert np.array_equal(c_m.data, cdata_before) and np.array_equal(c_m.mask, cmask_before)
    assert attribution is not None


# ==============================================================================================================
# 9. nullable pandas dtypes (Int64 / Float64 / boolean with pd.NA, object columns with None / pd.NA) are NaN too
# ==============================================================================================================
def _nullable_series(data, mask, dtype):
    """pandas Series of `dtype` holding `data`, with pd.NA at the masked positions."""
    values = pd.array(np.where(mask, np.nan, data).tolist(), dtype=dtype) if dtype != 'object' else None
    if dtype == 'object':
        return pd.Series([pd.NA if m else v for v, m in zip(data, mask)], dtype=object)
    return pd.Series(values)


@pytest.mark.parametrize('dtype', ['Float64', 'object'])
def test_asfloat_nullable_and_masked_inputs_are_nan(dtype):
    from utils import asfloat
    data = np.array([1.0, 2.0, 3.0, 4.0])
    mask = np.array([False, True, False, True])
    want = np.where(mask, np.nan, data)
    assert_close(asfloat(_nullable_series(data, mask, dtype)), want, rtol=0, what=f'Series {dtype}')
    assert_close(asfloat(pd.DataFrame({'a': _nullable_series(data, mask, dtype)})), want.reshape(-1, 1), rtol=0,
                 what=f'DataFrame {dtype}')
    assert_close(asfloat([v if not m else pd.NA for v, m in zip(data, mask)]), want, rtol=0, what='list with pd.NA')
    assert_close(asfloat([None if m else v for v, m in zip(data, mask)]), want, rtol=0, what='list with None')
    assert_close(asfloat(pd.array([1, pd.NA, 3], dtype='Int64')), np.array([1.0, np.nan, 3.0]), rtol=0,
                 what='Int64 extension array')
    assert_close(asfloat(pd.array([True, pd.NA, False], dtype='boolean')), np.array([1.0, np.nan, 0.0]), rtol=0,
                 what='boolean extension array')
    # a list of masked arrays keeps the masks (e.g. the per-study covariance matrices of a meta-analysis)
    mats = [np.ma.masked_array(np.eye(2), mask=[[0, 1], [1, 0]]), np.ma.masked_array(2 * np.eye(2))]
    got = asfloat(mats)
    assert got.shape == (2, 2, 2) and np.isnan(got[0, 0, 1]) and np.isnan(got[0, 1, 0]) and got[1, 0, 1] == 0.0
    # copy=True never aliases, and the masked input is left alone
    plain, masked = np.array([1.0, 2.0]), np.ma.masked_array([1.0, 2.0], mask=[0, 1])
    assert not np.shares_memory(asfloat(plain, copy=True), plain)
    assert np.shares_memory(asfloat(plain), plain)
    asfloat(masked)
    assert masked.mask.tolist() == [False, True] and masked.data.tolist() == [1.0, 2.0]
    assert np.isnan(asfloat(np.ma.masked)).all()


@pytest.mark.parametrize('dtype', ['Float64', 'object'])
def test_onebasis_nullable_series_equals_r_na(dtype):
    from basis import OneBasis
    x, _, mask = series(300, [10, 120, 250])
    ref = r_onebasis(nan_coded(x, mask), 'ns', {'df': 4})
    assert_onebasis_matches(OneBasis(_nullable_series(x, mask, dtype), 'ns', df=4), ref, f'Series {dtype}')


@pytest.mark.parametrize('dtype', ['Float64', 'object'])
def test_mvmeta_nullable_dataframe_outcomes_are_missing_like_na(dtype):
    """y as a pandas DataFrame of a nullable dtype (pd.NA = a missing outcome): the fit is R's on the NA-coded y."""
    y, S, X = sim(40, 3, 2, 0.3, 11)
    y_mask, s_mask = _mv_masks(S)
    ref = r_mvmeta(nan_coded(y, y_mask), np.where(s_mask, np.nan, S), X)
    y_df = pd.DataFrame({j: _nullable_series(y[:, j], y_mask[:, j], dtype) for j in range(3)})
    S_na = np.where(s_mask, np.nan, S)
    m = py_mvmeta(y_df, S_na, X)
    assert np.isnan(m.y).sum() == len(MISSING_OUTCOMES)
    assert_mvmeta_matches(m, ref, f'nullable {dtype} outcomes')


def test_mvmeta_nullable_covariate_drops_the_study_like_na():
    y, S, X = sim(40, 3, 2, 0.3, 11)
    x_mask = mask_at(X.shape, [[3, 1]])
    ref = r_mvmeta(y, S, nan_coded(X, x_mask))
    X_df = pd.DataFrame({j: _nullable_series(X[:, j], x_mask[:, j], 'Float64') for j in range(2)})
    m = py_mvmeta(y, S, X_df)
    assert m.na_action is not None and m.na_action.tolist() == [3]
    assert_mvmeta_matches(m, ref, 'nullable covariate')


@pytest.mark.parametrize('dtype', ['Float64', 'object'])
def test_as_vector_nullable_series_cells_are_nan(dtype):
    from rpy2_glm import as_vector
    _, death, mask = series(200, [20, 77])
    out = as_vector(_nullable_series(death, mask, dtype), 200, 'y')
    assert_close(out, nan_coded(death, mask), rtol=0, what=f'as_vector {dtype}')
