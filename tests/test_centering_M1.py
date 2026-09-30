"""Centering / MMT: find_mmt_blup vs the R recipe of the Lancet script (02.secondstage.R:58-70), plus faithful baselines.

Theme M1: find_mmt_blup passes non-existent keyword arguments to BSplineBasis.
  basis-cont-5  centering.find_mmt_blup builds BSplineBasis(include_intercept=True, boundary_knots=x_range).  The class
                parameter is called Boundary_knots and has no include_intercept, so both keywords vanish into **kwargs
                and the basis takes its boundary knots from the 1st-99th percentile grid instead of range(x).
                bvar %*% blup, the MMT percentile/temperature and MultiLocationDLNM.calculate_pooled_mmts are then wrong
                (in the audit 19/25 US cities had a different MMT percentile; basis differences of 0.6-0.9).  The blanket
                try/except also turns any failure into a silent "median_fallback" instead of an error.

R reference (always computed by R at test time):
    predvar <- quantile(x, 1:99/100)
    bvar    <- onebasis(predvar, fun="bs", knots=quantile(x, c(10,75,90)/100), degree=2, Boundary.knots=range(x))
    minperc <- (1:99)[which.min(bvar %*% blup)];   mintemp <- quantile(x, minperc/100)

Known-defect tests (strict xfail until the fix lands)
  test_blup_basis_matrix_matches_r_onebasis        basis-cont-5   basis matrix on range(x) boundary knots, 6 US cities
  test_blup_mmt_matches_r_which_min                basis-cont-5   MMT percentile + temperature, 12 US cities
  test_blup_basis_option_combinations              basis-cont-5   degree 1-3 / 1-3 interior knots
  test_blup_basis_percentile_range                 basis-cont-5   search ranges (2,98), (5,95), (10,90)
  test_blup_basis_with_nan_in_x                    basis-cont-5   NaN in x (na.rm=TRUE in R)
  test_blup_mmt_realistic_reduced_coef_chicago     basis-cont-5   R crossreduce() coefficients of chicagoNMMAPS as "BLUP"
  test_blup_wrong_length_raises_like_r             basis-cont-5   R errors (non-conformable); Python must not return a median
  test_calculate_pooled_mmts_matches_r             basis-cont-5   public wrapper MultiLocationDLNM.calculate_pooled_mmts

Plain tests (already faithful, guard against regressions while the fixes land)
  test_blup_prediction_grid_matches_r_quantile     1st-99th percentile grid (also 5-95, 0-100, NaN in x), internal consistency
  test_blup_full_range_grid_matches_r              percentile_range=(0,100): grid range == range(x), whole result equals R
  test_blup_normal_path_is_not_a_fallback          no warning, method == 'blup_optimization'
  test_crosspred_explicit_cen_matches_r            11 cen values incl. min, max and out of range (+-40)
  test_recenter_basis_matches_r_crosspred_cen      recenter_basis(cen=15) == R crosspred(cb, cen=15)
  test_find_mmt_matches_r_which_min_on_identical_grid   full cross-basis coefficients, identical explicit grids
  test_find_mmt_reduced_coef_matches_r_which_min   reduced (BLUP-style) coefficients on 4 US cities
  test_find_mmt_flat_curve_takes_first_index       np.argmin == R which.min tie-breaking
"""
import os
from functools import lru_cache
from types import SimpleNamespace

import numpy as np
import pytest

from rhelpers import REPO, assert_close, chicago, known_defect, np2r, r, rget      # rhelpers first: it starts R
from rpy2.rinterface_lib.embedded import RRuntimeError

DATA_CSV = REPO / 'temperature_mortality_analysis' / 'data.csv'     # data, not code: always the real repo

# --- cities used below (all in data.csv, 5114 daily values each) -----------------------------------------------------
BASIS_CITIES = ['Akron', 'Corpus Christi', 'Kansas City', 'Nashville', 'Sacramento', 'Tulsa']
REDUCED_CITIES = ['Akron', 'Kansas City', 'Sacramento', 'Tulsa']
CEN_VALUES = [-10.0, 0.0, 5.5, 10.0, 15.0, 21.1, 'mean', 'min', 'max', 40.0, -40.0]   # 40/-40 lie outside the data range


# Several PyDLNM modules (CrossBasis, improved_glm, rpy2_glm) overwrite os.environ['R_HOME'] with the R 4.6 path; the R
# session started by rhelpers is already running R 4.5 and an R glm() fit then segfaults.  Pin the value seen at
# collection time (before any test ran) around every test of this module, including its own fixtures.
_GOOD_R_HOME = os.environ.get('R_HOME')


def _pin_r_home():
    if _GOOD_R_HOME is None:
        os.environ.pop('R_HOME', None)
    else:
        os.environ['R_HOME'] = _GOOD_R_HOME


@pytest.fixture(autouse=True)
def _keep_r_home():
    _pin_r_home()
    yield
    _pin_r_home()


# ============================================================ helpers ==============================================
@lru_cache(maxsize=1)
def _us_temperature():
    import pandas as pd
    df = pd.read_csv(DATA_CSV, usecols=['cityName', 'TMean'])
    return {c: g['TMean'].to_numpy(dtype=float) for c, g in df.groupby('cityName')}


def _city_series(name):
    return _us_temperature()[name].copy()


def _spread_cities(k):
    names = sorted(_us_temperature())
    return names[::max(len(names) // k, 1)][:k]


def _r_blup_recipe(x, blup, varper=(10, 75, 90), degree=2, prange=(1, 99)):
    """The 02.secondstage.R recipe, evaluated in R on the same numbers Python receives."""
    np2r('cm1_x', x)
    np2r('cm1_blup', blup)
    np2r('cm1_per', np.asarray(varper, dtype=float))
    np2r('cm1_pr', np.asarray(prange, dtype=float))
    r(f'cm1_deg <- {int(degree)}L')
    r('''
    cm1_pv  <- quantile(cm1_x, (cm1_pr[1]:cm1_pr[2])/100, na.rm=TRUE)
    cm1_bv  <- onebasis(cm1_pv, fun="bs", knots=quantile(cm1_x, cm1_per/100, na.rm=TRUE), degree=cm1_deg,
                        Boundary.knots=range(cm1_x, na.rm=TRUE))
    cm1_lp  <- as.numeric(cm1_bv %*% cm1_blup)
    cm1_i   <- which.min(cm1_lp)
    ''')
    i = int(rget('cm1_i')[0])
    predvar = rget('as.numeric(cm1_pv)')
    return dict(basis=rget('unclass(cm1_bv)'), predvar=predvar, risk=rget('cm1_lp'),
                percentile=int(prange[0]) + i - 1, mmt=float(predvar[i - 1]))


def _py_blup_mmt(x, blup, knots=None, degree=2, prange=(1, 99)):
    import warnings
    from centering import find_mmt_blup
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        res = find_mmt_blup(x, blup, fun='bs', knots=knots, degree=degree, percentile_range=prange)
    return res, [str(w.message) for w in caught]


def _assert_blup_result_matches_r(res, ref, what, check_basis=True):
    assert res.get('method') == 'blup_optimization', f'{what}: no BLUP result ({res.get("method")}: {res.get("error")})'
    if check_basis:
        assert_close(res['basis_matrix'], ref['basis'], rtol=1e-9, what=f'{what}: bs basis on the prediction grid')
        assert_close(res['risk_values'], ref['risk'], rtol=1e-9, what=f'{what}: bvar %*% blup')
    assert int(res['percentile']) == ref['percentile'], (
        f'{what}: MMT percentile Python {int(res["percentile"])} vs R {ref["percentile"]}')
    assert abs(float(res['mmt']) - ref['mmt']) <= 1e-9 * max(abs(ref['mmt']), 1.0), (
        f'{what}: MMT temperature Python {float(res["mmt"])!r} vs R {ref["mmt"]!r}')


@pytest.fixture(scope='module')
def chicago_model():
    """chicagoNMMAPS bs(deg 2, P10/75/90) x ns(logknots(21,3)) fitted in R; the same coef/vcov feed both sides."""
    from basis import CrossBasis
    from utils import logknots
    _pin_r_home()
    temp = chicago()['temp']
    np2r('cm1_temp', temp)
    r('''
    cm1_kv <- quantile(cm1_temp, c(.10, .75, .90))
    cm1_cb <- crossbasis(cm1_temp, lag=21, argvar=list(fun="bs", degree=2, knots=cm1_kv),
                         arglag=list(fun="ns", knots=logknots(21, 3)))
    cm1_mod <- glm(death ~ cm1_cb + ns(time, 7*14) + dow, family=quasipoisson(), data=chicagoNMMAPS)
    cm1_ind  <- grep("cm1_cb", names(coef(cm1_mod)))
    cm1_coef <- unname(coef(cm1_mod)[cm1_ind])
    cm1_vcov <- unname(vcov(cm1_mod)[cm1_ind, cm1_ind])
    cm1_red  <- crossreduce(cm1_cb, cm1_mod, type="overall", cen=median(cm1_temp))
    ''')
    kv = rget('as.numeric(cm1_kv)')
    cb = CrossBasis(temp, lag=21, argvar={'fun': 'bs', 'degree': 2, 'knots': kv},
                    arglag={'fun': 'ns', 'knots': logknots([0, 21], nk=3)})
    return SimpleNamespace(temp=temp, cb=cb, coef=rget('cm1_coef'), vcov=rget('cm1_vcov'),
                           red_coef=rget('as.numeric(coef(cm1_red))'))


def _cen_value(spec, temp):
    return {'mean': float(np.nanmean(temp)), 'min': float(np.nanmin(temp)), 'max': float(np.nanmax(temp))}.get(spec, spec)


def _r_crosspred(grid, cen):
    """R crosspred of the Chicago cross-basis at `grid`, centred at `cen`, with the fitted coef/vcov; result in cm1_p."""
    np2r('cm1_grid', grid)
    np2r('cm1_cen', np.array([cen], dtype=float))
    r('cm1_p <- suppressWarnings(crosspred(cm1_cb, coef=cm1_coef, vcov=cm1_vcov, model.link="log", at=cm1_grid, '
      'cen=cm1_cen))')


# ==================================== known defect: find_mmt_blup (M1, basis-cont-5) ===================================
@known_defect('M1', 'basis-cont-5', note='include_intercept/boundary_knots swallowed by **kwargs; basis uses range(predvar)')
@pytest.mark.parametrize('city', BASIS_CITIES)
def test_blup_basis_matrix_matches_r_onebasis(city):
    """The bs basis inside find_mmt_blup is R's onebasis(predvar, 'bs', knots, degree=2, Boundary.knots=range(x))."""
    x = _city_series(city)
    blup = np.random.default_rng(1000 + BASIS_CITIES.index(city)).normal(0.0, 0.1, 5)
    ref = _r_blup_recipe(x, blup)
    res, _ = _py_blup_mmt(x, blup)
    assert res.get('method') == 'blup_optimization', f'no BLUP result: {res.get("error")}'
    assert_close(res['basis_matrix'], ref['basis'], rtol=1e-9, what=f'{city}: bs basis on the prediction grid')
    assert_close(res['risk_values'], ref['risk'], rtol=1e-9, what=f'{city}: bvar %*% blup')


@known_defect('M1', 'basis-cont-5', note='wrong boundary knots => argmin of bvar %*% blup moves')
def test_blup_mmt_matches_r_which_min():
    """MMT percentile and temperature of find_mmt_blup (default knots, as used by calculate_pooled_mmts) equal R's."""
    cities = _spread_cities(12)
    rng = np.random.default_rng(2024)
    bad = []
    for city in cities:
        x = _city_series(city)
        blup = rng.normal(0.0, 0.1, 5)
        ref = _r_blup_recipe(x, blup)
        res, _ = _py_blup_mmt(x, blup)
        try:
            _assert_blup_result_matches_r(res, ref, city, check_basis=False)      # MMT only; basis tested above
        except AssertionError as e:
            bad.append(str(e).splitlines()[0])
    assert not bad, f'{len(bad)}/{len(cities)} cities differ from R:\n  ' + '\n  '.join(bad)


@known_defect('M1', 'basis-cont-5', note='same misspelled Boundary_knots for every degree / knot set')
@pytest.mark.parametrize('degree,varper', [(2, (10, 75, 90)), (3, (10, 75, 90)), (3, (25, 75)), (2, (50,)), (1, (33, 66))],
                         ids=['deg2-k3', 'deg3-k3', 'deg3-k2', 'deg2-k1', 'deg1-k2'])
def test_blup_basis_option_combinations(degree, varper):
    """Explicit knots/degree combinations: basis (degree + n_knots columns) and MMT equal R's for three cities."""
    rng = np.random.default_rng(31 * degree + len(varper))
    for city in ['Akron', 'Sacramento', 'Tulsa']:
        x = _city_series(city)
        blup = rng.normal(0.0, 0.1, degree + len(varper))
        ref = _r_blup_recipe(x, blup, varper=varper, degree=degree)
        res, _ = _py_blup_mmt(x, blup, knots=np.percentile(x[~np.isnan(x)], list(varper)), degree=degree)
        _assert_blup_result_matches_r(res, ref, f'{city} degree={degree} knots at P{list(varper)}')


@known_defect('M1', 'basis-cont-5', note='percentile_range other than (1, 99): grid range still replaces range(x)')
@pytest.mark.parametrize('prange', [(2, 98), (5, 95), (10, 90)], ids=['p2-98', 'p5-95', 'p10-90'])
def test_blup_basis_percentile_range(prange):
    """Search percentile ranges: R's (lo:hi)[which.min(bvar %*% blup)] with the basis on range(x) boundary knots."""
    rng = np.random.default_rng(prange[0] * 7 + prange[1])
    for city in ['Kansas City', 'Nashville']:
        x = _city_series(city)
        blup = rng.normal(0.0, 0.1, 5)
        ref = _r_blup_recipe(x, blup, prange=prange)
        res, _ = _py_blup_mmt(x, blup, prange=prange)
        _assert_blup_result_matches_r(res, ref, f'{city} percentile_range={prange}')


@known_defect('M1', 'basis-cont-5', note='NaN in x: R uses na.rm=TRUE for knots, grid and Boundary.knots')
def test_blup_basis_with_nan_in_x():
    """Missing temperatures are ignored for the grid, the knots and the boundary knots, as in R (na.rm=TRUE)."""
    x = _city_series('Tulsa')
    x[::17] = np.nan
    blup = np.random.default_rng(12).normal(0.0, 0.1, 5)
    ref = _r_blup_recipe(x, blup)
    res, _ = _py_blup_mmt(x, blup)
    _assert_blup_result_matches_r(res, ref, 'Tulsa with NaN')


@known_defect('M1', 'basis-cont-5', note='realistic reduced coefficients; wrong boundary knots')
def test_blup_mmt_realistic_reduced_coef_chicago(chicago_model):
    """Overall-cumulative coefficients from R crossreduce() on chicagoNMMAPS play the role of a BLUP."""
    x = chicago_model.temp
    blup = chicago_model.red_coef
    assert blup.shape == (5,)
    ref = _r_blup_recipe(x, blup)
    res, _ = _py_blup_mmt(x, blup)
    _assert_blup_result_matches_r(res, ref, 'chicagoNMMAPS reduced coef')


@known_defect('M1', 'basis-cont-5', note='blanket except returns a median fallback instead of raising')
def test_blup_wrong_length_raises_like_r():
    """R: bvar %*% blup with a wrong-length blup stops with 'non-conformable arguments'.  Python must raise as well,
    not report the median temperature as an MMT."""
    x = _city_series('Akron')
    bad_blup = np.random.default_rng(3).normal(0.0, 0.1, 4)          # basis has 5 columns
    with pytest.raises(RRuntimeError):
        _r_blup_recipe(x, bad_blup)
    with pytest.raises(ValueError):
        res, _ = _py_blup_mmt(x, bad_blup)
        pytest.fail(f'no error; returned method={res.get("method")!r} mmt={res.get("mmt")!r}')


@known_defect('M1', 'basis-cont-5', note='calculate_pooled_mmts calls find_mmt_blup(fun="bs", degree=2) with default knots')
def test_calculate_pooled_mmts_matches_r():
    """Public wrapper: per-region MMT (percentile, temperature) and the pooled median/mean equal the R workflow."""
    import contextlib
    import io
    import multi_location
    names = _spread_cities(8)
    rng = np.random.default_rng(5)
    blups, regions, ref = [], [], {}
    for nm in names:
        x = _city_series(nm)
        b = rng.normal(0.0, 0.12, 5)
        blups.append({'blup': b})
        regions.append({'glm_interface': SimpleNamespace(crossbasis=SimpleNamespace(x=x))})
        ref[nm] = _r_blup_recipe(x, b)
    stand_in = SimpleNamespace(blup_results=blups, region_results=regions, region_names=names, pooled_mmts=None)
    with contextlib.redirect_stdout(io.StringIO()):
        out = multi_location.MultiLocationDLNM.calculate_pooled_mmts(stand_in)
    py_perc = {n: int(out['region_mmts'][n]['percentile']) for n in names}
    r_perc = {n: ref[n]['percentile'] for n in names}
    assert py_perc == r_perc, 'per-region MMT percentile differs from R: ' + \
        ', '.join(f'{n}: Py {py_perc[n]} vs R {r_perc[n]}' for n in names if py_perc[n] != r_perc[n])
    r_temps = np.array([ref[n]['mmt'] for n in names])
    assert_close(np.array([out['region_mmts'][n]['mmt'] for n in names]), r_temps, rtol=1e-9, what='regional MMT temperature')
    assert_close(out['pooled_mmt_median'], np.median(r_temps), rtol=1e-9, what='pooled median MMT')
    assert_close(out['pooled_mmt_mean'], np.mean(r_temps), rtol=1e-9, what='pooled mean MMT')


# ============================================ plain: faithful today (find_mmt_blup) ================================
@pytest.mark.parametrize('prange,with_nan', [((1, 99), False), ((5, 95), False), ((0, 100), False), ((1, 99), True)],
                         ids=['p1-99', 'p5-95', 'p0-100', 'p1-99-nan'])
def test_blup_prediction_grid_matches_r_quantile(prange, with_nan):
    """predvar = quantile(x, prange/100, na.rm=TRUE) (type 7) and the result dict is self-consistent."""
    x = _city_series('Nashville')
    if with_nan:
        x[::17] = np.nan
    ncol = 5
    blup = np.random.default_rng(8).normal(0.0, 0.1, ncol)
    res, _ = _py_blup_mmt(x, blup, prange=prange)
    ref = _r_blup_recipe(x, blup, prange=prange)
    assert_close(res['predvar'], ref['predvar'], rtol=1e-12, what='prediction grid vs R quantile()')
    i = int(np.argmin(res['risk_values']))
    assert int(res['percentile']) == prange[0] + i
    assert float(res['mmt']) == float(res['predvar'][i])
    assert res['min_risk'] == np.min(res['risk_values'])
    assert_close(res['risk_values'], res['basis_matrix'] @ blup, rtol=1e-12, what='risk_values == basis_matrix %*% blup')


@pytest.mark.parametrize('city', ['Kansas City', 'Sacramento'])
def test_blup_full_range_grid_matches_r(city):
    """percentile_range=(0, 100): the prediction grid spans range(x), so the (mis-set) boundary knots coincide with R's
    Boundary.knots = range(x) and the whole result equals R's already today.  Must stay so after the fix."""
    x = _city_series(city)
    blup = np.random.default_rng(21 + len(city)).normal(0.0, 0.1, 5)
    ref = _r_blup_recipe(x, blup, prange=(0, 100))
    res, _ = _py_blup_mmt(x, blup, prange=(0, 100))
    _assert_blup_result_matches_r(res, ref, f'{city} percentile_range=(0, 100)')


def test_blup_normal_path_is_not_a_fallback():
    """With a valid blup no warning is issued and the BLUP branch (not the median fallback) is taken."""
    x = _city_series('Tulsa')
    blup = np.random.default_rng(9).normal(0.0, 0.1, 5)
    res, warns = _py_blup_mmt(x, blup)
    assert res['method'] == 'blup_optimization'
    assert 'error' not in res
    assert not [w for w in warns if 'BLUP' in w or 'fallback' in w], warns
    assert res['basis_matrix'].shape == (99, 5)


# ================================================ plain: centering with explicit cen ===============================
@pytest.mark.parametrize('cen', CEN_VALUES, ids=[str(c) for c in CEN_VALUES])
def test_crosspred_explicit_cen_matches_r(chicago_model, cen):
    """Explicit cen (inside, at the edges of, and outside the data range): fits, SEs, RRs and cen itself equal R's."""
    from prediction import crosspred
    temp = chicago_model.temp
    c = _cen_value(cen, temp)
    grid = np.arange(-20.0, 30.001, 0.5)
    _r_crosspred(grid, c)
    pp = crosspred(chicago_model.cb, coef=chicago_model.coef, vcov=chicago_model.vcov, model_link='log', at=grid, cen=c)
    assert_close(pp.predvar, rget('as.numeric(cm1_p$predvar)'), rtol=1e-12, what='predvar')
    for key in ('allfit', 'allse'):
        assert_close(getattr(pp, key), rget(f'as.numeric(cm1_p${key})'), rtol=1e-9, what=f'cen={c}: {key}')
    for key in ('matfit', 'matse'):
        assert_close(getattr(pp, key), rget(f'cm1_p${key}'), rtol=1e-9, what=f'cen={c}: {key}')
    for key in ('allRRfit', 'allRRlow', 'allRRhigh'):
        assert_close(getattr(pp, key), rget(f'as.numeric(cm1_p${key})'), rtol=1e-9, what=f'cen={c}: {key}')
    assert float(pp.cen) == pytest.approx(float(rget('as.numeric(cm1_p$cen)')[0]), rel=0, abs=0)
    # the reference point itself has zero effect (only when it lies on the prediction grid)
    if c in grid:
        j = int(np.where(grid == c)[0][0])
        assert abs(pp.allfit[j]) <= 1e-12 and abs(pp.allse[j]) <= 1e-12


def test_recenter_basis_matches_r_crosspred_cen(chicago_model):
    """recenter_basis(cen=15) keeps the basis matrix (cen is metadata, as in R) and crosspred of the recentered basis
    equals R's crosspred(cb, cen=15)."""
    from centering import recenter_basis
    from prediction import crosspred
    cb0 = chicago_model.cb
    before = np.array(cb0.basis, copy=True)
    cb1, info = recenter_basis(cb0, None, cen=15)
    assert info['method'] == 'manual' and info['value'] == 15
    assert np.array_equal(np.asarray(cb1.basis), before, equal_nan=True)
    assert np.array_equal(np.asarray(cb0.basis), before, equal_nan=True), 'original basis was modified'
    grid = np.arange(-20.0, 30.001, 0.5)
    _r_crosspred(grid, 15.0)
    pp = crosspred(cb1, coef=chicago_model.coef, vcov=chicago_model.vcov, model_link='log', at=grid)
    for key in ('allfit', 'allse'):
        assert_close(getattr(pp, key), rget(f'as.numeric(cm1_p${key})'), rtol=1e-9, what=f'recentered {key}')
    assert_close(pp.matfit, rget('cm1_p$matfit'), rtol=1e-9, what='recentered matfit')


# ================================================ plain: find_mmt on identical grids ===============================
GRIDS = {
    'half-degree': lambda t: np.arange(-20.0, 30.001, 0.5),
    'tenth-degree': lambda t: np.round(np.arange(-20.0, 30.0001, 0.1), 1),
    'coarse': lambda t: np.arange(-25.0, 33.01, 2.0),
    'data-range': lambda t: np.linspace(np.nanmin(t), np.nanmax(t), 137),
}


@pytest.mark.parametrize('grid_name', list(GRIDS))
@pytest.mark.parametrize('method', ['overall', 'lagspecific'])
def test_find_mmt_matches_r_which_min_on_identical_grid(chicago_model, grid_name, method):
    """find_mmt(at=grid) equals R's predvar[which.min(allfit)] (or matfit[, lag0]) on the same grid."""
    from centering import find_mmt
    grid = GRIDS[grid_name](chicago_model.temp)
    _r_crosspred(grid, 15.0)
    key = 'as.numeric(cm1_p$allfit)' if method == 'overall' else 'cm1_p$matfit[, 1]'
    r_fit = rget(key)
    r_mmt = float(grid[int(np.argmin(r_fit))])
    assert int(rget('which.min(%s)' % key)[0]) - 1 == int(np.argmin(r_fit))       # numpy argmin == R which.min
    res = find_mmt(chicago_model.cb, None, coef=chicago_model.coef, vcov=chicago_model.vcov, at=grid, method=method)
    assert_close(res['predvar'], grid, rtol=0, what='searched grid')
    assert float(res['mmt']) == r_mmt, f'MMT Python {float(res["mmt"])!r} vs R {r_mmt!r}'


@pytest.mark.parametrize('city', REDUCED_CITIES)
def test_find_mmt_reduced_coef_matches_r_which_min(city):
    """Reduced (BLUP-style, one coefficient per variable-basis column) path on the same grid as R's
    grid[which.min(bvar %*% blup)] with Boundary.knots = range(x)."""
    from basis import CrossBasis
    from centering import find_mmt
    from utils import logknots
    x = _city_series(city)
    rng = np.random.default_rng(700 + REDUCED_CITIES.index(city))
    blup = rng.normal(0.0, 0.1, 5)
    a = rng.normal(size=(5, 5))
    vc = 0.01 * (a @ a.T) + 1e-3 * np.eye(5)
    knots = np.percentile(x, [10, 75, 90])
    grid = np.linspace(np.nanmin(x), np.nanmax(x), 150)
    np2r('cm1_x', x); np2r('cm1_kn', knots); np2r('cm1_grid', grid); np2r('cm1_blup', blup)
    r('cm1_bv <- onebasis(cm1_grid, fun="bs", knots=cm1_kn, degree=2, Boundary.knots=range(cm1_x));'
      'cm1_lp <- as.numeric(cm1_bv %*% cm1_blup)')
    r_mmt = float(grid[int(np.argmin(rget('cm1_lp')))])
    cb = CrossBasis(x, lag=21, argvar={'fun': 'bs', 'degree': 2, 'knots': knots},
                    arglag={'fun': 'ns', 'knots': logknots([0, 21], nk=3)})
    res = find_mmt(cb, None, coef=blup, vcov=vc, at=grid)
    assert float(res['mmt']) == r_mmt, f'{city}: MMT Python {float(res["mmt"])!r} vs R {r_mmt!r}'


def test_find_mmt_flat_curve_takes_first_index(chicago_model):
    """All-zero coefficients: R which.min returns the first grid point, and so must find_mmt."""
    from centering import find_mmt
    grid = np.arange(-5.0, 5.01, 1.0)
    np2r('cm1_flat', np.zeros(len(grid)))
    assert int(rget('which.min(cm1_flat)')[0]) == 1
    res = find_mmt(chicago_model.cb, None, coef=np.zeros_like(chicago_model.coef), vcov=chicago_model.vcov, at=grid)
    assert float(res['mmt']) == float(grid[0])
