"""Centering module, theme M3: CenteringManager cache / compare_centering_strategies / recenter_at_mmt (centering-16),
find_mmt_blup robustness beyond the BSplineBasis kwargs (centering-17) and the find_mmt(method='lagspecific')
documentation (centering-20).

There is no R counterpart of CenteringManager (R has no cache), so the reference is R's own recipe applied to the same
inputs at test time:   crosspred(cb, coef, vcov, at=grid)  ->  predvar[which.min(allfit)]      (Lancet 02.secondstage.R)
and, for the BLUP path,   onebasis(predvar, fun, knots, Boundary.knots=range(x)) %*% blup  ->  which.min.

Theme M3
  centering-16  CenteringManager.find_mmt keeps ONE cached result and ignores its arguments: a later call with another
                `at` grid, another `method`, or (after the public attributes are re-assigned) another model / basis
                returns the first call's dict.  compare_centering_strategies() calls self.find_mmt() bare (default
                21-point grid, `at` never reaches the MMT search), keys the results f'cen_{value}' so equal centering
                values collapse into one entry while strategy_names keeps all of them, and recenter_at_mmt() bypasses
                the manager's history.
  centering-17  find_mmt_blup: `fun` is accepted but never used (always a B-spline), NaN in the BLUP makes np.argmin pick
                the first NaN (R: which.min -> integer(0)), a Python list for x raises TypeError (R: quantile() accepts
                a plain vector).  The other two sub-claims belong to other themes and are tested there:
                wrong-length BLUP -> median fallback is test_centering_M1.test_blup_wrong_length_raises_like_r, NaN in
                `at` is test_crosspred_G_H1b.test_find_mmt_with_nan_in_at.  The `fun` tests use percentile_range=(0, 100)
                on purpose: the grid then spans range(x), so the (M1) boundary-knot defect is invisible and only the
                (M3) ignored-`fun` defect is exercised.
  centering-20  find_mmt(method='lagspecific') searches the FIRST column of the prediction lag range (lag 3 for
                lag=[3,10]) and returns uncentred fit/se, but its inline comment says "sum of lag-specific effects at
                lag 0".  Behaviour is pinned by plain tests; the wording by one documentation test.

Known-defect tests (strict xfail until the fix lands)
  test_manager_find_mmt_follows_changed_grid              centering-16   A-grid then B-grid on one manager (R per grid)
  test_manager_find_mmt_is_order_independent              centering-16   fresh manager, B-grid then A-grid
  test_manager_find_mmt_follows_method                    centering-16   overall then lagspecific
  test_manager_find_mmt_not_stale_after_model_or_basis_change  centering-16   cm.model / cm.basis re-assigned
  test_compare_strategies_mmt_uses_prediction_grid        centering-16   MMT searched on the `at` grid of the predictions
  test_compare_strategies_keeps_one_result_per_strategy   centering-16   equal centering values must not collapse
  test_recenter_at_mmt_is_recorded_in_history             centering-16   recenter_at_mmt bypasses the manager's history
  test_blup_fun_ns_is_a_natural_spline                    centering-17   4-column BLUP for fun='ns' (R onebasis ns)
  test_blup_fun_is_not_silently_ignored                   centering-17   5-column BLUP fits bs(deg 1) and ns: R ns answer
  test_blup_unknown_fun_raises                            centering-17   R: match.fun error; Python builds a B-spline
  test_blup_nan_coefficient_is_an_error                   centering-17   R which.min -> integer(0); Python percentile 1
  test_blup_accepts_python_list_for_x                     centering-17   list x == ndarray x (R quantile on a vector)
  test_lagspecific_documentation_matches_behaviour        centering-20   comment/docstring say what column is used

Plain tests (already faithful, guard the neighbourhood while the fixes land)
  test_manager_first_find_mmt_matches_r                   first call of a fresh manager == R which.min
  test_manager_repeated_identical_call_is_consistent      same arguments twice -> same MMT
  test_find_mmt_function_with_model_object_matches_r      uncached function on two grids (defect is manager-only)
  test_recenter_at_mmt_uses_requested_grid_each_time      recenter_at_mmt(at=A), recenter_at_mmt(at=B) -> own MMT each
  test_recenter_at_value_history_and_copy                 history entry + get_centering_history() returns a copy
  test_compare_centering_matches_r_crosspred_per_value    compare_centering == R crosspred(cen=v) per value
  test_compare_strategies_automatic_values_match_r        mean / median / percentile_X / custom centring values
  test_find_mmt_lagspecific_uses_first_prediction_lag     lag=[0,8],[3,10],[1,6]: column 1 of R matfit, not allfit
  test_blup_zero_coefficients_take_first_percentile       flat risk -> which.min = first
  test_blup_list_coefficients_match_array                 BLUP given as a list
"""
import contextlib
import inspect
import io
import os
import re
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from rhelpers import assert_close, chicago, known_defect, np2r, r, rget      # rhelpers first: it starts R
from rpy2.rinterface_lib.embedded import RRuntimeError


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
@contextlib.contextmanager
def _quiet():
    """PyDLNM prints progress lines and warns about grids; neither matters here."""
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        yield


class _LogLinkModel:
    """Minimal fitted-model stand-in: cross-basis coefficients / vcov of a log-link GLM (what R's glm(quasipoisson)
    provides).  PyDLNM reads params / cov_params / family.link from statsmodels-like objects."""

    def __init__(self, coef, vcov):
        self.params = np.asarray(coef, dtype=float)
        self.cov_params = np.asarray(vcov, dtype=float)
        self.family = SimpleNamespace(link='log')


GRID_A = np.arange(-15.0, 30.001, 0.5)        # wide grid: R's MMT is 25 (death) on it
GRID_B = np.arange(-10.0, 10.001, 0.5)        # narrow grid that excludes that MMT: R's MMT is 10 (upper end)
GRID_C = np.arange(-14.75, 30.0, 0.5)         # offset grid (R's MMT 25.25): no integer / pretty() / linspace default grid has it


@pytest.fixture(scope='module')
def fits():
    """chicagoNMMAPS: bs(deg 2, P10/75/90) x ns(logknots) lag 10 cross-basis fitted in R for three outcomes, and a second,
    different cross-basis (ns x ns, lag 6).  The same coef / vcov feed R and PyDLNM."""
    from basis import CrossBasis
    from utils import logknots
    _pin_r_home()
    temp = chicago()['temp']
    kv = np.quantile(temp, [.10, .75, .90])
    lk = logknots([0, 10], nk=2)
    kv2 = np.quantile(temp, [.25, .50, .75])
    lk2 = logknots([0, 6], nk=2)
    np2r('cm3_x', temp); np2r('cm3_kv', kv); np2r('cm3_lk', lk); np2r('cm3_kv2', kv2); np2r('cm3_lk2', lk2)
    r('''
    cm3_cb <- crossbasis(cm3_x, lag=10, argvar=list(fun="bs", degree=2, knots=cm3_kv),
                         arglag=list(fun="ns", knots=cm3_lk))
    cm3_cb2 <- crossbasis(cm3_x, lag=6, argvar=list(fun="ns", knots=cm3_kv2), arglag=list(fun="ns", knots=cm3_lk2))
    for (o in c("death", "resp")) {
      m <- glm(as.formula(paste(o, "~ cm3_cb + ns(time, 7*14) + dow")), family=quasipoisson(), data=chicagoNMMAPS)
      ind <- grep("cm3_cb", names(coef(m)))
      assign(paste0("cm3_coef_", o), unname(coef(m)[ind]))
      assign(paste0("cm3_vcov_", o), unname(vcov(m)[ind, ind]))
    }
    m <- glm(death ~ cm3_cb2 + ns(time, 7*14) + dow, family=quasipoisson(), data=chicagoNMMAPS)
    ind <- grep("cm3_cb2", names(coef(m)))
    cm3_coef_b2 <- unname(coef(m)[ind]); cm3_vcov_b2 <- unname(vcov(m)[ind, ind])
    ''')
    with _quiet():
        cb = CrossBasis(temp, lag=10, argvar={'fun': 'bs', 'degree': 2, 'knots': kv},
                        arglag={'fun': 'ns', 'knots': lk})
        cb2 = CrossBasis(temp, lag=6, argvar={'fun': 'ns', 'knots': kv2}, arglag={'fun': 'ns', 'knots': lk2})
    _pin_r_home()
    return SimpleNamespace(
        temp=temp, cb=cb, cb2=cb2,
        coef=rget('cm3_coef_death'), vcov=rget('cm3_vcov_death'),
        coef_resp=rget('cm3_coef_resp'), vcov_resp=rget('cm3_vcov_resp'),
        coef_b2=rget('cm3_coef_b2'), vcov_b2=rget('cm3_vcov_b2'),
        model=_LogLinkModel(rget('cm3_coef_death'), rget('cm3_vcov_death')),
        model_resp=_LogLinkModel(rget('cm3_coef_resp'), rget('cm3_vcov_resp')),
        model_b2=_LogLinkModel(rget('cm3_coef_b2'), rget('cm3_vcov_b2')))


def _r_pred(cb, coef, vcov, grid, cen=None):
    """R crosspred of R object `cb` with coefficient objects `coef`/`vcov` (R names) on `grid`; result in cm3_p."""
    np2r('cm3_grid', grid)
    cen_arg = f', cen={float(cen)!r}' if cen is not None else f', cen={float(grid[0])!r}'
    r(f'cm3_p <- suppressWarnings(crosspred({cb}, coef={coef}, vcov={vcov}, model.link="log", at=cm3_grid{cen_arg}))')


def _r_mmt(cb, coef, vcov, grid, first_lag=False):
    """predvar[which.min(allfit)] of R's crosspred (or of the first column of matfit); MMT is invariant to cen."""
    _r_pred(cb, coef, vcov, grid)
    fit = rget('cm3_p$matfit[, 1]') if first_lag else rget('as.numeric(cm3_p$allfit)')
    pv = rget('as.numeric(cm3_p$predvar)')
    i = int(np.argmin(fit))
    assert int(rget('which.min(%s)' % ('cm3_p$matfit[, 1]' if first_lag else 'cm3_p$allfit'))[0]) - 1 == i
    return float(pv[i])


def _records(comparison):
    """All per-centering result records (dicts that carry `allfit`) of a compare_centering[_strategies] result,
    whatever container (dict keyed by name, list, nested) holds them.  One entry per holder: a record that is reachable
    under two keys (two strategies, or the legacy key as well) is counted twice."""
    out = []

    def walk(v):
        if isinstance(v, dict):
            if 'allfit' in v:
                out.append(v)
                return
            for w in v.values():
                walk(w)
        elif isinstance(v, (list, tuple)):
            for w in v:
                walk(w)
    walk(comparison)
    return out


# ===================================== centering-16: CenteringManager.find_mmt cache =================================
def test_manager_first_find_mmt_matches_r(fits):
    """Plain: the first call of a fresh manager equals R's which.min(allfit) on the same grid."""
    from centering import CenteringManager
    cm = CenteringManager(fits.cb, fits.model)
    with _quiet():
        res = cm.find_mmt(at=GRID_A)
    assert res['method'] == 'overall'
    assert float(res['mmt']) == _r_mmt('cm3_cb', 'cm3_coef_death', 'cm3_vcov_death', GRID_A)


def test_manager_repeated_identical_call_is_consistent(fits):
    """Plain: asking twice with the same arguments gives the same answer (true with or without a cache)."""
    from centering import CenteringManager
    cm = CenteringManager(fits.cb, fits.model)
    with _quiet():
        a = cm.find_mmt(at=GRID_A)
        b = cm.find_mmt(at=GRID_A)
    assert float(a['mmt']) == float(b['mmt'])
    assert_close(a['all_fit'], b['all_fit'], rtol=0, what='all_fit of a repeated identical call')


def test_find_mmt_function_with_model_object_matches_r(fits):
    """Plain: the uncached function find_mmt(basis, model, at=grid) is R-exact on two different grids, so the stale
    results below come from the manager's cache only."""
    from centering import find_mmt
    for grid in (GRID_A, GRID_B):
        with _quiet():
            res = find_mmt(fits.cb, fits.model, at=grid)
        assert_close(res['predvar'], grid, rtol=0, what='searched grid')
        assert float(res['mmt']) == _r_mmt('cm3_cb', 'cm3_coef_death', 'cm3_vcov_death', grid)


@known_defect('M3', 'centering-16', note='CenteringManager._mmt_cache is filled once and never looks at `at`')
def test_manager_find_mmt_follows_changed_grid(fits):
    """find_mmt(at=A) then find_mmt(at=B) on one manager: the second answer is R's MMT on grid B (10.0), not the
    cached MMT of grid A (25.0, which is not even inside grid B)."""
    from centering import CenteringManager
    mmt_a = _r_mmt('cm3_cb', 'cm3_coef_death', 'cm3_vcov_death', GRID_A)
    mmt_b = _r_mmt('cm3_cb', 'cm3_coef_death', 'cm3_vcov_death', GRID_B)
    assert mmt_a != mmt_b, 'test design: the two grids must have different R MMTs'
    cm = CenteringManager(fits.cb, fits.model)
    with _quiet():
        first = cm.find_mmt(at=GRID_A)
        second = cm.find_mmt(at=GRID_B)
    assert float(first['mmt']) == mmt_a
    assert float(second['mmt']) == mmt_b, f'2nd call MMT {float(second["mmt"])!r}: R on grid B gives {mmt_b!r}'
    assert_close(second['predvar'], GRID_B, rtol=0, what='grid searched by the 2nd call')


@known_defect('M3', 'centering-16', note='result depends on which call came first')
def test_manager_find_mmt_is_order_independent(fits):
    """Fresh manager, grid B first and grid A second: the answer for A must still be A's (R: 25.0)."""
    from centering import CenteringManager
    mmt_a = _r_mmt('cm3_cb', 'cm3_coef_death', 'cm3_vcov_death', GRID_A)
    mmt_b = _r_mmt('cm3_cb', 'cm3_coef_death', 'cm3_vcov_death', GRID_B)
    cm = CenteringManager(fits.cb, fits.model)
    with _quiet():
        first = cm.find_mmt(at=GRID_B)
        second = cm.find_mmt(at=GRID_A)
    assert float(first['mmt']) == mmt_b
    assert float(second['mmt']) == mmt_a, f'grid A asked second: MMT {float(second["mmt"])!r} vs R {mmt_a!r}'


@known_defect('M3', 'centering-16', note='cached dict of method="overall" is returned for method="lagspecific"')
def test_manager_find_mmt_follows_method(fits):
    """overall then lagspecific on the same grid: the 2nd result is the lag-specific one (method label, the curve of
    R's first matfit column, and its MMT, which differs from the overall MMT)."""
    from centering import CenteringManager
    args = ('cm3_cb', 'cm3_coef_death', 'cm3_vcov_death', GRID_A)
    mmt_overall = _r_mmt(*args)
    mmt_lag = _r_mmt(*args, first_lag=True)
    assert mmt_overall != mmt_lag, 'test design: overall and first-lag MMT must differ in R'
    lag_curve = rget('cm3_p$matfit[, 1]')                      # cm3_p still holds the R call made by _r_mmt
    cm = CenteringManager(fits.cb, fits.model)
    with _quiet():
        first = cm.find_mmt(at=GRID_A)
        second = cm.find_mmt(at=GRID_A, method='lagspecific')
    assert float(first['mmt']) == mmt_overall
    assert second['method'] == 'lagspecific', f'method of the 2nd result is {second["method"]!r}'
    assert_close(np.asarray(second['all_fit']) - second['all_fit'][0], lag_curve - lag_curve[0], rtol=1e-9,
                 what='lag-specific curve (up to the centring constant)')
    assert float(second['mmt']) == mmt_lag


@known_defect('M3', 'centering-16', note='cache not invalidated when manager.model / manager.basis are re-assigned')
@pytest.mark.parametrize('change', ['model_only', 'basis_and_model'])
def test_manager_find_mmt_not_stale_after_model_or_basis_change(fits, change):
    """The public attributes `model` (and `basis`) of the manager are re-assigned; a repeated find_mmt(at=grid) must be
    computed for the NEW model / basis, i.e. equal R's MMT of the new fit (refit on another outcome: 24.0 vs 25.0;
    another cross-basis: 19.0 vs 25.0)."""
    from centering import CenteringManager
    old = _r_mmt('cm3_cb', 'cm3_coef_death', 'cm3_vcov_death', GRID_A)
    if change == 'model_only':
        new_r = _r_mmt('cm3_cb', 'cm3_coef_resp', 'cm3_vcov_resp', GRID_A)
    else:
        new_r = _r_mmt('cm3_cb2', 'cm3_coef_b2', 'cm3_vcov_b2', GRID_A)
    assert new_r != old, 'test design: the new fit must have a different R MMT'
    cm = CenteringManager(fits.cb, fits.model)
    with _quiet():
        before = cm.find_mmt(at=GRID_A)
    assert float(before['mmt']) == old
    if change == 'model_only':
        cm.model = fits.model_resp
    else:
        cm.basis, cm.model = fits.cb2, fits.model_b2
    with _quiet():
        after = cm.find_mmt(at=GRID_A)
    assert float(after['mmt']) == new_r, f'after {change}: MMT {float(after["mmt"])!r} vs R {new_r!r} (stale: {old!r})'


# ============================= centering-16: compare_centering_strategies / compare_centering ========================
@known_defect('M3', 'centering-16', note="compare_centering_strategies calls self.find_mmt() bare: 21-point default grid")
def test_compare_strategies_mmt_uses_prediction_grid(fits):
    """compare_centering_strategies(['mmt'], at=grid): the MMT used as centring value is the one of the same `at` grid
    the predictions use (R: which.min of crosspred allfit on that grid = 25.25), not of a hidden default grid."""
    from centering import CenteringManager
    mmt_r = _r_mmt('cm3_cb', 'cm3_coef_death', 'cm3_vcov_death', GRID_C)
    assert mmt_r % 0.5 != 0, 'test design: the MMT must not lie on a round default grid'
    cm = CenteringManager(fits.cb, fits.model)
    with _quiet():
        cmp_ = cm.compare_centering_strategies(strategies=['mmt'], at=GRID_C)
    used = float(cmp_['summary']['centering_values'][0])
    assert used == mmt_r, f'MMT used for centring {used!r} vs R MMT on the prediction grid {mmt_r!r}'
    # and the prediction made with it is R's prediction centred there
    _r_pred('cm3_cb', 'cm3_coef_death', 'cm3_vcov_death', GRID_C, cen=mmt_r)
    recs = _records(cmp_)
    assert recs, 'no result record'
    for rec in recs:                                # one strategy: every record held is the MMT prediction
        assert_close(rec['allfit'], rget('as.numeric(cm3_p$allfit)'), rtol=1e-9, what='allfit centred at the MMT')


@known_defect('M3', 'centering-16', note="results keyed f'cen_{value}': equal centring values collapse, names are kept")
def test_compare_strategies_keeps_one_result_per_strategy(fits):
    """Median and a custom value equal to the median (and mean + equal custom mean) are different strategies: 4 names
    must come with (at least) 4 result records, so that every label can be mapped back to its result."""
    from centering import CenteringManager
    x = fits.temp[~np.isnan(fits.temp)]
    cm = CenteringManager(fits.cb, fits.model)
    with _quiet():
        cmp_ = cm.compare_centering_strategies(strategies=['mean', 'median'],
                                               custom_values=[float(np.median(x)), float(np.mean(x))], at=GRID_A)
    names = list(cmp_['strategy_names'])
    assert len(names) == 4, names
    assert len(_records(cmp_)) >= len(names), (
        f'{len(names)} strategies ({names}) but only {len(_records(cmp_))} result records: '
        f'keys {[k for k in cmp_ if str(k).startswith("cen_")]}')


def test_compare_centering_matches_r_crosspred_per_value(fits):
    """Plain: compare_centering(basis, model, values, at=grid): one record per distinct value whose allfit / allse /
    RR equal R's crosspred(cb, coef, vcov, cen=value, at=grid)."""
    from centering import compare_centering
    values = [-5.0, 10.0, 20.5, 25.0]
    with _quiet():
        res = compare_centering(fits.cb, fits.model, values, at=GRID_A)
    for v in values:
        rec = res[f'cen_{v}']
        assert rec['centering'] == v
        _r_pred('cm3_cb', 'cm3_coef_death', 'cm3_vcov_death', GRID_A, cen=v)
        assert_close(rec['predvar'], GRID_A, rtol=0, what=f'cen={v}: predvar')
        assert_close(rec['allfit'], rget('as.numeric(cm3_p$allfit)'), rtol=1e-9, what=f'cen={v}: allfit')
        assert_close(rec['allse'], rget('as.numeric(cm3_p$allse)'), rtol=1e-9, what=f'cen={v}: allse')
        assert_close(rec['allRRfit'], rget('as.numeric(cm3_p$allRRfit)'), rtol=1e-9, what=f'cen={v}: allRRfit')
    assert res['summary']['centering_values'] == values and res['summary']['successful_runs'] == len(values)


def test_compare_strategies_automatic_values_match_r(fits):
    """Plain: 'mean' / 'median' / 'percentile_25' / a custom value are R's mean(), median(), quantile(, .25) and the
    value itself, and each one's prediction is R's crosspred centred there (no MMT strategy, no collision)."""
    from centering import CenteringManager
    np2r('cm3_xs', fits.temp)
    expected = [float(rget('mean(cm3_xs)')[0]), float(rget('median(cm3_xs)')[0]),
                float(rget('quantile(cm3_xs, .25)')[0]), 12.5]
    cm = CenteringManager(fits.cb, fits.model)
    with _quiet():
        cmp_ = cm.compare_centering_strategies(strategies=['mean', 'median', 'percentile_25'], custom_values=[12.5],
                                               at=GRID_A)
    got = [float(v) for v in cmp_['summary']['centering_values']]
    assert_close(got, expected, rtol=1e-12, what='centring values')
    assert len(cmp_['strategy_names']) == 4
    recs = _records(cmp_)
    assert len(recs) >= 4
    for v in expected:
        _r_pred('cm3_cb', 'cm3_coef_death', 'cm3_vcov_death', GRID_A, cen=v)
        ref = rget('as.numeric(cm3_p$allfit)')
        assert any(np.allclose(rec['allfit'], ref, rtol=1e-9, atol=1e-12) for rec in recs), \
            f'no result record equals the R prediction centred at {v}'


# ======================================== centering-16: recenter_at_mmt vs manager history ===========================
@known_defect('M3', 'centering-16', note='recenter_at_mmt calls recenter_basis directly: no history entry')
def test_recenter_at_mmt_is_recorded_in_history(fits):
    """recenter_at_value() appends to the manager's history; recenter_at_mmt() must too (one code path), with the MMT
    that R finds on the same grid as the centring value."""
    from centering import CenteringManager
    mmt_r = _r_mmt('cm3_cb', 'cm3_coef_death', 'cm3_vcov_death', GRID_A)
    cm = CenteringManager(fits.cb, fits.model)
    with _quiet():
        rb, info = cm.recenter_at_mmt(at=GRID_A)
    assert float(rb.argvar['cen']) == mmt_r
    hist = cm.get_centering_history()
    assert len(hist) == 1, f'history after recenter_at_mmt: {hist!r}'
    assert float(hist[0]['value']) == mmt_r and 'mmt' in str(hist[0].get('method', '')).lower(), hist[0]


def test_recenter_at_mmt_uses_requested_grid_each_time(fits):
    """Plain: recenter_at_mmt(at=A) and then recenter_at_mmt(at=B) on the same manager centre at R's MMT of each grid
    (it does not go through the stale cache; a fix that routes it through find_mmt must keep this)."""
    from centering import CenteringManager
    cm = CenteringManager(fits.cb, fits.model)
    for grid in (GRID_A, GRID_B, GRID_A):
        with _quiet():
            rb, info = cm.recenter_at_mmt(at=grid)
        mmt_r = _r_mmt('cm3_cb', 'cm3_coef_death', 'cm3_vcov_death', grid)
        assert float(rb.argvar['cen']) == mmt_r and info['method'] == 'mmt' and float(info['value']) == mmt_r


def test_recenter_at_value_history_and_copy(fits):
    """Plain: recenter_at_value records {'method': 'manual', 'value': v}; get_centering_history() hands out a copy."""
    from centering import CenteringManager
    cm = CenteringManager(fits.cb, fits.model)
    assert cm.get_centering_history() == []
    with _quiet():
        rb, info = cm.recenter_at_value(12.0)
    assert float(rb.argvar['cen']) == 12.0 and info['method'] == 'manual' and info['value'] == 12.0
    hist = cm.get_centering_history()
    assert len(hist) == 1 and hist[0]['method'] == 'manual' and hist[0]['value'] == 12.0
    hist.append({'method': 'x', 'value': 0.0})
    assert len(cm.get_centering_history()) == 1, 'get_centering_history() must return a copy of the list'


# ============================================ centering-17: find_mmt_blup ============================================
def _x_series(which):
    t = chicago()['temp']
    return {'a': t[:2600].copy(), 'b': t[2600:].copy()}[which]


def _r_blup_full_range(x, blup, fun, knots, degree=2):
    """R: onebasis(quantile(x, (0:100)/100), fun, knots, Boundary.knots=range(x)) %*% blup and the which.min."""
    np2r('cm3_bx', x); np2r('cm3_bk', knots); np2r('cm3_bblup', blup)
    extra = f', degree={int(degree)}L' if fun == 'bs' else ''
    r(f'''
    cm3_bpv <- quantile(cm3_bx, (0:100)/100, na.rm=TRUE)
    cm3_bbv <- onebasis(cm3_bpv, fun="{fun}", knots=cm3_bk{extra}, Boundary.knots=range(cm3_bx, na.rm=TRUE))
    cm3_blp <- as.numeric(cm3_bbv %*% cm3_bblup)
    cm3_bi  <- which.min(cm3_blp)
    ''')
    i = int(rget('cm3_bi')[0]) - 1
    pv = rget('as.numeric(cm3_bpv)')
    return dict(risk=rget('cm3_blp'), percentile=i, mmt=float(pv[i]), ncol=int(rget('ncol(cm3_bbv)')[0]))


def _py_blup(x, blup, **kw):
    from centering import find_mmt_blup
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        res = find_mmt_blup(x, blup, **kw)
    return res, [str(w.message) for w in caught]


@known_defect('M3', 'centering-17', note="`fun` is never used: fun='ns' builds a B-spline with the wrong column count")
@pytest.mark.parametrize('which,seed', [('a', 11), ('b', 12)])
def test_blup_fun_ns_is_a_natural_spline(which, seed):
    """fun='ns' with 3 knots has 4 columns in R's onebasis: a 4-coefficient BLUP gives R's risk curve and MMT (grid
    percentiles 0..100 = range(x), so the boundary knots cannot differ).  Python returns the median fallback."""
    x = _x_series(which)
    knots = np.percentile(x, [10, 75, 90])
    blup = np.random.default_rng(seed).normal(0.0, 0.1, 4)
    ref = _r_blup_full_range(x, blup, 'ns', knots)
    assert ref['ncol'] == 4
    res, _ = _py_blup(x, blup, fun='ns', knots=knots, percentile_range=(0, 100))
    assert res.get('method') == 'blup_optimization', f'no BLUP result: {res.get("method")} ({res.get("error")})'
    assert_close(res['risk_values'], ref['risk'], rtol=1e-9, what='ns basis %*% blup')
    assert int(res['percentile']) == ref['percentile'] and float(res['mmt']) == ref['mmt']


@known_defect('M3', 'centering-17', note="`fun='ns'` silently returns the B-spline answer (same column count)")
def test_blup_fun_is_not_silently_ignored():
    """4 interior knots: bs(degree=1) and ns both have 5 columns, so a 5-coefficient BLUP is conformable with either and
    nothing fails -- but the curves differ.  fun='ns' must give R's ns risk curve, not the bs(degree=1) one."""
    x = _x_series('a')
    knots = np.percentile(x, [5, 30, 60, 90])
    blup = np.random.default_rng(5).normal(0.0, 0.1, 5)
    ref_ns = _r_blup_full_range(x, blup, 'ns', knots)
    ref_bs = _r_blup_full_range(x, blup, 'bs', knots, degree=1)
    assert ref_ns['ncol'] == ref_bs['ncol'] == 5
    assert np.abs(ref_ns['risk'] - ref_bs['risk']).max() > 1e-3, 'test design: R ns and bs(1) curves must differ'
    res, _ = _py_blup(x, blup, fun='ns', degree=1, knots=knots, percentile_range=(0, 100))
    assert res.get('method') == 'blup_optimization'
    assert_close(res['risk_values'], ref_ns['risk'], rtol=1e-9, what='fun="ns" risk curve')
    assert int(res['percentile']) == ref_ns['percentile'] and float(res['mmt']) == ref_ns['mmt']


@known_defect('M3', 'centering-17', note='unsupported `fun` is accepted and a B-spline is built')
def test_blup_unknown_fun_raises():
    """R: onebasis(fun="nosuchfun") stops (match.fun).  find_mmt_blup must reject it instead of answering with a
    B-spline."""
    x = _x_series('a')
    knots = np.percentile(x, [10, 75, 90])
    blup = np.random.default_rng(1).normal(0.0, 0.1, 5)
    np2r('cm3_ux', x[:200])
    with pytest.raises(RRuntimeError):
        r('onebasis(cm3_ux, fun="nosuchfun")')
    with pytest.raises((ValueError, NotImplementedError)):
        res, _ = _py_blup(x, blup, fun='nosuchfun', knots=knots)
        pytest.fail(f'no error; returned method={res.get("method")!r} mmt={res.get("mmt")!r}')


@known_defect('M3', 'centering-17', note='np.argmin returns the first NaN: MMT at percentile 1')
@pytest.mark.parametrize('pos', [0, 2, 4])
def test_blup_nan_coefficient_is_an_error(pos):
    """A NaN BLUP coefficient makes every risk value NaN; R's which.min() then returns integer(0) (no MMT) and the
    Lancet script's assignment fails.  Python must not report the first percentile as the MMT."""
    x = _x_series('a')
    blup = np.random.default_rng(3).normal(0.0, 0.1, 5)
    blup[pos] = np.nan
    np2r('cm3_nx', x); np2r('cm3_nblup', blup)
    r('''
    cm3_npv <- quantile(cm3_nx, 1:99/100)
    cm3_nbv <- onebasis(cm3_npv, fun="bs", knots=quantile(cm3_nx, c(10, 75, 90)/100), degree=2,
                        Boundary.knots=range(cm3_nx))
    cm3_nlp <- as.numeric(cm3_nbv %*% cm3_nblup)
    ''')
    assert bool(rget('all(is.na(cm3_nlp))')[0]) and int(rget('length(which.min(cm3_nlp))')[0]) == 0
    with pytest.raises(ValueError):
        res, _ = _py_blup(x, blup)
        pytest.fail(f'no error; returned method={res.get("method")!r} percentile={res.get("percentile")!r} '
                    f'mmt={res.get("mmt")!r}')


@known_defect('M3', 'centering-17', note='x[~np.isnan(x)] on a list raises TypeError before the try block')
@pytest.mark.parametrize('prange', [(1, 99), (0, 100)])
def test_blup_accepts_python_list_for_x(prange):
    """R's quantile() takes a plain vector; a Python list for x must give the same result as the ndarray (and so not
    raise TypeError)."""
    x = _x_series('b')
    blup = np.random.default_rng(8).normal(0.0, 0.1, 5)
    ref, _ = _py_blup(x, blup, percentile_range=prange)
    res, _ = _py_blup(x.tolist(), blup, percentile_range=prange)
    assert res['method'] == ref['method']
    assert_close(res['risk_values'], ref['risk_values'], rtol=0, what='risk values, list vs ndarray')
    assert res['percentile'] == ref['percentile'] and float(res['mmt']) == float(ref['mmt'])


def test_blup_list_coefficients_match_array():
    """Plain: the BLUP itself may be given as a list (matmul accepts it)."""
    x = _x_series('a')
    blup = np.random.default_rng(9).normal(0.0, 0.1, 5)
    ref, _ = _py_blup(x, blup)
    res, _ = _py_blup(x, blup.tolist())
    assert res['method'] == 'blup_optimization'
    assert_close(res['risk_values'], ref['risk_values'], rtol=0, what='risk values, list vs ndarray blup')
    assert res['percentile'] == ref['percentile'] and float(res['mmt']) == float(ref['mmt'])


def test_blup_zero_coefficients_take_first_percentile():
    """Plain: a flat risk curve -> which.min picks the first grid point, i.e. the 1st percentile (np.argmin agrees)."""
    x = _x_series('a')
    np2r('cm3_zx', x)
    assert int(rget('which.min(rep(0, 99))')[0]) == 1
    res, _ = _py_blup(x, np.zeros(5))
    assert res['method'] == 'blup_optimization' and int(res['percentile']) == 1
    assert float(res['mmt']) == float(rget('quantile(cm3_zx, 0.01)')[0])


# ============================================ centering-20: lagspecific ==============================================
LAG_CASES = [(0, 8), (3, 10), (1, 6)]


@pytest.mark.parametrize('lag', LAG_CASES, ids=lambda t: f'lag{t[0]}-{t[1]}')
def test_find_mmt_lagspecific_uses_first_prediction_lag(lag):
    """Plain: method='lagspecific' searches the FIRST column of the prediction lag range (lag 3 for lag=[3,10]): its
    curve equals R's matfit[, 1] up to the centring constant, its MMT is which.min of that column, and it is neither
    R's overall (summed) curve nor the lag-0 curve when the range starts later."""
    from basis import CrossBasis
    from centering import find_mmt
    from utils import logknots
    x = chicago()['temp']
    kv = np.quantile(x, [.10, .75, .90])
    lk = logknots(list(lag), nk=2)
    np2r('cm3_lx', x); np2r('cm3_lkv', kv); np2r('cm3_llk', lk)
    r(f'cm3_lcb <- crossbasis(cm3_lx, lag=c({lag[0]}, {lag[1]}), argvar=list(fun="bs", degree=2, knots=cm3_lkv), '
      f'arglag=list(fun="ns", knots=cm3_llk))')
    n = int(rget('ncol(cm3_lcb)')[0])
    coef = np.random.default_rng(9 + lag[0]).normal(size=n) * 0.04
    vcov = np.eye(n) * 0.001
    np2r('cm3_lcoef', coef); np2r('cm3_lvcov', vcov)
    r(f'cm3_lvcov <- matrix(cm3_lvcov, {n})')
    grid = np.arange(-10.0, 30.001, 1.0)
    _r_pred('cm3_lcb', 'cm3_lcoef', 'cm3_lvcov', grid)
    mat = rget('cm3_p$matfit')
    allfit = rget('as.numeric(cm3_p$allfit)')
    assert list(r('colnames(cm3_p$matfit)'))[0] == f'lag{lag[0]}'
    with _quiet():
        cb = CrossBasis(x, lag=list(lag), argvar={'fun': 'bs', 'degree': 2, 'knots': kv},
                        arglag={'fun': 'ns', 'knots': lk})
        res = find_mmt(cb, None, coef=coef, vcov=vcov, at=grid, method='lagspecific')
    assert res['method'] == 'lagspecific'
    assert_close(np.asarray(res['all_fit']) - res['all_fit'][0], mat[:, 0] - mat[0, 0], rtol=1e-9,
                 what=f'lag-specific curve vs R matfit[, "lag{lag[0]}"]')
    assert float(res['mmt']) == float(grid[int(np.argmin(mat[:, 0]))])
    assert np.abs((np.asarray(res['all_fit']) - res['all_fit'][0]) - (allfit - allfit[0])).max() > 1e-3, (
        'lagspecific must not silently become the summed (overall) curve')


@known_defect('M3', 'centering-20', note="inline comment says 'sum of lag-specific effects at lag 0'; first lag is used")
def test_lagspecific_documentation_matches_behaviour():
    """The code searches the first lag of the prediction lag range (matfit[:, 0]: lag3 for lag=[3,10]; see the plain test
    above), not lag 0 and not a sum.  The comment must not claim that, and the docstring (or an explicit `lag`
    argument) must tell the user which lag 'lagspecific' uses."""
    import centering
    src = ' '.join(inspect.getsource(centering.find_mmt).lower().split())
    assert 'sum of lag-specific effects at lag 0' not in src, 'stale inline comment about a sum at lag 0 is still there'
    doc = ' '.join((inspect.getdoc(centering.find_mmt) or '').lower().split())
    says_which_lag = re.search(r'(first|lowest|minimum|smallest|earliest)\s+(prediction\s+)?(lag|column)', doc) is not None
    has_lag_argument = 'lag' in inspect.signature(centering.find_mmt).parameters
    assert says_which_lag or has_lag_argument, (
        "find_mmt docstring does not say which lag method='lagspecific' uses (it is the first lag of the "
        'prediction lag range) and there is no explicit `lag` argument')
