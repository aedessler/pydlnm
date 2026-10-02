"""crosspred centering (theme G) and prediction grid (theme H1b): PyDLNM vs R dlnm::crosspred.

R's crosspred() builds its exposure grid with mkat() and its centering value with mkcen(); PyDLNM's
CrossPred._setup_predictions is an ad-hoc re-implementation of both. Values at a given exposure point are right
(1e-16); what differs is WHICH points are predicted (grid) and WHICH reference the RR/fit is relative to (cen).

Theme G   (mkcen: auto-centering, cen=TRUE/FALSE, cen dropped when the var basis has intercept=TRUE)
  basis-cont-9      cen unspecified -> R sets median(pretty(range)) for ns/bs/poly; cen=TRUE means the same, cen=FALSE
                    means none, logical cen is ignored for lin/strata/thr/integer
  basis-cont-11     var basis with intercept=TRUE: R sets cen=NULL even if the user supplies a number
Theme H1b (mkat: prediction grid)
  crosspred-core-3  default grid = pretty(range, n=50) inside [from,to]; by= grid starts at min(pretty) and never
                    passes `to`; vector `at` -> sort(unique(at)) with NA dropped
  crosspred-grid-2  same (default 21-point linspace, from/to without by, by-grid start and overshoot)
  crosspred-grid-11 vector `at` sorted / de-duplicated / NA-stripped, scalar `at` accepted
  centering-5       same grid defects, seen through find_mmt (argmin over the grid; NaN in `at`)
  validation-audit-7  np.arange idiom overshoots `to` in crosspred from/to/by; default grid; `at` (its seqlag()/bylag
                    half is theme H1a: extra lag column when bylag does not divide the lag range)

Test design
* Each G test passes an explicit sorted-unique `at` and each H1b test an explicit numeric `cen`, so a test fails for
  exactly one defect; the two "fully default call" tests need both fixes.
* Bases have explicit knots AND explicit Boundary.knots (var and lag), so basis-construction defaults and the
  lag-Boundary.knots defect (theme A2) are not confounds. thr is only checked through the reported `cen`.
* Not covered here (other defects mask them): intercept=TRUE on a CrossBasis var basis (CrossBasis cannot be built),
  matrix-valued `at` (theme H2).
* The reference is always computed by R at run time (crosspred / dlnm:::seqlag); nothing is copied from Python output.
* known_defect tests assert the R-faithful behaviour and are strict xfails today; the un-marked tests guard
  behaviour that is already faithful (explicit cen, explicit sorted at, aligned from/to/by grids, ...).
"""
import contextlib
import functools
import io
import warnings

import numpy as np
import pytest

from rhelpers import assert_close, chicago, known_defect, np2r, r, rget

LAG = 5                       # short lag keeps every crosspred call in the millisecond range
FIELDS = ('predvar', 'matfit', 'matse', 'allfit', 'allse')


# ---------------------------------------------------------------------------------------------------------------
# helpers: paired R / Python bases with fixed coef, vcov; R and Python crosspred wrappers; comparison
# ---------------------------------------------------------------------------------------------------------------
def _temp(data):
    """'chicago': R's chicagoNMMAPS temperature; 'warm' / 'narrow': affine variants whose range (and so pretty()
    grid and auto-centering value) differs: 6.7..36.7 and -33.2..-29.7."""
    t = chicago()['temp']
    if data == 'chicago':
        return t
    if data == 'warm':
        return 0.5 * t + 20.0
    return -33.2 + 3.5 * (t - t.min()) / (t.max() - t.min())


AT_FOR = {'chicago': np.arange(-20.0, 30.0, 2.5), 'warm': np.arange(8.0, 36.0, 2.0),
          'narrow': np.arange(-33.0, -29.8, 0.2)}       # sorted, unique, inside each data range
CEN_FOR = {'chicago': 15.0, 'warm': 25.0, 'narrow': -31.0}


def _coefs(n, seed):
    rng = np.random.default_rng(seed)
    coef = rng.normal(0.0, 0.05, n)
    a = rng.normal(0.0, 1.0, (n, n)) * 0.01
    return coef, a @ a.T + np.eye(n) * 1e-5


class Case:
    """A basis built identically in R (global `g1_<tag>_b`) and PyDLNM, with fixed coef / vcov pushed to R."""

    def __init__(self, tag, py, seed):
        self.tag, self.py = tag, py
        self.rb, self.rcoef, self.rvcov = f'g1_{tag}_b', f'g1_{tag}_coef', f'g1_{tag}_vcov'
        n = int(rget(f'ncol({self.rb})')[0])
        assert tuple(py.shape) == (len(py.x), n), f'{tag}: basis shape differs between R and Python'
        self.coef, self.vcov = _coefs(n, seed)
        np2r(self.rcoef, self.coef)
        np2r(self.rvcov, self.vcov)
        r(f'{self.rvcov} <- matrix({self.rvcov}, {n})')


def _quiet(fn, *a, **kw):
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return fn(*a, **kw)


@functools.lru_cache(maxsize=None)
def make_cb(fun, data='chicago', cen=None):
    """cross-basis: var = bs(deg 2)/ns with 3 quantile knots, lag = ns with 2 log-knots (both boundary knots explicit)."""
    from basis import CrossBasis
    from utils import logknots
    x = _temp(data)
    tag = f'cb_{fun}_{data}' + ('' if cen is None else f'_cen{cen:g}')
    np2r('g1_x', x)
    np2r('g1_kv', np.quantile(x, [.10, .75, .90]))
    np2r('g1_bk', np.array([x.min(), x.max()]))
    np2r('g1_lk', logknots([0, LAG], nk=2))
    deg = ', degree=2' if fun == 'bs' else ''
    rcen = '' if cen is None else f', cen={cen:g}'
    r(f'g1_{tag}_b <- crossbasis(g1_x, lag={LAG}, argvar=list(fun="{fun}", knots=g1_kv, Boundary.knots=g1_bk{deg}{rcen}),'
      f' arglag=list(fun="ns", knots=g1_lk, Boundary.knots=c(0,{LAG})))')
    argvar = {'fun': fun, 'knots': np.quantile(x, [.10, .75, .90]), 'Boundary_knots': np.array([x.min(), x.max()])}
    if fun == 'bs':
        argvar['degree'] = 2
    if cen is not None:
        argvar['cen'] = cen
    arglag = {'fun': 'ns', 'knots': logknots([0, LAG], nk=2), 'Boundary_knots': np.array([0.0, LAG])}
    return Case(tag, _quiet(CrossBasis, x, lag=LAG, argvar=argvar, arglag=arglag), seed=3)


_ONE = {   # tag -> (R onebasis arguments, PyDLNM OneBasis keyword arguments); ns carries resolved knots/boundary
    'lin': ('fun="lin"', dict(fun='lin')),
    'lin_int': ('fun="lin", intercept=TRUE', dict(fun='lin', intercept=True)),
    'poly2': ('fun="poly", degree=2', dict(fun='poly', degree=2)),
    'poly2_int': ('fun="poly", degree=2, intercept=TRUE', dict(fun='poly', degree=2, intercept=True)),
    'ns': ('fun="ns", knots=g1_kv, Boundary.knots=g1_bk', dict(fun='ns')),
    'ns_int': ('fun="ns", knots=g1_kv, Boundary.knots=g1_bk, intercept=TRUE', dict(fun='ns', intercept=True)),
    'strata': ('fun="strata", breaks=c(0,10,20)', dict(fun='strata', breaks=np.array([0.0, 10.0, 20.0]))),
    'thr': ('fun="thr", thr.value=15', dict(fun='thr', thr_value=15.0)),
}


@functools.lru_cache(maxsize=None)
def make_one(kind, data='chicago'):
    from basis import OneBasis
    x = _temp(data)
    rargs, pkw = _ONE[kind]
    pkw = dict(pkw)
    if kind.startswith('ns'):
        pkw.update(knots=np.quantile(x, [.10, .75, .90]), Boundary_knots=np.array([x.min(), x.max()]))
    tag = f'ob_{kind}_{data}'
    np2r('g1_x', x)
    np2r('g1_kv', np.quantile(x, [.10, .75, .90]))
    np2r('g1_bk', np.array([x.min(), x.max()]))
    r(f'g1_{tag}_b <- onebasis(g1_x, {rargs})')
    return Case(tag, _quiet(OneBasis, x, **pkw), seed=1)


def _r_arg(v):
    if isinstance(v, (bool, np.bool_)):
        return 'TRUE' if v else 'FALSE'
    if np.ndim(v) == 0:
        return repr(float(v))
    return 'c(' + ','.join('NA' if np.isnan(z) else repr(float(z)) for z in np.ravel(v)) + ')'


def r_crosspred(case, **kw):
    """R's crosspred on the case's basis; kw in PyDLNM names (from_val/to_val -> from/to). Returns a dict."""
    names = {'from_val': 'from', 'to_val': 'to'}
    args = ''.join(f', {names.get(k, k)}={_r_arg(v)}' for k, v in kw.items())
    r(f'g1cp <- suppressMessages(crosspred({case.rb}, coef={case.rcoef}, vcov={case.rvcov}, model.link="log"{args}))')
    ref = {'predvar': rget('as.numeric(g1cp$predvar)'), 'matfit': rget('g1cp$matfit'), 'matse': rget('g1cp$matse'),
           'allfit': rget('as.numeric(g1cp$allfit)'), 'allse': rget('as.numeric(g1cp$allse)'),
           'cen': None if bool(r('is.null(g1cp$cen)')[0]) else float(r('g1cp$cen')[0])}
    return ref


def py_crosspred(case, **kw):
    from prediction import crosspred
    return _quiet(crosspred, case.py, coef=case.coef, vcov=case.vcov, model_link='log', **kw)


def assert_cen(py_cen, ref_cen, what='cen'):
    """R's pred$cen is NULL when uncentered; a bool is never a valid centering value in the result."""
    if ref_cen is None:
        assert py_cen is None, f'{what}: R is uncentered (cen=NULL) but PyDLNM cen={py_cen!r}'
    else:
        assert py_cen is not None and not isinstance(py_cen, (bool, np.bool_)), \
            f'{what}: R centres at {ref_cen} but PyDLNM cen={py_cen!r}'
        assert abs(float(py_cen) - ref_cen) <= 1e-12 * max(1.0, abs(ref_cen)), \
            f'{what}: R cen={ref_cen} vs PyDLNM cen={float(py_cen)}'


def check(case, fields=FIELDS, **kw):
    """Run crosspred in R and PyDLNM with identical arguments and compare grid, centering and all fits."""
    ref = r_crosspred(case, **kw)
    py = py_crosspred(case, **kw)
    assert_close(py.predvar, ref['predvar'], rtol=1e-12, what='predvar (prediction grid)')
    assert_cen(py.cen, ref['cen'])
    for f in fields:
        if f != 'predvar':
            assert_close(getattr(py, f), ref[f], rtol=1e-8, what=f)
    return py, ref


AT = np.arange(-20.0, 30.0, 2.5)          # sorted, unique, inside the data range: leaves the grid out of the picture
CEN = 15.0                                 # explicit numeric centering: leaves mkcen out of the picture

# ===============================================================================================================
# THEME G: mkcen
# ===============================================================================================================


# ---- basis-cont-9: cen unspecified -> median(pretty(range)) --------------------------------------------------------
@pytest.mark.parametrize('data', ['chicago', 'warm', 'narrow'])
@pytest.mark.parametrize('fun', ['bs', 'ns'])
def test_default_cen_crossbasis(fun, data):
    py, ref = check(make_cb(fun, data), at=AT_FOR[data])
    assert ref['cen'] is not None                       # R really auto-centres here (sanity of the reference)


@pytest.mark.parametrize('kind', ['ns', 'poly2'])
def test_default_cen_onebasis(kind):
    py, ref = check(make_one(kind), at=AT)
    assert ref['cen'] is not None


def test_default_cen_reduced_coefficients_route():
    """crosspred(cb, coef=<var-basis coefficients>) is R's crosspred(onebasis, coef, vcov); cen unspecified."""
    cb = make_cb('ns')
    ob = make_one('ns')                                 # same fun/knots/Boundary.knots as cb's var basis
    coef, vcov = _coefs(ob.coef.size, 5)
    np2r('g1_rc', coef)
    np2r('g1_rv', vcov)
    r(f'g1_rv <- matrix(g1_rv, {coef.size})')
    r(f'g1rp <- suppressMessages(crosspred({ob.rb}, coef=g1_rc, vcov=g1_rv, model.link="log", at={_r_arg(AT)}))')
    from prediction import crosspred
    py = _quiet(crosspred, cb.py, coef=coef, vcov=vcov, model_link='log', at=AT)
    ref_cen = None if bool(r('is.null(g1rp$cen)')[0]) else float(r('g1rp$cen')[0])
    assert_cen(py.cen, ref_cen)
    assert_close(py.allfit, rget('as.numeric(g1rp$allfit)'), what='allfit')
    assert_close(py.allse, rget('as.numeric(g1rp$allse)'), what='allse')


@pytest.mark.parametrize('cen', [True, False])
@pytest.mark.parametrize('fun', ['bs', 'ns'])
def test_cen_logical_crossbasis(fun, cen):
    """R: cen=TRUE -> same as unspecified (auto value); cen=FALSE -> uncentered (pred$cen is NULL)."""
    check(make_cb(fun), at=AT, cen=cen)


@pytest.mark.parametrize('cen', [True, False])
@pytest.mark.parametrize('kind', ['lin', 'strata', 'thr'])
def test_cen_logical_ignored_for_lin_strata_thr(kind, cen):
    # thr: only the reported cen is compared (OneBasis thr predictions differ from R for an unrelated reason)
    py, ref = check(make_one(kind), fields=() if kind == 'thr' else FIELDS, at=AT, cen=cen)
    assert ref['cen'] is None


# ---- basis-cont-11: intercept=TRUE in the var basis nullifies cen -----------------------------------------------------
@pytest.mark.parametrize('kind', ['lin_int', 'poly2_int', 'ns_int'])
def test_intercept_basis_drops_explicit_cen(kind):
    py, ref = check(make_one(kind), at=AT, cen=18.0)
    assert ref['cen'] is None                           # R silently ignores cen here


# ---- neighbours that are already faithful ---------------------------------------------------------------------------
@pytest.mark.parametrize('cen', [-5.0, 10, 15.0])
@pytest.mark.parametrize('fun', ['bs', 'ns'])
def test_explicit_cen_crossbasis(fun, cen):
    py, ref = check(make_cb(fun), at=AT, cen=cen)
    assert py.cen == cen


@pytest.mark.parametrize('kind', ['ns', 'poly2', 'lin', 'strata'])
def test_explicit_cen_onebasis(kind):
    check(make_one(kind), at=AT, cen=18.0)


def test_explicit_cen_is_reported_for_thr():
    check(make_one('thr'), fields=(), at=AT, cen=18.0)


@pytest.mark.parametrize('kind', ['lin', 'strata', 'thr'])
def test_no_autocentering_for_lin_strata_thr(kind):
    """lin/strata/thr/integer are never auto-centred by R; unspecified cen stays uncentered."""
    py, ref = check(make_one(kind), fields=() if kind == 'thr' else FIELDS, at=AT)
    assert ref['cen'] is None and py.cen is None


@pytest.mark.parametrize('kind', ['lin_int', 'poly2_int', 'ns_int'])
def test_intercept_basis_unspecified_cen_stays_uncentered(kind):
    py, ref = check(make_one(kind), at=AT)
    assert ref['cen'] is None and py.cen is None


def test_cen_stored_in_argvar_is_used_and_can_be_overridden():
    cb = make_cb('bs', cen=12)
    py, ref = check(cb, at=AT)                          # stored cen honoured by both
    assert ref['cen'] == 12.0
    check(cb, at=AT, cen=20.0)                          # explicit cen overrides it


# ===============================================================================================================
# THEME H1b: mkat (prediction grid)
# ===============================================================================================================
# ---- default grid: pretty(range, n=50) restricted to [from, to] --------------------------------------------------------
@pytest.mark.parametrize('kind,data', [('cb_bs', 'chicago'), ('cb_ns', 'chicago'), ('cb_bs', 'warm'),
                                       ('cb_bs', 'narrow'), ('ob_lin', 'chicago')])
def test_default_grid(kind, data):
    c = make_one('lin', data) if kind == 'ob_lin' else make_cb(kind[3:], data)
    py, ref = check(c, cen=CEN_FOR[data])
    assert ref['predvar'].size > 21                     # sanity: R's grid is the finer pretty(n=50) grid


_FROM_TO = [
    dict(from_val=0.0, to_val=25.0),                    # R 51 points (step 0.5), Python 21
    dict(from_val=0.0),
    dict(to_val=25.0),
    dict(from_val=5.0),
]


@pytest.mark.parametrize('kw', _FROM_TO, ids=lambda k: ','.join(f'{a}={b:g}' for a, b in k.items()))
def test_from_to_without_by_grid(kw):
    check(make_cb('bs'), cen=CEN, **kw)


_BY_DEFECT = [
    dict(by=1.0),                                       # R starts at -26 (first pretty value), Python at -26.667
    dict(by=0.5),
    dict(by=0.1),                                       # R 600 points, Python 601
    dict(by=7.0),                                       # Python predicts at 36.3 > data max 33.3, R never above `to`
    dict(from_val=-3.3, to_val=25.0, by=2.0),           # R starts at -3, Python at -3.3 (no common point)
    dict(from_val=-10.3, to_val=25.7, by=0.5),
    dict(from_val=0.05, to_val=30.02, by=0.1),          # Python's last point 30.05 > to
    dict(from_val=-3.0, to_val=31.0, by=2.5),
    dict(from_val=20.0, to_val=26.0, by=10.0),          # R: single point 20; Python also 30
]


@pytest.mark.parametrize('kw', _BY_DEFECT, ids=lambda k: ','.join(f'{a}={b:g}' for a, b in k.items()))
def test_by_grid(kw):
    check(make_cb('bs'), cen=CEN, **kw)


@pytest.mark.parametrize('kw', [dict(from_val=0.05, to_val=30.02, by=0.1), dict(from_val=-3.0, to_val=31.0, by=2.5),
                                dict(by=7.0), dict(by=0.7)], ids=lambda k: ','.join(f'{a}={b:g}' for a, b in k.items()))
def test_by_grid_never_exceeds_upper_limit(kw):
    py = py_crosspred(make_cb('bs'), cen=CEN, **kw)
    upper = kw.get('to_val', make_cb('bs').py.range[1])
    assert py.predvar.max() <= upper + 1e-9, f'grid reaches {py.predvar.max()} > to={upper} (R seq() never does)'
    assert py.predvar.size == r_crosspred(make_cb('bs'), cen=CEN, **kw)['predvar'].size


_BY_ALIGNED = [   # Python's arange start (from) coincides with R's min(pretty): both grids are identical today
    dict(from_val=0.0, to_val=30.0, by=0.5),
    dict(from_val=0.0, to_val=30.0, by=5.0),
    dict(from_val=0.0, to_val=10.0, by=0.5),
    dict(from_val=-10.0, to_val=30.0, by=0.25),
]


@pytest.mark.parametrize('kw', _BY_ALIGNED, ids=lambda k: ','.join(f'{a}={b:g}' for a, b in k.items()))
def test_by_grid_aligned_with_R(kw):
    check(make_cb('bs'), cen=CEN, **kw)


@pytest.mark.parametrize('kw', [dict(by=0.2), dict(from_val=-33.1, to_val=-30.05, by=0.2), dict(from_val=-32.5)],
                         ids=lambda k: ','.join(f'{a}={b:g}' for a, b in k.items()))
def test_grid_on_narrow_exposure_range(kw):
    check(make_cb('bs', 'narrow'), cen=CEN_FOR['narrow'], **kw)


# ---- `at`: vector -> sort(unique(at)), NA dropped; scalar accepted -----------------------------------------------------
_NAN = np.nan
_AT_DEFECT = {
    'unsorted': np.array([12.0, -5.0, 20.0, 3.0, 25.0, 0.0, 8.0]),
    'decreasing': np.arange(29.0, -21.0, -1.0),
    'duplicates': np.array([0.0, 5.0, 5.0, 10.0, 10.0, 10.0, 20.0]),
    'nan': np.array([0.0, 5.0, _NAN, 10.0, 20.0]),
    'unsorted_dup_nan': np.array([20.0, -5.0, 10.0, 10.0, _NAN, 0.0, 30.0, _NAN]),
}


@pytest.mark.parametrize('name', list(_AT_DEFECT))
def test_at_is_sorted_unique_nan_free(name):
    py, ref = check(make_cb('bs'), cen=CEN, at=_AT_DEFECT[name])
    assert ref['predvar'].size == np.unique(_AT_DEFECT[name][~np.isnan(_AT_DEFECT[name])]).size


@pytest.mark.parametrize('at', [5.0, 5, np.float64(-3.5)], ids=['float', 'int', 'np.float64'])
def test_scalar_at(at):
    py, ref = check(make_cb('bs'), cen=CEN, at=at)
    assert py.predvar.shape == (1,)


def test_at_takes_precedence_over_from_to_by():
    """R's mkat ignores from/to/by when `at` is given; so must PyDLNM."""
    check(make_cb('bs'), cen=CEN, at=AT, from_val=0.0, to_val=10.0, by=1.0)


def test_at_input_array_is_not_modified():
    """Guards the sort/unique port: the caller's `at` must not be sorted or de-duplicated in place."""
    at = np.array([12.0, -5.0, 20.0, 3.0, 25.0, 0.0, 8.0, 8.0])
    before = at.copy()
    py_crosspred(make_cb('bs'), cen=CEN, at=at)
    assert np.array_equal(at, before)


@pytest.mark.parametrize('kind', ['array', 'list', 'series', 'int_array', 'single'])
def test_sorted_unique_at_forms_match_R(kind):
    import pandas as pd
    at = {'array': AT, 'list': list(AT), 'series': pd.Series(AT), 'int_array': np.arange(-20, 30),
          'single': np.array([5.0])}[kind]
    ref_at = np.asarray(at, dtype=float)
    ref = r_crosspred(make_cb('bs'), cen=CEN, at=ref_at)
    py = py_crosspred(make_cb('bs'), cen=CEN, at=at)
    for f in FIELDS:
        assert_close(getattr(py, f), ref[f], rtol=1e-8, what=f)


def test_explicit_at_beyond_training_range_and_cumul():
    """Sorted unique `at` extending 3 beyond the data range (training boundary knots reused) with cumulative fits."""
    c = make_cb('ns')
    at = np.arange(-30.0, 37.0, 3.0)
    ref = r_crosspred(c, cen=CEN, at=at, cumul=True)
    py = py_crosspred(c, cen=CEN, at=at, cumul=True)
    for f in FIELDS:
        assert_close(getattr(py, f), ref[f], rtol=1e-8, what=f)
    assert_close(py.cumfit, rget('g1cp$cumfit'), rtol=1e-8, what='cumfit')
    assert_close(py.cumse, rget('g1cp$cumse'), rtol=1e-8, what='cumse')


# ---- seqlag / bylag: R seq() never exceeds `to` -------------------------------------------------------------------------
_SEQLAG_BAD = [([0, 5], 0.3), ([0, 5], 0.4), ([0, 5], 0.7), ([2, 7], 0.6), ([0, 21], 2.0)]
_SEQLAG_OK = [([0, 5], 0.5), ([0, 5], 0.25), ([0, 5], 1.0), ([0, 21], 1.0), ([3, 15], 1.0)]


def _r_seqlag(lag, by):
    return rget(f'dlnm:::seqlag(c({lag[0]},{lag[1]}), by={by!r})')


@pytest.mark.parametrize('lag,by', _SEQLAG_BAD, ids=lambda v: str(v).replace(' ', ''))
def test_seqlag_never_exceeds_upper_lag(lag, by):
    from utils import seqlag
    assert_close(np.asarray(seqlag(lag, by), dtype=float), _r_seqlag(lag, by), rtol=1e-12, what=f'seqlag({lag}, by={by})')


@pytest.mark.parametrize('lag,by', _SEQLAG_OK, ids=lambda v: str(v).replace(' ', ''))
def test_seqlag_matches_R_when_step_divides_range(lag, by):
    from utils import seqlag
    assert_close(np.asarray(seqlag(lag, by), dtype=float), _r_seqlag(lag, by), rtol=1e-12, what=f'seqlag({lag}, by={by})')


# make_cb's lag basis has explicit Boundary.knots = lag range, so only the lag GRID can differ between R and Python.
@pytest.mark.parametrize('bylag', [0.3, 0.4, 2.0])
def test_bylag_columns_match_R(bylag):
    check(make_cb('ns'), at=AT, cen=CEN, bylag=bylag)


@pytest.mark.parametrize('bylag', [0.5, 0.25])
def test_bylag_dividing_the_lag_range_matches_R(bylag):
    check(make_cb('ns'), at=AT, cen=CEN, bylag=bylag)


# ---- find_mmt inherits the grid (centering-5) ------------------------------------------------------------------------------
def _r_mmt(case, **kw):
    r_crosspred(case, **kw)
    return float(rget('g1cp$predvar[which.min(g1cp$allfit)]')[0])


def _py_mmt(case, **kw):
    from centering import find_mmt
    return float(_quiet(find_mmt, case.py, None, coef=case.coef, vcov=case.vcov, **kw)['mmt'])


@pytest.mark.parametrize('kw', [dict(), dict(by=1.0), dict(by=0.7)], ids=['default', 'by1', 'by0.7'])
def test_find_mmt_grid_matches_R_argmin(kw):
    c = make_cb('bs')
    assert _py_mmt(c, **kw) == pytest.approx(_r_mmt(c, cen=CEN, **kw), abs=1e-9)


def test_find_mmt_with_nan_in_at():
    c = make_cb('bs')
    at = np.array([-10.0, 0.0, 5.0, _NAN, 12.0, 20.0, 28.0])
    assert _py_mmt(c, at=at) == pytest.approx(_r_mmt(c, cen=CEN, at=at), abs=1e-9)


def test_find_mmt_on_explicit_sorted_grid_matches_R_argmin():
    c = make_cb('bs')
    at = np.round(np.arange(-20.0, 30.0, 0.1), 1)
    assert _py_mmt(c, at=at) == pytest.approx(_r_mmt(c, cen=CEN, at=at), abs=1e-9)


# ===============================================================================================================
# G + H1b together: the bare crosspred(cb, coef=, vcov=) call that every R tutorial starts with
# ===============================================================================================================
@pytest.mark.parametrize('fun', ['bs', 'ns'])
def test_fully_default_call(fun):
    py, ref = check(make_cb(fun))
    assert ref['cen'] is not None and ref['predvar'].size > 21     # R auto-centres and uses the pretty(n=50) grid
