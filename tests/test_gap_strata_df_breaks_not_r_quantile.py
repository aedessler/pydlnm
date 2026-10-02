"""Gap test [strata_df_breaks_not_r_quantile]: strata(df=...) WITHOUT explicit breaks (var basis and lag basis).

When strata() is asked for df degrees of freedom and no breaks, dlnm derives the cut points as quantiles of x

    R:        breaks <- quantile(x, 1/(df-intercept+1)*1:(df-intercept), na.rm=TRUE)       (type 7, R's arithmetic)
    PyDLNM:   breaks = np.quantile(x_clean, np.arange(1, k+1)/(k+1))                        (basis_functions.py, StrataBasis)

The two differ in rounding: the probability vectors ((1/5)*3 = 0.6000000000000001 versus 3/5 = 0.6, ...) and R's
(1-h)*x[lo] + h*x[hi] interpolation versus numpy's lerp. The breaks then differ by ~1e-15, which is harmless for a
continuous exposure but not for the discrete strata() basis: cut(x, right=FALSE) puts a value that equals a break into
the upper stratum, so any observation that ties with a break lands in a different stratum in R and in PyDLNM, and whole
design-matrix columns change. That happens

  * for lag bases: arglag=list(fun="strata", df=5) on seqlag(lag) puts a break on an integer lag for lag = 10, 15, 20,
    30, ... 60 (k = df - intercept = 4 and 9..11), and the lag the break sits on moves to the next stratum;
  * for var bases on integer-valued / rounded exposures or whenever (n-1)*p is an integer (for example 731 or 2021 days
    of the Chicago temperature series, df = 4).

Everything is differential: R (dlnm 2.4.10, through rpy2) is run at test time on identical inputs, nothing is copied
from Python output.  The existing strata tests (test_basis_S1_discrete.py, test_crossbasis_B.py) use continuous
random-normal exposures or explicit breaks, so they cannot see this.

Layout
  plain tests            guard what is already faithful (no ties at a break; explicit controls) and must keep passing;
  @known_defect tests    assert R's behaviour where PyDLNM differs today (strict xfail: when the fix lands they XPASS
                         and the markers must be deleted in the same commit).

Fix (not applied here): probs = (1/(k+1)) * np.arange(1, k+1) and R's type-7 arithmetic,
    index = 1 + (n-1)*p; lo = floor(index); hi = ceil(index); q = x[lo]; where index > lo and x[hi] != q:
    q = (1-h)*x[lo] + h*x[hi], h = index - lo      (on sorted x without NaN).
"""
import contextlib
import copy
import io
import os
import warnings

import numpy as np
import pytest

from rhelpers import assert_close, chicago, known_defect, max_rel_diff, np2r, r, rget

import rpy2.rinterface_lib.embedded as _emb

RRuntimeError = _emb.RRuntimeError

# CrossBasis construction overwrites os.environ['R_HOME'] (audit finding Q2); pin the working one, as the other tests do
_GOOD_R_HOME = os.path.dirname(str(r('.Library')[0]))
r('invisible(chol2inv(chol(diag(2) + 1))); invisible(solve(diag(2))); invisible(qr(diag(2)))')


@pytest.fixture(autouse=True)
def _keep_r_home():
    os.environ['R_HOME'] = _GOOD_R_HOME
    yield
    os.environ['R_HOME'] = _GOOD_R_HOME


# --------------------------------------------------------------------------------------------------------------
# helpers: the same call in R (dlnm) and in PyDLNM on identical inputs
# --------------------------------------------------------------------------------------------------------------
def _flag(b):
    return 'TRUE' if b else 'FALSE'


@contextlib.contextmanager
def _quiet():
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        yield


def _temp(n, rounded=False):
    t = chicago()['temp'][:n].copy()
    return np.round(t) if rounded else t


def r_strata(x, df, intercept, ref=1):
    """R onebasis(x, 'strata', df=, ref=, intercept=) -> (matrix, breaks attribute)."""
    np2r('gst_x', x)
    r(f'gst_b <- suppressWarnings(onebasis(gst_x, fun="strata", df={int(df)}, ref={int(ref)}, intercept={_flag(intercept)}))')
    return rget('matrix(as.numeric(gst_b), nrow=nrow(gst_b))'), rget('as.numeric(attr(gst_b, "breaks"))')


def py_strata(x, df, intercept, ref=1):
    """PyDLNM OneBasis(x, fun='strata', ...) -> (matrix, breaks attribute)."""
    from basis import OneBasis
    with _quiet():
        ob = OneBasis(np.asarray(x, dtype=float), fun='strata', df=df, ref=ref, intercept=intercept)
    brk = ob.attributes.get('breaks')
    return np.asarray(ob.basis, dtype=float), (np.array([]) if brk is None else np.atleast_1d(np.asarray(brk, dtype=float)))


def strata_mismatch(x, df, intercept, ref=1, label=''):
    """None if PyDLNM reproduces R's strata basis for this exposure (same matrix, or an error exactly where R errors);
    otherwise a message that says how they differ."""
    x = np.asarray(x, dtype=float)
    tag = f'{label}strata(df={df}, intercept={intercept}, ref={ref}) n={len(x)}'
    try:
        rm, rb = r_strata(x, df, intercept, ref)
        rerr = None
    except RRuntimeError as e:
        rm, rb, rerr = None, None, str(e).strip().splitlines()[-1]
    try:
        pm, pb = py_strata(x, df, intercept, ref)
        perr = None
    except Exception as e:                                                          # noqa: BLE001
        pm, pb, perr = None, None, f'{type(e).__name__}: {e}'
    if rm is None:
        return None if pm is None else f'{tag}: R raises ({rerr}) but PyDLNM returns shape {pm.shape}'
    if pm is None:
        return f'{tag}: R returns shape {rm.shape}, PyDLNM raises {perr}'
    if pm.shape != rm.shape:
        return f'{tag}: shape PyDLNM {pm.shape} vs R {rm.shape}'
    if not np.array_equal(np.isnan(pm), np.isnan(rm)):
        return f'{tag}: NaN pattern differs'
    rows = int((np.nan_to_num(pm) != np.nan_to_num(rm)).any(axis=1).sum())
    if rows:
        dbrk = float(np.abs(pb - rb).max()) if pb.shape == rb.shape and rb.size else float('nan')
        return f'{tag}: {rows} rows fall in a different stratum (break difference {dbrk:.2e})'
    return None


def assert_no_mismatch(cases, what):
    """cases: iterable of argument tuples for strata_mismatch; one failing assertion listing the first few."""
    cases = list(cases)
    msgs = [m for m in (strata_mismatch(*c) for c in cases) if m]
    assert not msgs, f'{what}: {len(msgs)} of {len(cases)} configurations differ from R, e.g.\n  ' + '\n  '.join(msgs[:6])


def _seqlag(a, L):
    from utils import seqlag
    return np.asarray(seqlag([a, L]), dtype=float)


def r_crossbasis(x, lag, var_df, arglag):
    """R crossbasis(x, lag, argvar=ns(df=var_df) or strata(df=var_df), arglag=strata(...)); matrix, stored in gst_cb."""
    np2r('gst_x', x)
    rv = f'list(fun="{var_df[0]}", df={var_df[1]})'
    rl = f'list(fun="{arglag[0]}", df={arglag[1]}' + ('' if len(arglag) < 3 else f', intercept={_flag(arglag[2])}') + ')'
    r(f'gst_cb <- suppressWarnings(crossbasis(gst_x, lag={lag}, argvar={rv}, arglag={rl}))')
    return rget('unclass(gst_cb)')


def py_crossbasis(x, lag, var_df, arglag):
    from basis import CrossBasis
    av = {'fun': var_df[0], 'df': var_df[1]}
    al = {'fun': arglag[0], 'df': arglag[1]}
    if len(arglag) > 2:
        al['intercept'] = arglag[2]
    with _quiet():
        return CrossBasis(np.asarray(x, dtype=float), lag=lag, argvar=copy.deepcopy(av), arglag=copy.deepcopy(al))


def assert_crossbasis_matches_r(x, lag, var_df, arglag, rtol=1e-10):
    ref = r_crossbasis(x, lag, var_df, arglag)
    cb = py_crossbasis(x, lag, var_df, arglag)
    what = f'crossbasis(n={len(x)}, lag={lag}, argvar={var_df}, arglag={arglag})'
    assert_close(np.asarray(cb.basis, dtype=float), ref, rtol=rtol, what=what)
    return cb, ref


LAG_STRATA_DF5 = ('strata', 5)                   # R: intercept defaults to TRUE for a lag basis -> k = 4 quantile breaks
LAG_STRATA_DF4_NOINT = ('strata', 4, False)      # k = 4 as well
LAG_STRATA_DF3 = ('strata', 3)                   # k = 2


# ==============================================================================================================
# 1. plain guards: what is already faithful (no observation ties with a break)
# ==============================================================================================================
@pytest.mark.parametrize('seed', [1, 2, 3])
def test_quantile_breaks_agree_with_r_to_rounding_on_continuous_data(seed):
    """The derived breaks are R's quantile() up to rounding error (here: the number and values, rtol 1e-12)."""
    rng = np.random.default_rng(seed)
    for n in (25, 60, 131, 400):
        x = rng.normal(15.0, 8.0, n)
        for df in range(2, 11):
            for intercept in (False, True):
                rm, rb = r_strata(x, df, intercept)
                pm, pb = py_strata(x, df, intercept)
                assert pb.shape == rb.shape == (df - int(intercept),), f'n={n} df={df} ic={intercept}: breaks shape'
                assert_close(pb, rb, rtol=1e-12, what=f'breaks n={n} df={df} ic={intercept}')


def _is_prime(m):
    return m > 1 and all(m % d for d in range(2, int(m ** 0.5) + 1))


def test_continuous_exposure_with_no_break_on_an_observation_matches_r():
    """Continuous random-normal exposures of length n = 1 + (a prime >= 13): (n-1)*j/(k+1) is never an integer for
    k = 1..10, so no break can coincide with an observation and PyDLNM equals R for every df / intercept."""
    rng = np.random.default_rng(10)
    cases = []
    for m in range(13, 212):
        if _is_prime(m):
            x = rng.normal(15.0, 8.0, m + 1)
            cases += [(x, df, ic, 1) for df in range(2, 11) for ic in (False, True)]
    assert len(cases) > 500
    assert_no_mismatch(cases, 'continuous exposure, n-1 prime')


@pytest.mark.parametrize('a', [0, 1])
def test_seqlag_lag_basis_unaffected_quantile_counts_match_r(a):
    """Lag basis strata on seqlag([a, L]): k = df - intercept in 1,2,3,5,6,7,8 never puts a break on an integer lag in
    these ranges (a in 0,1, L up to 60), so PyDLNM equals R there (k = 4 and 9..11 are the defect, tested below)."""
    cases = []
    for L in range(a + 1, 61):
        lagv = _seqlag(a, L)
        for k in (1, 2, 3, 5, 6, 7, 8):
            for intercept in (False, True):
                cases.append((lagv, k + int(intercept), intercept, 1))
    assert_no_mismatch(cases, f'seqlag([{a}, L]) lag basis')


@pytest.mark.parametrize('lag', [5, 7, 14, 21, 25, 28])
@pytest.mark.parametrize('arglag', [LAG_STRATA_DF5, LAG_STRATA_DF4_NOINT, LAG_STRATA_DF3],
                         ids=['df5', 'df4_noint', 'df3'])
def test_crossbasis_strata_lag_without_integer_break_matches_r(lag, arglag):
    """Control for the failing lags below: same construction, but no break falls on an integer lag."""
    assert_crossbasis_matches_r(_temp(1500), lag, ('ns', 4), arglag)


@pytest.mark.parametrize('lag', [10, 20, 30, 60])
def test_crossbasis_strata_lag_df3_matches_r_at_the_lags_where_df5_fails(lag):
    """df = 3 (k = 2) on the lags where k = 4 goes wrong: faithful, so the defect is specific to the break arithmetic."""
    assert_crossbasis_matches_r(_temp(1500), lag, ('ns', 4), LAG_STRATA_DF3)


@pytest.mark.parametrize('n', [500, 1000, 1461, 3000, 5114])
def test_chicago_temperature_strata_without_tie_at_a_break_matches_r(n):
    """Real exposure series where no observation equals a break in R: faithful for df = 2..6, both intercept settings."""
    x = _temp(n)
    assert_no_mismatch([(x, df, ic, 1) for df in range(2, 7) for ic in (False, True)], f'chicago temperature n={n}')


def test_nan_exposure_without_ties_matches_r():
    """quantile(na.rm=TRUE): missing values are dropped before the breaks are derived (continuous data)."""
    rng = np.random.default_rng(7)
    x = rng.normal(15.0, 8.0, 200)
    x[rng.choice(200, 12, replace=False)] = np.nan
    assert_no_mismatch([(x, df, ic, 1) for df in range(2, 9) for ic in (False, True)], 'NaN exposure, no ties')


def test_breaks_not_unique_raises_in_python_exactly_where_r_raises():
    """Zero-inflated exposures (precipitation-like): when R's quantile breaks coincide R stops with 'breaks' are not
    unique; PyDLNM must stop as well, and must not stop where R does not (the matrices are not compared here)."""
    rng = np.random.default_rng(3)
    n_both_ok = n_both_err = 0
    for n in (60, 100, 250):
        for frac0 in (0.3, 0.5, 0.7, 0.9):
            x = np.where(rng.uniform(size=n) < frac0, 0.0, rng.gamma(2.0, 3.0, n))
            for df in range(2, 7):
                try:
                    r_strata(x, df, False)
                    rerr = False
                except RRuntimeError:
                    rerr = True
                try:
                    py_strata(x, df, False)
                    perr = False
                except Exception:                                                    # noqa: BLE001
                    perr = True
                assert rerr == perr, f'n={n} frac0={frac0} df={df}: R raises={rerr}, PyDLNM raises={perr}'
                n_both_err += rerr
                n_both_ok += not rerr
    assert n_both_err > 0 and n_both_ok > 0          # the grid exercises both outcomes


@pytest.mark.parametrize('k', [4, 9, 10])
def test_continuous_exposure_with_break_on_an_observation_matches_r(k):
    """Even a fully continuous exposure is affected: when (n-1)/(k+1) is an integer the j-th break is exactly an
    observed value in R (index 1 + (n-1)*j/(k+1) integer) and PyDLNM's 1-ulp different break sends that observation
    (and the ones tied with it) to the neighbouring stratum.  n = 101 (k = 4 and 9) and n = 100 (k = 10) are round sample sizes."""
    rng = np.random.default_rng(20 + k)
    cases = []
    for n in range(k + 2, 302):
        if (n - 1) % (k + 1) == 0:
            x = rng.normal(15.0, 8.0, n)
            cases += [(x, k, False, 1), (x, k + 1, True, 1)]
    assert_no_mismatch(cases, f'continuous exposure, k={k}, (n-1) multiple of {k + 1}')


# ==============================================================================================================
# 2. lag basis: strata(df) on seqlag(lag)                                        (R: break on an integer lag)
# ==============================================================================================================
LAG_FAIL_CASES = [(10, 5, True), (15, 4, False), (20, 5, True), (30, 4, False), (60, 5, True),
                  (10, 9, False), (20, 10, True)]        # (L, df, intercept): k = 4 and 9


@pytest.mark.parametrize('L, df, intercept', LAG_FAIL_CASES)
def test_onebasis_strata_on_seqlag_matches_r(L, df, intercept):
    """onebasis(seqlag(c(0, L)), 'strata', df=) - the exact call crossbasis() makes for arglag=list(fun='strata')."""
    msg = strata_mismatch(_seqlag(0, L), df, intercept, label=f'seqlag(0,{L}) ')
    assert msg is None, msg


def test_onebasis_strata_on_seqlag_grid_matches_r():
    """lag = 5, 10, ..., 60 with (df, intercept) in {(5,T), (4,F), (3,T)}: all faithful only after the fix."""
    cases = [(_seqlag(0, L), df, ic, 1, f'seqlag(0,{L}) ')
             for L in range(5, 61, 5) for df, ic in ((5, True), (4, False), (3, True))]
    assert_no_mismatch(cases, 'seqlag lag basis grid')


def test_onebasis_strata_lag_breaks_are_bit_identical_to_r():
    """The root cause: R's break is 0x1.8000000000001p+2 (6.000000000000001) for seqlag(0, 10), df=5, intercept."""
    lagv = _seqlag(0, 10)
    rm, rb = r_strata(lagv, 5, True)
    pm, pb = py_strata(lagv, 5, True)
    assert np.array_equal(pb, rb), f'breaks PyDLNM {pb.tolist()} vs R {rb.tolist()} (difference {np.abs(pb - rb).max():.2e})'


def test_strata_class_matches_r_strata_function_on_integer_range():
    """Direct use of the basis function: dlnm:::strata(0:10, df=4) versus StrataBasis(df=4)(0:10)."""
    from basis_functions import StrataBasis
    x = np.arange(11.0)
    np2r('gst_xs', x)
    r('gst_s <- dlnm:::strata(gst_xs, df=4)')          # strata() is internal to dlnm (not exported)
    rm = rget('matrix(as.numeric(gst_s), nrow=nrow(gst_s))')
    rb = rget('as.numeric(attr(gst_s, "breaks"))')
    sb = StrataBasis(df=4)
    pm = np.asarray(sb(x), dtype=float)
    assert pm.shape == rm.shape
    assert np.array_equal(pm, rm), f'{int((pm != rm).any(axis=1).sum())} rows in a different stratum; ' \
                                   f'R breaks {rb.tolist()} vs PyDLNM {np.asarray(sb.attributes["breaks"]).tolist()}'


# ==============================================================================================================
# 3. CrossBasis with a strata lag basis (argvar ns, lag over seqlag(lag))
# ==============================================================================================================
@pytest.mark.parametrize('lag', [10, 15, 20, 30, 60])
@pytest.mark.parametrize('arglag', [LAG_STRATA_DF5, LAG_STRATA_DF4_NOINT], ids=['df5', 'df4_noint'])
def test_crossbasis_strata_lag_with_break_on_integer_lag_matches_r(lag, arglag):
    """Chicago temperature (n=1500), argvar ns df=4, arglag strata df=5 (or df=4 without intercept): k = 4 breaks of
    the lag grid 0..lag; the cross-basis differs from R's by whole columns of the lag-stratum products."""
    assert_crossbasis_matches_r(_temp(1500), lag, ('ns', 4), arglag)


# ==============================================================================================================
# 4. downstream: crosspred / crossreduce with a strata lag basis (same coef / vcov in R and Python)
# ==============================================================================================================
def _coef_vcov(p, seed=1):
    rng = np.random.default_rng(seed)
    coef = rng.normal(0.0, 0.05, p)
    a = rng.normal(0.0, 1.0, (p, p)) * 0.01
    return coef, a @ a.T + np.eye(p) * 1e-5


AT = np.arange(-10.0, 30.0 + 1e-9, 5.0)
CEN = 15.0


def _strata_lag_design(lag):
    x = _temp(1500)
    ref = r_crossbasis(x, lag, ('ns', 4), LAG_STRATA_DF5)
    cb = py_crossbasis(x, lag, ('ns', 4), LAG_STRATA_DF5)
    p = ref.shape[1]
    assert tuple(cb.shape) == ref.shape
    coef, vcov = _coef_vcov(p)
    np2r('gst_coef', coef)
    np2r('gst_vcov', vcov)
    r(f'gst_vcov <- matrix(gst_vcov, {p})')
    return cb, coef, vcov


def _crosspred_vs_r(lag):
    from prediction import crosspred
    cb, coef, vcov = _strata_lag_design(lag)
    np2r('gst_at', AT)
    r(f'gst_cp <- suppressMessages(crosspred(gst_cb, coef=gst_coef, vcov=gst_vcov, model.link="log", at=gst_at, '
      f'cen={CEN}, cumul=TRUE))')
    with _quiet():
        cp = crosspred(cb, coef=coef, vcov=vcov, model_link='log', at=AT, cen=CEN, cumul=True)
    for name, rexpr, val in (('matfit', 'gst_cp$matfit', cp.matfit), ('matse', 'gst_cp$matse', cp.matse),
                             ('allfit', 'as.numeric(gst_cp$allfit)', cp.allfit),
                             ('allse', 'as.numeric(gst_cp$allse)', cp.allse),
                             ('cumfit', 'gst_cp$cumfit', cp.cumfit)):
        assert_close(np.asarray(val, dtype=float), rget(rexpr), rtol=1e-8, what=f'crosspred {name} (lag={lag})')


def _crossreduce_overall_vs_r(lag):
    from crossreduce import crossreduce
    cb, coef, vcov = _strata_lag_design(lag)
    r(f'gst_rd <- suppressMessages(crossreduce(gst_cb, coef=gst_coef, vcov=gst_vcov, model.link="log", '
      f'type="overall", cen={CEN}))')
    with _quiet():
        rd = crossreduce(cb, coef=coef, vcov=vcov, model_link='log', type='overall', cen=CEN)
    pc = np.ravel(np.asarray(rd.coefficients if hasattr(rd, 'coefficients') else rd.coef, dtype=float))
    assert_close(pc, rget('as.numeric(gst_rd$coefficients)'), rtol=1e-8, what=f'crossreduce overall coef (lag={lag})')
    pv = np.asarray(rd.vcov, dtype=float)
    assert_close(pv, rget('matrix(as.numeric(gst_rd$vcov), nrow=nrow(gst_rd$vcov))'), rtol=1e-8,
                 what=f'crossreduce overall vcov (lag={lag})')


def test_crosspred_strata_lag_without_integer_break_matches_r():
    """Control: lag 21 (no break on an integer lag) - the harness reproduces R's crosspred to rounding error."""
    _crosspred_vs_r(21)


def test_crossreduce_overall_strata_lag_without_integer_break_matches_r():
    """Control: lag 21 - reduced coefficients and vcov equal R's."""
    _crossreduce_overall_vs_r(21)


@pytest.mark.parametrize('lag', [20, 30])
def test_crosspred_strata_lag_with_break_on_integer_lag_matches_r(lag):
    """Same coef / vcov in R and PyDLNM, lag 20 / 30: the lag where R's break sits (20 * 3/5 = 12 -> 12.000000000000002)
    is assigned to a different stratum, so the lag-response, its cumulative and the overall effect all move."""
    _crosspred_vs_r(lag)


@pytest.mark.parametrize('lag', [20, 30])
def test_crossreduce_overall_strata_lag_with_break_on_integer_lag_matches_r(lag):
    """The overall reduction (what mvmeta pools) uses the lag basis at seqlag(lag): reduced coefficients differ from R."""
    _crossreduce_overall_vs_r(lag)


# ==============================================================================================================
# 5. var basis: strata(df) on an exposure with ties
# ==============================================================================================================
def test_strata_df4_on_integers_0_to_10_matches_r():
    """x = 0:10, df=4: R's breaks are [2, 4, 6.000000000000001, 8] (PyDLNM: 6.0) so the observation 6 falls into stratum 3."""
    msg = strata_mismatch(np.arange(11.0), 4, False)
    assert msg is None, msg


@pytest.mark.parametrize('name, x, df, intercept',
                         [('0:10', np.arange(11.0), 4, False), ('chicago[:731]', _temp(731), 4, False),
                          ('chicago[:731]', _temp(731), 5, True)], ids=['0to10_df4', 'chi731_df4', 'chi731_df5ic'])
def test_var_strata_breaks_are_bit_identical_to_r(name, x, df, intercept):
    rm, rb = r_strata(x, df, intercept)
    pm, pb = py_strata(x, df, intercept)
    assert np.array_equal(pb, rb), f'{name}: breaks PyDLNM {pb.tolist()} vs R {rb.tolist()}'


def test_integer_valued_exposure_grid_matches_r():
    """Integer-valued exposures (n = 5..200, df = 2..10, both intercept settings): 12-15 % of the configurations put a
    tied observation into a different stratum than R."""
    rng = np.random.default_rng(11)
    cases = []
    for n in range(5, 201, 3):
        for df in range(2, 11):
            x = rng.integers(0, 30, n).astype(float)
            cases.append((x, df, bool(df % 2), 1, 'int-valued '))
    assert_no_mismatch(cases, 'integer-valued exposure')


def test_one_decimal_exposure_grid_matches_r():
    """Exposures rounded to 0.1 (thermometer resolution) - n = 5..200, df = 2..10."""
    rng = np.random.default_rng(12)
    cases = []
    for n in range(5, 201, 3):
        for df in range(2, 11):
            x = np.round(rng.normal(15.0, 8.0, n), 1)
            cases.append((x, df, bool(df % 2), 1, '1-decimal '))
    assert_no_mismatch(cases, '1-decimal exposure')


@pytest.mark.parametrize('n, df, intercept', [(731, 4, False), (731, 5, True), (2021, 4, False), (2021, 5, True),
                                              (2441, 4, False)])
def test_chicago_temperature_strata_with_tie_at_a_break_matches_r(n, df, intercept):
    """Two years (731 days) and other realistic series lengths of the Chicago temperature with df=4: the break at
    the 60th percentile equals an observed value in R (index (n-1)*3/5 integer) and PyDLNM files it differently."""
    msg = strata_mismatch(_temp(n), df, intercept, label='chicago ')
    assert msg is None, msg


def test_integer_rounded_chicago_temperature_strata_matches_r():
    """Temperature recorded in whole degrees (116 rows differ at n=2441, df=4 in the audit)."""
    assert_no_mismatch([(_temp(n, rounded=True), df, ic, 1, 'rounded ') for n in (731, 2441, 3650)
                        for df in (3, 4, 5, 6) for ic in (False, True)], 'integer-rounded chicago temperature')


def test_chicago_temperature_series_length_sweep_matches_r():
    """Sweep the series length (n = 300..1200, step 7) with df=5, intercept: a handful of lengths tie with a break."""
    t = chicago()['temp']
    assert_no_mismatch([(t[:n].copy(), 5, True, 1, 'chicago ') for n in range(300, 1200, 7)], 'length sweep')


def test_nan_exposure_with_ties_at_a_break_matches_r():
    """0:10 with three missing days mixed in (df = 4): quantile(na.rm=TRUE) sees 11 values, breaks as for x = 0:10."""
    rng = np.random.default_rng(2)
    x = np.concatenate([np.arange(11.0), [np.nan, np.nan, np.nan]])
    x = x[rng.permutation(len(x))]
    msg = strata_mismatch(x, 4, False, label='NaN + ties ')
    assert msg is None, msg


@pytest.mark.parametrize('n, rounded', [(731, False), (2021, False), (2441, True)],
                         ids=['chi731', 'chi2021', 'chi2441_rounded'])
def test_crossbasis_strata_var_basis_with_tie_at_a_break_matches_r(n, rounded):
    """argvar=list(fun='strata', df=4) (the exposure categories), arglag=ns(df=3): the var part of the cross-basis
    inherits the wrong stratum of the tied observations, so whole var x lag column blocks differ."""
    assert_crossbasis_matches_r(_temp(n, rounded=rounded), 3, ('strata', 4), ('ns', 3))


def test_crossbasis_strata_var_basis_without_tie_matches_r():
    """Control for the test above: n = 1000, no tied break."""
    assert_crossbasis_matches_r(_temp(1000), 3, ('strata', 4), ('ns', 3))
