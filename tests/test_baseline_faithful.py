"""BASELINE: the validated happy path of PyDLNM versus R dlnm 2.4.10 / mvmeta 1.0.3 / mixmeta 1.2.0.

Nothing in this module is a known defect: there is no @known_defect test here. Every test asserts behaviour that is
ALREADY faithful to R and that the audit fixes (themes A1-S1, see audit_handoff_2026-09-30/findings/root_cause_themes.md)
must not break, so the module is the regression backbone while the fixes land. All reference numbers are computed by R
at test run time on inputs that both sides receive (rpy2); nothing is copied from Python output and there are no
scaling constants. Deterministic algebra is compared to <= 1e-10 relative (mostly 1e-11/1e-12; the audit measured
1e-14..1e-16), optimiser-limited quantities (MVMeta fits) to 1e-6 against R run at reltol=1e-14.

Areas and what they guard (source: audit STATUS.md section 1 and findings/coverage_by_area.md, "verified_ok")
   1  Cross-basis      CrossBasis matrix for bs x ns(logknots) (the validated design) and four further explicit-knot
                       designs, E&W / US series; NaN rows, df, lag, range, column names
   2  OneBasis         ns / bs / lin / poly with explicit knots, intercept, Boundary.knots, degree, scale; NaN rows
   3  crosspred        full cross-basis coefficients (identical coef/vcov fed to R): every output field incl. cumul,
                       bylag, links, ci.level, cen, `at` beyond the range; OneBasis lin/poly; a real R quasi-Poisson
                       fit; a statsmodels quasi-Poisson model
   4  reduced crosspred  crosspred(CrossBasis, coef=<reduced/BLUP-like>, vcov=...): overall fields versus
                       R crosspred(onebasis, coef, vcov)
   5  crossreduce      type="overall" reduction (coef, vcov), independence of cen, consistency with full crosspred
   6  first-stage GLM  ImprovedGLMInterface (quasi-Poisson / Poisson, dow + ns(date)) on England & Wales regions vs R
                       glm, incl. the reduced coefficients
   7  MVMeta           REML/ML objective and analytic REML gradient at identical Psi, BLUP formula at identical
                       estimates, fits versus tightly converged R
   8  centering        explicit cen (11 values incl. out of range), cen stored in argvar, recenter_basis, find_mmt vs
                       which.min on identical grids (full and reduced coefficients)
   9  attrdl           per-observation AF (dir='forw') with explicit-knot bases and a log-link model, ranges,
                       NaN in x; totals at constant exposure
  10  utils            logknots, mklag, seqlag (dividing step), exphist
  11  pipelines        README Quick Start block 1, a US city first stage, the England & Wales second stage
  12  seasonality      harmonic basis (span) and non-cyclic seasonal spline

Every R object created here lives in R's global environment under a name starting with `bf_` / `bfm_`.
Run:  python -m pytest tests/test_baseline_faithful.py -q      (see tests/README.md for the R 4.5 environment)
"""
import contextlib
import io
import os
import warnings
from functools import lru_cache
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from rhelpers import REPO, assert_close, chicago, max_rel_diff, np2r, r, r2np, rget

EW_CSV = REPO / '2015_gasparrini_Lancet_Rcodedata-master' / 'regEngWales.csv'
US_CSV = REPO / 'temperature_mortality_analysis' / 'data.csv'
ATTRDL_R = REPO / '2015_gasparrini_Lancet_Rcodedata-master' / 'attrdl.R'

RTOL_BASIS = 1e-12        # bases and cross-bases: identical arithmetic (measured <= 1e-13)
RTOL_PRED = 1e-12         # crosspred / crossreduce / attrdl on identical coef/vcov (measured <= 1e-15)
RTOL_GLM = 1e-10          # first-stage GLM (measured <= 3e-13)

# --------------------------------------------------------------------------------------------------------------
# environment guards
# --------------------------------------------------------------------------------------------------------------
# Several PyDLNM modules (CrossBasis, improved_glm, rpy2_glm) overwrite os.environ['R_HOME'] (audit finding Q2). The R
# session started by rhelpers already runs the correct R; a later La_* call (vcov.glm, solve, chol2inv) then dlopens the
# wrong R's modules and segfaults. Pin the value R itself reports around every test and warm LAPACK up.
_GOOD_R_HOME = os.path.dirname(str(r('.Library')[0]))


def _pin_r_home():
    os.environ['R_HOME'] = _GOOD_R_HOME


_pin_r_home()
r('invisible(chol2inv(chol(diag(2) + 1))); invisible(solve(diag(2))); invisible(qr(diag(2)))')


@pytest.fixture(scope='module', autouse=True)
def _module_env():
    _pin_r_home()
    yield
    _pin_r_home()


@pytest.fixture(autouse=True)
def _keep_r_home():
    _pin_r_home()
    yield
    _pin_r_home()


@contextlib.contextmanager
def _quiet():
    """PyDLNM prints progress lines (ImprovedGLMInterface, CrossPred on reduced coefficients)."""
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        yield


def _rbool(flag):
    return 'TRUE' if flag else 'FALSE'


# --------------------------------------------------------------------------------------------------------------
# helpers: identical inputs for R and Python
# --------------------------------------------------------------------------------------------------------------
def _coef_vcov(p, seed, scale=0.05):
    """Deterministic coefficient vector and SPD covariance matrix."""
    rng = np.random.default_rng(seed)
    coef = rng.normal(0, scale, p)
    A = rng.normal(0, 1, (p, p))
    vcov = A @ A.T * scale ** 2 / p * 0.05 + np.eye(p) * 1e-5
    return coef, (vcov + vcov.T) / 2


@lru_cache(maxsize=None)
def _series(name):
    """Temperature series: 'chicago', an England & Wales region code ('London', 'Wales', ...) or a US city name."""
    if name == 'chicago':
        return chicago()['temp']
    ew = pd.read_csv(EW_CSV, usecols=['regnames', 'tmean'])
    if name in set(ew['regnames']):
        return ew.loc[ew['regnames'] == name, 'tmean'].to_numpy(dtype=float)
    us = pd.read_csv(US_CSV, usecols=['cityName', 'TMean'])
    return us.loc[us['cityName'] == name, 'TMean'].to_numpy(dtype=float)


# (var fun, degree, knot probabilities) x lag x (lag fun, number of log knots): every basis argument explicit, so the
# cross-basis is fully specified (no data-derived df defaults, theme A1) and the R and Python objects are identical.
CONFIGS = {
    'bs2_ns_L21': dict(var=('bs', 2, (.10, .75, .90)), lag=21, arglag=('ns', 3)),          # the validated design
    'bs3_int_L6': dict(var=('bs', 3, (.25, .75)), lag=6, arglag=('integer', None)),
    'ns_ns_L10': dict(var=('ns', None, (.10, .50, .90)), lag=10, arglag=('ns', 2)),
    'bs2_ns_lag2_10': dict(var=('bs', 2, (.20, .60)), lag=(2, 10), arglag=('ns', 2)),       # minimum lag > 0
    'ns_int_L3': dict(var=('ns', None, (.10, .50, .90)), lag=3, arglag=('integer', None)),  # Europe-style
}
_DESIGNS = {}


def _lag_r(lag):
    return f'c({lag[0]},{lag[1]})' if isinstance(lag, tuple) else f'{lag}'


def _design(name, series='chicago'):
    """R crossbasis `bf_cb_<key>` and onebasis `bf_ob_<key>` (in R's global env) plus the identical PyDLNM CrossBasis.
    Knots are computed once, in R, and handed to both sides."""
    key = f'{name}_{series}'.replace('-', '_').replace('&', '_').replace(' ', '_')
    if key in _DESIGNS:
        return _DESIGNS[key]
    from basis import CrossBasis
    cfg = CONFIGS[name]
    temp = _series(series)
    np2r(f'bf_temp_{key}', temp)
    fun, degree, probs = cfg['var']
    r(f'bf_kv_{key} <- quantile(bf_temp_{key}, c({", ".join(repr(p) for p in probs)}), na.rm=TRUE)')
    kv = rget(f'bf_kv_{key}')
    argvar = dict(fun=fun, knots=kv)
    r_argvar = f'fun="{fun}", knots=bf_kv_{key}'
    if degree:
        argvar['degree'] = degree
        r_argvar += f', degree={degree}'
    lagfun, nk = cfg['arglag']
    lag = cfg['lag']
    if lagfun == 'ns':
        r(f'bf_lk_{key} <- logknots({_lag_r(lag)}, nk={nk})')
        arglag = dict(fun='ns', knots=rget(f'bf_lk_{key}'))
        r_arglag = f'fun="ns", knots=bf_lk_{key}'
    else:
        arglag, r_arglag = dict(fun='integer'), 'fun="integer"'
    r(f'bf_cb_{key} <- crossbasis(bf_temp_{key}, lag={_lag_r(lag)}, argvar=list({r_argvar}), arglag=list({r_arglag}))')
    r(f'bf_ob_{key} <- do.call("onebasis", c(list(x=bf_temp_{key}), attr(bf_cb_{key}, "argvar")))')
    cb = CrossBasis(temp, lag=list(lag) if isinstance(lag, tuple) else lag, argvar=argvar, arglag=arglag)
    d = SimpleNamespace(key=key, cb=cb, temp=temp, cb_r=f'bf_cb_{key}', ob_r=f'bf_ob_{key}',
                        p=int(rget(f'ncol(bf_cb_{key})')[0]), pvar=int(rget(f'ncol(bf_ob_{key})')[0]),
                        lagfun=lagfun, lag=lag, kv=np.array(kv, copy=True), lagk=arglag.get('knots'))
    assert_close(np.asarray(cb.basis), rget(f'unclass(bf_cb_{key})'), rtol=RTOL_BASIS, what=f'{key} cross-basis')
    _DESIGNS[key] = d
    return d


def _assert_fields(pp, robj, fields, rtol=RTOL_PRED, what=''):
    """Every named field of the R object `robj` against the same attribute of the PyDLNM prediction; one assertion
    listing all disagreements (R's dotted names map to underscores, ci.level -> ci_level)."""
    bad = []
    for nm in fields:
        py = getattr(pp, nm.replace('.', '_'), None)
        if py is None:
            bad.append(f'{nm}: missing in Python')
            continue
        ref = rget(f'unname({robj}${nm})')
        py = np.atleast_1d(np.asarray(py, dtype=float))
        ref = np.atleast_1d(ref)
        if py.shape != ref.shape:
            bad.append(f'{nm}: shape Python {py.shape} vs R {ref.shape}')
        elif not np.array_equal(np.isnan(py), np.isnan(ref)):
            bad.append(f'{nm}: NaN pattern differs')
        elif max_rel_diff(py, ref) > rtol:
            bad.append(f'{nm}: max relative diff {max_rel_diff(py, ref):.2e} > {rtol:.0e}')
    assert not bad, '; '.join(bad)


def _r_pred_fields(robj):
    """Fields R's crosspred object carries (everything except the call and the model description)."""
    return [n for n in list(r(f'names({robj})')) if n not in ('call', 'model.class', 'model.link')]


# ==============================================================================================================
# 1  Cross-basis
# ==============================================================================================================
def test_validated_crossbasis_bs2_ns_logknots_matches_r():
    """The validated design exactly as used by the pipelines: Python's logknots and quantile knots, bs degree 2 knots
    P10/75/90 x ns(logknots(21,3)); NaN rows, df, lag, range, column names as in R."""
    from basis import CrossBasis
    from utils import logknots
    temp = chicago()['temp']
    np2r('bf_temp', temp)
    r('bf_kv <- quantile(bf_temp, c(.10,.75,.90));'
      'bf_cbR <- crossbasis(bf_temp, lag=21, argvar=list(fun="bs", degree=2, knots=bf_kv),'
      ' arglag=list(fun="ns", knots=logknots(21,3)))')
    kv = rget('bf_kv')
    cb = CrossBasis(temp, lag=21, argvar={'fun': 'bs', 'degree': 2, 'knots': kv},
                    arglag={'fun': 'ns', 'knots': logknots([0, 21], nk=3)})
    cb_r = rget('unclass(bf_cbR)')
    assert_close(np.asarray(cb.basis), cb_r, rtol=RTOL_BASIS, what='crossbasis')
    assert cb.shape == cb_r.shape == (len(temp), 25)
    assert int(np.isnan(cb.basis).any(axis=1).sum()) == 21
    assert np.isnan(cb.basis[:21]).all() and not np.isnan(cb.basis[21:]).any()
    assert tuple(cb.df) == tuple(int(v) for v in rget('attr(bf_cbR, "df")'))
    assert [int(v) for v in cb.lag] == [int(v) for v in rget('attr(bf_cbR, "lag")')]
    assert_close(np.asarray(cb.range, dtype=float), rget('attr(bf_cbR, "range")'), rtol=0, what='range')
    assert list(cb.colnames) == list(r('colnames(bf_cbR)'))


@pytest.mark.parametrize('name', list(CONFIGS))
def test_crossbasis_explicit_knots_matches_r(name):
    """Explicit-knot designs: bs degree 2/3 and ns variable bases x ns(logknots) or integer lag bases, lag 3..21 and a
    minimum lag > 0. The _design() builder asserts the matrix; this test adds attributes and NaN-row pattern."""
    d = _design(name)
    cb = d.cb
    cb_r = rget(f'unclass({d.cb_r})')
    assert np.array_equal(np.isnan(np.asarray(cb.basis)), np.isnan(cb_r)), 'NaN-row pattern differs'
    assert tuple(cb.df) == tuple(int(v) for v in rget(f'attr({d.cb_r}, "df")'))
    assert [int(v) for v in cb.lag] == [int(v) for v in rget(f'attr({d.cb_r}, "lag")')]
    assert list(cb.colnames) == list(r(f'colnames({d.cb_r})'))


@pytest.mark.parametrize('series', ['London', 'N-East', 'Akron'])
def test_validated_design_on_real_series_matches_r(series):
    """England & Wales regions and a US city with the validated design (60 series were checked in the audit)."""
    d = _design('bs2_ns_L21', series)
    assert d.cb.shape == (len(d.temp), 25)


# ==============================================================================================================
# 2  OneBasis
# ==============================================================================================================
ONEBASIS_CASES = {
    'ns_3knots': dict(fun='ns', knots=(.10, .50, .90)),
    'ns_1knot_intercept': dict(fun='ns', knots=(.50,), intercept=True),
    'ns_boundary_wider': dict(fun='ns', knots=(.25, .75), boundary=(-5.0, 5.0)),
    'ns_boundary_narrower': dict(fun='ns', knots=(.25, .75), boundary=(0.1, -0.1)),
    'bs2_3knots': dict(fun='bs', degree=2, knots=(.10, .75, .90)),
    'bs3_2knots_intercept': dict(fun='bs', degree=3, knots=(.30, .70), intercept=True),
    'bs1_1knot': dict(fun='bs', degree=1, knots=(.50,)),
    'bs2_boundary_wider': dict(fun='bs', degree=2, knots=(.25, .75), boundary=(-5.0, 5.0)),
    'lin': dict(fun='lin'),
    'lin_intercept': dict(fun='lin', intercept=True),
    'poly3': dict(fun='poly', degree=3),
    'poly2_intercept_scale': dict(fun='poly', degree=2, intercept=True, scale=30.0),
}


@pytest.mark.parametrize('case', list(ONEBASIS_CASES))
def test_onebasis_matches_r(case):
    """OneBasis versus R onebasis on a deterministic sample of 60 Chicago temperatures (some ties, extremes included).
    Knots are quantiles computed in R; `boundary=(lo, hi)` are offsets added to min/max (narrower when lo > 0)."""
    from basis import OneBasis
    spec = dict(ONEBASIS_CASES[case])
    fun = spec.pop('fun')
    probs = spec.pop('knots', None)
    boundary = spec.pop('boundary', None)
    rng = np.random.default_rng(3)
    x = np.sort(rng.choice(chicago()['temp'], 60, replace=False))
    x = np.append(x, [x[0], x[-1]])                                  # ties at the extremes
    np2r('bf_x', x)
    kw, rkw = dict(spec), [f'{k}={_rbool(v) if isinstance(v, bool) else repr(v)}' for k, v in spec.items()]
    if probs is not None:
        r(f'bf_kn <- quantile(bf_x, c({", ".join(repr(p) for p in probs)}))')
        kw['knots'] = rget('bf_kn')
        rkw.append('knots=bf_kn')
    if boundary is not None:
        bk = np.array([x.min() + boundary[0], x.max() + boundary[1]])
        np2r('bf_bk', bk)
        kw['Boundary_knots'] = bk
        rkw.append('Boundary.knots=bf_bk')
    r(f'bf_ob <- suppressWarnings(onebasis(bf_x, fun="{fun}"{"".join(", " + a for a in rkw)}))')
    with _quiet():
        ob = OneBasis(x, fun=fun, **kw)
    ref = rget('unclass(bf_ob)')
    assert_close(np.asarray(ob.basis), ref, rtol=RTOL_BASIS, what=f'OneBasis {case}')
    assert ob.shape == ref.shape


def test_onebasis_keeps_nan_rows_like_r():
    """NaN in x gives NaN rows (and nothing else changes) for ns, bs and lin."""
    from basis import OneBasis
    rng = np.random.default_rng(4)
    x = rng.normal(15, 8, 80)
    x[[0, 17, 40, 79]] = np.nan
    np2r('bf_x', x)
    r('bf_kn <- quantile(bf_x, c(.25, .75), na.rm=TRUE)')
    kn = rget('bf_kn')
    for fun, extra, rextra in [('ns', {}, ''), ('bs', {'degree': 2}, ', degree=2'), ('lin', {}, '')]:
        kws = {} if fun == 'lin' else {'knots': kn}
        rk = '' if fun == 'lin' else ', knots=bf_kn'
        ref = rget(f'unclass(onebasis(bf_x, fun="{fun}"{rk}{rextra}))')
        with _quiet():
            ob = OneBasis(x, fun=fun, **kws, **extra)
        assert_close(np.asarray(ob.basis), ref, rtol=RTOL_BASIS, what=f'{fun} with NaN')
        assert np.array_equal(np.isnan(ob.basis).any(axis=1), np.isnan(x))


# ==============================================================================================================
# 3  crosspred, full cross-basis coefficients (identical coef/vcov)
# ==============================================================================================================
AT = np.arange(-20.0, 30.0 + 1e-9, 0.5)

PRED_OPTIONS = {
    'default_log': dict(link='log'),
    'cumul_log': dict(link='log', cumul=True),
    'cumul_logit_ci90': dict(link='logit', cumul=True, ci_level=0.90),
    'bylag_half_log': dict(link='log', bylag=0.5),
    'cumul_bylag_half': dict(link='log', cumul=True, bylag=0.5),
    'identity_link': dict(link=None),
    'identity_cumul_ci99': dict(link=None, cumul=True, ci_level=0.99),
    'cen_minus10': dict(link='log', cen=-10.0),
    'cen_at_40_out_of_range': dict(link='log', cen=40.0, cumul=True),
}


def _full_pred_pair(d, opt, seed=11, at=AT):
    """(Python prediction, name of R crosspred object) for the same coef/vcov."""
    from prediction import crosspred
    coef, vcov = _coef_vcov(d.p, seed)
    np2r('bf_coef', coef)
    np2r('bf_vcov', vcov)
    np2r('bf_at', at)
    link, cumul, bylag = opt.get('link'), opt.get('cumul', False), opt.get('bylag', 1.0)
    ci, cen = opt.get('ci_level', 0.95), opt.get('cen', 15.0)
    ls = f', model.link="{link}"' if link else ''
    r(f'bf_pR <- suppressWarnings(crosspred({d.cb_r}, coef=bf_coef, vcov=bf_vcov{ls}, at=bf_at, cen={cen!r}, '
      f'bylag={bylag!r}, cumul={_rbool(cumul)}, ci.level={ci!r}))')
    with _quiet():
        pp = crosspred(d.cb, coef=coef, vcov=vcov, model_link=link, at=at, cen=cen, bylag=bylag, cumul=cumul,
                       ci_level=ci)
    return pp, 'bf_pR'


@pytest.mark.parametrize('opt', list(PRED_OPTIONS))
def test_crosspred_validated_design_all_fields_match_r(opt):
    """bs2 x ns(logknots(21,3)), identical random coef/vcov: every output field of R's crosspred (predvar, cen, lag,
    bylag, coefficients, vcov, matfit/matse/mat*, allfit/allse/all*, cumfit/cumse/cum* if requested, ci.level)."""
    d = _design('bs2_ns_L21')
    pp, robj = _full_pred_pair(d, PRED_OPTIONS[opt])
    fields = _r_pred_fields(robj)
    assert 'allfit' in fields and 'matfit' in fields
    _assert_fields(pp, robj, fields, what=opt)


@pytest.mark.parametrize('name', ['bs3_int_L6', 'ns_ns_L10', 'bs2_ns_lag2_10', 'ns_int_L3'])
def test_crosspred_explicit_knot_designs_match_r(name):
    """Other explicit-knot designs (bs degree 3 + integer lag, ns + ns lag, minimum lag 2, Europe-style ns + integer
    lag 0..3), with cumul and the log link."""
    d = _design(name)
    pp, robj = _full_pred_pair(d, dict(link='log', cumul=True), seed=7)
    _assert_fields(pp, robj, _r_pred_fields(robj), what=name)


def test_crosspred_at_beyond_training_range_matches_r():
    """`at` beyond the training range: bs/ns reuse the training Boundary.knots (extrapolation identical to R)."""
    d = _design('bs2_ns_L21')
    lo, hi = float(np.nanmin(d.temp)), float(np.nanmax(d.temp))
    at = np.linspace(lo - 5, hi + 5, 23)
    pp, robj = _full_pred_pair(d, dict(link='log', cumul=True, cen=15.0), at=at)
    _assert_fields(pp, robj, _r_pred_fields(robj), what='beyond range')


def test_crosspred_error_behaviour_like_r():
    """Errors R raises and Python raises too: no model and no coef/vcov, cumul with a lag sub-period, ci_level
    outside (0, 1), a decreasing lag pair."""
    from prediction import crosspred
    d = _design('bs2_ns_L21')
    coef, vcov = _coef_vcov(d.p, 3)
    np2r('bf_coef', coef)
    np2r('bf_vcov', vcov)
    cases = [
        ('crosspred(bf_cb_bs2_ns_L21_chicago, at=c(1,2))', dict(at=AT)),
        ('crosspred(bf_cb_bs2_ns_L21_chicago, coef=bf_coef, vcov=bf_vcov, at=c(1,2), lag=c(0,5), cumul=TRUE)',
         dict(coef=coef, vcov=vcov, at=AT, lag=[0, 5], cumul=True)),
        ('crosspred(bf_cb_bs2_ns_L21_chicago, coef=bf_coef, vcov=bf_vcov, at=c(1,2), ci.level=1)',
         dict(coef=coef, vcov=vcov, at=AT, ci_level=1.0)),
        ('crosspred(bf_cb_bs2_ns_L21_chicago, coef=bf_coef, vcov=bf_vcov, at=c(1,2), lag=c(3,2))',
         dict(coef=coef, vcov=vcov, at=AT, lag=[3, 2])),
    ]
    for rcode, kw in cases:
        with pytest.raises(Exception):
            r(rcode)
        with _quiet(), pytest.raises(ValueError):
            crosspred(d.cb, **kw)


def test_crosspred_does_not_mutate_inputs():
    from prediction import crosspred
    d = _design('bs2_ns_L21')
    coef, vcov = _coef_vcov(d.p, 5)
    c0, v0, a0 = coef.copy(), vcov.copy(), AT.copy()
    basis0 = np.array(d.cb.basis, copy=True)
    with _quiet():
        crosspred(d.cb, coef=coef, vcov=vcov, model_link='log', at=AT, cen=15.0, cumul=True)
    assert np.array_equal(coef, c0) and np.array_equal(vcov, v0) and np.array_equal(AT, a0)
    assert np.array_equal(np.asarray(d.cb.basis), basis0, equal_nan=True)
    assert np.array_equal(d.cb.argvar['knots'], d.kv)


@pytest.mark.parametrize('case', ['lin', 'lin_intercept', 'poly3', 'poly2_intercept_scale'])
def test_crosspred_of_onebasis_lin_and_poly_matches_r(case):
    """crosspred(OneBasis, coef=, vcov=) for the marginal lin/poly bases (poly carries its scale): lag-free fields,
    log link, explicit cen (audit: <= 7e-18)."""
    from basis import OneBasis
    from prediction import crosspred
    spec = dict(ONEBASIS_CASES[case])
    fun = spec.pop('fun')
    x = _series('chicago')
    x = x[~np.isnan(x)][:1500]
    np2r('bf_x', x)
    rkw = ''.join(f', {k}={_rbool(v) if isinstance(v, bool) else repr(v)}' for k, v in spec.items())
    r(f'bf_ob <- onebasis(bf_x, fun="{fun}"{rkw})')
    with _quiet():
        ob = OneBasis(x, fun=fun, **spec)
    p = ob.shape[1]
    coef, vcov = _coef_vcov(p, 83, scale=0.02)
    np2r('bf_coef', coef)
    np2r('bf_vcov', vcov)
    at = np.arange(-10.0, 30.0 + 1e-9, 2.5)
    np2r('bf_at', at)
    cen = 10.0 if 'intercept' not in case else None                 # R drops cen when the basis has an intercept
    rcen = f', cen={cen!r}' if cen is not None else ''
    r(f'bf_pR <- crosspred(bf_ob, coef=bf_coef, vcov=bf_vcov, model.link="log", at=bf_at{rcen})')
    with _quiet():
        pp = crosspred(ob, coef=coef, vcov=vcov, model_link='log', at=at, cen=cen)
    _assert_fields(pp, 'bf_pR', ['predvar', 'coefficients', 'vcov', 'allfit', 'allse', 'allRRfit', 'allRRlow',
                                 'allRRhigh', 'matfit', 'matse'], what=case)


# --- a real fitted quasi-Poisson model ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _ew_data(region):
    """England & Wales region in R (`bf_d_<region>`, sorted as in the file) and as Python objects."""
    tag = region.replace('-', '_').replace('&', '_')
    r(f'''bf_ew <- read.csv("{EW_CSV}", row.names=1); bf_ew$date <- as.Date(bf_ew$date)
          bf_d_{tag} <- bf_ew[bf_ew$regnames == "{region}", ]; bf_d_{tag}$dow <- factor(bf_d_{tag}$dow)''')
    days = rget(f'as.numeric(bf_d_{tag}$date)')
    return SimpleNamespace(tag=tag, region=region, r=f'bf_d_{tag}',
                           tmean=rget(f'as.numeric(bf_d_{tag}$tmean)'),
                           death=rget(f'as.numeric(bf_d_{tag}$death)'),
                           dates=pd.Series(pd.to_datetime(days, unit='D')))


@lru_cache(maxsize=None)
def _first_stage(region):
    """R quasi-Poisson first stage of the Lancet analysis for one region (01.firststage.R) plus the identical PyDLNM
    CrossBasis. R objects: bf_cb_fs_<tag>, bf_m_<tag>, bf_red_<tag>."""
    from basis import CrossBasis
    from utils import logknots
    e = _ew_data(region)
    t = e.tag
    np2r(f'bf_tm_{t}', e.tmean)
    r(f'''bf_kv_{t} <- quantile(bf_tm_{t}, c(.10,.75,.90), na.rm=TRUE)
          bf_cb_fs_{t} <- crossbasis(bf_tm_{t}, lag=21, argvar=list(fun="bs", degree=2, knots=bf_kv_{t}),
                                     arglag=list(knots=logknots(21,3)))
          bf_m_{t} <- glm(death ~ bf_cb_fs_{t} + dow + ns(date, df=8*length(unique(year))), {e.r},
                          family=quasipoisson, na.action="na.exclude")
          bf_ix_{t} <- grep("^bf_cb_fs_", names(coef(bf_m_{t})))
          bf_cen_{t} <- mean(bf_tm_{t}, na.rm=TRUE)
          bf_red_{t} <- crossreduce(bf_cb_fs_{t}, bf_m_{t}, cen=bf_cen_{t})''')
    kv = rget(f'bf_kv_{t}')
    cb = CrossBasis(e.tmean, lag=21, argvar={'fun': 'bs', 'degree': 2, 'knots': kv},
                    arglag={'fun': 'ns', 'knots': logknots([0, 21], nk=3)})
    return SimpleNamespace(e=e, tag=t, cb=cb, cb_r=f'bf_cb_fs_{t}', m_r=f'bf_m_{t}', red_r=f'bf_red_{t}',
                           coef=rget(f'unname(coef(bf_m_{t})[bf_ix_{t}])'),
                           vcov=rget(f'unname(vcov(bf_m_{t})[bf_ix_{t}, bf_ix_{t}])'),
                           red_coef=rget(f'as.numeric(bf_red_{t}$coefficients)'),
                           red_vcov=rget(f'unname(bf_red_{t}$vcov)'),
                           cen=float(rget(f'bf_cen_{t}')[0]))


def test_crosspred_real_r_glm_coefficients_match_r():
    """Real quasi-Poisson coef/vcov of London (bs2 x ns(logknots(21,3)), dow + ns(date)) fed to both sides; R's crosspred
    is called with the coefficients so the comparison isolates crosspred; cumul=TRUE."""
    from prediction import crosspred
    fs = _first_stage('London')
    np2r('bf_coef', fs.coef)
    np2r('bf_vcov', fs.vcov)
    at = np.arange(-5.0, 28.0 + 1e-9, 0.5)
    np2r('bf_at', at)
    r(f'bf_pR <- crosspred({fs.cb_r}, coef=bf_coef, vcov=bf_vcov, model.link="log", at=bf_at, '
      f'cen={fs.cen!r}, cumul=TRUE)')
    with _quiet():
        pp = crosspred(fs.cb, coef=fs.coef, vcov=fs.vcov, model_link='log', at=at, cen=fs.cen, cumul=True)
    _assert_fields(pp, 'bf_pR', _r_pred_fields('bf_pR'), what='London real coefficients')


def test_crosspred_with_statsmodels_quasipoisson_model_matches_r():
    """model=<statsmodels GLM(Poisson).fit(scale='X2')> on R's own design matrix (cross-basis columns first, named
    like R's): coefficients and standard errors of every lag-specific, overall and cumulative field agree with R's
    crosspred(cb, glm) (both IRLS fits run to a fixed point; bound 1e-9, measured ~1e-13). Only the fit/se fields are compared
    because how the RR fields are labelled for a statsmodels model is a separate topic (theme I2)."""
    from prediction import crosspred
    g = _chicago_glm()
    at = np.arange(-15.0, 30.0 + 1e-9, 1.0)
    np2r('bf_at', at)
    r(f'bf_pR <- crosspred({g.d.cb_r}, bf_m_chi, at=bf_at, cen=15, cumul=TRUE)')
    with _quiet():
        pp = crosspred(g.d.cb, model=g.res, at=at, cen=15.0, cumul=True)
    _assert_fields(pp, 'bf_pR', ['predvar', 'cen', 'lag', 'coefficients', 'vcov', 'matfit', 'matse', 'allfit',
                                 'allse', 'cumfit', 'cumse'], rtol=1e-9, what='statsmodels model')


_CHI_GLM = {}


def _chicago_glm():
    """R quasi-Poisson glm of Chicago (bs2 x ns(logknots(21,3)) + dow + ns(time, 28)) and a statsmodels
    GLM(Poisson).fit(scale='X2') on R's identical design matrix, cross-basis columns first and named like R's
    ('cbv1.l1', ...). R's summary dispersion uses the working weights of the last IRLS iteration's START, so with R's
    default stopping rule (4 iterations here) it is only exact to ~1e-8; R is therefore iterated to a fixed point
    (epsilon tiny, 12 iterations, warning suppressed) and statsmodels run to tol=1e-13. The two implementations then
    agree at ~1e-11."""
    if 'g' not in _CHI_GLM:
        sm = pytest.importorskip('statsmodels.api')
        d = _design('bs2_ns_L21')
        r(f'''bf_dat <- data.frame(death=chicagoNMMAPS$death, dow=chicagoNMMAPS$dow,
                                   time=seq_along(chicagoNMMAPS$death))
              bf_m_chi <- suppressWarnings(glm(death ~ {d.cb_r} + dow + ns(time, df=28), data=bf_dat,
                              family=quasipoisson, na.action="na.exclude",
                              control=glm.control(epsilon=1e-300, maxit=12)))
              bf_mm <- model.matrix(bf_m_chi); bf_yy <- bf_m_chi$y
              bf_isc <- grep("^bf_cb_", colnames(bf_mm))
              bf_mm <- bf_mm[, c(bf_isc, setdiff(seq(ncol(bf_mm)), bf_isc))]''')
        X = rget('unname(bf_mm)')
        names = ['cb' + n[n.rindex('v'):] if n.startswith('bf_cb_') else n for n in map(str, r('colnames(bf_mm)'))]
        exog = pd.DataFrame(X, columns=names)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = sm.GLM(rget('as.numeric(bf_yy)'), exog, family=sm.families.Poisson()).fit(
                scale='X2', tol=1e-13, maxiter=200)
        _CHI_GLM['g'] = SimpleNamespace(d=d, res=res)
    return _CHI_GLM['g']


# ==============================================================================================================
# 4  crosspred on reduced (BLUP-style) coefficients: overall fields
# ==============================================================================================================
# In R the reduced route is crosspred(onebasis, coef=, vcov=), whose lag range is c(0,0); PyDLNM keeps the CrossBasis lag
# range for the lag-specific and cumulative fields, so only the OVERALL curve (allfit, allse, allRR*) is comparable. That
# is the quantity the Lancet pipeline uses (BLUP curves, MMT, attributable risk).
REDUCED_FIELDS = ['predvar', 'cen', 'coefficients', 'vcov', 'allfit', 'allse']
REDUCED_FIELDS_LOG = REDUCED_FIELDS + ['allRRfit', 'allRRlow', 'allRRhigh']
REDUCED_FIELDS_LIN = REDUCED_FIELDS + ['alllow', 'allhigh']


def _reduced_pair(d, seed, link='log', ci=0.95, cen=15.0, at=AT):
    from prediction import crosspred
    coef, vcov = _coef_vcov(d.pvar, seed, scale=0.3)
    np2r('bf_rcoef', coef)
    np2r('bf_rvcov', vcov)
    np2r('bf_at', at)
    ls = f', model.link="{link}"' if link else ''
    r(f'bf_pR <- suppressWarnings(crosspred({d.ob_r}, coef=bf_rcoef, vcov=bf_rvcov{ls}, at=bf_at, cen={cen!r}, '
      f'ci.level={ci!r}))')
    with _quiet():
        pp = crosspred(d.cb, coef=coef, vcov=vcov, model_link=link, at=at, cen=cen, ci_level=ci)
    return pp


@pytest.mark.parametrize('name', ['bs2_ns_L21', 'bs3_int_L6', 'ns_ns_L10', 'bs2_ns_lag2_10', 'ns_int_L3'])
def test_reduced_coefficient_crosspred_overall_fields_match_r(name):
    """crosspred(cb, coef=reduced, vcov=reduced) vs R crosspred(onebasis, ...): the overall curve and its CI."""
    d = _design(name)
    pp = _reduced_pair(d, seed=17)
    _assert_fields(pp, 'bf_pR', REDUCED_FIELDS_LOG, what=name)


@pytest.mark.parametrize('link,ci,cen', [('log', 0.90, 10.0), ('logit', 0.99, 20.0), (None, 0.95, 15.0)])
def test_reduced_coefficient_crosspred_links_ci_levels_and_cen_match_r(link, ci, cen):
    d = _design('bs2_ns_L21')
    pp = _reduced_pair(d, seed=23, link=link, ci=ci, cen=cen)
    _assert_fields(pp, 'bf_pR', REDUCED_FIELDS_LIN if link is None else REDUCED_FIELDS_LOG,
                   what=f'link={link} ci={ci} cen={cen}')


def test_reduced_coefficient_crosspred_at_beyond_range_matches_r():
    d = _design('bs2_ns_L21')
    lo, hi = float(np.nanmin(d.temp)), float(np.nanmax(d.temp))
    at = np.linspace(lo - 4, hi + 4, 31)
    pp = _reduced_pair(d, seed=29, at=at)
    _assert_fields(pp, 'bf_pR', REDUCED_FIELDS_LOG, what='reduced, at beyond range')


def test_reduced_crosspred_with_real_london_reduction_matches_r():
    """The validated route on real numbers: R's crossreduce coefficients of London with a deterministic BLUP-like
    perturbation, crosspred at the England & Wales grid with the log link and cen = R's minimum (which.min)."""
    from prediction import crosspred
    fs = _first_stage('London')
    rng = np.random.default_rng(31)
    blup = fs.red_coef + rng.normal(0, 0.02, fs.red_coef.shape)
    np2r('bf_rcoef', blup)
    np2r('bf_rvcov', fs.red_vcov)
    lo, hi = float(fs.e.tmean.min()), float(fs.e.tmean.max())
    at = np.linspace(lo, hi, 60)
    np2r('bf_at', at)
    ob = f'bf_ob_fs_{fs.tag}'
    r(f'{ob} <- do.call("onebasis", c(list(x=bf_tm_{fs.tag}), attr({fs.cb_r}, "argvar")))')
    r(f'bf_p0 <- crosspred({ob}, coef=bf_rcoef, vcov=bf_rvcov, model.link="log", at=bf_at, cen=median(bf_at))')
    mmt = float(rget('bf_at[which.min(bf_p0$allfit)]')[0])
    r(f'bf_pR <- crosspred({ob}, coef=bf_rcoef, vcov=bf_rvcov, model.link="log", at=bf_at, cen={mmt!r})')
    with _quiet():
        pp = crosspred(fs.cb, coef=blup, vcov=fs.red_vcov, model_link='log', at=at, cen=mmt)
    _assert_fields(pp, 'bf_pR', REDUCED_FIELDS_LOG, what='London reduced')
    assert np.isclose(float(pp.allRRfit[int(np.argmin(np.abs(at - mmt)))]), 1.0, rtol=0, atol=1e-15)


# ==============================================================================================================
# 5  crossreduce type="overall"
# ==============================================================================================================
@pytest.mark.parametrize('name', list(CONFIGS))
def test_crossreduce_overall_matches_r(name):
    """M = I (x) 1'B_lag: reduced coefficients and vcov versus R crossreduce(cb, coef=, vcov=, type='overall'); the
    reduction does not depend on cen (R and Python)."""
    from crossreduce import crossreduce
    d = _design(name)
    coef, vcov = _coef_vcov(d.p, 41)
    np2r('bf_coef', coef)
    np2r('bf_vcov', vcov)
    r(f'bf_red <- crossreduce({d.cb_r}, coef=bf_coef, vcov=bf_vcov, type="overall", model.link="log", cen=15)')
    r(f'bf_red2 <- crossreduce({d.cb_r}, coef=bf_coef, vcov=bf_vcov, type="overall", model.link="log", cen=-3)')
    rc, rv = rget('as.numeric(bf_red$coefficients)'), rget('unname(bf_red$vcov)')
    for cen in (15.0, -3.0, None):
        with _quiet():
            red = crossreduce(d.cb, coef=coef, vcov=vcov, cen=cen)
        assert_close(red.coef, rc, rtol=RTOL_PRED, what=f'{name} reduced coef (cen={cen})')
        assert_close(red.vcov, rv, rtol=RTOL_PRED, what=f'{name} reduced vcov (cen={cen})')
    assert_close(rget('as.numeric(bf_red2$coefficients)'), rc, rtol=1e-13, what='R: reduction independent of cen')


@pytest.mark.parametrize('name', ['bs2_ns_L21', 'ns_ns_L10', 'bs2_ns_lag2_10', 'ns_int_L3'])
def test_reduced_and_full_crosspred_overall_curves_agree(name):
    """Consistency of the two routes inside PyDLNM (they agree with R to ~1e-15 each): the overall curve of the full
    coefficients equals the overall curve of the reduced coefficients."""
    from crossreduce import crossreduce
    from prediction import crosspred
    d = _design(name)
    coef, vcov = _coef_vcov(d.p, 43)
    with _quiet():
        red = crossreduce(d.cb, coef=coef, vcov=vcov)
        full = crosspred(d.cb, coef=coef, vcov=vcov, model_link='log', at=AT, cen=15.0)
        reduced = crosspred(d.cb, coef=red.coef, vcov=red.vcov, model_link='log', at=AT, cen=15.0)
    assert_close(reduced.allfit, full.allfit, rtol=RTOL_PRED, what=f'{name} allfit')
    assert_close(reduced.allse, full.allse, rtol=RTOL_PRED, what=f'{name} allse')


def test_crossreduce_input_validation_like_r():
    """Wrong-length coef/vcov and non-CrossBasis input are rejected (R stops as well)."""
    from basis import OneBasis
    from crossreduce import crossreduce
    d = _design('bs2_ns_L21')
    coef, vcov = _coef_vcov(d.p, 3)
    np2r('bf_coef', coef[:-1])
    np2r('bf_vcov', vcov[:-1, :-1])
    with pytest.raises(Exception):
        r(f'crossreduce({d.cb_r}, coef=bf_coef, vcov=bf_vcov)')
    with _quiet(), pytest.raises(ValueError):
        crossreduce(d.cb, coef=coef[:-1], vcov=vcov[:-1, :-1])
    with _quiet(), pytest.raises(ValueError):
        crossreduce(d.cb)
    with _quiet(), pytest.raises((TypeError, ValueError)):
        crossreduce(OneBasis(np.arange(10.0), fun='lin'), coef=coef, vcov=vcov)


# ==============================================================================================================
# 6  first-stage GLM (quasi-Poisson, dow + ns(date)) on England & Wales regions
# ==============================================================================================================
def _fit_py_first_stage(fs, family='quasipoisson', dfseas=8):
    from improved_glm import ImprovedGLMInterface
    with _quiet():
        g = ImprovedGLMInterface(fs.cb)
        g.fit_dlnm_model(fs.e.death, fs.e.dates, dfseas=dfseas, family=family)
    return g


@pytest.mark.parametrize('region', ['London', 'Wales'])
def test_first_stage_quasipoisson_glm_matches_r(region):
    """ImprovedGLMInterface (death ~ cb + dow + ns(date, 8 per year), quasi-Poisson, na.exclude) vs R glm on the same
    region: cross-basis coefficients and vcov (audit: <= 3e-13 for all 10 regions and 106 US cities)."""
    fs = _first_stage(region)
    g = _fit_py_first_stage(fs)
    assert g.cb_coef.shape == fs.coef.shape == (25,)
    assert_close(g.cb_coef, fs.coef, rtol=RTOL_GLM, what=f'{region} cb coef')
    assert_close(g.cb_vcov, fs.vcov, rtol=RTOL_GLM, what=f'{region} cb vcov')


@pytest.mark.parametrize('region', ['London', 'Wales'])
def test_first_stage_reduced_coefficients_match_r(region):
    """crossreduce(cb, model, cen=mean(tmean)) as in 01.firststage.R: the interface's reduction (coef and vcov) vs R,
    and the module-level crossreduce given R's own cross-basis block."""
    from crossreduce import crossreduce
    fs = _first_stage(region)
    g = _fit_py_first_stage(fs)
    with _quiet():
        red = g.crossreduce(cen=fs.cen)
        red_from_r = crossreduce(fs.cb, coef=fs.coef, vcov=fs.vcov, cen=fs.cen)
    assert red.coef.shape == fs.red_coef.shape == (5,)
    assert_close(red.coef, fs.red_coef, rtol=RTOL_GLM, what=f'{region} reduced coef')
    assert_close(red.vcov, fs.red_vcov, rtol=RTOL_GLM, what=f'{region} reduced vcov')
    assert_close(red_from_r.coef, fs.red_coef, rtol=RTOL_PRED, what=f'{region} reduction of R coefficients')
    assert_close(red_from_r.vcov, fs.red_vcov, rtol=RTOL_PRED, what=f'{region} reduction of R vcov')


def test_first_stage_poisson_family_matches_r():
    """Family dispatch: Poisson (dispersion fixed at 1) vs R glm(family=poisson) on London."""
    fs = _first_stage('London')
    t = fs.tag
    r(f'''bf_mp <- glm(death ~ {fs.cb_r} + dow + ns(date, df=8*length(unique(year))), {fs.e.r}, family=poisson,
                       na.action="na.exclude")
          bf_ixp <- grep("^bf_cb_fs_", names(coef(bf_mp)))''')
    g = _fit_py_first_stage(fs, family='poisson')
    assert_close(g.cb_coef, rget('unname(coef(bf_mp)[bf_ixp])'), rtol=RTOL_GLM, what='Poisson coef')
    assert_close(g.cb_vcov, rget('unname(vcov(bf_mp)[bf_ixp, bf_ixp])'), rtol=RTOL_GLM, what='Poisson vcov')


def test_first_stage_does_not_mutate_inputs():
    fs = _first_stage('London')
    basis0 = np.array(fs.cb.basis, copy=True)
    y0, d0 = fs.e.death.copy(), fs.e.dates.copy()
    _fit_py_first_stage(fs)
    assert np.array_equal(np.asarray(fs.cb.basis), basis0, equal_nan=True)
    assert np.array_equal(fs.e.death, y0) and fs.e.dates.equals(d0)


# ==============================================================================================================
# 7  MVMeta (mvmeta 1.0.3): algebra at identical Psi, BLUP formula, fits
# ==============================================================================================================
_R_META_DEFS = '''
bfm_lists <- function(X, y, Sv) {
  k <- ncol(y); m <- nrow(y); p <- ncol(X); nay <- is.na(y)
  list(Xlist = lapply(seq(m), function(i) diag(1, k)[!nay[i, ], , drop = FALSE] %x% X[i, , drop = FALSE]),
       ylist = lapply(seq(m), function(i) y[i, ][!nay[i, ]]),
       Slist = lapply(seq(m), function(i) mixmeta:::xpndMat(Sv[i, ])[!nay[i, ], !nay[i, ], drop = FALSE]),
       nalist = lapply(seq(m), function(i) nay[i, ]), k = k, m = m, p = p, nall = sum(!nay))
}
bfm_prof <- function(par, L, what) {
  f <- get(what, envir = asNamespace("mvmeta"))
  f(par, L$Xlist, L$ylist, L$Slist, L$nalist, L$k, L$m, L$p, L$nall, "unstr", NULL)
}
bfm_formula_fit <- function(y, Sv, X, method, control) {
  df <- data.frame(id = seq_len(nrow(y))); df$y <- y; df$Sv <- Sv; df$X <- X
  mvmeta(y ~ X - 1, S = Sv, data = df, method = method, control = control)
}
'''
R_TIGHT = 'list(maxiter=20000, reltol=1e-14)'
# (n studies, k outcomes, p meta-predictors, tau, seed): simulated meta-analyses with an interior optimum
META_CFGS = [(30, 2, 2, 0.4, 5), (40, 3, 2, 0.3, 11), (25, 4, 1, 0.3, 7), (40, 5, 3, 0.3, 21)]
META_IDS = [f'n{c[0]}k{c[1]}p{c[2]}' for c in META_CFGS]


@pytest.fixture(scope='module')
def _meta_r():
    """R packages and helper functions for the meta-analysis tests (skips if mvmeta / mixmeta are missing)."""
    for pkg in ('mvmeta', 'mixmeta'):
        if not bool(r(f'isTRUE(suppressWarnings(requireNamespace("{pkg}", quietly=TRUE)))')[0]):
            pytest.skip(f'R package {pkg} not installed')
    r('suppressMessages({library(mvmeta); library(mixmeta)})')
    r(_R_META_DEFS)


@pytest.fixture(scope='module', autouse=True)
def _single_threaded_blas():
    """The matrices are tiny; BLAS worker threads only spin. Numerical results are unaffected."""
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        yield
        return
    with threadpool_limits(limits=1, user_api='blas'):
        yield


def _meta_sim(n, k, p=1, tau=0.3, seed=0, sw=0.2, corr=0.5):
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
        dd = sw * np.exp(rng.normal(scale=0.5, size=k))
        S[i] = np.outer(dd, dd) * (np.full((k, k), corr) + (1 - corr) * np.eye(k))
    y = np.array([X[i] @ beta + rng.multivariate_normal(np.zeros(k), Psi + S[i]) for i in range(n)])
    return y, S, X


def _vech_rows(S):
    """R's layout of S: lower triangle of every S_i, column by column."""
    n, k, _ = S.shape
    idx = [(a, b) for b in range(k) for a in range(b, k)]
    return np.array([[S[i, a, b] for (a, b) in idx] for i in range(n)])


def _meta_push(y, S, X):
    np2r('bfm_y', y)
    np2r('bfm_Sv', _vech_rows(S))
    np2r('bfm_X', X)


def _r_meta_fit(y, S, X, method='reml', control=R_TIGHT):
    """mvmeta.fit (the estimation engine of mvmeta()); coefficient matrix (p,k), outcome-major vcov, Psi, logLik."""
    _meta_push(y, S, X)
    r(f'bfm_fit <- mvmeta:::mvmeta.fit(as.matrix(bfm_X), as.matrix(bfm_y), as.matrix(bfm_Sv), '
      f'method="{method}", control={control})')
    p, k = X.shape[1], y.shape[1]
    return dict(coef=np.asarray(rget('bfm_fit$coefficients')).reshape(p, k), vcov=rget('bfm_fit$vcov'),
                psi=rget('bfm_fit$Psi'), loglik=float(rget('as.numeric(bfm_fit$logLik)').ravel()[0]))


def _py_meta_fit(y, S, X, method='reml'):
    from meta_analysis import MVMeta
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return MVMeta(method=method).fit(y, S, X)


def _py_lists(y, S, X):
    n, k = y.shape
    return [np.kron(np.eye(k), X[i:i + 1]) for i in range(n)], list(y), list(S)


def _r_par(L):
    """R's parameter order for a lower-Cholesky factor: lower triangle column by column."""
    k = L.shape[0]
    return np.array([L[a, b] for b in range(k) for a in range(b, k)])


@pytest.mark.parametrize('cfg', META_CFGS, ids=META_IDS)
def test_mvmeta_objective_and_reml_gradient_match_r_at_identical_psi(cfg, _meta_r):
    """Negative REML and ML profile log-likelihood and the analytic REML gradient equal mvmeta:::remlprof.fn /
    mlprof.fn / remlprof.gr at the same Psi (random lower-Cholesky parameters; Python's row-major and R's column-wise
    parameter orders are mapped explicitly). Audit: objective 2.5e-14, gradient 3.2e-13 relative (342 evaluations)."""
    import meta_analysis as ma
    n, k, p, tau, seed = cfg
    y, S, X = _meta_sim(n, k, p, tau, seed)
    _meta_push(y, S, X)
    r('bfm_L <- bfm_lists(as.matrix(bfm_X), as.matrix(bfm_y), as.matrix(bfm_Sv))')
    Xl, yl, Sl = _py_lists(y, S, X)
    rng = np.random.default_rng(1000 + seed)
    ti = np.tril_indices(k)
    order_py = list(zip(*ti))
    order_r = [(a, b) for b in range(k) for a in range(b, k)]
    for rep in range(3):
        L = np.tril(rng.normal(size=(k, k))) * tau
        par_py = L[ti]
        np2r('bfm_par', _r_par(L))
        for what, fn in [('remlprof.fn', ma._reml_fn), ('mlprof.fn', ma._ml_fn)]:
            ref = float(r(f'bfm_prof(bfm_par, bfm_L, "{what}")')[0])
            assert_close(np.array([-fn(par_py, k, Xl, yl, Sl)]), np.array([ref]), rtol=1e-10,
                         what=f'{what} at rep {rep}')
        g_py = -ma._reml_gr(par_py, k, Xl, yl, Sl)                # gradient of the log-likelihood, Python order
        g_py_in_r = np.array([g_py[order_py.index(ab)] for ab in order_r])
        g_r = r2np(r('bfm_prof(bfm_par, bfm_L, "remlprof.gr")')).ravel()
        assert_close(g_py_in_r, g_r, rtol=1e-10, what=f'remlprof.gr at rep {rep}')


@pytest.mark.parametrize('cfg', META_CFGS, ids=META_IDS)
def test_mvmeta_blup_formula_matches_r_at_identical_estimates(cfg, _meta_r):
    """blup() is exact algebra: with PyDLNM's Psi / coefficients / vcov injected into R's mvmeta object, R's
    blup.mvmeta(vcov=TRUE) and PyDLNM's blup() agree to rounding (point estimates and their vcov)."""
    from meta_analysis import blup
    n, k, p, tau, seed = cfg
    y, S, X = _meta_sim(n, k, p, tau, seed)
    m = _py_meta_fit(y, S, X)
    _meta_push(y, S, X)
    r('bfm_mv <- bfm_formula_fit(as.matrix(bfm_y), as.matrix(bfm_Sv), as.matrix(bfm_X), "reml", list())')
    np2r('bfm_psi', m.psi)
    np2r('bfm_coef', m.coefficients)
    np2r('bfm_vcov', m.vcov)
    r('bfm_mv$Psi <- bfm_psi; bfm_mv$coefficients <- bfm_coef; bfm_mv$vcov <- bfm_vcov')
    r('bfm_bl <- blup(bfm_mv, vcov=TRUE)')
    ref_b = np.array([rget(f'as.numeric(bfm_bl[[{i + 1}]]$blup)') for i in range(n)])
    ref_v = np.array([rget(f'unname(bfm_bl[[{i + 1}]]$vcov)') for i in range(n)])
    res = blup(m, vcov=True)
    assert_close(np.array([x['blup'] for x in res]), ref_b, rtol=1e-10, what='BLUP')
    assert_close(np.array([x['vcov'] for x in res]), ref_v, rtol=1e-10, what='BLUP vcov')


META_FITS = [(META_CFGS[0], 'reml'), (META_CFGS[0], 'ml'), (META_CFGS[2], 'reml'), (META_CFGS[3], 'reml')]
META_FIT_IDS = ['n30k2p2-reml', 'n30k2p2-ml', 'n25k4p1-reml', 'n40k5p3-reml']


@pytest.mark.parametrize('cfg,method', META_FITS, ids=META_FIT_IDS)
def test_mvmeta_fit_agrees_with_tightly_converged_r(cfg, method, _meta_r):
    """R run at reltol=1e-14: PyDLNM's coefficients (p,k), vcov (outcome-major), Psi and logLik agree to <= 1e-6
    relative (audit: median 2e-10, max 6.5e-8), and PyDLNM never stops at a worse optimum. Agreement with R's DEFAULT
    control is optimiser-limited (~1e-5) and is not asserted here."""
    n, k, p, tau, seed = cfg
    y, S, X = _meta_sim(n, k, p, tau, seed)
    ref = _r_meta_fit(y, S, X, method)
    m = _py_meta_fit(y, S, X, method)
    assert m.coefficients.shape == (p, k)
    assert_close(m.coefficients, ref['coef'], rtol=1e-6, what=f'{method} coefficients')
    assert_close(m.vcov, ref['vcov'], rtol=1e-6, what=f'{method} vcov')
    assert_close(m.psi, ref['psi'], rtol=1e-6, what=f'{method} Psi')
    assert abs(m.loglik - ref['loglik']) <= 1e-8 * max(1.0, abs(ref['loglik'])), (m.loglik, ref['loglik'])
    assert m.loglik >= ref['loglik'] - 1e-8 * abs(ref['loglik']), 'PyDLNM stopped at a worse optimum than R'


# ==============================================================================================================
# 8  centering: explicit cen, recentered basis, minimum-mortality value on identical grids
# ==============================================================================================================
CEN_VALUES = [-10.0, 0.0, 5.5, 10.0, 15.0, 21.1, 'mean', 'min', 'max', 40.0, -40.0]     # 40 / -40 lie outside the data
CEN_FIELDS = ['cen', 'predvar', 'allfit', 'allse', 'matfit', 'matse', 'allRRfit', 'allRRlow', 'allRRhigh']


@pytest.mark.parametrize('cen', CEN_VALUES, ids=[str(c) for c in CEN_VALUES])
def test_crosspred_explicit_cen_matches_r(cen):
    """Explicit reference values, incl. mean/min/max of the data and values outside the range (audit: <= 1.8e-15)."""
    d = _design('bs2_ns_L21')
    val = {'mean': float(np.nanmean(d.temp)), 'min': float(np.nanmin(d.temp)),
           'max': float(np.nanmax(d.temp))}.get(cen, cen)
    at = np.arange(-15.0, 30.0 + 1e-9, 1.0)
    pp, robj = _full_pred_pair(d, dict(link='log', cen=val), seed=13, at=at)
    _assert_fields(pp, robj, CEN_FIELDS, what=f'cen={cen}')
    i = int(np.argmin(np.abs(at - val)))
    if at[i] == val:                                                 # exactly zero effect at the reference
        assert pp.allfit[i] == 0.0 and pp.allse[i] == 0.0


def test_recenter_basis_matches_r_crosspred_cen():
    """recenter_basis(cb, None, cen=15) leaves the cross-basis matrix untouched (cen is metadata, as in R) and
    crosspred on it, without a cen argument, equals R's crosspred(cb, cen=15); the original basis is not mutated."""
    from centering import recenter_basis
    from prediction import crosspred
    d = _design('bs2_ns_L21')
    coef, vcov = _coef_vcov(d.p, 19)
    np2r('bf_coef', coef)
    np2r('bf_vcov', vcov)
    np2r('bf_at', AT)
    r(f'bf_pR <- crosspred({d.cb_r}, coef=bf_coef, vcov=bf_vcov, model.link="log", at=bf_at, cen=15)')
    basis0 = np.array(d.cb.basis, copy=True)
    with _quiet():
        cb2, _ = recenter_basis(d.cb, None, cen=15.0)
        pp = crosspred(cb2, coef=coef, vcov=vcov, model_link='log', at=AT)
    assert np.array_equal(np.asarray(cb2.basis), basis0, equal_nan=True)
    assert np.array_equal(np.asarray(d.cb.basis), basis0, equal_nan=True)
    _assert_fields(pp, 'bf_pR', CEN_FIELDS, what='recentered basis')


def test_cen_stored_in_argvar_is_used_and_overridable_like_r():
    """cen given in argvar is retained, does not change the basis matrix, is used by crosspred when no cen is passed and
    is overridden by an explicit cen (R: crossbasis(argvar=list(..., cen=)))."""
    from basis import CrossBasis
    from prediction import crosspred
    d = _design('bs2_ns_L21')
    kv = np.array(d.kv, copy=True)
    np2r('bf_kvc', kv)
    r(f'bf_cbc <- crossbasis(bf_temp_{d.key}, lag=21, argvar=list(fun="bs", degree=2, knots=bf_kvc, cen=12), '
      f'arglag=list(fun="ns", knots=bf_lk_{d.key}))')
    lagk = np.array(d.lagk, copy=True)
    with _quiet():
        cbc = CrossBasis(d.temp, lag=21, argvar={'fun': 'bs', 'degree': 2, 'knots': kv, 'cen': 12.0},
                         arglag={'fun': 'ns', 'knots': lagk})
    assert_close(np.asarray(cbc.basis), rget('unclass(bf_cbc)'), rtol=RTOL_BASIS, what='cross-basis with cen')
    assert_close(np.asarray(cbc.basis), np.asarray(d.cb.basis), rtol=0, what='cen does not change the matrix')
    coef, vcov = _coef_vcov(d.p, 37)
    np2r('bf_coef', coef)
    np2r('bf_vcov', vcov)
    np2r('bf_at', AT)
    for cen_arg, rcen in [(None, ''), (7.0, ', cen=7')]:
        r(f'bf_pR <- crosspred(bf_cbc, coef=bf_coef, vcov=bf_vcov, model.link="log", at=bf_at{rcen})')
        with _quiet():
            pp = crosspred(cbc, coef=coef, vcov=vcov, model_link='log', at=AT, cen=cen_arg)
        _assert_fields(pp, 'bf_pR', CEN_FIELDS, what=f'argvar cen, crosspred cen={cen_arg}')


def _r_which_min(robj_fit, grid):
    """R's minimum of an overall curve on `grid` (first index wins, which.min)."""
    np2r('bf_grid', grid)
    return float(rget(f'bf_grid[which.min({robj_fit})]')[0])


@pytest.mark.parametrize('seed', [1, 2, 3])
def test_find_mmt_matches_r_which_min_on_identical_grid(seed):
    """find_mmt on an explicit grid returns R's which.min minimum; the (uncentered) curve differs from R's centered one
    only by its value at the reference."""
    from centering import find_mmt
    d = _design('bs2_ns_L21')
    coef, vcov = _coef_vcov(d.p, 50 + seed, scale=0.08)
    grid = np.linspace(-15.0, 30.0, 451)
    np2r('bf_coef', coef)
    np2r('bf_vcov', vcov)
    np2r('bf_at', grid)
    r(f'bf_pR <- crosspred({d.cb_r}, coef=bf_coef, vcov=bf_vcov, model.link="log", at=bf_at, cen=15)')
    with _quiet():
        res = find_mmt(d.cb, None, coef=coef, vcov=vcov, at=grid)
    assert res['mmt'] == _r_which_min('bf_pR$allfit', grid)
    i15 = int(np.argmin(np.abs(grid - 15.0)))
    assert grid[i15] == 15.0
    assert_close(res['all_fit'] - res['all_fit'][i15], rget('as.numeric(bf_pR$allfit)'), rtol=1e-10,
                 what='centered curve')


def test_find_mmt_reduced_coefficients_matches_r_which_min():
    """MMT of a reduced (BLUP-like) curve on an explicit grid equals R's which.min (10/10 E&W regions and 106/106 US cities
    in the audit); here real London reduced coefficients with a deterministic perturbation."""
    from centering import find_mmt
    fs = _first_stage('London')
    rng = np.random.default_rng(61)
    blup = fs.red_coef + rng.normal(0, 0.02, fs.red_coef.shape)
    lo, hi = float(fs.e.tmean.min()), float(fs.e.tmean.max())
    grid = np.linspace(lo, hi, 300)
    np2r('bf_rcoef', blup)
    np2r('bf_rvcov', fs.red_vcov)
    np2r('bf_at', grid)
    ob = f'bf_ob_fs_{fs.tag}'
    r(f'{ob} <- do.call("onebasis", c(list(x=bf_tm_{fs.tag}), attr({fs.cb_r}, "argvar")))')
    r(f'bf_pR <- crosspred({ob}, coef=bf_rcoef, vcov=bf_rvcov, model.link="log", at=bf_at, cen=median(bf_at))')
    with _quiet():
        res = find_mmt(fs.cb, None, coef=blup, vcov=fs.red_vcov, at=grid)
    assert res['mmt'] == _r_which_min('bf_pR$allfit', grid)


def test_find_mmt_flat_curve_takes_first_index_like_which_min():
    """A curve with an exactly flat minimum: np.argmin and R's which.min both return the first index."""
    from centering import find_mmt
    d = _design('bs2_ns_L21')
    grid = np.linspace(-5.0, 25.0, 61)
    coef = np.zeros(d.p)
    vcov = np.eye(d.p) * 1e-4
    with _quiet():
        res = find_mmt(d.cb, None, coef=coef, vcov=vcov, at=grid)
    np2r('bf_coef', coef)
    np2r('bf_vcov', vcov)
    np2r('bf_at', grid)
    r(f'bf_pR <- crosspred({d.cb_r}, coef=bf_coef, vcov=bf_vcov, model.link="log", at=bf_at, cen=15)')
    assert res['mmt'] == _r_which_min('bf_pR$allfit', grid) == grid[0]


# ==============================================================================================================
# 9  attrdl: per-observation attributable fraction, explicit-knot bases, log link supplied
# ==============================================================================================================
class _LogLinkModel:
    """Minimal fitted-model stand-in: cross-basis coefficients/vcov of a log-link GLM (what R's glm(family=quasipoisson)
    provides), which is how the link reaches attrdl today."""

    def __init__(self, coef, vcov):
        self.params = np.asarray(coef, dtype=float)
        self.cov_params = np.asarray(vcov, dtype=float)
        self.family = SimpleNamespace(link='log')


def _load_r_attrdl():
    """Source Gasparrini's attrdl.R (Lancet 2015 code, needs tsModel) once into a private environment."""
    if not bool(r('exists("bf_attr", envir=globalenv(), inherits=FALSE)')[0]):
        if not bool(r('isTRUE(suppressWarnings(requireNamespace("tsModel", quietly=TRUE)))')[0]):
            pytest.skip('R package tsModel not installed')
        r('suppressMessages(library(tsModel))')
        r(f'bf_attr <- new.env(); sys.source("{ATTRDL_R}", envir=bf_attr)')


# (variable basis, degree, knot probs) x (lag range, lag basis, arglag knots) on a 600-day Chicago window
ATTR_CFGS = {
    'bs2_nslog_L10': dict(var=('bs', 2, (.10, .75, .90)), lag=10, arglag='nslog2', nan_x=False),
    'ns_integer_lag2_6': dict(var=('ns', None, (.10, .50, .90)), lag=(2, 6), arglag='integer', nan_x=False),
    'bs3_nsknots_L8_nan': dict(var=('bs', 3, (.25, .75)), lag=8, arglag='nslog1', nan_x=True),
}
ATTR_RANGES = ['none', 'cold', 'heat', 'mid']
_ATTR = {}


def _attr_case(name):
    """R crossbasis `bf_acb_<name>` + Python CrossBasis on the same 600-day window (and NaN pattern), coef/vcov, cases."""
    if name in _ATTR:
        return _ATTR[name]
    from basis import CrossBasis
    cfg = ATTR_CFGS[name]
    dd = chicago()
    x = dd['temp'][1000:1600].copy()
    cases = dd['death'][1000:1600].astype(float)
    if cfg['nan_x']:
        x[[3, 50, 51, 300, 599]] = np.nan
    np2r(f'bf_ax_{name}', x)
    fun, degree, probs = cfg['var']
    r(f'bf_akv_{name} <- quantile(bf_ax_{name}, c({", ".join(repr(p) for p in probs)}), na.rm=TRUE)')
    kv = rget(f'bf_akv_{name}')
    argvar, r_argvar = dict(fun=fun, knots=kv), f'fun="{fun}", knots=bf_akv_{name}'
    if degree:
        argvar['degree'] = degree
        r_argvar += f', degree={degree}'
    lag = cfg['lag']
    if cfg['arglag'] == 'integer':
        arglag, r_arglag = dict(fun='integer'), 'fun="integer"'
    else:
        nk = int(cfg['arglag'][-1])
        r(f'bf_alk_{name} <- logknots({_lag_r(lag)}, nk={nk})')
        arglag, r_arglag = dict(fun='ns', knots=rget(f'bf_alk_{name}')), f'fun="ns", knots=bf_alk_{name}'
    r(f'bf_acb_{name} <- crossbasis(bf_ax_{name}, lag={_lag_r(lag)}, argvar=list({r_argvar}), arglag=list({r_arglag}))')
    with _quiet():                                       # NaN in x makes numpy warn inside CrossBasis
        cb = CrossBasis(x, lag=list(lag) if isinstance(lag, tuple) else lag, argvar=argvar, arglag=arglag)
    assert_close(np.asarray(cb.basis), rget(f'unclass(bf_acb_{name})'), rtol=RTOL_BASIS, what=f'{name} cross-basis')
    coef, vcov = _coef_vcov(cb.shape[1], 71, scale=0.04)
    xr = x[~np.isnan(x)]
    cen = round(float(np.quantile(xr, 0.6)), 2)
    ranges = {'none': None, 'cold': np.array([-100.0, cen]), 'heat': np.array([cen, 100.0]),
              'mid': np.array([float(np.quantile(xr, .2)), float(np.quantile(xr, .6))])}
    _ATTR[name] = SimpleNamespace(name=name, cb=cb, x=x, cases=cases, coef=coef, vcov=vcov, cen=cen, ranges=ranges,
                                  cb_r=f'bf_acb_{name}', x_r=f'bf_ax_{name}')
    return _ATTR[name]


def _r_attrdl(a, typ, tot, rng=None, x=None, cases=None):
    """R attrdl (dir='forw') on the case's coef/vcov (always interpreted as log-RR by R)."""
    _load_r_attrdl()
    np2r('bf_acases', a.cases if cases is None else cases)
    np2r('bf_acoef', a.coef)
    np2r('bf_avcov', a.vcov)
    np2r('bf_acen', [a.cen])
    x_expr = a.x_r
    if x is not None:
        np2r('bf_ax_const', x)
        x_expr = 'bf_ax_const'
    rng_s = 'NULL'
    if rng is not None:
        np2r('bf_arng', rng)                          # exact doubles: no decimal-literal round trip
        rng_s = 'bf_arng'
    r(f'bf_ares <- bf_attr$attrdl({x_expr}, {a.cb_r}, bf_acases, coef=bf_acoef, vcov=bf_avcov, type="{typ}", '
      f'dir="forw", tot={_rbool(tot)}, cen=bf_acen, range={rng_s})')
    return np.atleast_1d(rget('as.numeric(bf_ares)'))


@pytest.mark.parametrize('rng_name', ATTR_RANGES)
@pytest.mark.parametrize('name', list(ATTR_CFGS))
def test_attrdl_per_observation_af_matches_r(name, rng_name):
    """type='af', dir='forw', tot=False: 1 - exp(-X_centered %*% coef) per day, with values outside `range` forced to the
    reference. Explicit-knot bases, lag ranges [0,L] / [2,6], integer and ns lag bases, NaN in x (R keeps NaN rows,
    PyDLNM drops them: compared on the valid rows). Audit: 330 configurations, max 2.8e-16."""
    from attribution import attrdl
    a = _attr_case(name)
    rng = a.ranges[rng_name]
    ref = _r_attrdl(a, 'af', False, rng)
    with _quiet():
        res = attrdl(np.array(a.x), a.cb, np.array(a.cases), model=_LogLinkModel(a.coef, a.vcov), type='af',
                     dir='forw', tot=False, cen=a.cen, range=rng)
    valid = ~np.isnan(a.x)
    assert ref.shape == a.x.shape and np.array_equal(np.isnan(ref), ~valid)
    af = np.asarray(res['af'], dtype=float)
    if af.shape == a.x.shape:                                         # R's convention: one entry per day, NaN kept
        ref_al, x_al = ref, a.x
    else:                                                             # PyDLNM today: rows with NaN x are dropped
        assert af.shape == (int(valid.sum()),)
        ref_al, x_al = ref[valid], a.x[valid]
    assert_close(af, ref_al, rtol=RTOL_PRED, what=f'{name}/{rng_name} per-day AF')
    if rng is not None:                                               # outside the range: exactly no attribution
        outside = (x_al < rng[0]) | (x_al > rng[1])
        assert outside.any() and np.all(af[outside] == 0.0)


@pytest.mark.parametrize('typ', ['af', 'an'])
@pytest.mark.parametrize('x0', [5.0, 27.0])
def test_attrdl_total_at_constant_exposure_matches_r(typ, x0):
    """Totals at a constant exposure, where the algorithms cannot differ in the treatment of cases: every day carries the
    same AF, so total AF = AF and total AN = AF * sum(cases) in R (attrdl.R:139-145) and in PyDLNM."""
    from attribution import attrdl
    a = _attr_case('bs2_nslog_L10')
    n = len(a.x)
    x = np.full(n, x0)
    cases = np.random.default_rng(11).poisson(50, n).astype(float)
    ref = _r_attrdl(a, typ, True, None, x=x, cases=cases)
    with _quiet():
        res = attrdl(x, a.cb, cases, model=_LogLinkModel(a.coef, a.vcov), type=typ, dir='forw', tot=True, cen=a.cen)
    assert ref.shape == (1,)
    assert_close(np.atleast_1d(float(res[typ + '_total'])), ref, rtol=RTOL_PRED, what=f'{typ} total at x={x0}')


def test_attrdl_does_not_mutate_inputs():
    from attribution import attrdl
    a = _attr_case('ns_integer_lag2_6')
    x, cases = np.array(a.x, copy=True), np.array(a.cases, copy=True)
    x0, c0, basis0 = x.copy(), cases.copy(), np.array(a.cb.basis, copy=True)
    with _quiet():
        attrdl(x, a.cb, cases, model=_LogLinkModel(a.coef, a.vcov), type='af', dir='forw', tot=False, cen=a.cen)
    assert np.array_equal(x, x0) and np.array_equal(cases, c0)
    assert np.array_equal(np.asarray(a.cb.basis), basis0, equal_nan=True)


# ==============================================================================================================
# 10  utils: logknots, mklag, seqlag, exphist
# ==============================================================================================================
LOGKNOTS_CASES = [
    ('21, nk=3', 'logknots(21, nk=3)', dict(x=21, nk=3)),
    ('[2,21], nk=2', 'logknots(c(2,21), nk=2)', dict(x=[2, 21], nk=2)),
    ('30, nk=5', 'logknots(30, nk=5)', dict(x=30, nk=5)),
    ('10, nk=1', 'logknots(10, nk=1)', dict(x=10, nk=1)),
    ('[0,14], nk=4', 'logknots(c(0,14), nk=4)', dict(x=[0, 14], nk=4)),
    ('21, ns df=5', 'logknots(21, fun="ns", df=5)', dict(x=21, fun='ns', df=5)),
    ('21, bs df=7 deg3', 'logknots(21, fun="bs", df=7, degree=3)', dict(x=21, fun='bs', df=7, degree=3)),
    ('[1,8], strata df=4', 'logknots(c(1,8), fun="strata", df=4)', dict(x=[1, 8], fun='strata', df=4)),
    ('21, ns df=4 no intercept', 'logknots(21, fun="ns", df=4, intercept=FALSE)',
     dict(x=21, fun='ns', df=4, intercept=False)),
]


@pytest.mark.parametrize('label,rexpr,kw', LOGKNOTS_CASES, ids=[c[0] for c in LOGKNOTS_CASES])
def test_logknots_matches_r(label, rexpr, kw):
    from utils import logknots
    assert_close(np.asarray(logknots(**kw), dtype=float), rget(f'as.numeric({rexpr})'), rtol=1e-13,
                 what=f'logknots {label}')


@pytest.mark.parametrize('lag', [0, 5, 21, -3, [2, 8], [3, 3], [-5, -1], [0, 30]], ids=str)
def test_mklag_matches_r(lag):
    from utils import mklag
    rl = f'c({",".join(str(v) for v in lag)})' if isinstance(lag, list) else f'{lag}'
    ref = rget(f'as.numeric(dlnm:::mklag({rl}))')
    assert_close(np.asarray(mklag(lag), dtype=float), ref, rtol=0, what='mklag')


@pytest.mark.parametrize('lag,by', [([0, 5], 1), ([0, 21], 1), ([2, 12], 1), ([0, 10], 0.5), ([0, 5], 0.25),
                                    ([3, 9], 2), ([0, 6], 3)], ids=str)
def test_seqlag_matches_r_when_the_step_divides_the_range(lag, by):
    from utils import seqlag
    ref = rget(f'as.numeric(dlnm:::seqlag(c({lag[0]},{lag[1]}), {by}))')
    assert_close(np.asarray(seqlag(lag, by), dtype=float), ref, rtol=1e-14, what='seqlag')


@pytest.mark.parametrize('lag', [[0, 3], [2, 5], 3, -2, 12], ids=str)
def test_exphist_matches_r(lag):
    """Exposure histories with fill=NA, default times: same values, same NaN pattern, same shape."""
    from utils import exphist
    x = np.random.default_rng(8).normal(15, 6, 30)
    np2r('bf_xe', x)
    rl = f'c({",".join(str(v) for v in lag)})' if isinstance(lag, list) else f'{lag}'
    ref = rget(f'unname(exphist(bf_xe, lag={rl}, fill=NA))')
    py = exphist(x, lag=lag, fill=np.nan)
    assert_close(py, ref, rtol=1e-14, what=f'exphist lag={lag}')


# ==============================================================================================================
# 11  validated end-to-end pipelines (README Quick Start, 106 US cities first stage, England & Wales second stage)
# ==============================================================================================================
def test_readme_quickstart_single_location_pipeline_matches_r():
    """README Quick Start block 1 on London, step for step: CrossBasis -> ImprovedGLMInterface.fit_dlnm_model ->
    crosspred(coef=glm.cb_coef, vcov=glm.cb_vcov, model_link='log', at=np.arange(min, max, 0.5), cen=mean) versus R
    crossbasis -> glm -> crosspred(cb, model, at, cen) (audit: allRRfit 1.7e-14). Bound 1e-9."""
    from prediction import crosspred
    fs = _first_stage('London')
    g = _fit_py_first_stage(fs)
    tm = fs.e.tmean
    pred_temps = np.arange(np.nanmin(tm), np.nanmax(tm), 0.5)
    np2r('bf_at', pred_temps)
    r(f'bf_pR <- crosspred({fs.cb_r}, {fs.m_r}, at=bf_at, cen={fs.cen!r})')
    with _quiet():
        pp = crosspred(basis=fs.cb, coef=g.cb_coef, vcov=g.cb_vcov, model_link='log', at=pred_temps, cen=fs.cen)
    _assert_fields(pp, 'bf_pR', ['predvar', 'cen', 'allfit', 'allse', 'allRRfit', 'allRRlow', 'allRRhigh',
                                 'matRRfit', 'matRRlow', 'matRRhigh'], rtol=1e-9, what='README quick start')


def test_us_city_first_stage_matches_r():
    """A US city of the 106-city analysis (temperature_mortality_analysis/run_analysis.py vs run_analysis_R.R): dates from the
    yyyymmdd column, day-of-week from the date in Python and from factor(DOW) in R (same partition of the days), seasonal
    df from the number of calendar years. Cross-basis block and crossreduce output vs R (audit: 3e-13 over 106 cities)."""
    from basis import CrossBasis
    from crossreduce import crossreduce
    from improved_glm import ImprovedGLMInterface
    from utils import logknots
    city = 'Akron'
    us = pd.read_csv(US_CSV, usecols=['cityName', 'Date', 'DOW', 'Year', 'TMean', 'Death'])
    us = us[us['cityName'] == city].copy()
    us['date'] = pd.to_datetime(us['Date'].astype(int).astype(str), format='%Y%m%d')
    us = us.sort_values('date').reset_index(drop=True)
    assert not us['TMean'].isna().any() and not us['Death'].isna().any()
    np2r('bf_us_tm', us['TMean'].to_numpy(float))
    np2r('bf_us_y', us['Death'].to_numpy(float))
    np2r('bf_us_dow', us['DOW'].to_numpy(float))
    np2r('bf_us_days', (us['date'] - pd.Timestamp('1970-01-01')).dt.days.to_numpy(float))
    r('''bf_us_d <- data.frame(Death=bf_us_y, DOW=factor(bf_us_dow), date=as.Date(bf_us_days, origin="1970-01-01"))
         bf_us_ny <- length(unique(format(bf_us_d$date, "%Y")))
         bf_us_kv <- quantile(bf_us_tm, c(.10,.75,.90), na.rm=TRUE)
         bf_us_cb <- crossbasis(bf_us_tm, lag=21, argvar=list(fun="bs", degree=2, knots=bf_us_kv),
                                arglag=list(fun="ns", knots=logknots(c(0,21), nk=3)))
         bf_us_m <- glm(Death ~ bf_us_cb + DOW + ns(date, df=8*bf_us_ny), data=bf_us_d, family=quasipoisson,
                        na.action=na.exclude)
         bf_us_ix <- grep("^bf_us_cb", names(coef(bf_us_m)))
         bf_us_red <- crossreduce(bf_us_cb, bf_us_m, cen=mean(bf_us_tm, na.rm=TRUE))''')
    kv = rget('bf_us_kv')
    cb = CrossBasis(us['TMean'].to_numpy(float), lag=21, argvar={'fun': 'bs', 'knots': kv, 'degree': 2},
                    arglag={'fun': 'ns', 'knots': logknots([0, 21], nk=3)})
    assert_close(np.asarray(cb.basis), rget('unclass(bf_us_cb)'), rtol=RTOL_BASIS, what='US cross-basis')
    with _quiet():
        g = ImprovedGLMInterface(cb)
        g.fit_dlnm_model(us['Death'].to_numpy(float), us['date'], dfseas=8, family='quasipoisson')
        red = g.crossreduce(cen=float(us['TMean'].mean()))
    assert_close(g.cb_coef, rget('unname(coef(bf_us_m)[bf_us_ix])'), rtol=RTOL_GLM, what='US cb coef')
    assert_close(g.cb_vcov, rget('unname(vcov(bf_us_m)[bf_us_ix, bf_us_ix])'), rtol=RTOL_GLM, what='US cb vcov')
    assert_close(red.coef, rget('as.numeric(bf_us_red$coefficients)'), rtol=RTOL_GLM, what='US reduced coef')
    assert_close(red.vcov, rget('unname(bf_us_red$vcov)'), rtol=RTOL_GLM, what='US reduced vcov')


@lru_cache(maxsize=None)
def _ew_first_stage_all():
    """01.firststage.R for all ten England & Wales regions, in R only (crossbasis + glm + crossreduce), and the
    meta-predictors of 02.secondstage.R: y (10,5), S (10,5,5), X = [1, mean tmean, range tmean]."""
    if not bool(r('isTRUE(suppressWarnings(requireNamespace("tsModel", quietly=TRUE)))')[0]):
        pytest.skip('R package tsModel not installed')
    r(f'''bf_ew_all <- read.csv("{EW_CSV}", row.names=1); bf_ew_all$date <- as.Date(bf_ew_all$date)
          bf_regs <- sort(as.character(unique(bf_ew_all$regnames)))
          bf_fsall <- lapply(bf_regs, function(x) {{
            d <- bf_ew_all[bf_ew_all$regnames == x, ]
            cb <- crossbasis(d$tmean, lag=21, argvar=list(fun="bs", degree=2,
                             knots=quantile(d$tmean, c(10,75,90)/100, na.rm=TRUE)), arglag=list(knots=logknots(21,3)))
            mod <- glm(death ~ cb + dow + ns(date, df=8*length(unique(year))), d, family=quasipoisson,
                       na.action="na.exclude")
            red <- crossreduce(cb, mod, cen=mean(d$tmean, na.rm=TRUE))
            list(coef=coef(red), vcov=vcov(red), avg=mean(d$tmean, na.rm=TRUE), rng=diff(range(d$tmean, na.rm=TRUE)))
          }})''')
    n = int(rget('length(bf_fsall)')[0])
    y = rget('do.call(rbind, lapply(bf_fsall, function(f) unname(f$coef)))')
    S = np.array([rget(f'unname(bf_fsall[[{i + 1}]]$vcov)') for i in range(n)])
    avg = rget('sapply(bf_fsall, function(f) f$avg)').ravel()
    rng = rget('sapply(bf_fsall, function(f) f$rng)').ravel()
    return y, S, np.column_stack([np.ones(n), avg, rng])


def test_england_wales_second_stage_agrees_with_r(_meta_r):
    """The validated second stage: 10 regions, k=5 reduced coefficients, meta-regression on the intercept, average
    temperature and temperature range (p=3, REML), then BLUPs with vcov, from R's first-stage output. Against R at
    reltol=1e-14 the fit, the BLUPs and their vcov agree to <= 1e-6 (flat REML surface, rank-deficient Psi); against
    R's default control to <= 1e-4 (R's own stopping error, so the bound must not be tightened)."""
    from meta_analysis import blup
    y, S, X = _ew_first_stage_all()
    assert y.shape == (10, 5) and S.shape == (10, 5, 5) and X.shape == (10, 3)
    m = _py_meta_fit(y, S, X)
    res = blup(m, vcov=True)
    b, bv = np.array([x['blup'] for x in res]), np.array([x['vcov'] for x in res])
    for control, tol in [(R_TIGHT, 1e-6), ('list()', 1e-4)]:
        ref = _r_meta_fit(y, S, X, 'reml', control)
        assert_close(m.coefficients, ref['coef'], rtol=tol, what=f'E&W coefficients ({control})')
        assert_close(m.vcov, ref['vcov'], rtol=tol, what=f'E&W vcov ({control})')
        assert_close(m.psi, ref['psi'], rtol=tol, what=f'E&W Psi ({control})')
        r(f'bfm_mv <- bfm_formula_fit(as.matrix(bfm_y), as.matrix(bfm_Sv), as.matrix(bfm_X), "reml", {control});'
          'bfm_bl <- blup(bfm_mv, vcov=TRUE)')
        ref_b = np.array([rget(f'as.numeric(bfm_bl[[{i + 1}]]$blup)') for i in range(10)])
        ref_v = np.array([rget(f'unname(bfm_bl[[{i + 1}]]$vcov)') for i in range(10)])
        assert_close(b, ref_b, rtol=tol, what=f'E&W BLUP ({control})')
        assert_close(bv, ref_v, rtol=tol, what=f'E&W BLUP vcov ({control})')


# ==============================================================================================================
# 12  seasonal bases (span comparisons: independent of the column order)
# ==============================================================================================================
def _span_residual(A, B):
    """Largest residual when the columns of A are regressed on the columns of B (0 if span(A) is inside span(B))."""
    coef = np.linalg.lstsq(B, A, rcond=None)[0]
    return float(np.abs(A - B @ coef).max())


@pytest.mark.parametrize('nh,period', [(1, 365.25), (3, 365.25), (2, 52.18)])
def test_harmonic_seasonal_basis_spans_tsmodel_harmonic(nh, period):
    """HarmonicSeasonalBasis and tsModel::harmonic span the same sin/cos space (R groups sin then cos, PyDLNM interleaves
    them), so each basis reproduces the other exactly."""
    from seasonality import HarmonicSeasonalBasis
    if not bool(r('isTRUE(suppressWarnings(requireNamespace("tsModel", quietly=TRUE)))')[0]):
        pytest.skip('R package tsModel not installed')
    r('suppressMessages(library(tsModel))')
    t = np.arange(0.0, 900.0)
    np2r('bf_t', t)
    ref = rget(f'unname(as.matrix(harmonic(bf_t, nfreq={nh}, period={period!r})))')
    py = np.asarray(HarmonicSeasonalBasis(n_harmonics=nh, period=period)(t), dtype=float)
    assert py.shape == ref.shape == (900, 2 * nh)
    assert _span_residual(py, ref) < 1e-10 and _span_residual(ref, py) < 1e-10


@pytest.mark.parametrize('df,period', [(4, 365.25), (6, 365.25), (5, 52.18)])
def test_noncyclic_seasonal_spline_matches_r_ns_of_position_in_period(df, period):
    """SeasonalSplineBasis(cyclic=False) is splines::ns(t %% period, df) (audit: 4.9e-15)."""
    from seasonality import SeasonalSplineBasis
    t = np.arange(0.0, 1100.0)
    np2r('bf_t', t)
    ref = rget(f'unname(unclass(ns(bf_t %% {period!r}, df={df})))')
    with _quiet():
        py = SeasonalSplineBasis(df=df, period=period, cyclic=False)(t)
    assert_close(np.asarray(py, dtype=float), ref, rtol=1e-12, what='seasonal ns')
