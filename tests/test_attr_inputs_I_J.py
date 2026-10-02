"""attrdl / getlink / cross-basis coefficient selection versus R (themes I1, I2, J of the 2026-09 audit).

Reference: R attrdl (2015_gasparrini_Lancet_Rcodedata-master/attrdl.R, sourced into the embedded R session) and R
dlnm (crosspred, getlink) on identical inputs. All R numbers are computed at test run time.

Themes and findings (tests marked ``known_defect`` assert the R-faithful behaviour and fail until the fix lands):

  J   attr-point-1  attrdl(coef=, vcov=) with no model has no link information: PyDLNM uses the log-RR as if it were the
                    RR, so the Lancet-style reduced/BLUP call (03.attr.R) and every full-coefficient coef=/vcov= call
                    return garbage. R always applies exp() to coef/vcov input.
  I2  attr-point-5  getlink() classifies every statsmodels GLM as 'identity' (family.link objects have no .name and the
                    class-name table maps 'GLM' -> identity), and attrdl never checks the link. R: poisson/quasipoisson
                    -> log, binomial -> logit, and attrdl stops unless the link is log or logit.
  I1  attr-point-6  CrossPred(model=...) takes the first ncol(basis) coefficients POSITIONALLY. R picks the cross-basis
                    coefficients by name (crosspred.R / attrdl.R), so intercept-first designs (statsmodels add_constant,
                    patsy, R formulas) silently give wrong results; R stops when the block cannot be identified or
                    has the wrong length.

Comparison design. attrdl has other, separately tracked differences from R (forward moving average of cases and the
total rescaling, dir='back', the AF denominator with range=, NaN alignment). To keep every test here about the input
handling only, and valid before and after those other fixes, the comparisons are restricted to quantities on which the
algorithms agree by construction:
  * per-observation AF (type='af', tot=False), which does not involve cases;
  * per-observation AN with constant cases, on the rows that R's forward moving average keeps (the first n - lag rows);
  * total AN/AF at a constant exposure, where the AF is the same number on every row so weights and row exclusions
    cancel, and total AN = AF * sum(cases).
The neighbouring behaviour that is already faithful (log-linked model objects, explicit coef with model_link='log',
cb-first designs, getlink for interfaces/discrete/Gaussian models, input validation) is covered by plain tests.

Environment notes. The tests need R packages dlnm and tsModel and statsmodels (skipped otherwise). PyDLNM modules
overwrite os.environ['R_HOME'] (audit finding Q2), and R then segfaults on its next lazy LAPACK load if it was started
from a different R; _restore_r_home() undoes that around every PyDLNM call so this module does not depend on Q2.
"""
import contextlib
import io
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rhelpers import assert_close, chicago, known_defect, np2r, r, rget

REPO = Path(__file__).resolve().parents[1]
RTOL = 1e-9          # deterministic identical arithmetic agrees to ~1e-15; well inside the 1e-8 bound

# =====================================================================================================================
# helpers
# =====================================================================================================================


def _restore_r_home():
    """PyDLNM modules overwrite os.environ['R_HOME'] when they are imported or used (audit finding Q2). R loads its
    LAPACK library lazily from R_HOME, so a session started from another R segfaults on its next glm/vcov/solve.
    The running R's real home is in .Library (fixed at start-up); put it back. A no-op once Q2 is fixed."""
    home = str(r('dirname(.Library)')[0])
    if os.environ.get('R_HOME') != home:
        os.environ['R_HOME'] = home


@contextlib.contextmanager
def _quiet():
    """Run PyDLNM code without its progress prints or warnings (they are noise here), then undo the R_HOME overwrite."""
    try:
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter('ignore')
            yield
    finally:
        _restore_r_home()


@pytest.fixture(autouse=True)
def _guard_r_home():
    _restore_r_home()
    yield
    _restore_r_home()


def _load_r_attrdl():
    """Source R's Lancet attrdl.R once into a private environment (needs tsModel); called as attrdl_IJ$attrdl."""
    _restore_r_home()
    if not bool(r('exists("attrdl_IJ", envir=globalenv(), inherits=FALSE)')[0]):
        if not bool(r('nzchar(system.file(package="tsModel"))')[0]):
            pytest.skip('R package tsModel not installed')
        r('suppressMessages(library(tsModel))')
        path = REPO / '2015_gasparrini_Lancet_Rcodedata-master' / 'attrdl.R'
        r(f'attrdl_IJ <- new.env(); sys.source("{path}", envir=attrdl_IJ)')


def r_error(code):
    """Message of the R error raised by `code`, or None if it runs without error."""
    _restore_r_home()
    res = r(f'tryCatch({{ {code}; NULL }}, error=function(e) conditionMessage(e))')
    return str(res[0]) if len(res) else None


def assert_same(py, ref, what, rtol=RTOL):
    """assert_close plus a finiteness check (assert_close ignores entries that are non-finite on either side)."""
    py = np.atleast_1d(np.asarray(py, dtype=float))
    ref = np.atleast_1d(np.asarray(ref, dtype=float))
    assert py.shape == ref.shape, f'{what}: shape Python {py.shape} vs R {ref.shape}'
    assert np.array_equal(np.isfinite(py), np.isfinite(ref)), (
        f'{what}: finite pattern differs (Python has {int((~np.isfinite(py)).sum())} non-finite entries, '
        f'R {int((~np.isfinite(ref)).sum())}); e.g. Python {py.ravel()[:3]} vs R {ref.ravel()[:3]}')
    assert_close(py, ref, rtol=rtol, what=what)


def r_attrdl(cb_name, x, cases, coef, vcov, typ, tot, cen, rng=None):
    """R attrdl (dir='forw') on coef/vcov; returns a float array."""
    _load_r_attrdl()
    np2r('x_IJ', x)
    np2r('cases_IJ', cases)
    np2r('coef_IJ', coef)
    np2r('vcov_IJ', np.atleast_2d(vcov))
    np2r('cen_IJ', [cen])
    rng_s = 'NULL'
    if rng is not None:
        np2r('rng_IJ', rng)          # exact doubles (no decimal-literal round trip)
        rng_s = 'rng_IJ'
    r(f'res_IJ <- attrdl_IJ$attrdl(x_IJ, {cb_name}, cases_IJ, coef=coef_IJ, vcov=vcov_IJ, type="{typ}", dir="forw", '
      f'tot={"TRUE" if tot else "FALSE"}, cen=cen_IJ, range={rng_s})')
    return np.atleast_1d(rget('as.numeric(res_IJ)'))


def py_attrdl(cbP, x, cases, typ, tot, cen, rng=None, **kw):
    """PyDLNM attrdl (dir='forw'); returns the requested per-observation array or the total as a float array."""
    from attribution import attrdl
    with _quiet():
        res = attrdl(np.array(x), cbP, np.array(cases), type=typ, dir='forw', tot=tot, cen=cen, range=rng, **kw)
    return np.atleast_1d(np.asarray(res[typ + '_total'] if tot else res[typ], dtype=float))


# comparison modes: name -> (type, tot)
_MODES = {'af_obs': ('af', False), 'an_obs': ('an', False), 'an_tot': ('an', True), 'af_tot': ('af', True)}
_CASES_CONST = 25.0


def attr_pair(ds, mode, cen, ref_coef, ref_vcov, rng=None, x0=None, **py_kw):
    """(Python, R) results for one comparison mode. R always gets coef/vcov (name-selected block); Python gets what
    py_kw says (model=..., or coef=/vcov=)."""
    typ, tot = _MODES[mode]
    n = len(ds.temp)
    if mode == 'af_obs':
        x, cases = ds.temp, ds.death
    elif mode == 'an_obs':
        x, cases = ds.temp, np.full(n, _CASES_CONST)
    else:                                                    # totals at a constant exposure
        x = np.full(n, ds.temp_q[x0] if isinstance(x0, str) else x0)
        cases = np.random.default_rng(11).poisson(50, n).astype(float)
    ref = r_attrdl(ds.cb_name, x, cases, ref_coef, ref_vcov, typ, tot, cen, rng)
    py = py_attrdl(ds.cbP, x, cases, typ, tot, cen, rng, **py_kw)
    if mode == 'an_obs':                                     # R's forward moving average leaves the last `lag` rows NA
        head = n - int(ds.cbP.lag[1])
        py, ref = py[:head], ref[:head]
    return py, ref


class _Dataset:
    """R and PyDLNM objects built from identical inputs (R glm on the full series gives realistic coefficients)."""


def _build_full(tag, r_data_code, temp_col, seed):
    """Chicago / England&Wales series with the Lancet cross-basis (bs deg 2, knots 10/75/90 x ns logknots(21,3))."""
    _restore_r_home()
    from basis import CrossBasis
    from utils import logknots
    r(r_data_code)
    r(f'''
    temp_{tag} <- as.numeric(d_{tag}${temp_col}); death_{tag} <- as.numeric(d_{tag}$death)
    kv_{tag} <- quantile(temp_{tag}, c(.10,.75,.90), na.rm=TRUE)
    cb_{tag} <- crossbasis(temp_{tag}, lag=21, argvar=list(fun="bs", degree=2, knots=kv_{tag}),
                           arglag=list(fun="ns", knots=logknots(21,3)))
    m_{tag} <- glm(death_{tag} ~ cb_{tag} + dow + ns(time, df=28), data=d_{tag}, family=quasipoisson)
    ii_{tag} <- grep("^cb_{tag}", names(coef(m_{tag})))
    coef_{tag} <- unname(coef(m_{tag})[ii_{tag}]); vcov_{tag} <- unname(vcov(m_{tag})[ii_{tag}, ii_{tag}])
    at_{tag} <- quantile(temp_{tag}, 1:99/100, na.rm=TRUE)
    red_{tag} <- crossreduce(cb_{tag}, m_{tag}, type="overall", at=at_{tag}, cen=median(temp_{tag}))
    mmt_{tag} <- red_{tag}$predvar[which.min(red_{tag}$fit)]
    ''')
    ds = _Dataset()
    ds.tag, ds.cb_name = tag, f'cb_{tag}'
    ds.temp, ds.death = rget(f'temp_{tag}'), rget(f'death_{tag}')
    kv = rget(f'kv_{tag}')
    with _quiet():
        ds.cbP = CrossBasis(ds.temp, lag=21, argvar={'fun': 'bs', 'degree': 2, 'knots': kv},
                            arglag={'fun': 'ns', 'knots': logknots([0, 21], nk=3)})
    assert_close(np.asarray(ds.cbP.basis), rget(f'unclass(cb_{tag})'), rtol=1e-12, what='cross-basis precondition')
    ds.coef_full, ds.vcov_full = rget(f'coef_{tag}'), rget(f'vcov_{tag}')
    assert ds.coef_full.shape == (25,)
    # reduced (variable-basis) coefficients: R's overall reduction plus a deterministic BLUP-like perturbation
    rr = np.random.default_rng(seed)
    coef_red = rget(f'as.numeric(red_{tag}$coefficients)')
    ds.coef_red = coef_red + rr.normal(0.0, 0.03, coef_red.shape)
    ds.vcov_red = rget(f'unname(red_{tag}$vcov)')
    ds.cen = float(rget(f'as.numeric(mmt_{tag})')[0])
    ds.temp_q = {'cold': float(np.quantile(ds.temp, 0.05)), 'hot': float(np.quantile(ds.temp, 0.95))}
    ds.ranges = {'glob': None, 'cold': np.array([-100.0, ds.cen]), 'heat': np.array([ds.cen, 100.0])}
    return ds


@pytest.fixture(scope='module')
def chi():
    """Chicago 1987-2000, lag 21: 25-column cross-basis, R quasi-Poisson fit."""
    return _build_full('IJchi', 'data(chicagoNMMAPS, package="dlnm"); d_IJchi <- chicagoNMMAPS', 'temp', seed=1)


@pytest.fixture(scope='module')
def ew_london():
    """England & Wales, London (the 03.attr.R data): same cross-basis as the Lancet first stage."""
    csv = REPO / '2015_gasparrini_Lancet_Rcodedata-master' / 'regEngWales.csv'
    code = (f'd_IJew <- read.csv("{csv}", row.names=1); d_IJew <- d_IJew[d_IJew$regnames=="London",]; '
            'd_IJew <- d_IJew[order(as.Date(d_IJew$date)),]; d_IJew$dow <- factor(d_IJew$dow)')
    return _build_full('IJew', code, 'tmean', seed=2)


# ---- small Chicago subset for the statsmodels (I1/I2) tests ---------------------------------------------------------

_SUB_N, _SUB_LAG = 1500, 10


@pytest.fixture(scope='module')
def sub():
    """First 1500 Chicago days, lag 10, bs(deg 2, knots 10/75/90) x ns(logknots(10,2)): 20 cross-basis columns."""
    _restore_r_home()
    from basis import CrossBasis
    from utils import logknots
    d = chicago()
    ds = _Dataset()
    ds.temp, ds.death = d['temp'][:_SUB_N].copy(), d['death'][:_SUB_N].copy()
    kv = np.quantile(ds.temp, [.1, .75, .9])
    np2r('temp_IJsub', ds.temp)
    np2r('death_IJsub', ds.death)
    np2r('kv_IJsub', kv)
    r(f'cb_IJsub <- crossbasis(temp_IJsub, lag={_SUB_LAG}, argvar=list(fun="bs", degree=2, knots=kv_IJsub), '
      f'arglag=list(fun="ns", knots=logknots({_SUB_LAG},2)))')
    ds.cb_name = 'cb_IJsub'
    with _quiet():
        ds.cbP = CrossBasis(ds.temp, lag=_SUB_LAG, argvar={'fun': 'bs', 'degree': 2, 'knots': kv},
                            arglag={'fun': 'ns', 'knots': logknots([0, _SUB_LAG], nk=2)})
    assert_close(np.asarray(ds.cbP.basis), rget('unclass(cb_IJsub)'), rtol=1e-12, what='cross-basis precondition')
    ds.rows = ~np.isnan(np.asarray(ds.cbP.basis)).any(axis=1)
    ds.k = ds.cbP.basis.shape[1]
    tt = np.arange(_SUB_N)
    ds.seas = np.column_stack([np.sin(2 * np.pi * tt / 365.25), np.cos(2 * np.pi * tt / 365.25),
                               np.sin(4 * np.pi * tt / 365.25), np.cos(4 * np.pi * tt / 365.25)])
    ds.temp_q = {'cold': float(np.quantile(ds.temp, 0.05)), 'hot': float(np.quantile(ds.temp, 0.95))}
    ds.grid = np.arange(-5.0, 31.0, 1.0)
    return ds


def _design(ds, layout, named=True):
    """Design matrix pieces in the requested order. named=True gives R-style column names ('cb' + 'v1.l1', ...)."""
    cbcols = ['cb' + c for c in ds.cbP.colnames]
    parts = {
        'const': (['(Intercept)'], np.ones((int(ds.rows.sum()), 1))),
        'cb': (cbcols, np.asarray(ds.cbP.basis)[ds.rows]),
        'seas': (['seas1', 'seas2', 'seas3', 'seas4'], ds.seas[ds.rows]),
        'seas_a': (['seas1', 'seas2'], ds.seas[ds.rows][:, :2]),
        'seas_b': (['seas3', 'seas4'], ds.seas[ds.rows][:, 2:]),
    }
    order = {'cb_first': ['cb', 'const', 'seas'],
             'const_first': ['const', 'cb', 'seas'],        # statsmodels add_constant / patsy / R formula order
             'seas_first': ['seas', 'const', 'cb'],
             'cb_middle': ['const', 'seas_a', 'cb', 'seas_b'],
             'cb_only': ['cb']}[layout]
    names = [n for p in order for n in parts[p][0]]
    X = np.column_stack([parts[p][1] for p in order])
    idx = np.array([i for i, n in enumerate(names) if n in cbcols])
    return (pd.DataFrame(X, columns=names) if named else X), idx


def _sm():
    return pytest.importorskip('statsmodels.api')


def _fit_sm(ds, layout, named=True, family='poisson'):
    """statsmodels GLM fit; returns (fit, cb coef block, cb vcov block) with the block taken by construction."""
    sm = _sm()
    X, idx = _design(ds, layout, named)
    y = ds.death[ds.rows]
    fam = {'poisson': sm.families.Poisson(), 'binomial': sm.families.Binomial(),
           'gaussian': sm.families.Gaussian()}[family]
    if family == 'binomial':
        y = (y > np.median(y)).astype(float)
    with _quiet():
        fit = sm.GLM(y, X, family=fam).fit()
    coef = np.asarray(fit.params)[idx]
    vcov = np.asarray(fit.cov_params())[np.ix_(idx, idx)]
    return fit, coef, vcov


def _r_crosspred(ds, coef, vcov, cen, grid):
    _restore_r_home()
    np2r('coefp_IJ', coef)
    np2r('vcovp_IJ', vcov)
    np2r('gridp_IJ', grid)
    np2r('cenp_IJ', [cen])
    r(f'cp_IJ <- crosspred({ds.cb_name}, coef=coefp_IJ, vcov=vcovp_IJ, model.link="log", at=gridp_IJ, cen=cenp_IJ)')
    return {k: rget(f'cp_IJ${k}') for k in ('allfit', 'allse', 'matfit', 'matse', 'allRRfit')}


def _py_crosspred(ds, cen, grid, **kw):
    from prediction import CrossPred
    with _quiet():
        return CrossPred(ds.cbP, at=grid, cen=cen, **kw)


def _stub_log_model(coef, vcov):
    """Duck-typed stand-in for PyDLNM's own GLM interface (getlink -> 'log' by class name, exposes only the
    cross-basis block), exactly as the repo's rpy2 interfaces do."""
    class ImprovedGLMInterface:
        def __init__(self, cb_coef, cb_vcov):
            self.cb_coef, self.cb_vcov = np.asarray(cb_coef), np.asarray(cb_vcov)
    return ImprovedGLMInterface(coef, vcov)


# =====================================================================================================================
# Theme J -- attr-point-1: attrdl(coef=, vcov=) without a model (the Lancet / BLUP call)
# =====================================================================================================================

_RANGES = ['glob', 'cold', 'heat']


@pytest.mark.parametrize('kind,mode,rng_name',
                         [('full', 'af_obs', r_) for r_ in _RANGES] + [('full', 'an_obs', 'glob')] +
                         [('reduced', m_, r_) for m_ in ('af_obs', 'an_obs') for r_ in _RANGES])
def test_coef_vcov_only_per_observation_matches_R(chi, kind, mode, rng_name):
    """attrdl(x, cb, cases, coef=, vcov=, cen=, range=) per observation: full 25-coefficient and reduced (BLUP-style,
    5 coefficients) input, glob / cold (-100, cen] / heat [cen, 100) ranges, R: af = 1 - exp(-X coef)."""
    coef, vcov = (chi.coef_full, chi.vcov_full) if kind == 'full' else (chi.coef_red, chi.vcov_red)
    rng = chi.ranges[rng_name]
    py, ref = attr_pair(chi, mode, chi.cen, coef, vcov, rng=rng, coef=coef, vcov=vcov)
    assert_same(py, ref, f'attrdl {kind} coef {mode} {rng_name}')


@pytest.mark.parametrize('kind,mode,x0', [('full', 'an_tot', 'hot'), ('full', 'af_tot', 'cold'),
                                          ('reduced', 'an_tot', 'cold'), ('reduced', 'an_tot', 'hot'),
                                          ('reduced', 'af_tot', 'hot')])
def test_coef_vcov_only_totals_at_constant_exposure_match_R(chi, kind, mode, x0):
    """Total AN / AF for a constant exposure (5th and 95th percentile of temperature): the AF is one number for every
    row, so the total is independent of how cases are weighted and R's AN = AF * sum(cases)."""
    coef, vcov = (chi.coef_full, chi.vcov_full) if kind == 'full' else (chi.coef_red, chi.vcov_red)
    py, ref = attr_pair(chi, mode, chi.cen, coef, vcov, x0=x0, coef=coef, vcov=vcov)
    assert_same(py, ref, f'attrdl {kind} coef {mode} x={x0}')


@pytest.mark.parametrize('rng_name', _RANGES)
@pytest.mark.parametrize('mode', ['af_obs', 'an_obs'])
def test_lancet_03_attr_call_england_wales_london_matches_R(ew_london, mode, rng_name):
    """The 03.attr.R call pattern on the paper's own data: London, reduced coef/vcov, cen = MMT of the reduced curve,
    glob / cold / heat."""
    ds = ew_london
    py, ref = attr_pair(ds, mode, ds.cen, ds.coef_red, ds.vcov_red, rng=ds.ranges[rng_name],
                        coef=ds.coef_red, vcov=ds.vcov_red)
    assert_same(py, ref, f'Lancet-style attrdl London {mode} {rng_name}')


@pytest.mark.parametrize('x0', ['cold', 'hot'])
def test_lancet_03_attr_call_total_an_matches_R(ew_london, x0):
    ds = ew_london
    py, ref = attr_pair(ds, 'an_tot', ds.cen, ds.coef_red, ds.vcov_red, x0=x0, coef=ds.coef_red, vcov=ds.vcov_red)
    assert_same(py, ref, f'Lancet-style total AN London x={x0}')


# ---- neighbouring behaviour that is already faithful (plain tests) ---------------------------------------------------

@pytest.mark.parametrize('mode,rng_name,x0', [('af_obs', 'glob', None), ('af_obs', 'heat', None),
                                              ('an_obs', 'cold', None), ('an_tot', None, 'hot'),
                                              ('af_tot', None, 'cold')])
def test_log_linked_model_object_matches_R(chi, mode, rng_name, x0):
    """attrdl(model=<PyDLNM GLM interface, link 'log'>) already agrees with R attrdl on the same full coefficients
    (per-observation AF/AN with ranges, totals at constant exposure)."""
    model = _stub_log_model(chi.coef_full, chi.vcov_full)
    rng = chi.ranges[rng_name] if rng_name else None
    py, ref = attr_pair(chi, mode, chi.cen, chi.coef_full, chi.vcov_full, rng=rng, x0=x0, model=model)
    assert_same(py, ref, f'attrdl(model=log stub) {mode} {rng_name or x0}')


def test_crosspred_explicit_coef_with_log_link_matches_R(chi):
    """CrossPred(coef=, vcov=, model_link='log') vs R crosspred(coef=, vcov=, model.link='log'): allfit, allse and
    allRRfit on the observed exposures (this is the machinery a link-aware attrdl must use)."""
    grid = np.sort(np.unique(chi.temp))[::25]
    ref = _r_crosspred(chi, chi.coef_full, chi.vcov_full, chi.cen, grid)
    pred = _py_crosspred(chi, chi.cen, grid, coef=chi.coef_full, vcov=chi.vcov_full, model_link='log')
    assert_same(pred.allfit, ref['allfit'], 'allfit')
    assert_same(pred.allse, ref['allse'], 'allse')
    assert_same(pred.allRRfit, ref['allRRfit'], 'allRRfit')


def test_crosspred_reduced_coef_with_log_link_matches_R(chi):
    """Reduced (BLUP-style) coefficients through CrossPred(cb, coef=, vcov=, model_link='log') vs R crosspred on the
    matching one-dimensional basis (the validated England & Wales BLUP route): allfit, allse and allRRfit."""
    grid = np.linspace(chi.temp.min(), chi.temp.max(), 40)
    np2r('coefp_IJ', chi.coef_red)
    np2r('vcovp_IJ', chi.vcov_red)
    np2r('gridp_IJ', grid)
    np2r('cenp_IJ', [chi.cen])
    r(f'ob_IJ <- onebasis(temp_IJchi, fun="bs", degree=2, knots=kv_IJchi); '
      'cp_IJ <- crosspred(ob_IJ, coef=coefp_IJ, vcov=vcovp_IJ, model.link="log", at=gridp_IJ, cen=cenp_IJ)')
    ref = {k: rget(f'cp_IJ${k}') for k in ('allfit', 'allse', 'allRRfit')}
    pred = _py_crosspred(chi, chi.cen, grid, coef=chi.coef_red, vcov=chi.vcov_red, model_link='log')
    for key in ref:
        assert_same(getattr(pred, key), ref[key], f'reduced crosspred {key}')


def test_attrdl_input_validation_like_R(sub):
    """R stops when x and cases have different lengths ("'x' and 'cases' not consistent") and when neither model nor
    coef/vcov is given ("arguments 'basis' do not match 'model' or 'coef'-'vcov'"); PyDLNM raises ValueError in both."""
    from attribution import attrdl
    _load_r_attrdl()
    np2r('x_IJ', sub.temp)
    np2r('cases_IJ', sub.death[:-3])
    np2r('coef_IJ', np.zeros(sub.k))
    np2r('vcov_IJ', np.eye(sub.k))
    np2r('cen_IJ', [15.0])
    msg = r_error(f'attrdl_IJ$attrdl(x_IJ, {sub.cb_name}, cases_IJ, coef=coef_IJ, vcov=vcov_IJ, cen=cen_IJ, '
                  'type="af", dir="forw")')
    assert msg is not None and "not consistent" in msg, msg
    with _quiet(), pytest.raises(ValueError):
        attrdl(sub.temp, sub.cbP, sub.death[:-3], model=_stub_log_model(np.zeros(sub.k), np.eye(sub.k)),
               type='af', dir='forw', cen=15.0)
    np2r('cases_IJ', sub.death)
    msg = r_error(f'attrdl_IJ$attrdl(x_IJ, {sub.cb_name}, cases_IJ, cen=cen_IJ, type="af", dir="forw")')
    assert msg is not None and "do not match" in msg, msg
    with _quiet(), pytest.raises(ValueError):
        attrdl(sub.temp, sub.cbP, sub.death, type='af', dir='forw', cen=15.0)


def test_attrdl_does_not_modify_its_inputs(sub):
    """x, cases and the basis attributes are left untouched (audit-verified faithful)."""
    import copy
    x, cases = sub.temp.copy(), sub.death.copy()
    argvar, arglag = copy.deepcopy(sub.cbP.argvar), copy.deepcopy(sub.cbP.arglag)
    py_attrdl(sub.cbP, x, cases, 'an', True, 15.0, model=_stub_log_model(np.full(sub.k, 0.01), 1e-4 * np.eye(sub.k)))
    assert np.array_equal(x, sub.temp) and np.array_equal(cases, sub.death)
    for a, b in ((argvar, sub.cbP.argvar), (arglag, sub.cbP.arglag)):
        assert a.keys() == b.keys()
        for key in a:
            assert np.array_equal(np.asarray(a[key]), np.asarray(b[key]))


# =====================================================================================================================
# Theme I2 -- attr-point-5: link detection for statsmodels GLMs, and R's log/logit requirement
# =====================================================================================================================

def _r_getlink(family):
    """dlnm:::getlink for an R glm of the given family (R reads model$family$link)."""
    _restore_r_home()
    r(f'fake_IJ <- structure(list(family={family}()), class=c("glm","lm"))')
    return str(r('dlnm:::getlink(fake_IJ, class(fake_IJ))')[0])


def _tiny_glm_data(family, n=80):
    rr = np.random.default_rng(5)
    x = rr.normal(size=n)
    eta = 0.3 + 0.4 * x
    y = {'poisson': lambda: rr.poisson(np.exp(eta)).astype(float),
         'binomial': lambda: (rr.uniform(size=n) < 1 / (1 + np.exp(-eta))).astype(float),
         'gaussian': lambda: eta + rr.normal(size=n)}[family]()
    return y, np.column_stack([np.ones(n), x])


@pytest.mark.parametrize('kind', ['poisson', 'quasipoisson', 'binomial'])
def test_getlink_statsmodels_glm_matches_R(kind):
    """R getlink: poisson/quasipoisson -> 'log', binomial -> 'logit' (quasipoisson = Poisson family with scale='X2' in
    statsmodels). PyDLNM reports 'identity' for every statsmodels GLM."""
    sm = _sm()
    from model_utils import getlink
    y, X = _tiny_glm_data('binomial' if kind == 'binomial' else 'poisson')
    family = sm.families.Binomial() if kind == 'binomial' else sm.families.Poisson()
    with _quiet():
        fit = sm.GLM(y, X, family=family).fit(scale='X2' if kind == 'quasipoisson' else None)
    assert getlink(fit) == _r_getlink(kind)


def test_getlink_statsmodels_discrete_logit_matches_R():
    sm = _sm()
    from model_utils import getlink
    y, X = _tiny_glm_data('binomial')
    with _quiet():
        fit = sm.Logit(y, X).fit(disp=0)
    assert getlink(fit) == _r_getlink('binomial')


def test_getlink_neighbours_that_are_already_faithful():
    """Gaussian GLM -> identity and discrete Poisson -> log agree with R's getlink; an explicit model_link wins (R's
    model.link argument); PyDLNM's own rpy2 GLM interfaces (quasi-Poisson fits) -> log as R's quasipoisson."""
    sm = _sm()
    from model_utils import getlink
    y, X = _tiny_glm_data('gaussian')
    with _quiet():
        fit_g = sm.GLM(y, X, family=sm.families.Gaussian()).fit()
    assert getlink(fit_g) == _r_getlink('gaussian') == 'identity'
    y, X = _tiny_glm_data('poisson')
    with _quiet():
        fit_p = sm.Poisson(y, X).fit(disp=0)
    assert getlink(fit_p) == _r_getlink('poisson') == 'log'
    assert getlink(fit_g, model_link='log') == 'log'
    for cls_name in ('DLNMGLMInterface', 'Rpy2GLMInterface', 'ImprovedGLMInterface'):
        assert getlink(type(cls_name, (), {})()) == _r_getlink('quasipoisson')


@pytest.mark.parametrize('cen', [15.0, 15.05])
@pytest.mark.parametrize('mode,x0', [('af_obs', None), ('an_obs', None), ('an_tot', 'hot'), ('af_tot', 'cold')])
@pytest.mark.parametrize('family', ['poisson', 'binomial'])
def test_attrdl_statsmodels_glm_matches_R(sub, family, mode, x0, cen):
    """attrdl(model=<statsmodels GLM, cross-basis columns first>) vs R attrdl on the same coefficients. R accepts log
    (Poisson) and logit (Binomial) links and computes RR = exp(eta). cen=15 coincides with observed temperatures
    (log-RR = 0 there), cen=15.05 does not."""
    fit, coef, vcov = _fit_sm(sub, 'cb_first', named=True, family=family)
    py, ref = attr_pair(sub, mode, cen, coef, vcov, x0=x0, model=fit)
    assert_same(py, ref, f'attrdl(statsmodels {family}) {mode} cen={cen}')


def test_attrdl_rejects_non_log_logit_model_like_R(sub):
    """R attrdl(model=<gaussian glm>) stops with "'model' must have a log or logit link function"; PyDLNM must raise
    too instead of returning numbers."""
    _load_r_attrdl()
    r(f'gauss_IJ <- glm(death_IJsub ~ {sub.cb_name}, family=gaussian)')
    msg = r_error(f'attrdl_IJ$attrdl(temp_IJsub, {sub.cb_name}, death_IJsub, model=gauss_IJ, type="af", dir="forw", '
                  'cen=15)')
    assert msg is not None and 'log or logit' in msg, msg
    from attribution import attrdl
    fit, _, _ = _fit_sm(sub, 'cb_first', named=True, family='gaussian')
    with _quiet(), pytest.raises(ValueError):
        attrdl(sub.temp, sub.cbP, sub.death, model=fit, type='af', dir='forw', cen=15.0)


# =====================================================================================================================
# Theme I1 -- attr-point-6: cross-basis coefficients are selected by name in R, by position in PyDLNM
# =====================================================================================================================

_INTERCEPT_FIRST = ['const_first', 'seas_first', 'cb_middle']     # cross-basis block does not start the design


@pytest.mark.parametrize('layout', _INTERCEPT_FIRST)
def test_crosspred_model_selects_crossbasis_block_by_name(sub, layout):
    """CrossPred(model=<statsmodels fit with named columns>) vs R crosspred on the name-selected cross-basis
    coefficients: allfit, allse, matfit, matse over -5..30 C centred at 15 C."""
    fit, coef, vcov = _fit_sm(sub, layout, named=True)
    ref = _r_crosspred(sub, coef, vcov, 15.0, sub.grid)
    pred = _py_crosspred(sub, 15.0, sub.grid, model=fit)
    for key, val in (('allfit', pred.allfit), ('allse', pred.allse), ('matfit', pred.matfit), ('matse', pred.matse)):
        assert_same(val, ref[key], f'crosspred {key} layout={layout}')


@pytest.mark.parametrize('mode,x0', [('af_obs', None), ('an_tot', 'hot'), ('af_tot', 'cold')])
@pytest.mark.parametrize('layout', _INTERCEPT_FIRST)
def test_attrdl_model_with_intercept_first_design_matches_R(sub, layout, mode, x0):
    """attrdl(model=<statsmodels Poisson fit, design [const, cb, seasonal] and other orders>) vs R attrdl on the
    name-selected cross-basis block. The link is forced to 'log' (fit.link) so that only the coefficient selection is
    exercised (link detection is attr-point-5)."""
    fit, coef, vcov = _fit_sm(sub, layout, named=True)
    fit.link = 'log'
    py, ref = attr_pair(sub, mode, 15.0, coef, vcov, x0=x0, model=fit)
    assert_same(py, ref, f'attrdl(model, {layout}) {mode}')


@pytest.mark.parametrize('layout', ['const_first', 'seas_first'])
def test_unnamed_intercept_first_design_never_silently_wrong(sub, layout):
    """A numpy design carries no coefficient names, so the cross-basis block cannot be identified. R stops ('coef/vcov
    not consistent with basis matrix' when no name matches); PyDLNM may either raise or use an explicit index, but must
    not return numbers computed from the wrong coefficients."""
    fit, coef, vcov = _fit_sm(sub, layout, named=False)
    ref = _r_crosspred(sub, coef, vcov, 15.0, sub.grid)
    try:
        pred = _py_crosspred(sub, 15.0, sub.grid, model=fit)
    except ValueError:
        return
    assert_same(pred.allfit, ref['allfit'], f'crosspred allfit, unnamed {layout}')


@pytest.mark.parametrize('entry', ['crosspred', 'attrdl'])
def test_explicit_coef_longer_than_basis_is_rejected_like_R(sub, entry):
    """coef=/vcov= with 3 extra trailing coefficients: R crosspred stops ('coef/vcov not consistent with basis matrix')
    and R attrdl stops ("arguments 'basis' do not match 'model' or 'coef'-'vcov'"); PyDLNM truncates to the first
    ncol(basis) entries and returns numbers."""
    _load_r_attrdl()
    rr = np.random.default_rng(3)
    coef = np.r_[rr.normal(0, 0.02, sub.k), [0.1, -0.2, 0.3]]
    vcov = 1e-4 * np.eye(sub.k + 3)
    np2r('coef_IJ', coef)
    np2r('vcov_IJ', vcov)
    np2r('x_IJ', sub.temp)
    np2r('cases_IJ', sub.death)
    np2r('cen_IJ', [15.0])
    np2r('grid_IJ', sub.grid)
    r_call, phrase = {
        'crosspred': (f'crosspred({sub.cb_name}, coef=coef_IJ, vcov=vcov_IJ, model.link="log", at=grid_IJ, cen=15)',
                      'not consistent with basis matrix'),
        'attrdl': (f'attrdl_IJ$attrdl(x_IJ, {sub.cb_name}, cases_IJ, coef=coef_IJ, vcov=vcov_IJ, type="af", '
                   'dir="forw", cen=cen_IJ)', 'do not match'),
    }[entry]
    msg = r_error(r_call)
    assert msg is not None and phrase in msg, f'R {entry} must stop with "{phrase}", got: {msg}'
    with _quiet(), pytest.raises(ValueError):
        if entry == 'crosspred':
            _py_crosspred(sub, 15.0, sub.grid, coef=coef, vcov=vcov, model_link='log')
        else:
            from attribution import attrdl
            attrdl(sub.temp, sub.cbP, sub.death, coef=coef, vcov=vcov, type='af', dir='forw', cen=15.0)


# ---- neighbouring behaviour that is already faithful (plain tests) ---------------------------------------------------

@pytest.mark.parametrize('layout,named', [('cb_first', True), ('cb_only', False), ('cb_only', True)])
def test_crossbasis_first_or_alone_matches_R(sub, layout, named):
    """Designs whose cross-basis block comes first (named), or that consist of the block alone (named or not), are
    handled correctly; they must keep working once coefficients are selected by name."""
    fit, coef, vcov = _fit_sm(sub, layout, named=named)
    ref = _r_crosspred(sub, coef, vcov, 15.0, sub.grid)
    pred = _py_crosspred(sub, 15.0, sub.grid, model=fit)
    for key, val in (('allfit', pred.allfit), ('allse', pred.allse), ('matfit', pred.matfit)):
        assert_same(val, ref[key], f'crosspred {key} {layout} named={named}')


def test_unnamed_crossbasis_first_design_matches_R_or_raises(sub):
    """Unnamed cb-first design: the positional guess happens to be right today; after by-name selection the block cannot
    be identified without names, so raising is also acceptable. Either way never a wrong number."""
    fit, coef, vcov = _fit_sm(sub, 'cb_first', named=False)
    ref = _r_crosspred(sub, coef, vcov, 15.0, sub.grid)
    try:
        pred = _py_crosspred(sub, 15.0, sub.grid, model=fit)
    except ValueError:
        return
    assert_same(pred.allfit, ref['allfit'], 'crosspred allfit, unnamed cb-first')


def test_attrdl_model_cb_first_design_matches_R(sub):
    """attrdl(model=<statsmodels fit, named design [cb, const, seasonal]>, link forced to log) vs R, per-observation
    AF: unaffected by the coefficient-selection defect today and must stay so."""
    fit, coef, vcov = _fit_sm(sub, 'cb_first', named=True)
    fit.link = 'log'
    py, ref = attr_pair(sub, 'af_obs', 15.0, coef, vcov, model=fit)
    assert_same(py, ref, 'attrdl(model, cb_first) af_obs')
