"""attribution.py versus R's attrdl (Gasparrini 2015, 2015_gasparrini_Lancet_Rcodedata-master/attrdl.R).

Audit themes and findings turned into differential regression tests (R computes the reference at run time):

  K   attribution.attrdl is not R attrdl
        attr-point-2   dir='back' silently equals dir='forw' (no lagged exposure history)
        attr-point-3   dir='forw': no forward moving average of the cases, total AN not rescaled to the observed
                       cases, NaN rows dropped instead of kept in place
        attr-point-4   type='af'/'both' with a range divides by the in-range cases (AF heat + AF cold != AF total)
        attr-point-7   cen=None: coarse-grid MMT substituted and basis argvar['cen'] ignored (R: use it, else error)
        attr-point-9   attrdl(sim=True) crashes for type='an'/'both' (also attr_heat_cold, attr_by_percentiles and
                       AttributionManager.summary_report with sim=True)
        attr-point-12  attr_heat_cold / attr_by_percentiles return silent zeros when x has a NaN
        attr-point-13  attr_by_percentiles bins are closed on both ends: tied values are counted in two bins
        attr-point-14  tot=False output not aligned with the input (NaN rows dropped, no NaN for incomplete forward
                       windows); all-zero cases give 0.0 where R gives NaN            (confirmed by ONE verifier only)
        attr-point-15  argument handling: invalid type silently returns nothing, no partial matching of dir,
                       matrix x / cases give a cryptic numpy crash
        attr-point-16  attrdl_proper is dead code that ignores cen/vcov and does not reproduce R
        attr-point-17  sim=True rebuilds the coefficient-independent design matrix for every draw
  K2  attr-sim-13    attr_heat_cold splits at the 2.5/97.5 percentiles instead of the MMT (and shares the NaN
                     threshold defect with attr-point-12)

Plain (unmarked) tests guard what is already faithful today: per-observation AF (dir='forw', tot=False) for
explicit-knot bases (330 random configurations, a Chicago fit with ranges, tied range end points), additivity of the
heat and cold AN, percentile thresholds, zero-effect and input handling, dir validation and the tiny-simulation
machinery for type='af'.

Conventions
  * a model-like object named ImprovedGLMInterface (cb_coef / cb_vcov) is passed so that the link is 'log' (the link
    finding attr-point-1 is not exercised here);
  * R attrdl is sourced into a private R environment (aK_env); every R variable of this module starts with aK_;
  * R's attrdl has no counterpart for attr_heat_cold / attr_by_percentiles: there the R-faithful property (R's own
    identity, e.g. heat + cold = total, or the per-observation AF of the whole series) is computed in R and demanded
    of PyDLNM;
  * tests that isolate one finding avoid the others: per-observation AF and identities do not depend on the forward
    moving average of the cases (attr-point-3); the totals versus R do.
"""
import contextlib
import copy
import io
import os
import warnings

import numpy as np
import pytest
import rpy2.robjects as ro

from rhelpers import REPO, assert_close, chicago, known_defect, np2r, r, rget

ATTRDL_R = REPO / '2015_gasparrini_Lancet_Rcodedata-master' / 'attrdl.R'

# Several PyDLNM modules overwrite os.environ['R_HOME'] with a path under which a later R glm() segfaults here.
# R keeps its start-up library path in .Library, so the correct value can always be recovered from it.
_R_HOME = os.path.dirname(str(r('.Library')[0]))


def _fix_r_home():
    os.environ['R_HOME'] = _R_HOME


@pytest.fixture(autouse=True)
def _r_home_guard():
    _fix_r_home()
    yield
    _fix_r_home()


# ----------------------------------------------------------------------------------------------------------------
# helpers: one series + one cross-basis + fixed coefficients, evaluated by R attrdl and by PyDLNM attrdl
# ----------------------------------------------------------------------------------------------------------------
class ImprovedGLMInterface:
    """Minimal stand-in for PyDLNM's fitted-model wrapper (same class name => getcoef/getvcov/getlink work, link 'log')."""

    def __init__(self, coef, vcov):
        self.cb_coef = np.asarray(coef, dtype=float)
        self.cb_vcov = np.asarray(vcov, dtype=float)


def _require_r_package(pkg):
    # (rhelpers.require_r_packages indexes the invisible result of requireNamespace, which rpy2 returns as None)
    if not bool(r(f'isTRUE(suppressWarnings(requireNamespace("{pkg}", quietly=TRUE)))')[0]):
        pytest.skip(f'R package {pkg} not installed')


def _load_r_attrdl():
    _require_r_package('tsModel')
    if not bool(r('exists("aK_env")')[0]):
        ro.globalenv['aK_src'] = str(ATTRDL_R)
        r('aK_env <- new.env(); sys.source(aK_src, envir=aK_env)')


def _r_arglist(prefix, spec):
    """R list(...) literal for a basis-argument dict; arrays are pushed to R as aK_<prefix>_<key>."""
    items = []
    for k, v in spec.items():
        rk = 'Boundary.knots' if k == 'Boundary_knots' else k
        if isinstance(v, str):
            items.append(f'{rk}="{v}"')
        elif isinstance(v, (bool, np.bool_)):
            items.append(f'{rk}={"TRUE" if v else "FALSE"}')
        elif isinstance(v, (int, np.integer)):
            items.append(f'{rk}={int(v)}')
        else:
            np2r(f'aK_{prefix}_{k}', np.atleast_1d(np.asarray(v, dtype=float)))
            items.append(f'{rk}=aK_{prefix}_{k}')
    return 'list(' + ', '.join(items) + ')'


def _lag_literal(lag):
    return f'{int(lag)}' if np.ndim(lag) == 0 else f'c({int(lag[0])},{int(lag[1])})'


def _quiet(f, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with contextlib.redirect_stdout(io.StringIO()):
            return f(*args, **kwargs)


def _py_crossbasis(x, lag, var, lagb):
    from basis import CrossBasis
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            return CrossBasis(x, lag=lag if np.ndim(lag) == 0 else list(lag), argvar=dict(var), arglag=dict(lagb))
    finally:
        _fix_r_home()


class Scenario:
    """Series x, cases, a cross-basis (built in R and in PyDLNM from one specification) and fixed coef/vcov."""

    def __init__(self, x, cases, lag, var, lagb, coef=None, vcov=None, seed=0):
        self.x = np.array(x, dtype=float)
        self.cases = np.array(cases, dtype=float)
        self.lag, self.var, self.lagb = lag, var, lagb
        self.cb = _py_crossbasis(self.x, lag, var, lagb)
        k = self.cb.shape[1]
        # coef=None: small deterministic random coefficients (R evaluates the reference on the same values)
        self.coef = np.random.default_rng(seed).normal(0, .04, k) if coef is None else np.array(coef, dtype=float)
        assert len(self.coef) == k
        self.vcov = np.eye(k) * 1e-4 if vcov is None else np.array(vcov, dtype=float)
        self.cen = None

    def replace(self, **kw):
        d = dict(x=self.x, cases=self.cases, lag=self.lag, var=self.var, lagb=self.lagb, coef=self.coef, vcov=self.vcov)
        d.update(kw)
        new = Scenario(**d)
        new.cen = self.cen
        return new

    # ---- R side
    def push(self):
        np2r('aK_x', self.x)
        np2r('aK_cases', self.cases)
        np2r('aK_coef', self.coef)
        np2r('aK_vcov', self.vcov)
        # plain vectors: a 1-d array coming from numpy makes `coef + matrix` non-conformable in attrdl's simulation
        r('aK_x <- as.numeric(aK_x); aK_cases <- as.numeric(aK_cases); aK_coef <- as.numeric(aK_coef)')
        r(f'aK_cb <- crossbasis(aK_x, lag={_lag_literal(self.lag)}, argvar={_r_arglist("var", self.var)}, '
          f'arglag={_r_arglist("lag", self.lagb)})')

    def r_attrdl(self, *, type='an', dir='forw', tot=True, cen=None, rng=None, x=None, cases=None, sim=False, nsim=5):
        """R attrdl on this scenario; returns a 1-d float array. x / cases override the series (e.g. by matrices)."""
        _load_r_attrdl()
        self.push()
        xn, cn = 'aK_x', 'aK_cases'
        if x is not None:
            np2r('aK_xalt', x)
            xn = 'aK_xalt'
        if cases is not None:
            np2r('aK_casesalt', cases)
            cn = 'aK_casesalt'
        args = [xn, 'aK_cb', cn, 'coef=aK_coef', 'vcov=aK_vcov', f'type="{type}"', f'dir="{dir}"',
                f'tot={"TRUE" if tot else "FALSE"}']
        if cen is not None:
            np2r('aK_cen', np.array([cen], dtype=float))
            args.append('cen=aK_cen')
        if rng is not None:
            np2r('aK_rng', np.array(rng, dtype=float))
            args.append('range=aK_rng')
        if sim:
            args += ['sim=TRUE', f'nsim={int(nsim)}']
        return np.atleast_1d(rget('aK_env$attrdl(' + ', '.join(args) + ')')).astype(float)

    # ---- PyDLNM side
    def py_attrdl(self, x=None, cases=None, **kw):
        import attribution
        return _quiet(attribution.attrdl, self.x if x is None else x, self.cb, self.cases if cases is None else cases,
                      model=ImprovedGLMInterface(self.coef, self.vcov), **kw)

    def py_call(self, fname, **kw):
        """PyDLNM wrapper function (attr_heat_cold, attr_by_percentiles, ...) with this scenario's inputs."""
        import attribution
        return _quiet(getattr(attribution, fname), self.x, self.cb, self.cases,
                      model=ImprovedGLMInterface(self.coef, self.vcov), **kw)


def _ranges(x, cen):
    xf = x[np.isfinite(x)]
    return {'none': None, 'cold': (-100.0, cen), 'heat': (cen, 100.0),
            'mid': (float(np.quantile(xf, .2)), float(np.quantile(xf, .6)))}


def _as_array(v):
    return np.atleast_1d(np.asarray(v, dtype=float))


def _out(res, type, tot):
    """The requested output of a PyDLNM attrdl result as a 1-d array (total or per observation)."""
    return _as_array(res[type + '_total'] if tot else res[type])


def _lagmat(v, lags):
    """Matrix whose column j is v shifted by lags[j] (lags < 0: future values); NaN where undefined (tsModel Lag)."""
    n = len(v)
    out = np.full((n, len(lags)), np.nan)
    for j, l in enumerate(lags):
        if l >= 0:
            out[l:, j] = v[:n - l]
        else:
            out[:n + l, j] = v[-l:]
    return out


def _r_error(fn):
    """Message of the R error raised by fn(); fails the test if R does not raise."""
    from rpy2.rinterface_lib.embedded import RRuntimeError
    try:
        fn()
    except RRuntimeError as e:
        return str(e)
    raise AssertionError('R attrdl was expected to raise an error')


def _on_finite_x(py, ref, x):
    """Rows with a finite exposure. The implementation either keeps missing rows in place as NaN (as R does) or drops
    them; both are accepted here so that this regression guard survives the alignment fix (attr-point-14)."""
    ok = np.isfinite(x)
    py = _as_array(py)
    if py.shape[0] == len(x):
        py = py[ok]
    assert py.shape[0] == int(ok.sum()), f'output has {py.shape[0]} rows for {len(x)} observations ({int(ok.sum())} finite)'
    return py, _as_array(ref)[ok]


def _rel(a, b):
    return abs(float(a) - float(b)) / max(abs(float(b)), 1e-300)


# ----------------------------------------------------------------------------------------------------------------
# fixtures
# ----------------------------------------------------------------------------------------------------------------
LAG = 14


@pytest.fixture(scope='module')
def chi():
    """Chicago NMMAPS: quasi-Poisson DLNM fitted in R on the whole series (bs(deg 2, knots at the 10/75/90 percentiles)
    x ns(logknots(14, 3)), lag 14, 25 coefficients); the attribution is evaluated on the first 1000 days with cen = the
    minimum of the overall curve (so that heat and cold both carry excess risk)."""
    _fix_r_home()   # this module-scoped fixture runs before the function-scoped autouse guard
    ch = chicago()
    np2r('aK_tf', ch['temp'])
    np2r('aK_yf', ch['death'])
    vk = rget('quantile(aK_tf, c(.10,.75,.90))')
    lk = rget(f'logknots({LAG}, 3)')
    np2r('aK_vk0', vk)
    np2r('aK_lk0', lk)
    r(f'aK_cbf <- crossbasis(aK_tf, lag={LAG}, argvar=list(fun="bs", degree=2, knots=aK_vk0), '
      f'arglag=list(fun="ns", knots=aK_lk0))')
    r('aK_datf <- data.frame(y=aK_yf, tt=seq_along(aK_yf))')
    r('aK_mf <- glm(y ~ aK_cbf + splines::ns(tt, df=35), data=aK_datf, family=quasipoisson, na.action=na.exclude)')
    r('aK_iif <- grep("aK_cbf", names(coef(aK_mf)))')
    coef = rget('unname(coef(aK_mf)[aK_iif])')
    vcov = rget('unname(vcov(aK_mf)[aK_iif, aK_iif])')
    n = 1000
    sc = Scenario(ch['temp'][:n], ch['death'][:n], LAG, {'fun': 'bs', 'degree': 2, 'knots': vk},
                  {'fun': 'ns', 'knots': lk}, coef, vcov)
    sc.push()
    r('aK_g <- seq(quantile(aK_x, .01), quantile(aK_x, .99), by=0.1)')
    r('aK_cp <- crosspred(aK_cb, coef=aK_coef, vcov=aK_vcov, model.link="log", at=aK_g, cen=median(aK_x))')
    sc.cen = float(rget('aK_g[which.min(aK_cp$allfit)]')[0])
    return sc


@pytest.fixture(scope='module')
def small(chi):
    """A short series with a 15-coefficient model (cheap for the tests that call CrossPred many times)."""
    x = chi.x[:300]
    sc = chi.replace(x=x, cases=chi.cases[:300], lag=5,
                     var={'fun': 'bs', 'degree': 2, 'knots': np.quantile(x, [.25, .5, .75])},
                     lagb={'fun': 'ns', 'df': 3}, coef=np.zeros(15), vcov=np.eye(15))
    k = len(sc.coef)
    sc.coef, sc.vcov = np.linspace(-.02, .03, k), np.eye(k) * 1e-4
    sc.cen = round(float(np.quantile(x, .6)), 2)
    return sc


# ================================================================================================================
# PLAIN TESTS: behaviour that is already R-faithful today
# ================================================================================================================
def _random_config(seed, temp0, death0):
    """One random explicit-knot configuration (bs deg 1-3 / ns variable basis, ns / ns-df / integer lag basis)."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(150, 400))
    s0 = int(rng.integers(0, len(temp0) - n))
    x, y = temp0[s0:s0 + n].copy(), death0[s0:s0 + n].copy()
    L = int(rng.integers(2, 9))
    m = min(int(rng.choice([0, 0, 0, 1, 2])), L - 2)
    lag = [m, L] if m > 0 else L
    nl = L - m + 1
    kq = np.sort(rng.choice([.1, .25, .5, .75, .9], size=int(rng.integers(1, 4)), replace=False))
    knots = np.quantile(x, kq)
    vf = str(rng.choice(['bs2', 'bs3', 'bs1', 'ns']))
    var = {'fun': 'bs', 'degree': int(vf[2]), 'knots': knots} if vf.startswith('bs') else {'fun': 'ns', 'knots': knots}
    lf = str(rng.choice(['ns_knots', 'ns_df', 'integer']))
    if lf == 'ns_knots' and nl >= 4:
        lagb = {'fun': 'ns', 'knots': np.quantile(np.arange(m, L + 1), [.33, .66])}
    elif lf == 'integer':
        lagb = {'fun': 'integer'}
    else:
        lagb = {'fun': 'ns', 'df': max(1, min(3, nl - 1))}
    pat = str(rng.choice(['none', 'none', 'xlead', 'xrand', 'czeros']))
    if pat == 'xlead':
        x[:int(rng.integers(1, 10))] = np.nan
    elif pat == 'xrand':
        x[rng.choice(n, int(n * .03) + 1, replace=False)] = np.nan
    elif pat == 'czeros':
        j = int(rng.integers(0, n - 30))
        y[j:j + 25] = 0
    cen = round(float(np.quantile(x[np.isfinite(x)], rng.uniform(.3, .8))), 2)
    rname = str(rng.choice(['none', 'cold', 'heat', 'mid']))
    return dict(x=x, y=y, lag=lag, var=var, lagb=lagb, cen=cen, rng=_ranges(x, cen)[rname],
                label=f'seed{seed} n={n} lag={lag} {vf} {lf} {pat} range={rname}')


N_RANDOM_CONFIGS = 330


@pytest.mark.slow
def test_per_obs_af_forw_matches_r_random_configs():
    """330 random explicit-knot configurations (bs degree 1-3 / ns variable basis with 1-3 knots, ns-knots / ns-df /
    integer lag basis, lag ranges [0..2, 2..8], NaN in x or runs of zero cases, ranges none/cold/heat/mid):
    per-observation AF (type='af', dir='forw', tot=False) equals R's to <= 1e-8. This is the part of attrdl that is
    faithful today (centering, range-to-cen, boundary knots)."""
    ch = chicago()
    temp0, death0 = ch['temp'], ch['death'].astype(float)
    worst, bad = 0.0, []
    for seed in range(N_RANDOM_CONFIGS):
        cfg = _random_config(seed, temp0, death0)
        sc = Scenario(cfg['x'], cfg['y'], cfg['lag'], cfg['var'], cfg['lagb'], seed=1000 + seed)
        ref = sc.r_attrdl(type='af', dir='forw', tot=False, cen=cfg['cen'], rng=cfg['rng'])
        py = sc.py_attrdl(type='af', dir='forw', tot=False, cen=cfg['cen'], range=cfg['rng'])['af']
        py, ref = _on_finite_x(py, ref, cfg['x'])
        d = float(np.max(np.abs(py - ref)))
        worst = max(worst, d)
        if not d <= 1e-8:
            bad.append((cfg['label'], d))
    assert not bad, f'{len(bad)}/{N_RANDOM_CONFIGS} configurations differ from R (worst {worst:.2e}): {bad[:5]}'


@pytest.mark.parametrize('rname', ['none', 'cold', 'heat', 'mid'])
def test_chicago_per_obs_af_forw_matches_r(chi, rname):
    """Chicago fit (25 coefficients): per-observation AF for the whole range, cold, heat and an interior range."""
    rng = _ranges(chi.x, chi.cen)[rname]
    ref = chi.r_attrdl(type='af', dir='forw', tot=False, cen=chi.cen, rng=rng)
    py = chi.py_attrdl(type='af', dir='forw', tot=False, cen=chi.cen, range=rng)['af']
    py, ref = _on_finite_x(py, ref, chi.x)
    assert_close(py, ref, rtol=1e-8, what=f'per-observation AF, range {rname}')


def test_range_end_points_are_inclusive_like_r(chi):
    """R sets x[x < range[1] | x > range[2]] to cen, i.e. a value equal to an end point stays in. Chicago temperatures
    are tied (F converted to C), so end points sitting on tied observations are a sharp test of that convention."""
    vals, counts = np.unique(chi.x, return_counts=True)
    lo = vals[np.argmax((vals > np.quantile(chi.x, .2)) & (counts >= 3))]
    hi = vals[np.argmax((vals > np.quantile(chi.x, .8)) & (counts >= 3))]
    assert (chi.x == lo).sum() >= 3 and (chi.x == hi).sum() >= 3
    ref = chi.r_attrdl(type='af', dir='forw', tot=False, cen=chi.cen, rng=(lo, hi))
    py = chi.py_attrdl(type='af', dir='forw', tot=False, cen=chi.cen, range=(float(lo), float(hi)))['af']
    py, ref = _on_finite_x(py, ref, chi.x)
    assert_close(py, ref, rtol=1e-8, what='per-observation AF with tied range end points')
    assert np.count_nonzero(ref) > 0.3 * len(ref)


def test_heat_plus_cold_an_equals_total_an(chi):
    """03.attr.R splits at cen with range=c(-100,cen) and c(cen,100); cold + heat = total in R and in PyDLNM."""
    cen = chi.cen
    r_tot, r_cold, r_heat = (chi.r_attrdl(type='an', tot=True, cen=cen, rng=rg)[0]
                             for rg in (None, (-100., cen), (cen, 100.)))
    assert _rel(r_cold + r_heat, r_tot) <= 1e-10
    py_tot, py_cold, py_heat = (float(chi.py_attrdl(type='an', tot=True, cen=cen, range=rg)['an_total'])
                                for rg in (None, (-100., cen), (cen, 100.)))
    assert _rel(py_cold + py_heat, py_tot) <= 1e-10, 'heat + cold AN != total AN'


@pytest.mark.parametrize('typ', ['an', 'af'])
def test_zero_coefficients_give_zero_attribution(chi, typ):
    sc = chi.replace(coef=np.zeros(len(chi.coef)))
    ref = sc.r_attrdl(type=typ, tot=True, cen=chi.cen)
    py = _out(sc.py_attrdl(type=typ, tot=True, cen=chi.cen), typ, True)
    assert ref[0] == 0.0
    assert py[0] == 0.0


def test_inputs_are_not_modified_and_series_are_accepted(chi):
    """attrdl must not change x, cases or the basis arguments; pandas Series (int cases) give the same numbers."""
    import pandas as pd
    import attribution
    x0, c0 = chi.x.copy(), chi.cases.copy()
    argvar0, arglag0 = copy.deepcopy(chi.cb.argvar), copy.deepcopy(chi.cb.arglag)
    rng = (chi.cen, 100.0)
    base = chi.py_attrdl(type='af', tot=False, cen=chi.cen, range=rng)['af']
    idx = pd.date_range('1987-01-01', periods=len(x0), freq='D')
    via_series = _quiet(attribution.attrdl, pd.Series(x0, index=idx), chi.cb, pd.Series(c0.astype(np.int32), index=idx),
                        model=ImprovedGLMInterface(chi.coef, chi.vcov), type='af', tot=False, cen=chi.cen,
                        range=rng)['af']
    assert np.array_equal(chi.x, x0) and np.array_equal(chi.cases, c0)
    for before, after in ((argvar0, chi.cb.argvar), (arglag0, chi.cb.arglag)):
        assert before.keys() == after.keys()
        assert all(np.array_equal(np.asarray(before[k]), np.asarray(after[k])) for k in before)
    assert_close(np.asarray(via_series), np.asarray(base), rtol=1e-12, what='pandas Series input')


def test_sim_af_with_zero_vcov_reproduces_point_estimate(small):
    """type='af' simulation works today: with a zero covariance every draw is the point estimate, in R and in PyDLNM
    (tiny nsim, deterministic). The structure of the returned CI is checked as well."""
    nsim = 5
    sc = small.replace(vcov=np.zeros((len(small.coef),) * 2))
    r_point = sc.r_attrdl(type='af', tot=True, cen=small.cen)
    r_sim = sc.r_attrdl(type='af', tot=True, cen=small.cen, sim=True, nsim=nsim)
    assert r_sim.shape == (nsim,) and np.allclose(r_sim, r_point[0], rtol=1e-12, atol=0)
    res = sc.py_attrdl(type='af', tot=True, cen=small.cen, sim=True, nsim=nsim)
    draws = np.asarray(res['sim_results']['simulations']['af_total'])
    assert draws.shape == (nsim,)
    assert np.allclose(draws, res['af_total'], rtol=1e-12, atol=0)
    assert _rel(res['ci']['af_total_low'], res['af_total']) <= 1e-12
    assert _rel(res['ci']['af_total_high'], res['af_total']) <= 1e-12


def test_percentile_thresholds_match_r_quantile(chi):
    """attr_by_percentiles thresholds are R's quantile(x, p/100) (type 7) on a complete series."""
    pr = [(0, 5), (5, 50), (50, 95), (95, 100)]
    chi.push()
    r_thr = {p: float(rget(f'quantile(aK_x, {p}/100)')[0]) for p in {q for pair in pr for q in pair}}
    table = chi.py_call('attr_by_percentiles', cen=chi.cen, percentile_ranges=pr)['summary_table']
    assert np.allclose(table['low_threshold'], [r_thr[lo] for lo, _ in pr], rtol=1e-12, atol=0)
    assert np.allclose(table['high_threshold'], [r_thr[hi] for _, hi in pr], rtol=1e-12, atol=0)


@pytest.mark.parametrize('typ', ['an', 'af'])
def test_invalid_dir_is_an_error_like_r(chi, typ):
    msg = _r_error(lambda: chi.r_attrdl(type=typ, dir='sideways', tot=True, cen=chi.cen))
    assert 'dir' in msg or 'arg' in msg
    with pytest.raises(ValueError):
        chi.py_attrdl(type=typ, dir='sideways', tot=True, cen=chi.cen)


# ================================================================================================================
# KNOWN DEFECTS: these assert the R-faithful behaviour and fail today (strict xfail)
# ================================================================================================================

# ---- attr-point-2: dir='back' -------------------------------------------------------------------------------------
BACK_CASES = [(typ, tot, rname) for typ in ('an', 'af') for tot in (True, False)
              for rname in ('none', 'cold', 'heat', 'mid')]


@pytest.mark.parametrize('typ,tot,rname', BACK_CASES,
                         ids=[f'{t}-{"tot" if o else "obs"}-{n}' for t, o, n in BACK_CASES])
def test_dir_back_matches_r(chi, typ, tot, rname):
    """R dir='back': at = Lag(x, 0:lag) (after x outside the range is set to cen), Xpredall = sum over the lags of the
    centred tensor basis; the per-observation output is NaN in the first lag[2] rows."""
    rng = _ranges(chi.x, chi.cen)[rname]
    ref = chi.r_attrdl(type=typ, dir='back', tot=tot, cen=chi.cen, rng=rng)
    py = _out(chi.py_attrdl(type=typ, dir='back', tot=tot, cen=chi.cen, range=rng), typ, tot)
    assert_close(py, ref, rtol=1e-8, what=f"dir='back' {typ} tot={tot} range {rname}")


@pytest.mark.parametrize('rname', ['none', 'heat'])
def test_dir_back_differs_from_forw_like_r(chi, rname):
    """R distinguishes the two perspectives (per-observation AF, which does not involve the forward average of the
    cases, so this stays independent of attr-point-3); PyDLNM returns the same numbers for both."""
    rng = _ranges(chi.x, chi.cen)[rname]
    r_forw = chi.r_attrdl(type='af', dir='forw', tot=False, cen=chi.cen, rng=rng)
    r_back = chi.r_attrdl(type='af', dir='back', tot=False, cen=chi.cen, rng=rng)
    ok = np.isfinite(r_forw) & np.isfinite(r_back)
    assert np.max(np.abs(r_back[ok] - r_forw[ok])) > 1e-3, 'R itself must distinguish the perspectives on this input'
    py_forw = _as_array(chi.py_attrdl(type='af', dir='forw', tot=False, cen=chi.cen, range=rng)['af'])
    py_back = _as_array(chi.py_attrdl(type='af', dir='back', tot=False, cen=chi.cen, range=rng)['af'])
    assert py_forw.shape == py_back.shape == r_forw.shape
    ok = np.isfinite(py_forw) & np.isfinite(py_back)
    d = float(np.max(np.abs(py_back[ok] - py_forw[ok])))
    assert d > 1e-3, f"dir='back' and dir='forw' give the same per-observation AF (max difference {d:.1e})"


# ---- attr-point-3: forward moving average of the cases, rescaling to the observed cases, NaN kept in place ----------
@pytest.mark.parametrize('typ', ['an', 'af'])
def test_forw_totals_match_r(chi, typ):
    """R dir='forw', tot=TRUE: cases_t -> forward mean over the lag window, af = sum(an)/sum(cases_fwd) over the rows
    with a complete window, an = af * (all observed cases)."""
    ref = chi.r_attrdl(type=typ, dir='forw', tot=True, cen=chi.cen)
    py = _out(chi.py_attrdl(type=typ, dir='forw', tot=True, cen=chi.cen), typ, True)
    assert_close(py, ref, rtol=1e-8, what=f'forw total {typ}')


@pytest.mark.parametrize('typ', ['an', 'af'])
@pytest.mark.parametrize('rname', ['cold', 'heat', 'mid'])
def test_forw_totals_with_range_match_r(chi, typ, rname):
    """Heat / cold / interior range: x outside the range is set to cen (null risk) but the denominator keeps all cases."""
    rng = _ranges(chi.x, chi.cen)[rname]
    ref = chi.r_attrdl(type=typ, dir='forw', tot=True, cen=chi.cen, rng=rng)
    py = _out(chi.py_attrdl(type=typ, dir='forw', tot=True, cen=chi.cen, range=rng), typ, True)
    assert_close(py, ref, rtol=1e-8, what=f'forw total {typ}, range {rname}')


def test_forw_per_obs_an_uses_forward_average(chi):
    ref = chi.r_attrdl(type='an', dir='forw', tot=False, cen=chi.cen)
    assert int(np.isnan(ref).sum()) == LAG, 'R: the last lag[2] rows have an incomplete forward window'
    py = _as_array(chi.py_attrdl(type='an', dir='forw', tot=False, cen=chi.cen)['an'])
    assert py.shape == ref.shape
    ok = np.isfinite(ref)
    assert_close(py[ok], ref[ok], rtol=1e-8, what='per-observation AN on the rows with a complete window')


@pytest.mark.parametrize('where', ['x', 'cases'])
def test_forw_total_an_with_missing_data_matches_r(chi, where):
    rng_ = np.random.default_rng(5)
    x, cases = chi.x.copy(), chi.cases.copy()
    target = x if where == 'x' else cases
    target[rng_.choice(len(x), 30, replace=False)] = np.nan
    sc = chi.replace(x=x, cases=cases)
    ref = sc.r_attrdl(type='an', dir='forw', tot=True, cen=chi.cen)
    py = _out(sc.py_attrdl(type='an', dir='forw', tot=True, cen=chi.cen), 'an', True)
    assert_close(py, ref, rtol=1e-8, what=f'forw total AN with NaN in {where}')


# ---- attr-point-4: AF denominator with a range ---------------------------------------------------------------------
@pytest.mark.parametrize('rname', ['cold', 'heat', 'mid'])
def test_af_total_denominator_is_all_observed_cases(chi, rname):
    """R: an = af * den, so af_total = an_total / (all observed cases) for every range. (The identity does not involve
    the forward average of attr-point-3.)"""
    rng = _ranges(chi.x, chi.cen)[rname]
    r_an = chi.r_attrdl(type='an', tot=True, cen=chi.cen, rng=rng)[0]
    r_af = chi.r_attrdl(type='af', tot=True, cen=chi.cen, rng=rng)[0]
    assert _rel(r_an / chi.cases.sum(), r_af) <= 1e-12, 'R identity af = an / den'
    res = chi.py_attrdl(type='both', tot=True, cen=chi.cen, range=rng)
    assert _rel(res['af_total'], res['an_total'] / chi.cases.sum()) <= 1e-10, \
        f"af_total {res['af_total']:.6f} vs an_total/all cases {res['an_total'] / chi.cases.sum():.6f}"


def test_heat_af_plus_cold_af_equals_total_af(chi):
    cen = chi.cen
    r_af = [chi.r_attrdl(type='af', tot=True, cen=cen, rng=rg)[0] for rg in (None, (-100., cen), (cen, 100.))]
    assert _rel(r_af[1] + r_af[2], r_af[0]) <= 1e-10, 'R: cold AF + heat AF = total AF'
    py_af = [float(chi.py_attrdl(type='af', tot=True, cen=cen, range=rg)['af_total'])
             for rg in (None, (-100., cen), (cen, 100.))]
    assert _rel(py_af[1] + py_af[2], py_af[0]) <= 1e-10, \
        f'cold {py_af[1]:.5f} + heat {py_af[2]:.5f} != total {py_af[0]:.5f}'


def test_wrapper_af_totals_use_all_observed_cases(chi):
    den = chi.cases.sum()
    hc = chi.py_call('attr_heat_cold', cen=chi.cen)
    bp = chi.py_call('attr_by_percentiles', cen=chi.cen, percentile_ranges=[(0, 5), (95, 100)])
    parts = [hc['cold']['results'], hc['heat']['results'], bp['pct_0_5']['results'], bp['pct_95_100']['results']]
    for part in parts:
        assert part['an_total'] > 0
        assert _rel(part['af_total'], part['an_total'] / den) <= 1e-10, \
            f"af_total {part['af_total']:.5f} != an_total/all cases {part['an_total'] / den:.5f}"


# ---- attr-point-7: cen ---------------------------------------------------------------------------------------------
def test_cen_stored_in_the_basis_is_used(chi):
    """R: cen missing -> attr(basis, 'argvar')$cen. (Per-observation AF is used so that the forward average of
    attr-point-3 plays no role.)"""
    sc = chi.replace(var={**chi.var, 'cen': chi.cen})
    ref = sc.r_attrdl(type='af', dir='forw', tot=False, cen=None)
    res = sc.py_attrdl(type='af', dir='forw', tot=False)
    assert_close(_as_array(res['af']), ref, rtol=1e-8, what='AF with cen taken from the basis')
    assert res['metadata']['centering'] == chi.cen


def test_missing_cen_is_an_error_like_r(chi):
    msg = _r_error(lambda: chi.r_attrdl(type='an', dir='forw', tot=True))
    assert 'cen' in msg
    with pytest.raises((ValueError, TypeError), match='cen'):
        chi.py_attrdl(type='an', dir='forw', tot=True)


# ---- attr-point-9: sim=True crash ---------------------------------------------------------------------------------
@pytest.mark.parametrize('typ', ['an', 'both', None], ids=['an', 'both', 'default-type'])
def test_sim_an_returns_nsim_draws(small, typ):
    """R attrdl(sim=TRUE, tot=TRUE) returns nsim simulated totals (the Lancet script uses type='an', sim=T)."""
    nsim = 6
    r_sim = small.r_attrdl(type='an', tot=True, cen=small.cen, sim=True, nsim=nsim)
    assert r_sim.shape == (nsim,) and np.all(np.isfinite(r_sim))
    kw = {} if typ is None else {'type': typ}
    res = small.py_attrdl(tot=True, cen=small.cen, sim=True, nsim=nsim, **kw)
    sims = res['sim_results']['simulations']
    for key in ['an_total'] + (['af_total'] if typ == 'both' else []):
        draws = np.asarray(sims[key])
        assert draws.shape == (nsim,) and np.all(np.isfinite(draws))
        lo, hi = res['ci'][key + '_low'], res['ci'][key + '_high']
        assert lo <= hi
        assert draws.min() - 1e-9 * abs(lo) <= lo and hi <= draws.max() + 1e-9 * abs(hi)


def test_sim_an_with_zero_vcov_reproduces_point_estimate(small):
    nsim = 4
    sc = small.replace(vcov=np.zeros((len(small.coef),) * 2))
    r_point = sc.r_attrdl(type='an', tot=True, cen=small.cen)
    r_sim = sc.r_attrdl(type='an', tot=True, cen=small.cen, sim=True, nsim=nsim)
    assert r_sim.shape == (nsim,) and np.allclose(r_sim, r_point[0], rtol=1e-12, atol=0)
    res = sc.py_attrdl(type='an', tot=True, cen=small.cen, sim=True, nsim=nsim)
    draws = np.asarray(res['sim_results']['simulations']['an_total'])
    assert draws.shape == (nsim,) and np.allclose(draws, res['an_total'], rtol=1e-12, atol=0)


def test_sim_wrappers_do_not_crash(small):
    import attribution
    nsim = 3
    hc = small.py_call('attr_heat_cold', cen=small.cen, sim=True, nsim=nsim)
    assert hc['cold']['results']['ci']['an_total_low'] <= hc['cold']['results']['ci']['an_total_high']
    bp = small.py_call('attr_by_percentiles', cen=small.cen, percentile_ranges=[(0, 10), (90, 100)], sim=True,
                       nsim=nsim)
    assert set(bp['summary_table'].columns) >= {'an_total_low', 'an_total_high'}
    mgr = attribution.AttributionManager(small.x, small.cb, small.cases,
                                         model=ImprovedGLMInterface(small.coef, small.vcov))
    rep = _quiet(mgr.summary_report, sim=True, nsim=nsim)
    assert 'total' in rep and 'heat_cold' in rep


# ---- attr-point-12 (+ attr-sim-13): NaN in the exposure -------------------------------------------------------------
def _with_nan_x(chi):
    x = chi.x.copy()
    x[500] = np.nan
    return chi.replace(x=x)


def test_percentile_bins_ignore_nan_exposure(chi):
    """R (Gasparrini scripts): thresholds are quantile(x, p, na.rm=TRUE); attrdl itself skips the missing exposure."""
    sc = _with_nan_x(chi)
    pr = [(0, 5), (5, 50), (50, 95), (95, 100)]
    sc.push()
    r_thr = {p: float(rget(f'quantile(aK_x, {p}/100, na.rm=TRUE)')[0]) for p in {q for pair in pr for q in pair}}
    res = sc.py_call('attr_by_percentiles', cen=chi.cen, percentile_ranges=pr)
    table = res['summary_table']
    for col, want in (('low_threshold', [r_thr[lo] for lo, _ in pr]), ('high_threshold', [r_thr[hi] for _, hi in pr])):
        got = np.asarray(table[col], dtype=float)
        assert np.allclose(got, want, rtol=1e-12), f'{col} {got} vs R quantile(na.rm=TRUE) {np.asarray(want)}'
    an = np.asarray(table['an_total'], dtype=float)
    assert np.all(np.isfinite(an)) and np.all(an > 0), f'bins with NaN exposure: an_total {an}'


def test_attr_heat_cold_with_nan_exposure_is_not_zero(chi):
    """A missing temperature must not silence the function (thresholds via na.rm=TRUE) nor make every observation both
    cold and heat: R's cold and heat pieces are disjoint, so their sum cannot exceed the total."""
    sc = _with_nan_x(chi)
    r_heat = sc.r_attrdl(type='an', tot=True, cen=chi.cen, rng=(chi.cen, 100.))[0]
    assert r_heat > 0, 'R: heat attributable number is positive on this input'
    total = float(sc.py_attrdl(type='an', tot=True, cen=chi.cen)['an_total'])
    res = sc.py_call('attr_heat_cold', cen=chi.cen)
    cold, heat = res['summary']['cold_an_total'], res['summary']['heat_an_total']
    assert np.isfinite(cold) and np.isfinite(heat) and cold > 0 and heat > 0, f'cold {cold}, heat {heat}'
    assert cold + heat <= total * (1 + 1e-8), f'cold {cold:.2f} + heat {heat:.2f} exceeds the total {total:.2f}'


# ---- attr-point-13: percentile bins ---------------------------------------------------------------------------------
PARTITION = [(0, 1), (1, 5), (5, 10), (10, 50), (50, 90), (90, 95), (95, 99), (99, 100)]


def test_percentile_bins_partition_the_observations(chi):
    """Contiguous bins covering the 0-100 percentiles must count every observation once: the per-observation AF
    summed over the bins equals R's per-observation AF of the whole series."""
    thr = np.percentile(chi.x, sorted({p for b in PARTITION for p in b}))
    assert sum(int((chi.x == t).sum()) for t in thr[1:-1]) > 0, 'this series has values tied at a shared threshold'
    res = chi.py_call('attr_by_percentiles', cen=chi.cen, percentile_ranges=PARTITION)
    af_sum = sum(np.asarray(res[f'pct_{lo}_{hi}']['results']['af'], dtype=float) for lo, hi in PARTITION)
    ref = chi.r_attrdl(type='af', dir='forw', tot=False, cen=chi.cen)
    assert_close(af_sum, ref, rtol=1e-8, what='sum over percentile bins of the per-observation AF')


def test_percentile_bin_an_totals_add_up(chi):
    res = chi.py_call('attr_by_percentiles', cen=chi.cen, percentile_ranges=PARTITION)
    whole = float(chi.py_attrdl(type='an', tot=True, cen=chi.cen)['an_total'])
    bins = sum(res[f'pct_{lo}_{hi}']['results']['an_total'] for lo, hi in PARTITION)
    assert _rel(bins, whole) <= 1e-10, f'sum of the bins {bins:.4f} vs total {whole:.4f}'


# ---- attr-point-14: alignment of tot=False output and NaN handling (ONE verifier only) -----------------------------
@pytest.mark.parametrize('where', ['x', 'cases'])
def test_per_obs_af_is_aligned_with_the_input(chi, where):
    """R returns one AF per input row (length n); NaN exactly where the exposure is missing (AF does not use cases)."""
    x, cases = chi.x.copy(), chi.cases.copy()
    (x if where == 'x' else cases)[[100, 400, 401]] = np.nan
    sc = chi.replace(x=x, cases=cases)
    ref = sc.r_attrdl(type='af', dir='forw', tot=False, cen=chi.cen)
    assert ref.shape == x.shape
    py = _as_array(sc.py_attrdl(type='af', dir='forw', tot=False, cen=chi.cen)['af'])
    assert_close(py, ref, rtol=1e-8, what=f'per-observation AF with NaN in {where}')


def test_per_obs_an_has_nan_for_incomplete_forward_window(chi):
    """Constant cases make the forward moving average a no-op, so only the NaN handling differs from R."""
    sc = chi.replace(cases=np.full(len(chi.x), 20.0))
    ref = sc.r_attrdl(type='an', dir='forw', tot=False, cen=chi.cen)
    assert int(np.isnan(ref).sum()) == LAG
    py = _as_array(sc.py_attrdl(type='an', dir='forw', tot=False, cen=chi.cen)['an'])
    assert_close(py, ref, rtol=1e-8, what='per-observation AN, constant cases')


@pytest.mark.parametrize('typ', ['an', 'af'])
@pytest.mark.parametrize('rname', ['none', 'heat'])
def test_all_zero_cases_give_nan_like_r(chi, typ, rname):
    sc = chi.replace(cases=np.zeros(len(chi.x)))
    rng = _ranges(chi.x, chi.cen)[rname]
    ref = sc.r_attrdl(type=typ, dir='forw', tot=True, cen=chi.cen, rng=rng)
    assert np.isnan(ref).all(), 'R: 0/0'
    py = _out(sc.py_attrdl(type=typ, dir='forw', tot=True, cen=chi.cen, range=rng), typ, True)
    assert np.isnan(py).all(), f'python returns {py}'


# ---- attr-point-15: argument handling -------------------------------------------------------------------------------
@pytest.mark.parametrize('bad', ['AF', 'foo'])
def test_invalid_type_is_an_error_like_r(chi, bad):
    assert _r_error(lambda: chi.r_attrdl(type=bad, dir='forw', tot=True, cen=chi.cen))
    with pytest.raises((ValueError, TypeError)):
        chi.py_attrdl(type=bad, dir='forw', tot=True, cen=chi.cen)


@pytest.mark.parametrize('abbr,full', [('f', 'forw'), ('b', 'back')])
def test_dir_partial_matching_like_r(chi, abbr, full):
    r_full = chi.r_attrdl(type='af', dir=full, tot=False, cen=chi.cen)
    r_abbr = chi.r_attrdl(type='af', dir=abbr, tot=False, cen=chi.cen)
    assert_close(r_abbr, r_full, rtol=1e-12, what='R accepts the abbreviation')
    py_full = _as_array(chi.py_attrdl(type='af', dir=full, tot=False, cen=chi.cen)['af'])
    py_abbr = _as_array(chi.py_attrdl(type='af', dir=abbr, tot=False, cen=chi.cen)['af'])
    assert np.array_equal(py_abbr, py_full, equal_nan=True)


@pytest.mark.parametrize('which', ['x-matrix-back', 'cases-matrix-forw'])
def test_matrix_inputs_like_r_or_clear_error(chi, which):
    """R: x may be the matrix of lagged exposures (dir='back'), cases the matrix of future cases (dir='forw').
    Implementing them, or raising NotImplementedError, are the acceptable outcomes; a numpy broadcast crash is not."""
    lags = np.arange(LAG + 1)
    if which == 'x-matrix-back':
        xm = _lagmat(chi.x, lags)
        ref = chi.r_attrdl(type='an', dir='back', tot=True, cen=chi.cen, x=xm)
        call = lambda: chi.py_attrdl(x=xm, type='an', dir='back', tot=True, cen=chi.cen)   # noqa: E731
    else:
        cm = _lagmat(chi.cases, -lags)
        ref = chi.r_attrdl(type='an', dir='forw', tot=True, cen=chi.cen, cases=cm)
        call = lambda: chi.py_attrdl(cases=cm, type='an', dir='forw', tot=True, cen=chi.cen)   # noqa: E731
    try:
        res = call()
    except NotImplementedError:
        return
    assert_close(_out(res, 'an', True), ref, rtol=1e-8, what=which)


@pytest.mark.parametrize('which', ['x-matrix-forw', 'cases-matrix-back'])
def test_matrix_inputs_rejected_like_r_with_a_clear_message(chi, which):
    lags = np.arange(LAG + 1)
    if which == 'x-matrix-forw':
        xm = _lagmat(chi.x, lags)
        assert _r_error(lambda: chi.r_attrdl(type='an', dir='forw', tot=True, cen=chi.cen, x=xm))
        call = lambda: chi.py_attrdl(x=xm, type='an', dir='forw', tot=True, cen=chi.cen)   # noqa: E731
    else:
        cm = _lagmat(chi.cases, -lags)
        assert _r_error(lambda: chi.r_attrdl(type='an', dir='back', tot=True, cen=chi.cen, cases=cm))
        call = lambda: chi.py_attrdl(cases=cm, type='an', dir='back', tot=True, cen=chi.cen)   # noqa: E731
    with pytest.raises((ValueError, TypeError, NotImplementedError)) as ei:
        call()
    assert 'broadcast' not in str(ei.value) and 'operands' not in str(ei.value), str(ei.value)


# ---- attr-point-16: attrdl_proper -----------------------------------------------------------------------------------
def test_attrdl_proper_reproduces_r_or_is_removed(chi):
    """attrdl_proper(x, basis_matrix, cases, coef, vcov, cen, ...) advertises R-style attribution from a cross-basis
    matrix, i.e. the backward perspective. The finding's fix is to delete it (or make it a thin wrapper); a version
    that stays must reproduce R's dir='back' total for the uncentred CrossBasis matrix and the given cen."""
    import attribution
    proper = getattr(attribution, 'attrdl_proper', None)
    if proper is None:
        return
    ref = chi.r_attrdl(type='an', dir='back', tot=True, cen=chi.cen)
    out = proper(chi.x, np.asarray(chi.cb.basis), chi.cases, chi.coef, chi.vcov, cen=chi.cen, type='an')
    assert_close(np.atleast_1d(out), ref, rtol=1e-8, what='attrdl_proper an')


# ---- attr-point-17: the design matrix is rebuilt for every simulation draw --------------------------------------------
def test_simulation_does_not_rebuild_the_design_per_draw(small, monkeypatch):
    """R (attrdl.R, lines 105-172) builds Xpredall once and needs one matrix product per draw. The coefficient-
    independent design must therefore be built the same number of times whatever nsim is (counted deterministically
    through CrossPred construction instead of wall-clock time, which is noisy)."""
    from prediction import CrossPred
    calls = {'n': 0}
    original = CrossPred.__init__

    def counting_init(self, *args, **kwargs):
        calls['n'] += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(CrossPred, '__init__', counting_init)
    counts = {}
    for nsim in (2, 8):
        calls['n'] = 0
        small.py_attrdl(type='af', tot=True, cen=small.cen, sim=True, nsim=nsim)
        counts[nsim] = calls['n']
    assert counts[2] == counts[8], f'CrossPred objects built: {counts[2]} for nsim=2, {counts[8]} for nsim=8'


# ---- attr-sim-13 (K2): attr_heat_cold splits at percentiles, not at cen -----------------------------------------------
def test_attr_heat_cold_splits_at_cen_and_covers_the_total(chi):
    """03.attr.R: cold = range c(-100, cen), heat = range c(cen, 100), cold + heat = total (also in R). attr_heat_cold
    given a cen must therefore report the two sides of cen, not the tails beyond the 2.5th / 97.5th percentiles."""
    cen = chi.cen
    r_w, r_c, r_h = (chi.r_attrdl(type='an', tot=True, cen=cen, rng=rg)[0] for rg in (None, (-100., cen), (cen, 100.)))
    assert _rel(r_c + r_h, r_w) <= 1e-10, 'R: cold + heat = total'
    whole, cold_ref, heat_ref = (float(chi.py_attrdl(type='an', tot=True, cen=cen, range=rg)['an_total'])
                                 for rg in (None, (-100., cen), (cen, 100.)))
    res = chi.py_call('attr_heat_cold', cen=cen)
    cold, heat = res['summary']['cold_an_total'], res['summary']['heat_an_total']
    assert _rel(cold, cold_ref) <= 1e-8, f'cold {cold:.3f} vs attrdl(range=(-100, cen)) {cold_ref:.3f}'
    assert _rel(heat, heat_ref) <= 1e-8, f'heat {heat:.3f} vs attrdl(range=(cen, 100)) {heat_ref:.3f}'
    assert _rel(cold + heat, whole) <= 1e-8, f'cold + heat {cold + heat:.3f} vs total {whole:.3f}'


@pytest.mark.parametrize('side', ['cold', 'heat'])
def test_attr_heat_cold_matches_r_attrdl_ranges(chi, side):
    cen = chi.cen
    rng = (-100., cen) if side == 'cold' else (cen, 100.)
    ref = chi.r_attrdl(type='an', tot=True, cen=cen, rng=rng)[0]
    got = chi.py_call('attr_heat_cold', cen=cen)['summary'][f'{side}_an_total']
    assert _rel(got, ref) <= 1e-8, f'{side}: {got:.3f} vs R {ref:.3f}'
