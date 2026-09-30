"""GLM interfaces (theme O): ImprovedGLMInterface, Rpy2GLMInterface, DLNMGLMInterface versus R glm()/crossreduce().

Every test computes its reference in R (dlnm 2.4.10, splines, at run time) on R's chicagoNMMAPS (one plain test on the
England & Wales series) and compares PyDLNM on identical inputs.  The Python interfaces only differ from R glm() in how they ASSEMBLE the model, so the reference is
always `glm(y ~ cb + dow + ns(date, df=dfseas*length(unique(year))), family=..., na.action=na.exclude)` (plus whatever
the test varies), and the comparison is on the cross-basis block of coef/vcov, nobs, fitted values or crossreduce().

Theme O -- GLM interfaces
  glm-1      **kwargs (weights, offset, subset, ...) documented as "passed to R's glm()" are never read: the fit is
             silently the plain one (R changes the answer; R rejects an unknown glm() argument)
  glm-2      summary()/predict() evaluate the R *global name* `fitted_model`, so a later fit (or a user variable)
             silently changes what an earlier interface reports
  glm-3, crossreduce-1
             Rpy2GLMInterface.crossreduce / DLNMGLMInterface.crossreduce never return (numpy conversion error, then R
             name-regex mismatch, then a positional call that binds coef to `model`); `type` is never forwarded
  crossreduce-2
             crossreduce.crossreduce(<interface object>) raises AttributeError / TypeError although documented
  glm-6      documented family 'gamma' is pasted into R code (base::gamma) and crashes; the family string is not
             validated (R-code injection point)
  glm-7      day-of-week coefficients carry hard-coded labels dowTuesday..dowSunday that do not name the column
             (columns are Monday, Saturday, ...; Friday is the reference); the printed reference level is wrong
  glm-8      na.exclude is a no-op: DLNMGLMInterface.predict() has one entry per KEPT row, R pads with NA
  glm-10     input handling: y as list/(n,1)/DataFrame and dates as list/date objects crash (ImprovedGLMInterface);
             formula_vars are pasted raw into the R formula (silent overwrite of the response, 'cb.' capture, ...)
  glm-11     aliased (rank-deficient) cross-basis coefficients silently become NaN; R stops in crossreduce()
  glm-12     (nit) seasonal df = dfseas*n_years without round(): only the integer-dfseas agreement with R is guarded
             (plain tests; the fractional-product behaviour equals the literal R script and is documentation-level)
  glm-13     (nit) ImprovedGLMInterface.crossreduce docstring: only guarded through plain tests (overall reduction
             equals R; an unknown reduction type raises)

Tests decorated with @known_defect assert the R-faithful behaviour and fail today (strict xfail); the plain tests
guard neighbouring behaviour that is already faithful and must keep passing while the fixes land.

Where the audit lists two acceptable fixes ("honour or reject") the test asserts the R-faithful one:
  * weights/offset/subset are honoured like R (an unknown keyword must be rejected, as R does);
  * formula_vars that R cannot use safely may either be rejected up front (ValueError/TypeError) or handled so that the
    cross-basis fit is exactly R's;
  * an aliased fit must be rejected at fit or crossreduce time (R errors in crossreduce).

Not asserted (no R-faithful behaviour to pin, or the API is ambiguous): glm.control()/`control=` (how a Python caller
would spell it is undecided), string dates (R's ns()/weekdays() reject them too), fractional dfseas (matches the literal
R script today; a deliberate switch to round() would change it), the weekly-data assumption of ImprovedGLMInterface.
"""
import contextlib
import io
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from rhelpers import assert_close, chicago, known_defect, max_abs_diff, np2r, r, r2np, rget
import rpy2.robjects as ro

THEME = 'O'
LAG = 21
DFSEAS = 8
KINDS = ('improved', 'rpy2', 'dlnm', 'fit_dlnm_model')     # entry points that fit a GLM with a cross-basis
CEN = 15.0                                                # centring value used for every crossreduce() call
WEEKDAYS = ('Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday')


def _warm_up_r_lapack():
    """Load R's lazily loaded LAPACK module while os.environ['R_HOME'] still points at the R that is embedded.
    Workaround for audit finding Q2: basis.py and the *_glm modules overwrite R_HOME with the
    /Library/Frameworks/R.framework/Resources path, and the first La_*() call afterwards (chol2inv in vcov.glm)
    dlopens the wrong R's modules/lapack and segfaults.  Remove once Q2 is fixed."""
    os.environ['R_HOME'] = os.path.dirname(str(r('.Library')[0]))      # the R_HOME R itself started with
    r('invisible(chol2inv(chol(diag(2) + 1))); invisible(solve(diag(2)))')


_warm_up_r_lapack()
_SENTINEL_R_HOME = os.environ['R_HOME']


@pytest.fixture(autouse=True, scope='module')
def _memoise_importr():
    """Speed only: every interface constructor calls importr('stats'/'splines'/'dlnm') (about 0.7 s in total).
    The wrappers are identical each time, so memoise them per module for the duration of this test module."""
    import improved_glm
    import rpy2_glm
    patched = []
    for mod in (improved_glm, rpy2_glm):
        orig = getattr(mod, 'importr', None)
        if orig is None:
            continue
        cache = {}

        def cached(name, *args, _orig=orig, _cache=cache, **kwargs):
            if args or kwargs:
                return _orig(name, *args, **kwargs)
            if name not in _cache:
                _cache[name] = _orig(name)
            return _cache[name]

        mod.importr = cached
        patched.append((mod, orig))
    yield
    for mod, orig in patched:
        mod.importr = orig


@pytest.fixture(autouse=True)
def _keep_process_r_home():
    """The modules under test rewrite os.environ['R_HOME'] at import / construction (finding Q2)."""
    os.environ['R_HOME'] = _SENTINEL_R_HOME
    yield
    os.environ['R_HOME'] = _SENTINEL_R_HOME


# --------------------------------------------------------------------------------------------------------------
# helpers: identical inputs for R and Python
# --------------------------------------------------------------------------------------------------------------
class Ctx:
    """A slice of chicagoNMMAPS with the same cross-basis in R (`go_<tag>_cb`) and Python, R-built covariates
    (`other` = day-of-week dummies + ns(date), for the interfaces that take user covariates) and the R data frame
    `go_<tag>_df`.  Knots are computed once in R and given to both sides."""

    def __init__(self, tag, rows):
        from basis import CrossBasis
        from utils import logknots
        ch = chicago()
        cvd = rget('as.numeric(chicagoNMMAPS$cvd)')
        self.tag, self.df, self.cbr, self.m = tag, f'go_{tag}_df', f'go_{tag}_cb', f'go_{tag}_m'
        self.temp = ch['temp'][rows].astype(float)
        self.y = {'death': ch['death'][rows].astype(float), 'cvd': cvd[rows]}
        self.date_days = ch['date'][rows]
        self.n = len(self.temp)
        self.dates = pd.Series(pd.to_datetime(self.date_days, unit='D'))
        np2r(f'{self.df}_temp', self.temp)
        np2r(f'{self.df}_death', self.y['death'])
        np2r(f'{self.df}_cvd', self.y['cvd'])
        np2r(f'{self.df}_date', self.date_days)
        wd = ', '.join(f'"{d}"' for d in WEEKDAYS)
        r(f'''
        {self.df} <- data.frame(date=as.Date({self.df}_date, origin="1970-01-01"), death={self.df}_death,
                                cvd={self.df}_cvd, temp={self.df}_temp)
        {self.df}$year <- as.integer(format({self.df}$date, "%Y"))
        {self.df}$dowf <- factor(c({wd})[as.POSIXlt({self.df}$date)$wday + 1L])
        go_{tag}_kv <- quantile({self.df}$temp, c(.10, .75, .90))
        {self.cbr} <- crossbasis({self.df}$temp, lag={LAG}, argvar=list(fun="bs", degree=2, knots=go_{tag}_kv),
                                 arglag=list(fun="ns", knots=logknots({LAG}, 3)))
        ''')
        kv = rget(f'go_{tag}_kv')
        with _quiet():
            self.cb = CrossBasis(self.temp, lag=LAG, argvar={'fun': 'bs', 'degree': 2, 'knots': kv},
                                 arglag={'fun': 'ns', 'knots': logknots([0, LAG], nk=3)})
        assert_close(np.asarray(self.cb.basis), rget(f'unclass({self.cbr})'), rtol=1e-12, what='test setup: cross-basis')
        self.other = rget(f'cbind(model.matrix(~dowf, {self.df})[, -1, drop=FALSE], '
                          f'ns({self.df}$date, df={DFSEAS}*length(unique({self.df}$year))))')
        self.other_names = [f'o{i}' for i in range(self.other.shape[1])]

    def r_fit(self, y='death', family='quasipoisson', dfseas=DFSEAS, weights=None, offset=None, subset=None, cb=None):
        """R glm(y ~ cb + dowf + ns(date, df=dfseas*n_years)) with na.exclude; the model stays in R as `go_<tag>_m`."""
        args = ''
        for key, val in (('weights', weights), ('offset', offset), ('subset', subset)):
            if val is not None:
                np2r(f'{self.m}_{key}', np.asarray(val, dtype=float))
                args += f', {key}=' + (f'as.logical({self.m}_{key})' if key == 'subset' else f'{self.m}_{key}')
        cb = cb or self.cbr
        r(f'{self.m} <- glm({y} ~ {cb} + dowf + ns(date, df={dfseas}*length(unique(year))), data={self.df}, '
          f'family={family}, na.action=na.exclude{args})')
        return self.reference(cb)

    def reference(self, cb=None):
        """Cross-basis block, nobs and (NA-padded) fitted values of the R model `go_<tag>_m`."""
        cb, m = cb or self.cbr, self.m
        idx = f'grep("^{cb}", names(coef({m})))'
        return SimpleNamespace(
            coef=rget(f'coef({m})[{idx}]'), vcov=rget(f'vcov({m})[{idx}, {idx}]'),
            nobs=int(rget(f'nobs({m})')[0]), fitted=rget(f'as.numeric(fitted({m}))'),
            names=list(r(f'names(coef({m}))')), all_coef=rget(f'coef({m})'),
            table=rget(f'summary({m})$coefficients'), model=m)

    def r_reduce(self, cen=CEN, cb=None, model=None):
        cb, model = cb or self.cbr, model or self.m
        cen = 'NULL' if cen is None else repr(float(cen))
        r(f'{self.m}_red <- crossreduce({cb}, {model}, cen={cen})')
        return rget(f'{self.m}_red$coef'), rget(f'{self.m}_red$vcov')


_CTX = {}


def ctx(name='small'):
    """'small' = 1987-1990 (n=1461), 'full' = 1987-2000 (n=5114), 'wkdays' = Monday-Friday rows of 'small'."""
    if name not in _CTX:
        if name == 'small':
            rows = np.arange(1461)
        elif name == 'full':
            rows = np.arange(len(chicago()['temp']))
        elif name == 'wkdays':
            dow = pd.Series(pd.to_datetime(chicago()['date'][:1461], unit='D')).dt.dayofweek.values
            rows = np.flatnonzero(dow < 5)
        else:
            raise KeyError(name)
        _CTX[name] = Ctx(name, rows)
    return _CTX[name]


@contextlib.contextmanager
def _quiet():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


class Fit:
    """A fitted PyDLNM interface with uniform accessors (cb block, R model object)."""

    def __init__(self, kind, iface):
        self.kind, self.iface = kind, iface

    @property
    def rpy2_iface(self):
        return self.iface if self.kind == 'rpy2' else getattr(self.iface, 'rpy2_interface', None)

    @property
    def cb_coef(self):
        return np.asarray(self.iface.get_crossbasis_coefficients()[0] if self.kind in ('dlnm', 'fit_dlnm_model')
                          else self.iface.cb_coef, dtype=float)

    @property
    def cb_vcov(self):
        return np.asarray(self.iface.get_crossbasis_coefficients()[1] if self.kind in ('dlnm', 'fit_dlnm_model')
                          else self.iface.cb_vcov, dtype=float)

    @property
    def r_model(self):
        return self.iface.r_model if self.kind in ('improved', 'rpy2') else self.iface.rpy2_interface.r_model

    def summary_table(self):
        s = self.iface.get_model_summary() if self.kind in ('improved', 'rpy2') \
            else self.iface.rpy2_interface.get_model_summary()
        return r2np(s.rx2('coefficients'))


def make_iface(kind, c):
    import glm_integration
    import improved_glm
    import rpy2_glm
    return {'improved': improved_glm.ImprovedGLMInterface, 'rpy2': rpy2_glm.Rpy2GLMInterface,
            'dlnm': glm_integration.DLNMGLMInterface}[kind](c.cb)


_DEFAULT = object()


def py_fit(kind, c, y='death', family='quasipoisson', dates=None, yvals=None, dfseas=DFSEAS, other=None,
           names=_DEFAULT, **kw):
    """Fit the same model as Ctx.r_fit through one of the interfaces.  `improved` builds dow + ns(date) itself;
    the others take the R-built covariates."""
    import glm_integration
    yv = c.y[y] if yvals is None else yvals
    other = c.other if other is None else other
    names = c.other_names if names is _DEFAULT else names
    with _quiet():
        if kind == 'improved':
            g = make_iface(kind, c)
            g.fit_dlnm_model(yv, c.dates if dates is None else dates, dfseas=dfseas, family=family, **kw)
        elif kind == 'rpy2' or kind == 'dlnm':
            g = make_iface(kind, c)
            g.fit_glm(yv, family=family, other_vars=other, formula_vars=names, **kw)
        elif kind == 'fit_dlnm_model':
            g = glm_integration.fit_dlnm_model(c.cb, yv, family=family, other_vars=other, formula_vars=names, **kw)
        else:
            raise KeyError(kind)
    return Fit(kind, g)


def as_coef_vcov(res):
    """crossreduce results come back as a dict (R branch) or a CrossReduce object (Python reducer)."""
    if isinstance(res, dict):
        return np.asarray(res['coef'], dtype=float), np.asarray(res['vcov'], dtype=float)
    return np.asarray(res.coef, dtype=float), np.asarray(res.vcov, dtype=float)


def assert_cb_block_matches(fit, ref, rtol=1e-8, what=''):
    assert_close(fit.cb_coef, ref.coef, rtol=rtol, what=f'{what} cross-basis coef')
    assert_close(fit.cb_vcov, ref.vcov, rtol=rtol, what=f'{what} cross-basis vcov')


def _expect_rejection(fn, what):
    """`fn` must stop with an error (R stops here).  While glm-3 is open the crossreduce() METHODS of the rpy2-based
    interfaces raise NotImplementedError for every input; that says nothing about aliasing and is not counted as a
    rejection."""
    try:
        fn()
    except NotImplementedError:
        pass
    except Exception:
        return
    pytest.fail(f'{what}: NaN coefficients handed out silently (R stops: "coef/vcov do not consistent")')


# --------------------------------------------------------------------------------------------------------------
# plain tests: behaviour that is already faithful (verified_ok of the glm / crossreduce audit areas)
# --------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('kind', KINDS)
def test_baseline_fit_matches_R_glm(kind):
    """bs2 x ns(logknots) lag 21, quasi-Poisson, dow + ns(date, 8/yr) on the full 14-year series: cross-basis block,
    dispersion-scaled vcov and nobs equal R's (all four ways of fitting the model)."""
    c = ctx('full')
    ref = c.r_fit()
    fit = py_fit(kind, c)
    assert ref.nobs == c.n - LAG
    assert_cb_block_matches(fit, ref, what=kind)
    assert int(r2np(ro.r['nobs'](fit.r_model))[0]) == ref.nobs


@pytest.mark.parametrize('kind', ('improved', 'rpy2'))
@pytest.mark.parametrize('family', ('poisson', 'quasipoisson', 'gaussian', 'Gamma'))
def test_baseline_families_match_R(kind, family):
    c = ctx('small')
    ref = c.r_fit(family=family)
    assert_cb_block_matches(py_fit(kind, c, family=family), ref, what=f'{kind}/{family}')


@pytest.mark.parametrize('kind', ('improved', 'dlnm'))
def test_baseline_missing_values_dropped_like_na_exclude(kind):
    """NaN in the response (scattered) on top of the lag-induced NaN rows: same rows dropped, same fit as R."""
    c = ctx('small')
    y = c.y['death'].copy()
    y[np.random.default_rng(20260929).choice(c.n, 25, replace=False)] = np.nan
    np2r('go_ynan', y)
    r(f'{c.df}$ynan <- go_ynan')
    ref = c.r_fit(y='ynan')
    fit = py_fit(kind, c, yvals=y)
    assert ref.nobs < c.n - LAG
    assert_cb_block_matches(fit, ref, what=kind)
    assert int(r2np(ro.r['nobs'](fit.r_model))[0]) == ref.nobs


def test_baseline_refit_is_reproducible():
    c = ctx('small')
    a, b = py_fit('improved', c), py_fit('improved', c)
    assert np.array_equal(a.cb_coef, b.cb_coef) and np.array_equal(a.cb_vcov, b.cb_vcov)


@pytest.mark.parametrize('dfseas, rows', [(8, 'full'), (2, 'full'), (3, 'small')])
def test_seasonal_basis_matches_R_ns_for_integer_dfseas(dfseas, rows):
    """glm-12 (plain part): create_seasonality_basis == ns(date, df=dfseas*length(unique(year))) for integer dfseas."""
    c = ctx(rows)
    with _quiet():
        basis = make_iface('improved', c).create_seasonality_basis(c.dates, dfseas)
    ref = rget(f'unclass(ns({c.df}$date, df={dfseas}*length(unique({c.df}$year))))')
    assert_close(basis, ref, rtol=1e-10, what=f'seasonal ns basis (dfseas={dfseas}, {rows})')


@pytest.mark.parametrize('as_numpy', (False, True))
def test_dow_dummies_equal_R_model_matrix_with_friday_reference(as_numpy):
    """glm-7 (plain part): the dummy matrix is R's model.matrix(~factor(weekday))[, -1] (alphabetical levels,
    Friday is the reference); only the LABELS attached in fit_dlnm_model are wrong."""
    c = ctx('small')
    dates = c.dates.values if as_numpy else c.dates
    with _quiet():
        dummies = make_iface('improved', c).create_dow_factors(dates)
    ref = rget(f'model.matrix(~dowf, {c.df})[, -1]')
    assert list(r(f'levels({c.df}$dowf)')) == sorted(WEEKDAYS)
    assert np.array_equal(np.asarray(dummies, dtype=float), ref)


def test_improved_crossreduce_and_module_reducer_match_R():
    """glm-13 (plain part): the Python reducer reproduces R's overall crossreduce(); by keyword and via the interface."""
    from crossreduce import crossreduce
    c = ctx('small')
    fit = py_fit('improved', c)
    c.r_fit()
    coef_r, vcov_r = c.r_reduce(CEN)
    for label, res in (('ImprovedGLMInterface.crossreduce', fit.iface.crossreduce(cen=CEN)),
                       ('crossreduce(cb, coef=, vcov=)', crossreduce(c.cb, coef=fit.cb_coef, vcov=fit.cb_vcov, cen=CEN))):
        coef_p, vcov_p = as_coef_vcov(res)
        assert_close(coef_p, coef_r, rtol=1e-8, what=f'{label} coef')
        assert_close(vcov_p, vcov_r, rtol=1e-8, what=f'{label} vcov')


def test_improved_crossreduce_unknown_type_raises():
    """R's match.arg() rejects an unknown reduction type; so must the Python reducer (glm-13, plain part)."""
    c = ctx('small')
    fit = py_fit('improved', c)
    with pytest.raises((ValueError, TypeError, KeyError)):
        fit.iface.crossreduce(cen=CEN, type='bogus')


def test_summary_right_after_fit_matches_R():
    """Neighbour of glm-2: right after a fit the summary table IS the R one (the defect needs a later fit)."""
    c = ctx('small')
    ref = c.r_fit()
    for kind in ('improved', 'rpy2'):
        fit = py_fit(kind, c)
        assert_close(fit.summary_table()[:, :2], ref.table[:, :2], rtol=1e-8, what=f'{kind} summary estimate/SE')


@pytest.mark.parametrize('label', ('DatetimeIndex', 'np.datetime64 array', 'tz-aware Series'))
def test_dates_input_types_accepted(label):
    """glm-10 (plain part): the date containers that already work give R's fit."""
    c = ctx('small')
    dates = {'DatetimeIndex': lambda: pd.DatetimeIndex(c.dates), 'np.datetime64 array': lambda: c.dates.values,
             'tz-aware Series': lambda: c.dates.dt.tz_localize('UTC')}[label]()
    assert_cb_block_matches(py_fit('improved', c, dates=dates), c.r_fit(), what=label)


@pytest.mark.parametrize('label', ('list', 'column vector'))
def test_rpy2_response_shapes_accepted(label):
    """glm-10 (plain part): Rpy2GLMInterface flattens the response (ImprovedGLMInterface does not, see below)."""
    c = ctx('small')
    y = list(c.y['death']) if label == 'list' else c.y['death'].reshape(-1, 1)
    assert_cb_block_matches(py_fit('rpy2', c, yvals=y), c.r_fit(), what=label)


def test_rpy2_covariates_without_names_and_as_dataframe():
    """glm-10 (plain part): unnamed covariates (formula_vars=None) and a DataFrame of covariates are accepted."""
    c = ctx('small')
    fit = py_fit('rpy2', c, other=pd.DataFrame(c.other), names=None)
    assert_cb_block_matches(fit, c.r_fit(), what='DataFrame covariates')


@pytest.mark.parametrize('label', ('int array', 'Series with shifted index'))
def test_improved_response_containers_that_already_work(label):
    """glm-10 (plain part): an integer array and a pandas Series with a non-default index are aligned by position."""
    c = ctx('small')
    y = c.y['death']
    yv = y.astype(np.int64) if label == 'int array' else pd.Series(y, index=np.arange(c.n) + 1000)
    assert_cb_block_matches(py_fit('improved', c, yvals=yv), c.r_fit(), what=f'y as {label}')


def test_inputs_are_not_modified_by_a_fit():
    c = ctx('small')
    y0, dates0, basis0 = c.y['death'].copy(), c.dates.copy(), np.array(c.cb.basis, copy=True)
    for kind in KINDS:
        py_fit(kind, c)
    assert np.array_equal(c.y['death'], y0) and c.dates.equals(dates0)
    assert np.array_equal(np.asarray(c.cb.basis), basis0, equal_nan=True)


@pytest.mark.parametrize('kind', ('improved', 'rpy2', 'dlnm'))
def test_cross_basis_block_of_an_earlier_fit_survives_a_later_fit(kind):
    """glm-2 (plain part): cb_coef/cb_vcov are extracted at fit time, so they stay right (only summary()/predict()
    read R globals)."""
    c = ctx('small')
    ref = c.r_fit()
    a = py_fit(kind, c)
    py_fit(kind, c, y='cvd')
    assert_cb_block_matches(a, ref, what=f'{kind} after a second fit')


@pytest.mark.parametrize('cen', (None, 5.0, CEN))
def test_overall_reduction_matches_R_for_any_centring(cen):
    """glm-13 (plain part): R's overall reduction does not depend on cen; neither does the Python reducer."""
    from crossreduce import crossreduce
    c = ctx('small')
    c.r_fit()
    coef_r, vcov_r = c.r_reduce(cen)
    fit = py_fit('improved', c)
    coef_p, vcov_p = as_coef_vcov(crossreduce(c.cb, coef=fit.cb_coef, vcov=fit.cb_vcov, cen=cen))
    assert_close(coef_p, coef_r, rtol=1e-8, what='reduced coef')
    assert_close(vcov_p, vcov_r, rtol=1e-8, what='reduced vcov')


@pytest.mark.parametrize('region', ('London', 'Wales'))
def test_england_wales_validated_path_matches_R(region):
    """The validated route of the README (England & Wales, bs2 x ns lag 21, quasi-Poisson, dow + ns(date, 8/yr)):
    ImprovedGLMInterface cross-basis block and crossreduce() equal R's for two of the ten regions."""
    from basis import CrossBasis
    from crossreduce import crossreduce  # noqa: F401  (import path used by ImprovedGLMInterface.crossreduce)
    from utils import logknots
    csv = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       '2015_gasparrini_Lancet_Rcodedata-master', 'regEngWales.csv')
    if not os.path.exists(csv):
        pytest.skip('England & Wales data not available')
    r(f'go_ew <- read.csv("{csv}", row.names=1); go_ew <- go_ew[go_ew$regnames == "{region}", ]; '
      f'go_ew$date <- as.Date(go_ew$date); '
      f'go_ew_kv <- quantile(go_ew$tmean, c(.10, .75, .90)); '
      f'go_ew_cb <- crossbasis(go_ew$tmean, lag={LAG}, argvar=list(fun="bs", degree=2, knots=go_ew_kv), '
      f'arglag=list(knots=logknots({LAG}, 3))); '
      f'go_ew_m <- glm(death ~ go_ew_cb + dow + ns(date, df=8*length(unique(year))), go_ew, '
      f'family=quasipoisson, na.action="na.exclude"); '
      f'go_ew_red <- crossreduce(go_ew_cb, go_ew_m, cen=mean(go_ew$tmean))')
    tmean, death = rget('go_ew$tmean'), rget('as.numeric(go_ew$death)')
    dates = pd.Series(pd.to_datetime(rget('as.numeric(go_ew$date)'), unit='D'))
    with _quiet():
        cb = CrossBasis(tmean, lag=LAG, argvar={'fun': 'bs', 'degree': 2, 'knots': rget('go_ew_kv')},
                        arglag={'fun': 'ns', 'knots': logknots([0, LAG], nk=3)})
        from improved_glm import ImprovedGLMInterface
        g = ImprovedGLMInterface(cb)
        g.fit_dlnm_model(death, dates, dfseas=DFSEAS)
    idx = 'grep("^go_ew_cb", names(coef(go_ew_m)))'
    assert_close(g.cb_coef, rget(f'coef(go_ew_m)[{idx}]'), rtol=1e-8, what=f'{region} cb coef')
    assert_close(g.cb_vcov, rget(f'vcov(go_ew_m)[{idx}, {idx}]'), rtol=1e-8, what=f'{region} cb vcov')
    coef_p, vcov_p = as_coef_vcov(g.crossreduce(cen=float(np.mean(tmean))))
    assert_close(coef_p, rget('go_ew_red$coef'), rtol=1e-8, what=f'{region} reduced coef')
    assert_close(vcov_p, rget('go_ew_red$vcov'), rtol=1e-8, what=f'{region} reduced vcov')


def _aliasing_fixture(lag_hi):
    """lag [1, lag_hi], ns(df=4) variable basis, lag basis ns(3 interior knots) + intercept = 5 columns: aliased when
    only 4 lags exist (lag_hi=4), full rank for 5 lags (lag_hi=5).  Knots come from R's logknots()."""
    from basis import CrossBasis
    c = ctx('small')
    lk = rget(f'logknots(c(1, {lag_hi}), 3)')
    np2r(f'go_lk{lag_hi}', lk)
    name = f'go_alias{lag_hi}_cb'
    r(f'{name} <- crossbasis({c.df}$temp, lag=c(1, {lag_hi}), argvar=list(fun="ns", df=4), '
      f'arglag=list(fun="ns", knots=go_lk{lag_hi}))')
    with _quiet():
        cb = CrossBasis(c.temp, lag=[1, lag_hi], argvar={'fun': 'ns', 'df': 4}, arglag={'fun': 'ns', 'knots': lk})
    assert_close(np.asarray(cb.basis), rget(f'unclass({name})'), rtol=1e-12, what='test setup: cross-basis')
    return c, cb, name


def test_full_rank_lag_design_has_no_missing_coefficients():
    """glm-11 (plain part): with as many lags as lag-basis columns nothing is aliased; fit and crossreduce equal R."""
    c, cb, name = _aliasing_fixture(5)
    ref = c.r_fit(cb=name)
    assert not np.isnan(ref.coef).any()
    with _quiet():
        from improved_glm import ImprovedGLMInterface
        g = ImprovedGLMInterface(cb)
        g.fit_dlnm_model(c.y['death'], c.dates, dfseas=DFSEAS)
    assert_close(g.cb_coef, ref.coef, rtol=1e-8, what='cb coef')
    coef_r, vcov_r = c.r_reduce(CEN, cb=name)
    coef_p, vcov_p = as_coef_vcov(g.crossreduce(cen=CEN))
    assert_close(coef_p, coef_r, rtol=1e-8, what='reduced coef')
    assert_close(vcov_p, vcov_r, rtol=1e-8, what='reduced vcov')


# --------------------------------------------------------------------------------------------------------------
# glm-1: weights / offset / subset / unknown keywords
# --------------------------------------------------------------------------------------------------------------
def _kwarg_case(c, label):
    """(python kwargs, R reference fit) for a documented glm() argument; guards against a vacuous test by requiring
    R's answer to differ from the plain fit."""
    rng = np.random.default_rng({'weights': 3, 'offset': 4, 'subset': 5}[label])
    plain = c.r_fit()
    if label == 'weights':
        w = rng.uniform(0.5, 2.0, c.n)
        kw, ref = dict(weights=w), c.r_fit(weights=w)
    elif label == 'offset':
        off = np.log(rng.uniform(0.8, 1.25, c.n))
        kw, ref = dict(offset=off), c.r_fit(offset=off)
    else:
        sub = np.arange(c.n) < int(0.6 * c.n)
        kw, ref = dict(subset=sub), c.r_fit(subset=sub)
    assert max_abs_diff(ref.coef, plain.coef) > 1e-3 or ref.nobs != plain.nobs, 'R: the argument changes nothing'
    return kw, ref


@known_defect(THEME, 'glm-1', note='kwargs are never interpolated into the R glm() call')
@pytest.mark.parametrize('kind', KINDS)
@pytest.mark.parametrize('label', ('weights', 'offset', 'subset'))
def test_glm_arguments_are_honoured_like_R(kind, label):
    c = ctx('small')
    kw, ref = _kwarg_case(c, label)
    fit = py_fit(kind, c, **kw)
    assert int(r2np(ro.r['nobs'](fit.r_model))[0]) == ref.nobs, f'{kind}: nobs with {label}'
    assert_cb_block_matches(fit, ref, what=f'{kind} {label}=')


@known_defect(THEME, 'glm-1', note='an invented keyword is accepted and ignored')
@pytest.mark.parametrize('kind', KINDS)
def test_unknown_glm_argument_is_rejected_like_R(kind):
    c = ctx('small')
    with pytest.raises(Exception, match='unused argument'):        # R: glm(..., bogus_kw=1) stops
        r(f'glm(death ~ {c.cbr}, data={c.df}, bogus_kw=1)')
    with pytest.raises((TypeError, ValueError)):
        py_fit(kind, c, bogus_kw=1)


# --------------------------------------------------------------------------------------------------------------
# glm-2: results must come from the interface's own R model, not from the R global name `fitted_model`
# --------------------------------------------------------------------------------------------------------------
def _report(fit):
    """What the interface reports about its own model: coefficient table (summary) or fitted values (predict)."""
    if fit.kind == 'dlnm':
        p = np.asarray(fit.iface.predict(), dtype=float)
        return p[~np.isnan(p)]
    return fit.summary_table()[:, :2]


@known_defect(THEME, 'glm-2', note='summary()/fitted() evaluated by the global name fitted_model')
@pytest.mark.parametrize('scenario', ('second_fit', 'user_variable'))
@pytest.mark.parametrize('kind', ('improved', 'rpy2', 'dlnm'))
def test_earlier_interface_is_not_disturbed_by_later_R_state(kind, scenario):
    c = ctx('small')
    ref = c.r_fit()                                    # the model interface A must keep reporting
    expected = ref.fitted[~np.isnan(ref.fitted)] if kind == 'dlnm' else ref.table[:, :2]
    a = py_fit(kind, c)
    assert_close(_report(a), expected, rtol=1e-8, what=f'{kind}: report right after the fit')
    try:
        if scenario == 'second_fit':
            py_fit(kind, c, y='cvd')                   # another model, same class, fitted afterwards
        else:
            r('fitted_model <- "a user variable"')     # the interface must not read R names it does not own
        assert_close(_report(a), expected, rtol=1e-8, what=f'{kind}: report after {scenario}')
    finally:
        r('if (exists("fitted_model", envir=globalenv())) rm("fitted_model", envir=globalenv())')


# --------------------------------------------------------------------------------------------------------------
# glm-3 / crossreduce-1 / crossreduce-2: crossreduce through the interface objects
# --------------------------------------------------------------------------------------------------------------
@known_defect(THEME, 'glm-3', 'crossreduce-1', note='Rpy2/DLNMGLMInterface.crossreduce never return a result')
@pytest.mark.parametrize('has_dlnm', (True, False))
@pytest.mark.parametrize('kind', ('rpy2', 'dlnm'))
def test_interface_crossreduce_method_matches_R(kind, has_dlnm):
    c = ctx('small')
    ref_fit = c.r_fit()
    coef_r, vcov_r = c.r_reduce(CEN)
    fit = py_fit(kind, c)
    assert_cb_block_matches(fit, ref_fit, what='fit')
    fit.rpy2_iface.has_dlnm = has_dlnm                  # False = the "R package dlnm not importable" branch
    coef_p, vcov_p = as_coef_vcov(fit.iface.crossreduce(cen=CEN))
    assert_close(coef_p, coef_r, rtol=1e-8, what='reduced coef')
    assert_close(vcov_p, vcov_r, rtol=1e-8, what='reduced vcov')


@known_defect(THEME, 'glm-3', 'crossreduce-1', note='cen=None path of the R-object branch')
def test_interface_crossreduce_without_cen_matches_R():
    c = ctx('small')
    c.r_fit()
    coef_r, vcov_r = c.r_reduce(None)
    fit = py_fit('rpy2', c)
    coef_p, vcov_p = as_coef_vcov(fit.iface.crossreduce())
    assert_close(coef_p, coef_r, rtol=1e-8, what='reduced coef')
    assert_close(vcov_p, vcov_r, rtol=1e-8, what='reduced vcov')


@known_defect(THEME, 'glm-3', 'crossreduce-1', note='type is never forwarded; the method crashes before validating it')
@pytest.mark.parametrize('kind', ('rpy2', 'dlnm'))
def test_interface_crossreduce_unknown_type_is_rejected_like_R(kind):
    c = ctx('small')
    fit = py_fit(kind, c)
    with pytest.raises((ValueError, TypeError, KeyError)):      # R: match.arg(type, c("overall","var","lag")) stops
        fit.iface.crossreduce(cen=CEN, type='bogus')


@known_defect(THEME, 'crossreduce-2', 'glm-3', note='module-level crossreduce() rejects the interface objects')
@pytest.mark.parametrize('kind', ('improved', 'rpy2', 'dlnm'))
def test_module_crossreduce_accepts_fitted_interface(kind):
    from crossreduce import crossreduce
    c = ctx('small')
    c.r_fit()
    coef_r, vcov_r = c.r_reduce(CEN)
    fit = py_fit(kind, c)
    coef_p, vcov_p = as_coef_vcov(crossreduce(fit.iface, cen=CEN))
    assert_close(coef_p, coef_r, rtol=1e-8, what='reduced coef')
    assert_close(vcov_p, vcov_r, rtol=1e-8, what='reduced vcov')


# --------------------------------------------------------------------------------------------------------------
# glm-6: family names
# --------------------------------------------------------------------------------------------------------------
@known_defect(THEME, 'glm-6', note="documented family 'gamma' is evaluated as base::gamma in R")
@pytest.mark.parametrize('kind', ('improved', 'rpy2', 'dlnm'))
def test_documented_family_gamma_gives_R_Gamma_fit(kind):
    c = ctx('small')
    with pytest.raises(Exception, match='argument'):                # R itself has no family called `gamma`
        r(f'glm(death ~ {c.cbr}, data={c.df}, family=gamma)')
    ref = c.r_fit(family='Gamma')
    assert_cb_block_matches(py_fit(kind, c, family='gamma'), ref, what=f'{kind} family="gamma"')


@known_defect(THEME, 'glm-6', note='the family string is pasted into R source (injection point)')
@pytest.mark.parametrize('kind', ('improved', 'rpy2'))
def test_family_string_with_extra_R_arguments_is_rejected(kind):
    """R's family argument is an object or a NAME: a string carrying further glm() arguments is an error there
    (get("poisson, weights=...") fails); PyDLNM pastes it into the call and silently fits a weighted model."""
    c = ctx('small')
    evil = 'poisson, weights=rep(c(1, 2), length.out=nrow(model_data))'
    with pytest.raises(Exception, match='not found'):
        r(f'glm(death ~ {c.cbr}, data={c.df}, family=get("{evil}"))')
    with pytest.raises((ValueError, TypeError)):
        py_fit(kind, c, family=evil)


# --------------------------------------------------------------------------------------------------------------
# glm-7: day-of-week labels
# --------------------------------------------------------------------------------------------------------------
@known_defect(THEME, 'glm-7', note='static labels dowTuesday..dowSunday on columns Monday, Saturday, Sunday, ...')
@pytest.mark.parametrize('rows', ('small', 'wkdays'))
def test_dow_coefficient_labels_name_their_weekday(rows):
    """Each coefficient labelled dow<Day> is R's effect of <Day> (reference = first level alphabetically)."""
    c = ctx(rows)
    ref = c.r_fit()
    ref_dow = {n[len('dowf'):]: v for n, v in zip(ref.names, ref.all_coef) if n.startswith('dowf')}
    fit = py_fit('improved', c)
    names = list(ro.r['names'](ro.r['coef'](fit.r_model)))
    dow = {n[len('dow'):]: v for n, v in zip(names, r2np(ro.r['coef'](fit.r_model))) if n.startswith('dow')}
    assert set(dow) == set(ref_dow), f'labels {sorted(dow)} vs R {sorted(ref_dow)}'
    for day, value in ref_dow.items():
        assert_close(dow[day], value, rtol=1e-8, what=f'coefficient labelled dow{day}')


@known_defect(THEME, 'glm-7', note='the printed reference level is the first date\'s weekday, not the alphabetical first')
def test_dow_reference_level_message_names_the_dropped_weekday(capsys):
    c = ctx('small')                                              # first date is a Thursday; R's reference is Friday
    ref_level = str(r(f'levels({c.df}$dowf)[1]')[0])
    assert c.dates.dt.day_name().iloc[0] != ref_level
    make_iface('improved', c).create_dow_factors(c.dates)
    out = capsys.readouterr().out
    assert f'reference: {ref_level}' in out, out.strip().splitlines()[-2:]


# --------------------------------------------------------------------------------------------------------------
# glm-8: na.exclude padding
# --------------------------------------------------------------------------------------------------------------
@known_defect(THEME, 'glm-8', note='predict() has one value per kept row; R pads the excluded rows with NA')
@pytest.mark.parametrize('nan_source', ('lag_rows', 'scattered_response'))
def test_predict_is_padded_to_the_input_length_like_na_exclude(nan_source):
    c = ctx('small')
    y = c.y['death'].copy()
    if nan_source == 'scattered_response':
        y[np.random.default_rng(11).choice(np.arange(LAG, c.n), 30, replace=False)] = np.nan
    np2r('go_ynan', y)
    r(f'{c.df}$ynan <- go_ynan')
    ref = c.r_fit(y='ynan')
    assert len(ref.fitted) == c.n and np.isnan(ref.fitted).sum() >= LAG
    pred = np.asarray(py_fit('dlnm', c, yvals=y).iface.predict(), dtype=float)
    assert_close(pred, ref.fitted, rtol=1e-8, what='predict() vs fitted() with na.exclude')


# --------------------------------------------------------------------------------------------------------------
# glm-10: input handling
# --------------------------------------------------------------------------------------------------------------
@known_defect(THEME, 'glm-10', note='ImprovedGLMInterface does not coerce y')
@pytest.mark.parametrize('label', ('list', 'column vector', 'DataFrame column'))
def test_improved_accepts_response_shapes_like_a_vector(label):
    c = ctx('small')
    y = c.y['death']
    yv = {'list': list(y), 'column vector': y.reshape(-1, 1), 'DataFrame column': pd.DataFrame({'death': y})}[label]
    assert_cb_block_matches(py_fit('improved', c, yvals=yv), c.r_fit(), what=f'y as {label}')


@known_defect(THEME, 'glm-10', note='dates given as list / datetime.date objects crash')
@pytest.mark.parametrize('label', ('list of datetime', 'Series of datetime.date'))
def test_improved_accepts_date_containers(label):
    c = ctx('small')
    dates = list(c.dates.dt.to_pydatetime()) if label == 'list of datetime' else c.dates.dt.date
    assert_cb_block_matches(py_fit('improved', c, dates=dates), c.r_fit(), what=label)


BAD_NAMES = {
    'cb. prefix': lambda names: ['cb.foo'] + names[1:],
    'name of the response': lambda names: ['death'] + names[1:],
    'name of a cross-basis column': lambda names: ['cb.v1.l1'] + names[1:],
    'duplicate names': lambda names: ['o0', 'o0'] + names[2:],
    'name with a space': lambda names: ['pm 10'] + names[1:],
    'name starting with a digit': lambda names: ['1st'] + names[1:],
}


@known_defect(THEME, 'glm-10', note='formula_vars are used raw as data-frame keys and formula terms')
@pytest.mark.parametrize('label', sorted(BAD_NAMES))
def test_covariate_names_do_not_change_the_cross_basis_fit(label):
    """The names of the covariates cannot change the cross-basis estimates in R.  PyDLNM must either reject a
    name it cannot use safely (ValueError/TypeError) or return exactly R's cross-basis fit."""
    c = ctx('small')
    ref = c.r_fit()
    names = BAD_NAMES[label](c.other_names)
    try:
        fit = py_fit('rpy2', c, names=names)
    except (ValueError, TypeError):
        return
    assert_cb_block_matches(fit, ref, what=label)


@known_defect(THEME, 'glm-10', note='wrong number of formula_vars gives an obscure R "object not found"')
def test_wrong_number_of_covariate_names_is_rejected_clearly():
    c = ctx('small')
    with pytest.raises((ValueError, TypeError)):
        py_fit('rpy2', c, names=c.other_names[:-1])


# --------------------------------------------------------------------------------------------------------------
# glm-11: aliased coefficients
# --------------------------------------------------------------------------------------------------------------
@known_defect(THEME, 'glm-11', note='aliased cb coefficients become NaN in cb_coef/cb_vcov without warning or error')
@pytest.mark.parametrize('kind', ('improved', 'rpy2'))
def test_aliased_cross_basis_coefficients_are_rejected_like_R(kind):
    from improved_glm import ImprovedGLMInterface
    from rpy2_glm import Rpy2GLMInterface
    c, cb, name = _aliasing_fixture(4)
    ref = c.r_fit(cb=name)                                          # R glm() itself tolerates aliasing ...
    assert np.isnan(ref.coef).sum() == 4
    with pytest.raises(Exception, match='do not consistent'):       # ... crossreduce() does not
        c.r_reduce(CEN, cb=name)

    def fit_and_reduce():
        with _quiet():
            if kind == 'improved':
                g = ImprovedGLMInterface(cb)
                g.fit_dlnm_model(c.y['death'], c.dates, dfseas=DFSEAS)
            else:
                g = Rpy2GLMInterface(cb)
                g.fit_glm(c.y['death'], other_vars=c.other, formula_vars=c.other_names)
            g.crossreduce(cen=CEN)

    _expect_rejection(fit_and_reduce, f'{kind}: aliased design')
