"""Gap-fill: numeric / timezone-aware `dates` silently misread by improved_glm._as_datetime_series.

Critic gap [numeric_dates_silently_misread]: ImprovedGLMInterface.fit_dlnm_model / create_dow_factors /
create_seasonality_basis / fit_enhanced_dlnm_model / MultiLocationDLNM.add_region_analysis turn `dates` into
datetime64 with pd.to_datetime, which reads INTEGERS AS NANOSECONDS SINCE 1970.  A numeric date vector (R Date day
numbers from as.numeric(Date) / chicagoNMMAPS$date through rpy2, Python ordinals, Excel serials, yyyymmdd integers,
epoch seconds/ms) then falls into one calendar year and on one weekday, so the model is fitted silently with
dfseas*1 seasonal df instead of dfseas*n_years and without the day-of-week dummies.  Separately a tz-aware date
vector crossing a DST change has 23/25-hour days, i.e. fractional-day spacing in the seasonal ns() basis.

The reference is R's own recipe on the SAME calendar days as a proper `Date` (00.prepdata.R / 01.firststage.R of the
Lancet code):
    glm(death ~ cb + dowf + ns(date, df = dfseas * length(unique(year))), family = quasipoisson, na.action = na.exclude)
R never takes numeric dates in this recipe: weekdays(<numeric>) stops ("no applicable method"), and the year count
comes from format(date, "%Y").  So there are exactly two R-faithful outcomes for a numeric date vector, and the
tests accept either:
  * a loud failure (any exception), or
  * the fit / basis / dummies that R gives for the dates the numbers encode (the `truth`, as.Date()).
A SILENT fit that differs from R is the defect.  The 'epoch nanoseconds' case is the one numeric encoding that
pandas reads correctly, so it must keep giving R's numbers (or be rejected).

Defects (one root cause, improved_glm.py:_as_datetime_series):
  * numeric day numbers / ordinals / yyyymmdd / Excel serials / epoch s / epoch ms: pd.to_datetime(int) == ns
    (-> tests marked known_defect('GAP', 'numeric_dates_silently_misread'))
  * tz-aware dates across a DST change: (dates - dates.min()) / Timedelta(days=1) is fractional
    (-> the DST tests are marked known_defect as well; UTC and Asia/Kolkata, which have no DST, are plain tests)

All cases use 3 full calendar years (1987-1989, n = 1096) of R's chicagoNMMAPS, bs2 x ns(logknots) lag 21,
dfseas = 6, quasi-Poisson, so that n_years = 3 distinguishes the correct reading (18 seasonal df + 6 weekday dummies)
from the 1970 misreading (6 seasonal df, no dummies).
"""
import contextlib
import io
import os

import numpy as np
import pandas as pd
import pytest

from rhelpers import assert_close, chicago, known_defect, max_rel_diff, np2r, r, r2np, rget
import rpy2.robjects as ro

GAP = 'GAP'
KEY = 'numeric_dates_silently_misread'
LAG = 21
DFSEAS = 6
N_ROWS = 1096                     # 1987-01-01 .. 1989-12-31 (three complete calendar years, 1988 is a leap year)
N_YEARS = 3
WEEKDAYS = ('Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday')
RTOL = 1e-8


# --------------------------------------------------------------------------------------------------------------
# harness workarounds shared with the other test modules (audit finding Q2: PyDLNM rewrites R_HOME)
# --------------------------------------------------------------------------------------------------------------
def _warm_up_r_lapack():
    os.environ['R_HOME'] = os.path.dirname(str(r('.Library')[0]))      # the R_HOME R itself started with
    r('invisible(chol2inv(chol(diag(2) + 1))); invisible(solve(diag(2)))')


_warm_up_r_lapack()
_SENTINEL_R_HOME = os.environ['R_HOME']


@pytest.fixture(autouse=True)
def _keep_process_r_home():
    os.environ['R_HOME'] = _SENTINEL_R_HOME
    yield
    os.environ['R_HOME'] = _SENTINEL_R_HOME


@pytest.fixture(autouse=True, scope='module')
def _memoise_importr():
    """Speed only: every interface constructor calls importr() (about 0.7 s); the wrappers are identical each time."""
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


@contextlib.contextmanager
def _quiet():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


# --------------------------------------------------------------------------------------------------------------
# the scenario: one R model on proper Dates, one Python cross-basis on the same exposure series
# --------------------------------------------------------------------------------------------------------------
class Scenario:
    def __init__(self):
        from basis import CrossBasis
        from utils import logknots
        ch = chicago()
        rows = slice(0, N_ROWS)
        self.days = ch['date'][rows].astype('int64')                   # R Date day numbers (days since 1970-01-01)
        self.temp = ch['temp'][rows].astype(float)
        self.death = ch['death'][rows].astype(float)
        self.n = len(self.days)
        self.truth = pd.Series(pd.to_datetime(self.days, unit='D'))   # the calendar days the numbers encode
        assert self.truth.iloc[0] == pd.Timestamp('1987-01-01') and self.truth.iloc[-1] == pd.Timestamp('1989-12-31')
        assert self.truth.dt.year.nunique() == N_YEARS

        np2r('nd_days', self.days)
        np2r('nd_temp', self.temp)
        np2r('nd_death', self.death)
        wd = ', '.join(f'"{d}"' for d in WEEKDAYS)
        r(f'''
        nd_date <- as.Date(nd_days, origin = "1970-01-01")
        nd_df <- data.frame(date = nd_date, death = nd_death, temp = nd_temp)
        nd_df$year <- as.integer(format(nd_df$date, "%Y"))
        nd_df$dowf <- factor(c({wd})[as.POSIXlt(nd_df$date)$wday + 1L])
        nd_kv <- quantile(nd_df$temp, c(.10, .75, .90))
        nd_cb <- crossbasis(nd_df$temp, lag = {LAG}, argvar = list(fun = "bs", degree = 2, knots = nd_kv),
                            arglag = list(fun = "ns", knots = logknots({LAG}, 3)))
        nd_m <- glm(death ~ nd_cb + dowf + ns(date, df = {DFSEAS} * length(unique(year))), data = nd_df,
                    family = quasipoisson, na.action = na.exclude)
        nd_red <- crossreduce(nd_cb, nd_m, cen = mean(nd_df$temp))
        ''')
        kv = rget('nd_kv')
        with _quiet():
            self.cb = CrossBasis(self.temp, lag=LAG, argvar={'fun': 'bs', 'degree': 2, 'knots': kv},
                                 arglag={'fun': 'ns', 'knots': logknots([0, LAG], nk=3)})
        assert_close(np.asarray(self.cb.basis), rget('unclass(nd_cb)'), rtol=1e-12, what='test setup: cross-basis')

        idx = 'grep("^nd_cb", names(coef(nd_m)))'
        self.ref_coef = rget(f'coef(nd_m)[{idx}]')
        self.ref_vcov = rget(f'vcov(nd_m)[{idx}, {idx}]')
        self.ref_all_coef = rget('coef(nd_m)')
        self.ref_red_coef, self.ref_red_vcov = rget('nd_red$coef'), rget('nd_red$vcov')
        self.ref_seasonal = rget(f'unclass(ns(nd_df$date, df = {DFSEAS} * length(unique(nd_df$year))))')
        self.ref_dow = rget('model.matrix(~dowf, nd_df)[, -1]')
        assert self.ref_dow.shape == (self.n, 6) and self.ref_seasonal.shape == (self.n, DFSEAS * N_YEARS)
        assert len(self.ref_all_coef) == 1 + self.ref_coef.size + 6 + DFSEAS * N_YEARS
        self.cen = float(np.mean(self.temp))


_SC = []


def sc():
    if not _SC:
        _SC.append(Scenario())
    return _SC[0]


def fit_improved(dates):
    """ImprovedGLMInterface.fit_dlnm_model on the scenario; returns the interface."""
    from improved_glm import ImprovedGLMInterface
    s = sc()
    g = ImprovedGLMInterface(s.cb)
    with _quiet():
        g.fit_dlnm_model(s.death, dates, dfseas=DFSEAS)
    return g


def assert_fit_is_R(g, what, rtol=RTOL):
    """The fitted interface equals R's glm on proper Dates: cross-basis block, the whole coefficient vector."""
    s = sc()
    all_coef = r2np(ro.r['coef'](g.r_model))
    assert all_coef.shape == s.ref_all_coef.shape, (
        f'{what}: {all_coef.size} coefficients in Python vs {s.ref_all_coef.size} in R '
        f'(day-of-week dummies: {getattr(g, "dow_columns", None)}) -- the model was assembled from misread dates')
    assert_close(g.cb_coef, s.ref_coef, rtol=rtol, what=f'{what}: cross-basis coef')
    assert_close(g.cb_vcov, s.ref_vcov, rtol=rtol, what=f'{what}: cross-basis vcov')
    assert_close(all_coef, s.ref_all_coef, rtol=rtol, what=f'{what}: all coefficients')


def raise_or_match(make, check, what):
    """`make()` must either stop with an exception (a loud failure is R-faithful for numeric dates) or return
    something `check` accepts as R's answer.  Anything else (a silent, different answer) fails."""
    try:
        got = make()
    except Exception:          # noqa: BLE001 -- any loud failure counts as a rejection
        return 'rejected'
    check(got, what)
    return 'matches R'


# --------------------------------------------------------------------------------------------------------------
# numeric encodings of the same 1096 calendar days
# --------------------------------------------------------------------------------------------------------------
def encodings():
    s = sc()
    idx = pd.DatetimeIndex(s.truth)
    return {
        'R day numbers int64': s.days.astype('int64'),
        'R day numbers float64': s.days.astype(float),
        'Python ordinals': np.array([t.toordinal() for t in idx], dtype='int64'),
        'yyyymmdd integers': np.array(idx.strftime('%Y%m%d'), dtype='int64'),
        'Excel serials': s.days.astype('int64') + 25569,
        'epoch seconds': s.days.astype('int64') * 86400,
        'epoch milliseconds': s.days.astype('int64') * 86400 * 1000,
        'epoch nanoseconds': s.days.astype('int64') * 86400 * 10 ** 9,
    }


def _as_param_marks(mark):
    """pytest.param(marks=...) cannot take the no-op usefixtures() mark that known_defect() returns when
    PYDLNM_XFAIL_OFF=1 is set: use an empty list then."""
    return [] if getattr(mark, 'name', '') == 'usefixtures' else [mark]


DEFECT_NUMERIC = known_defect(GAP, KEY, note='pd.to_datetime reads integers as nanoseconds since 1970: one year, one weekday')
NUMERIC_MARKS = _as_param_marks(DEFECT_NUMERIC)
NUMERIC_KINDS = [
    pytest.param(k, marks=NUMERIC_MARKS) if k != 'epoch nanoseconds' else k
    for k in ('R day numbers int64', 'R day numbers float64', 'Python ordinals', 'yyyymmdd integers', 'Excel serials',
              'epoch seconds', 'epoch milliseconds', 'epoch nanoseconds')
]
CONTAINERS = {
    'ndarray': lambda v: v,
    'Series': lambda v: pd.Series(v),
    'list': lambda v: [x.item() for x in v],
    'object Series': lambda v: pd.Series(list(v), dtype=object),
    'nullable Int64 Series': lambda v: pd.Series(np.asarray(v).astype('int64'), dtype='Int64'),
}


# --------------------------------------------------------------------------------------------------------------
# 0. reference behaviour and baseline guards (faithful today)
# --------------------------------------------------------------------------------------------------------------
def test_R_itself_refuses_numeric_dates_for_the_weekday_dummies():
    """The R recipe has no numeric-date path: weekdays() stops on a number, so 'reject' is the R-faithful behaviour."""
    sc()                                              # defines nd_days / nd_date in R
    with pytest.raises(Exception, match='(?i)no applicable method|weekdays'):
        r('weekdays(nd_days)')                        # nd_days = as.numeric(Date)
    r('invisible(weekdays(nd_date))')                 # ... while the Date itself is fine


def test_baseline_datetime_dates_give_R_fit():
    """Proper datetimes (the README route): every coefficient (cross-basis block, dummies, seasonal) equals R's."""
    g = fit_improved(sc().truth)
    assert g.dow_columns == [d for d in sorted(WEEKDAYS) if d != 'Friday']      # Friday is the reference level
    assert_fit_is_R(g, 'datetime64 Series')


def test_epoch_nanosecond_integers_are_the_one_numeric_encoding_pandas_reads_right():
    """Integers ARE datetimes when they are epoch nanoseconds (pandas' own convention): either rejected or R's fit."""
    v = encodings()['epoch nanoseconds']
    raise_or_match(lambda: fit_improved(v), lambda g, w: assert_fit_is_R(g, w), 'epoch nanoseconds')


# --------------------------------------------------------------------------------------------------------------
# 1. ImprovedGLMInterface.fit_dlnm_model with numeric dates: R's fit or an error, never a silent different model
# --------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('kind', NUMERIC_KINDS)
def test_fit_dlnm_model_numeric_dates_rejected_or_equal_to_R(kind):
    v = encodings()[kind]
    raise_or_match(lambda: fit_improved(v), lambda g, w: assert_fit_is_R(g, w), kind)


@pytest.mark.parametrize('container', [pytest.param(c, marks=NUMERIC_MARKS) for c in
                                       ('Series', 'list', 'object Series', 'nullable Int64 Series')])
def test_fit_dlnm_model_R_day_numbers_in_any_container(container):
    """R Date day numbers (as.numeric(date), the way chicagoNMMAPS$date arrives through rpy2) in every container."""
    v = CONTAINERS[container](encodings()['R day numbers int64'])
    raise_or_match(lambda: fit_improved(v), lambda g, w: assert_fit_is_R(g, w), f'R day numbers in {container}')


@pytest.mark.parametrize('kind', [pytest.param(k, marks=NUMERIC_MARKS)
                                  for k in ('R day numbers int64', 'Python ordinals', 'yyyymmdd integers')])
def test_create_seasonality_basis_numeric_dates_rejected_or_equal_to_R(kind):
    """ns(date, df = dfseas * length(unique(year))): with day numbers the year count must still be 3."""
    from improved_glm import ImprovedGLMInterface
    v, s = encodings()[kind], sc()
    iface = ImprovedGLMInterface(s.cb)

    def make():
        with _quiet():
            return iface.create_seasonality_basis(v, DFSEAS)

    raise_or_match(make, lambda b, w: assert_close(b, s.ref_seasonal, rtol=1e-10, what=w), f'seasonal basis, {kind}')


@pytest.mark.parametrize('kind', [pytest.param(k, marks=NUMERIC_MARKS)
                                  for k in ('R day numbers int64', 'Python ordinals', 'yyyymmdd integers')])
def test_create_dow_factors_numeric_dates_rejected_or_equal_to_R(kind):
    """model.matrix(~factor(weekday))[, -1]: with day numbers the dummies must not collapse to zero columns."""
    from improved_glm import ImprovedGLMInterface
    v, s = encodings()[kind], sc()
    iface = ImprovedGLMInterface(s.cb)

    def make():
        with _quiet():
            return iface.create_dow_factors(v, verbose=False)

    raise_or_match(make, lambda d, w: assert_close(d, s.ref_dow, rtol=0, what=w), f'weekday dummies, {kind}')


@known_defect(GAP, KEY, note='_as_datetime_series(np.arange(10957, 10967)) == 1970-01-01 00:00:00.000010957')
def test_as_datetime_series_R_day_numbers_2000_01_01():
    """The helper itself: 10957 is R's as.Date(10957) == 2000-01-01.  Reading it as 10957 ns is never right."""
    from improved_glm import _as_datetime_series
    want = r2np(ro.r('as.numeric(as.Date(10957:10966, origin = "1970-01-01"))'))

    def check(got, what):
        got_days = (got.to_numpy('datetime64[ns]').astype('int64') / 86400e9)
        assert_close(got_days, want, rtol=0, what=what)

    raise_or_match(lambda: _as_datetime_series(np.arange(10957, 10967)), check, '_as_datetime_series(R day numbers)')


# --------------------------------------------------------------------------------------------------------------
# 2. the wrappers that forward `dates`: fit_enhanced_dlnm_model and MultiLocationDLNM.add_region_analysis
# --------------------------------------------------------------------------------------------------------------
def _check_enhanced(res, what):
    s = sc()
    assert_close(res['coefficients'], s.ref_coef, rtol=RTOL, what=f'{what}: coefficients')
    assert_close(res['vcov'], s.ref_vcov, rtol=RTOL, what=f'{what}: vcov')
    assert_close(res['reduced']['coefficients'], s.ref_red_coef, rtol=RTOL, what=f'{what}: reduced coef')
    assert_close(res['reduced']['vcov'], s.ref_red_vcov, rtol=RTOL, what=f'{what}: reduced vcov')


def _enhanced(dates):
    from improved_glm import fit_enhanced_dlnm_model
    s = sc()
    with _quiet():
        return fit_enhanced_dlnm_model(s.cb, s.death, dates, dfseas=DFSEAS)


def _region(dates):
    from multi_location import MultiLocationDLNM
    s = sc()
    ml = MultiLocationDLNM()
    with _quiet():
        return ml.add_region_analysis('region-A', s.cb, s.death, dates, dfseas=DFSEAS)


def test_baseline_fit_enhanced_and_add_region_with_datetime_dates_match_R():
    """Plain guard: the wrappers equal R's fit and R's overall crossreduce() for proper datetimes."""
    s = sc()
    _check_enhanced(_enhanced(s.truth), 'fit_enhanced_dlnm_model')
    _check_enhanced(_region(s.truth), 'add_region_analysis')


@pytest.mark.parametrize('kind', [pytest.param(k, marks=NUMERIC_MARKS)
                                  for k in ('R day numbers int64', 'Python ordinals', 'yyyymmdd integers')])
def test_fit_enhanced_dlnm_model_numeric_dates_rejected_or_equal_to_R(kind):
    v = encodings()[kind]
    raise_or_match(lambda: _enhanced(v), _check_enhanced, f'fit_enhanced_dlnm_model, {kind}')


@pytest.mark.parametrize('kind', [pytest.param(k, marks=NUMERIC_MARKS)
                                  for k in ('R day numbers int64', 'R day numbers float64', 'yyyymmdd integers')])
def test_add_region_analysis_numeric_dates_rejected_or_equal_to_R(kind):
    v = encodings()[kind]
    raise_or_match(lambda: _region(v), _check_enhanced, f'add_region_analysis, {kind}')


# --------------------------------------------------------------------------------------------------------------
# 3. date containers that ARE dates: basis and dummies equal R's (cheap guards, no glm)
# --------------------------------------------------------------------------------------------------------------
DATE_CONTAINER_LABELS = ['datetime64[D] array', 'datetime64[s] array', 'datetime.date list', 'object Series of date',
                         'categorical Series', 'one-column DataFrame', 'ISO strings', 'yyyymmdd strings',
                         'noon timestamps', 'Series with shifted index', 'DatetimeIndex']


def _date_containers():
    t = sc().truth
    return {
        'datetime64[D] array': lambda: t.values.astype('datetime64[D]'),
        'datetime64[s] array': lambda: t.values.astype('datetime64[s]'),
        'datetime.date list': lambda: list(t.dt.date),
        'object Series of date': lambda: pd.Series(list(t.dt.date), dtype=object),
        'categorical Series': lambda: pd.Series(pd.Categorical(t)),
        'one-column DataFrame': lambda: pd.DataFrame({'d': t}),
        'ISO strings': lambda: t.dt.strftime('%Y-%m-%d'),
        'yyyymmdd strings': lambda: t.dt.strftime('%Y%m%d'),
        'noon timestamps': lambda: t + pd.Timedelta(hours=12),
        'Series with shifted index': lambda: pd.Series(t.values, index=np.arange(len(t)) + 500),
        'DatetimeIndex': lambda: pd.DatetimeIndex(t),
    }


@pytest.mark.parametrize('label', DATE_CONTAINER_LABELS)
def test_real_date_containers_give_R_basis_and_dummies(label):
    from improved_glm import ImprovedGLMInterface
    s = sc()
    dates = _date_containers()[label]()
    iface = ImprovedGLMInterface(s.cb)
    with _quiet():
        seasonal = iface.create_seasonality_basis(dates, DFSEAS)
        dummies = iface.create_dow_factors(dates, verbose=False)
    assert_close(seasonal, s.ref_seasonal, rtol=1e-10, what=f'{label}: seasonal ns basis')
    assert_close(dummies, s.ref_dow, rtol=0, what=f'{label}: weekday dummies')


# --------------------------------------------------------------------------------------------------------------
# 4. tz-aware dates across a DST change
#    local midnights of the same 1096 calendar days: R's integer Dates are evenly spaced; 23/25-hour days are not
# --------------------------------------------------------------------------------------------------------------
NO_DST = ['UTC', 'Asia/Kolkata']
DST = ['America/Chicago', 'Europe/Paris', 'Australia/Sydney']
DEFECT_DST = known_defect(GAP, KEY, note='tz-aware dates: (dates - min) / Timedelta(days=1) is fractional across a DST change')
DST_MARKS = _as_param_marks(DEFECT_DST)


def _tz_dates(tz, as_index=False):
    t = sc().truth.dt.tz_localize(tz)          # the calendar day is unchanged: 00:00 local time
    assert (t.dt.tz_localize(None) == sc().truth).all()
    return pd.DatetimeIndex(t) if as_index else t


@pytest.mark.parametrize('tz', NO_DST + DST)
def test_R_reduces_local_midnight_timestamps_to_the_same_integer_Dates(tz):
    """The R side of the expectation: as.Date() of local-midnight POSIXct (in that tz) is the integer Date again, so
    R's ns(date) basis is the evenly spaced one whatever the tz, DST included."""
    sc()
    r(f'nd_ct <- as.POSIXct(format(nd_date), tz = "{tz}"); nd_back <- as.Date(nd_ct, tz = "{tz}")')
    assert bool(r('identical(as.numeric(nd_back), as.numeric(nd_date))')[0])
    if tz in DST:
        step = rget('unique(diff(as.numeric(nd_ct)) / 3600)')
        assert set(np.round(step, 6)) >= {23.0, 24.0} or set(np.round(step, 6)) >= {24.0, 25.0}, step   # DST is crossed
    assert_close(rget(f'unclass(ns(nd_back, df = {DFSEAS} * 3))'), sc().ref_seasonal, rtol=1e-12, what=f'{tz}: R basis')


@pytest.mark.parametrize('tz', list(NO_DST) + [pytest.param(z, marks=DST_MARKS) for z in DST])
def test_tz_aware_dates_give_the_R_seasonal_basis_and_dummies(tz):
    """Same calendar days, tz-aware: the seasonal basis must be R's ns(Date); the weekday dummies are local."""
    from improved_glm import ImprovedGLMInterface
    s = sc()
    iface = ImprovedGLMInterface(s.cb)
    with _quiet():
        seasonal = iface.create_seasonality_basis(_tz_dates(tz), DFSEAS)
        dummies = iface.create_dow_factors(_tz_dates(tz), verbose=False)
    assert_close(dummies, s.ref_dow, rtol=0, what=f'{tz}: weekday dummies')
    assert_close(seasonal, s.ref_seasonal, rtol=1e-10, what=f'{tz}: seasonal basis (max rel diff '
                                                             f'{max_rel_diff(seasonal, s.ref_seasonal):.2e})')


@pytest.mark.parametrize('tz', ['Asia/Kolkata', pytest.param('America/Chicago', marks=DST_MARKS),
                                pytest.param('Australia/Sydney', marks=DST_MARKS)])
def test_tz_aware_dates_give_the_R_fit(tz):
    """Cross-basis coefficients of the full model: tz-aware local dates must equal R's fit on the integer Dates."""
    g = fit_improved(_tz_dates(tz))
    assert_fit_is_R(g, f'tz-aware Series ({tz})')


@pytest.mark.parametrize('tz', [pytest.param('America/Chicago', marks=DST_MARKS)])
def test_tz_aware_datetimeindex_gives_the_R_fit(tz):
    g = fit_improved(_tz_dates(tz, as_index=True))
    assert_fit_is_R(g, f'tz-aware DatetimeIndex ({tz})')


@DEFECT_DST
def test_tz_aware_result_does_not_depend_on_the_timezone_label():
    """Invariance statement of the defect: two tz labels for the same calendar days cannot give different seasonal
    bases (UTC has no DST, Chicago does) -- holds only when tz-aware dates are reduced to calendar days."""
    from improved_glm import ImprovedGLMInterface
    iface = ImprovedGLMInterface(sc().cb)
    with _quiet():
        a = iface.create_seasonality_basis(_tz_dates('UTC'), DFSEAS)
        b = iface.create_seasonality_basis(_tz_dates('America/Chicago'), DFSEAS)
    assert max_rel_diff(b, a) <= 1e-10, f'same calendar days, UTC vs America/Chicago: {max_rel_diff(b, a):.2e}'
