"""Gap: the Europe 2022 WEEKLY configuration through the packaged pipeline (ImprovedGLMInterface /
fit_enhanced_dlnm_model / multi_location_dlnm_analysis) versus R dlnm + mixmeta.

Every reference number is computed by R at test time (dlnm, splines, mixmeta) from the code of
europe_summer_2022_heat-main/code.R (Ballester et al., Nature Medicine 2023) on the same inputs PyDLNM receives.
The 103-region Europe panel (git-ignored, local) is used when it is on disk; otherwise a seeded synthetic weekly panel
with the same structure (24 regions x ISO weeks 2015-W01 ... 2022-W43) replaces it.  PYDLNM_WEEKLY_SYNTHETIC=1 forces
the synthetic panel.

The configuration (code.R): calibration window 2015-W01 ... 2020-W03 = 264 weeks, each week dated by its Thursday
(so the calendar year of the date is the ISO year: 6 "years"); cross-basis = ns(knots at the 10/50/90th percentiles,
Boundary.knots = range) x integer lag [0, 3]; quasi-Poisson GLM with
    ns(wop, df = round(DF_SEAS * n_weeks * 7 / 365.25)) + cross-basis      (DF_SEAS = 8 -> 40 df);
reduction at the minimum-mortality temperature found on a percentile grid restricted to [5 %, 100 %]; second stage
mixmeta(coef ~ TEMP_AVG + TEMP_IQR, reml) and BLUPs.

What the audit critic found and what this module establishes
  * ImprovedGLMInterface documents "the specification assumes a DAILY series": its seasonal df is
    dfseas * (number of calendar years in the dates) = 48 for the window above, i.e. the literal R formula of the
    Lancet 2015 code, `ns(date, df = dfseas * length(unique(year)))`.  Europe uses 40 (elapsed-time rule).  Run on
    the Europe weeks the packaged route therefore returns R's glm for the Lancet rule (faithful, asserted below) and
    NOT the Europe model: over the 103 regions the reduced coefficients differ from R's Europe first stage by a
    relative 0.12 (minimum), 0.57 (median) and 3.4 (maximum), and neither a warning nor any output is produced
    (measured while writing this module; the non-vacuity check below asserts that the two R rules really differ).
  * The expectation "raise or warn on non-daily input" is a design choice (R has no such function and fits
    whatever formula it is given), so it is NOT asserted here; what IS asserted is the R-faithfulness that holds:
      - the validated route for weekly data (Rpy2GLMInterface.fit_glm with the covariate ns(wop, df=round(...))
        built by OneBasis) reproduces the Europe first stage exactly: cross-basis, GLM block, reduction at the MMT,
        the percentile-grid MMT search;
      - whatever df rule the packaged route applies to weekly data must be one of the two named R rules and must
        equal R's glm under that rule (a third, unnamed rule would fail);
      - multi_location_dlnm_analysis on weekly data equals R's first stage + mixmeta BLUPs under its rule, and its
        per-region MMT equals R's 02.secondstage.R recipe applied to the same BLUPs;
      - the Europe second stage that the packaged class cannot express (IQR meta-predictor, MMT restricted to the
        5th-100th percentile of the PRED_PRC grid) is reproduced by the lower-level API (mvmeta/blup/OneBasis/
        crosspred).
  * Not asserted: week dates anchored on another weekday (ISO Monday: the calendar years of the dates become 7, df 56;
    R takes `year` from a data column, PyDLNM derives it from the dates); attributable numbers on weekly data
    (tests/test_attr_*.py).  The mixmeta-only control name `igls.inititer` of code.R is rejected by R's mvmeta too
    (asserted at the end).

Tolerances: 1e-8 relative for deterministic quantities; 1e-6 for the BLUPs (the REML optimum is optimiser-limited,
tests/test_metaanalysis_N.py theme N2; R is run with reltol = 1e-14); 1e-4 against R's default mixmeta control.
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

from rhelpers import REPO, assert_close, max_rel_diff, np2r, r, r2np, rget      # rhelpers first: it starts R

GAP = 'weekly_data_seasonal_df_rule'
EU_DATA = REPO / 'europe_summer_2022_heat-main' / 'data.csv'
FORCE_SYNTHETIC = bool(os.environ.get('PYDLNM_WEEKLY_SYNTHETIC'))
N_SYNTHETIC = 24
DF_SEAS = 8
LAG = [0, 3]
TOL = 1e-8
TOL_BLUP = 1e-6
R_TIGHT = 'list(maxiter = 20000, reltol = 1e-14)'


# --------------------------------------------------------------------------------------------------------------
# R home (audit finding Q2): PyDLNM modules overwrite os.environ['R_HOME'] while R 4.5 is embedded; R then segfaults
# when it lazily loads LAPACK.  Pin the home of the R that is running and load LAPACK now.
# --------------------------------------------------------------------------------------------------------------
def _running_r_home():
    return os.path.dirname(str(r('.Library')[0]))


_R_HOME = _running_r_home()
os.environ['R_HOME'] = _R_HOME
r('invisible(list(chol(diag(2)), chol2inv(chol(diag(2))), solve(diag(2)), eigen(diag(2)), svd(diag(2)), qr(diag(2))))')


@pytest.fixture(autouse=True)
def _keep_r_home():
    os.environ['R_HOME'] = _R_HOME
    yield
    os.environ['R_HOME'] = _R_HOME


@pytest.fixture(scope='module', autouse=True)
def _single_threaded_blas():
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        yield
        return
    with threadpool_limits(limits=1, user_api='blas'):
        yield


def require_r_packages(*pkgs):
    """Skip unless the R packages are installed, then attach them (rhelpers.require_r_packages indexes the invisible
    result of requireNamespace(), which rpy2 3.6 returns as None; hence this local copy)."""
    for p in pkgs:
        if not bool(r(f'isTRUE(suppressWarnings(requireNamespace("{p}", quietly=TRUE)))')[0]):
            pytest.skip(f'R package {p} not installed')
        r(f'suppressMessages(library({p}))')


@contextlib.contextmanager
def _quiet():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


# --------------------------------------------------------------------------------------------------------------
# data: the Europe weekly panel (or a seeded synthetic one with the same columns) and the R reference functions
# --------------------------------------------------------------------------------------------------------------
def _synthetic_csv(path, n_regions=N_SYNTHETIC, seed=20260930):
    """Seeded weekly panel: location, year (ISO), woy, temp, popu, mort for ISO weeks 2015-W01 ... 2022-W43 (414 weeks,
    Thursdays), with region-specific seasonality, heat / cold slopes and a slow trend."""
    rng = np.random.default_rng(seed)
    weeks = pd.date_range('2015-01-01', periods=414, freq='7D')
    iso = weeks.isocalendar()
    t = np.arange(len(weeks))
    rows = []
    for k in range(n_regions):
        mean, amp, phase = rng.uniform(6, 17), rng.uniform(6, 11), rng.uniform(-0.2, 0.2)
        temp = mean + amp * np.sin(2 * np.pi * (t / 52.1775 - 0.3 + phase)) + rng.normal(0, 2.0, len(t))
        heat, cold = rng.uniform(0.004, 0.03), rng.uniform(0.002, 0.012)
        eta = (np.log(rng.uniform(300, 2500)) + 0.12 * np.cos(2 * np.pi * t / 52.1775) + 1e-4 * t
               + heat * np.maximum(temp - 20, 0) + cold * np.maximum(8 - temp, 0) + rng.normal(0, 0.04, len(t)))
        mort = rng.poisson(np.exp(eta))
        rows.append(pd.DataFrame({'location': f'SYN{k + 1:03d}', 'year': iso['year'].to_numpy(),
                                  'woy': iso['week'].to_numpy(), 'temp': temp, 'popu': 1_000_000, 'mort': mort}))
    pd.concat(rows).to_csv(path, index=False)


@pytest.fixture(scope='module')
def eu(tmp_path_factory):
    """Loads the panel into R (`wk_cal`: the calibration window of code.R with the Thursday dates, `wk_regs`)."""
    if EU_DATA.exists() and not FORCE_SYNTHETIC:
        csv, source = EU_DATA, 'europe'
    else:
        csv, source = tmp_path_factory.mktemp('weekly') / 'synthetic.csv', 'synthetic'
        _synthetic_csv(csv)
    r(f'''
    wk_eu <- read.csv("{csv}")
    # Thursday of ISO week `woy` of ISO year `year` (what ISOweek2date(paste0(year, "-W", woy, "-4")) returns)
    wk_jan4 <- as.Date(paste0(wk_eu$year, "-01-04"))
    wk_eu$date <- wk_jan4 - ((as.POSIXlt(wk_jan4)$wday + 6) %% 7) + 7 * (wk_eu$woy - 1) + 3
    wk_cal <- wk_eu[as.Date("2015-01-01") <= wk_eu$date & wk_eu$date <= as.Date("2019-12-26") + 7 * {LAG[1]}, ]
    wk_regs <- unique(wk_cal$location)
    wk_PRED_PRC <- sort(unique(c(seq(0.0, 1.0, 0.1), seq(1.5, 5.0, 0.5), seq(6.0, 94.0, 1.0), seq(95.0, 98.5, 0.5),
                                 seq(99.0, 100.0, 0.1)) / 100))
    wk_i0 <- which(wk_PRED_PRC == 5 / 100); wk_i1 <- which(wk_PRED_PRC == 100 / 100)
    wk_VAR_PRC <- c(10, 50, 90) / 100
    ''')
    return SimpleNamespace(source=source, n_regions=int(rget('length(wk_regs)')[0]))


@pytest.fixture(scope='module', autouse=True)
def _r_defs(eu):
    """Europe code.R, 'Location-Specific Associations' and the MMT / BLUP steps, as R functions of one region's data
    frame `d` (columns date, year, temp, mort, wop).  rule = "europe": round(DF_SEAS * n * 7 / 365.25) on the week of
    the period; rule = "calendar": DF_SEAS * length(unique(year)) on the date (the daily-data rule of PyDLNM)."""
    r(f'''
    wk_stage1 <- function(d, rule = "europe") {{
      cb <- crossbasis(d$temp, c({LAG[0]}, {LAG[1]}),
                       argvar = list(fun = "ns", knots = quantile(d$temp, wk_VAR_PRC, na.rm = TRUE),
                                     Boundary.knots = range(d$temp, na.rm = TRUE)),
                       arglag = list(fun = "integer"))
      if (rule == "europe") {{
        dfs <- round({DF_SEAS} * nrow(d) * 7 / 365.25)
        m <- glm(mort ~ ns(wop, df = round({DF_SEAS} * length(wop) * 7 / 365.25)) + cb, data = d,
                 family = quasipoisson, na.action = "na.exclude")
      }} else {{
        dfs <- {DF_SEAS} * length(unique(d$year))
        m <- glm(mort ~ ns(date, df = {DF_SEAS} * length(unique(year))) + cb, data = d, family = quasipoisson,
                 na.action = "na.exclude")
      }}
      at <- quantile(d$temp, wk_PRED_PRC, na.rm = TRUE)
      cp <- suppressMessages(crosspred(cb, m, at = at))
      mmt <- cp$predvar[wk_i0 - 1 + which.min(cp$allRRfit[wk_i0:wk_i1])]
      red <- suppressMessages(crossreduce(cb, m, cen = mmt))
      red_mean <- suppressMessages(crossreduce(cb, m, cen = mean(d$temp, na.rm = TRUE)))
      idx <- grep("^cb", names(coef(m)))
      list(cb = unclass(cb), dfs = dfs, nobs = nobs(m), coef = coef(m)[idx], vcov = vcov(m)[idx, idx], at = at,
           allRRfit = cp$allRRfit, mmt = mmt, red_coef = coef(red), red_vcov = vcov(red),
           redmean_coef = coef(red_mean), redmean_vcov = vcov(red_mean))
    }}
    wk_region <- function(l) {{ d <- wk_cal[wk_cal$location == l, ]; d$wop <- seq_len(nrow(d)); d }}
    # MMT from a BLUP on the percentile grid (code.R, 'Best Linear Unbiased Predictions': onebasis + crosspred)
    wk_mmt_blup <- function(d, blup, bvcov) {{
      at <- quantile(d$temp, wk_PRED_PRC, na.rm = TRUE)
      bv <- onebasis(at, fun = "ns", knots = quantile(d$temp, wk_VAR_PRC, na.rm = TRUE),
                     Boundary.knots = range(d$temp, na.rm = TRUE))
      pm <- suppressMessages(crosspred(bv, coef = blup, vcov = bvcov, model.link = "log", at = at))
      as.numeric(at[wk_i0 - 1 + which.min(pm$allRRfit[wk_i0:wk_i1])])
    }}
    # 02.secondstage.R: percentile (1:99) of the minimum of bvar %*% blup on the region's percentile grid
    wk_mmt_second_stage <- function(d, blup) {{
      pv <- quantile(d$temp, 1:99 / 100, na.rm = TRUE)
      bv <- onebasis(pv, fun = "ns", knots = quantile(d$temp, wk_VAR_PRC, na.rm = TRUE),
                     Boundary.knots = range(d$temp, na.rm = TRUE))
      i <- which.min(bv %*% blup)
      c(i, as.numeric(pv[i]))
    }}
    ''')
    yield


class Region:
    """One region's calibration series as numpy arrays, identical in R (`wk_d`) and Python."""

    def __init__(self, index, scenario='clean'):
        self.index, self.scenario = index, scenario
        r(f'wk_d <- wk_region(wk_regs[{index + 1}])')
        if scenario != 'clean':                        # 4 scattered missing weeks after the lag-induced ones
            rng = np.random.default_rng(1000 + index)
            n = int(rget('nrow(wk_d)')[0])
            np2r('wk_idx', rng.choice(np.arange(6, n), 4, replace=False) + 1)          # R counts from 1
            column = 'temp' if scenario == 'nan_temp' else 'mort'
            r(f'wk_d${column}[wk_idx] <- NA')
        self.name = str(r(f'wk_regs[{index + 1}]')[0])
        self.temp = rget('wk_d$temp')
        self.mort = rget('as.numeric(wk_d$mort)')
        self.dates = pd.Series(pd.to_datetime(rget('as.numeric(wk_d$date)'), unit='D'))
        self.n = len(self.temp)
        self.knots = rget('quantile(wk_d$temp, wk_VAR_PRC, na.rm = TRUE)')
        self.bounds = rget('range(wk_d$temp, na.rm = TRUE)')

    def stage1(self, rule):
        """R reference (see wk_stage1) for this region's data frame, as a dict of numpy arrays / floats."""
        r(f'wk_s1 <- wk_stage1(wk_d, "{rule}")')
        ref = {key: rget(f'wk_s1${key}') for key in ('cb', 'coef', 'vcov', 'at', 'allRRfit', 'red_coef', 'red_vcov',
                                                     'redmean_coef', 'redmean_vcov')}
        ref.update(dfs=int(rget('wk_s1$dfs')[0]), nobs=int(rget('wk_s1$nobs')[0]), mmt=float(rget('wk_s1$mmt')[0]))
        return ref

    def crossbasis(self):
        from basis import CrossBasis
        with _quiet():
            return CrossBasis(self.temp, lag=LAG, argvar={'fun': 'ns', 'knots': self.knots,
                                                          'Boundary_knots': self.bounds},
                              arglag={'fun': 'integer'})


def _sample(eu):
    """Six regions spread over the panel."""
    step = max(1, eu.n_regions // 6)
    return list(range(0, eu.n_regions, step))[:6]


# --------------------------------------------------------------------------------------------------------------
# the configuration itself
# --------------------------------------------------------------------------------------------------------------
def test_calibration_window_is_the_europe_weekly_window(eu):
    """264 equally spaced Thursday weeks over 6 ISO years (2015-01-01 ... 2020-01-16): the two seasonal-df rules
    give 40 (Europe) and 48 (daily-data rule) -- the premise of the gap, asserted so that the tests below cannot be
    vacuous."""
    assert int(rget('as.integer(table(wk_cal$location))[1]')[0]) == 264
    assert set(rget('as.integer(table(wk_cal$location))')) == {264}
    assert set(rget('as.POSIXlt(wk_cal$date)$wday')) == {4}
    assert str(r('as.character(min(wk_cal$date))')[0]) == '2015-01-01'
    assert str(r('as.character(max(wk_cal$date))')[0]) == '2020-01-16'
    assert set(rget('diff(as.numeric(wk_cal$date[wk_cal$location == wk_regs[1]]))')) == {7.0}
    assert int(rget('length(unique(wk_cal$year))')[0]) == 6
    assert int(rget(f'round({DF_SEAS} * 264 * 7 / 365.25)')[0]) == 40
    assert int(rget(f'{DF_SEAS} * length(unique(wk_cal$year))')[0]) == 48


@pytest.mark.parametrize('scenario', ('clean', 'nan_temp'))
def test_weekly_europe_crossbasis_matches_R(eu, scenario):
    """ns(explicit knots + Boundary.knots) x integer lag [0, 3] on the weekly series (NaN in the exposure propagates
    through the lag rows like R's Lag())."""
    for index in _sample(eu):
        region = Region(index, scenario)
        ref = region.stage1('europe')
        cb = region.crossbasis()
        assert_close(np.asarray(cb.basis), ref['cb'], rtol=1e-12, what=f'{region.name} cross-basis')
        assert np.array_equal(np.isnan(np.asarray(cb.basis)), np.isnan(ref['cb']))


# --------------------------------------------------------------------------------------------------------------
# the validated route for weekly data: explicit covariates through Rpy2GLMInterface
# --------------------------------------------------------------------------------------------------------------
def _explicit_fit(region, dfs):
    """Europe first stage through PyDLNM building blocks: glm(mort ~ ns(wop, df=dfs) + cb) by Rpy2GLMInterface."""
    from basis import OneBasis
    from rpy2_glm import Rpy2GLMInterface
    cb = region.crossbasis()
    wop = OneBasis(np.arange(1, region.n + 1, dtype=float), fun='ns', df=dfs).basis
    with _quiet():
        g = Rpy2GLMInterface(cb)
        g.fit_glm(region.mort, other_vars=wop, formula_vars=[f'wop{j + 1}' for j in range(wop.shape[1])])
    return cb, g, wop


@pytest.mark.parametrize('scenario', ('clean', 'nan_temp', 'nan_mort'))
def test_explicit_route_reproduces_the_europe_first_stage(eu, scenario):
    """Cross-basis block of the GLM (coef, dispersion-scaled vcov, nobs) and the reduction at the R-chosen MMT equal
    R's; the overall reduction does not depend on the centring value (mean vs MMT)."""
    sample = _sample(eu)[:4] if scenario == 'clean' else _sample(eu)[:2]
    for index in sample:
        region = Region(index, scenario)
        ref = region.stage1('europe')
        cb, g, wop = _explicit_fit(region, ref['dfs'])
        assert wop.shape[1] == ref['dfs']
        what = f'{region.name}/{scenario}'
        assert_close(g.cb_coef, ref['coef'], rtol=TOL, what=f'{what} cross-basis coef')
        assert_close(g.cb_vcov, ref['vcov'], rtol=TOL, what=f'{what} cross-basis vcov')
        assert int(r2np(ro_nobs(g.r_model))[0]) == ref['nobs']
        at_mmt = g.crossreduce(cen=ref['mmt'])
        at_mean = g.crossreduce(cen=float(np.nanmean(region.temp)))
        assert_close(at_mmt.coef, ref['red_coef'], rtol=TOL, what=f'{what} reduced coef at the MMT')
        assert_close(at_mmt.vcov, ref['red_vcov'], rtol=TOL, what=f'{what} reduced vcov at the MMT')
        assert_close(at_mean.coef, ref['redmean_coef'], rtol=TOL, what=f'{what} reduced coef at the mean')
        assert_close(at_mean.coef, at_mmt.coef, rtol=1e-12, what=f'{what} reduction independent of cen')


def ro_nobs(model):
    import rpy2.robjects as ro
    return ro.r['nobs'](model)


def test_explicit_route_percentile_grid_mmt_matches_R(eu):
    """code.R: MMT = percentile-grid temperature with the lowest allRRfit within [5 %, 100 %] of PRED_PRC, from
    crosspred() of the first-stage model with default centring; the Python crosspred on the same coefficients gives
    the same RR curve and the same MMT."""
    from prediction import crosspred
    for index in _sample(eu):
        region = Region(index)
        ref = region.stage1('europe')
        cb, g, _ = _explicit_fit(region, ref['dfs'])
        pred = crosspred(cb, coef=g.cb_coef, vcov=g.cb_vcov, model_link='log', at=ref['at'])
        assert_close(pred.predvar, ref['at'], rtol=1e-12, what=f'{region.name} predvar')
        assert_close(pred.allRRfit, ref['allRRfit'], rtol=TOL, what=f'{region.name} allRRfit (default centring)')
        i0 = int(rget('wk_i0')[0]) - 1
        i1 = int(rget('wk_i1')[0])
        mmt = pred.predvar[i0 + int(np.argmin(pred.allRRfit[i0:i1]))]
        assert mmt == ref['mmt'], f'{region.name}: MMT {mmt} vs R {ref["mmt"]}'


# --------------------------------------------------------------------------------------------------------------
# the packaged route on weekly data
# --------------------------------------------------------------------------------------------------------------
def _packaged_fit(region):
    from improved_glm import ImprovedGLMInterface, fit_enhanced_dlnm_model
    cb = region.crossbasis()
    with _quiet():
        g = ImprovedGLMInterface(cb)
        g.fit_dlnm_model(region.mort, region.dates, dfseas=DF_SEAS)
        packaged = fit_enhanced_dlnm_model(cb, region.mort, region.dates, dfseas=DF_SEAS)
    return cb, g, packaged


@pytest.mark.parametrize('scenario', ('clean', 'nan_mort'))
def test_packaged_route_on_weekly_data_is_R_glm_under_a_named_seasonal_rule(eu, scenario):
    """Whatever seasonal-df rule ImprovedGLMInterface / fit_enhanced_dlnm_model apply to weekly dates must be one of
    the two R rules -- the Lancet rule `ns(date, df = dfseas * length(unique(year)))` (48 df here, what PyDLNM
    documents) or Europe's `round(dfseas * n * 7 / 365.25)` (40 df) -- and must reproduce R's glm under that rule
    (cross-basis block and reduced coefficients/vcov).  A third rule would fail.  Also guards that the two rules give
    different numbers, so that the comparison can tell them apart."""
    for index in _sample(eu)[:4]:
        region = Region(index, scenario)
        refs = {'calendar': region.stage1('calendar'), 'europe': region.stage1('europe')}
        assert refs['calendar']['dfs'] == 48 and refs['europe']['dfs'] == 40
        gap = max_rel_diff(refs['calendar']['red_coef'], refs['europe']['red_coef'])
        assert gap > 1e-2, f'{region.name}: the two R rules give the same reduced coefficients ({gap:.2e}): vacuous'
        cb, g, packaged = _packaged_fit(region)
        seasonal_columns = [str(n) for n in g.r_model.rx2('coefficients').names if 'total_df' in str(n)]
        found = {}
        for rule, ref in refs.items():
            found[rule] = max(max_rel_diff(g.cb_coef, ref['coef']), max_rel_diff(g.cb_vcov, ref['vcov']),
                              max_rel_diff(g.crossreduce(cen=ref['mmt']).coef, ref['red_coef']),
                              max_rel_diff(packaged['reduced']['coefficients'], ref['redmean_coef']))
        assert min(found.values()) <= TOL, (f'{region.name}: packaged route equals neither R rule (max relative '
                                            f'difference {found}); seasonal columns {len(seasonal_columns)}')
        # the documented rule today: one seasonal column per calendar year x dfseas, no day-of-week dummies
        if found['calendar'] <= TOL:
            assert len(seasonal_columns) == refs['calendar']['dfs']
        assert g.dow_columns == []                              # a single weekday (Thursday): no dummies, no crash
        assert not any(str(n).startswith('dow') for n in g.r_model.rx2('coefficients').names)


def test_seasonal_basis_of_weekly_dates_is_a_named_R_ns_basis(eu):
    """create_seasonality_basis(dates, dfseas) on the 264 weekly Thursday dates equals R's ns() with df = 48 (Lancet
    rule, documented) or 40 (Europe rule), to 1e-10, columns and all."""
    from improved_glm import ImprovedGLMInterface
    region = Region(_sample(eu)[0])
    cb = region.crossbasis()
    with _quiet():
        basis = ImprovedGLMInterface(cb).create_seasonality_basis(region.dates, DF_SEAS)
    candidates = {48: rget('unclass(ns(wk_d$date, df = 8 * length(unique(wk_d$year))))'),
                  40: rget('unclass(ns(wk_d$wop, df = round(8 * nrow(wk_d) * 7 / 365.25)))')}
    assert basis.shape[1] in candidates, f'{basis.shape[1]} seasonal columns: neither 48 (Lancet) nor 40 (Europe)'
    assert_close(basis, candidates[basis.shape[1]], rtol=1e-10, what='seasonal ns basis')


def test_weekly_dates_spacing_does_not_change_the_fit(eu):
    """The packaged seasonal basis depends on the dates only through their affine image: ns() is invariant to the
    origin and unit of x, so a weekly series dated by Thursday, by the week index, or shifted by 10 years gives the
    same cross-basis block (R: ns(wop) == ns(date) for equally spaced weeks)."""
    region = Region(_sample(eu)[1])
    cb = region.crossbasis()
    from improved_glm import ImprovedGLMInterface
    results = []
    for dates in (region.dates, region.dates - pd.Timedelta(days=7 * 52), region.dates + pd.Timedelta(days=3)):
        if dates.dt.year.nunique() != region.dates.dt.year.nunique():
            continue                                  # a shift that changes the number of calendar years is another rule
        with _quiet():
            g = ImprovedGLMInterface(cb)
            g.fit_dlnm_model(region.mort, dates, dfseas=DF_SEAS)
        results.append(g.cb_coef)
    assert len(results) >= 1
    for coef in results[1:]:
        assert_close(coef, results[0], rtol=1e-9, what='cross-basis coef under a shifted date origin')


# --------------------------------------------------------------------------------------------------------------
# multi_location_dlnm_analysis on weekly data
# --------------------------------------------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _all_regions(n_regions):
    return tuple(Region(i) for i in range(n_regions))


@lru_cache(maxsize=None)
def _r_all_stage1(n_regions, rule):
    """R first stage of every region under `rule` (coefficient matrix, list of vcov), R-side only."""
    r(f'wk_all_{rule} <- lapply(wk_regs, function(l) wk_stage1(wk_region(l), "{rule}"))')
    coef = rget(f't(sapply(wk_all_{rule}, function(s) s$red_coef))')
    vcov = [rget(f'wk_all_{rule}[[{i + 1}]]$red_vcov') for i in range(n_regions)]
    return coef, vcov


def _r_blup_table(n_regions):
    """BLUPs of the R mixmeta fit `wk_mv` as arrays (n, k) and (n, k, k)."""
    blup = rget('do.call(rbind, lapply(wk_bl, function(b) as.numeric(b$blup)))')
    vcov = rget('do.call(rbind, lapply(wk_bl, function(b) as.numeric(b$vcov)))')
    k = blup.shape[1]
    return blup, vcov.reshape(n_regions, k, k)


def test_multi_location_pipeline_on_weekly_data_matches_R_mixmeta(eu):
    """multi_location_dlnm_analysis on every region (weekly, ns x integer lag [0, 3]): first stage, Lancet
    meta-predictors (mean, range), REML meta-analysis + BLUPs and the per-region MMT percentiles equal R (first
    stage under the named seasonal rule PyDLNM applies, today the calendar-year rule; mixmeta(coef ~ avg + range,
    reml); R's 02.secondstage.R recipe for the MMT applied to the same BLUPs)."""
    from multi_location import multi_location_dlnm_analysis
    require_r_packages('mixmeta')
    n = eu.n_regions
    regions = _all_regions(n)
    data = [{'name': reg.name, 'crossbasis': reg.crossbasis(), 'y': reg.mort, 'dates': reg.dates} for reg in regions]
    with _quiet():
        analysis = multi_location_dlnm_analysis(data, method='reml', dfseas=DF_SEAS)

    # the packaged first stage must be R's under ONE of the two named seasonal rules (today: the calendar-year rule)
    coef_p = np.array([res['reduced']['coefficients'] for res in analysis.region_results])
    by_rule = {rule: _r_all_stage1(n, rule) for rule in ('calendar', 'europe')}
    gaps = {rule: max_rel_diff(coef_p, by_rule[rule][0]) for rule in by_rule}
    rule = min(gaps, key=gaps.get)
    assert gaps[rule] <= TOL, f'first-stage reduced coef equals neither R rule: max relative differences {gaps}'
    coef_r, vcov_r = by_rule[rule]
    assert_close(np.array([res['reduced']['vcov'] for res in analysis.region_results]), np.array(vcov_r), rtol=TOL,
                 what='first-stage reduced vcov')

    avg_r = rget('sapply(wk_regs, function(l) mean(wk_cal$temp[wk_cal$location == l], na.rm = TRUE))')
    rng_r = rget('sapply(wk_regs, function(l) diff(range(wk_cal$temp[wk_cal$location == l], na.rm = TRUE)))')
    assert_close([analysis.meta_predictors[reg.name]['avg_temp'] for reg in regions], avg_r, rtol=1e-12, what='avg')
    assert_close([analysis.meta_predictors[reg.name]['temp_range'] for reg in regions], rng_r, rtol=1e-12, what='range')

    np2r('wk_COEF', coef_r)
    np2r('wk_AVG', avg_r)
    np2r('wk_RNG', rng_r)
    r('wk_VC <- lapply(wk_regs, function(l) NULL)')
    for i, v in enumerate(vcov_r):
        np2r('wk_v', v)
        r(f'wk_VC[[{i + 1}]] <- wk_v')
    r(f'wk_mv <- mixmeta(wk_COEF ~ wk_AVG + wk_RNG, wk_VC, method = "reml", control = {R_TIGHT}); '
      f'wk_bl <- blup(wk_mv, vcov = TRUE)')
    blup_r, bvcov_r = _r_blup_table(n)
    blup_p = np.array([b['blup'] for b in analysis.blup_results])
    bvcov_p = np.array([b['vcov'] for b in analysis.blup_results])
    assert_close(blup_p, blup_r, rtol=TOL_BLUP, what='BLUP')
    assert_close(bvcov_p, bvcov_r, rtol=TOL_BLUP, what='BLUP vcov')

    percentiles = []
    for i, reg in enumerate(regions):
        r(f'wk_d <- wk_region(wk_regs[{i + 1}])')
        np2r('wk_b', blup_p[i])
        ref = rget('wk_mmt_second_stage(wk_d, wk_b)')
        got = analysis.pooled_mmts['region_mmts'][reg.name]
        assert got['percentile'] == int(ref[0]), f'{reg.name}: MMT percentile {got["percentile"]} vs R {int(ref[0])}'
        assert_close(got['mmt'], ref[1], rtol=1e-10, what=f'{reg.name} MMT temperature')
        percentiles.append(int(ref[0]))
    assert analysis.pooled_mmts['pooled_mmt_percentile_median'] == float(np.median(percentiles))


# --------------------------------------------------------------------------------------------------------------
# the Europe second stage the packaged class cannot express, through the lower-level API
# --------------------------------------------------------------------------------------------------------------
def _python_europe_first_stage(regions):
    """Europe first stage by PyDLNM building blocks, per region: reduced coef / vcov at the MMT found on the
    percentile grid restricted to [5 %, 100 %]."""
    from prediction import crosspred
    out = []
    i0, i1 = int(rget('wk_i0')[0]) - 1, int(rget('wk_i1')[0])
    for reg in regions:
        r(f'wk_d <- wk_region(wk_regs[{reg.index + 1}])')
        dfs = int(rget(f'round({DF_SEAS} * nrow(wk_d) * 7 / 365.25)')[0])
        cb, g, _ = _explicit_fit(reg, dfs)
        at = rget('quantile(wk_d$temp, wk_PRED_PRC, na.rm = TRUE)')
        pred = crosspred(cb, coef=g.cb_coef, vcov=g.cb_vcov, model_link='log', at=at)
        mmt = float(pred.predvar[i0 + int(np.argmin(pred.allRRfit[i0:i1]))])
        red = g.crossreduce(cen=mmt)
        out.append(SimpleNamespace(coef=np.asarray(red.coef, dtype=float), vcov=np.asarray(red.vcov, dtype=float),
                                   mmt=mmt, at=at))
    return out


def test_europe_second_stage_with_iqr_predictor_and_restricted_mmt_matches_R(eu):
    """mixmeta(COEF ~ TEMP_AVG + TEMP_IQR, reml) + blup() of code.R, then the BLUP-based MMT on the PRED_PRC grid
    restricted to [5 %, 100 %], on the Europe (round(dfseas * n * 7 / 365.25)) first stage: PyDLNM's explicit route
    + mvmeta + blup + OneBasis/crosspred equal R.  (First stage from PyDLNM, meta-analysis in both, same BLUP fed to
    both MMT searches.)"""
    from basis import OneBasis
    from meta_analysis import blup as py_blup, mvmeta
    from prediction import crosspred
    require_r_packages('mixmeta')
    n = eu.n_regions
    regions = _all_regions(n)
    stage1 = _python_europe_first_stage(regions)

    coef_r, vcov_r = _r_all_stage1(n, 'europe')
    assert_close(np.array([s.coef for s in stage1]), coef_r, rtol=TOL, what='Europe first-stage reduced coef')
    assert_close(np.array([s.vcov for s in stage1]), np.array(vcov_r), rtol=TOL, what='Europe first-stage reduced vcov')
    assert_close([s.mmt for s in stage1], rget('sapply(wk_all_europe, function(s) s$mmt)'), rtol=1e-12,
                 what='first-stage MMT')

    avg = np.array([np.nanmean(reg.temp) for reg in regions])
    iqr = np.array([np.nanpercentile(reg.temp, 75) - np.nanpercentile(reg.temp, 25) for reg in regions])
    assert_close(iqr, rget('sapply(wk_regs, function(l) IQR(wk_cal$temp[wk_cal$location == l], na.rm = TRUE))'),
                 rtol=1e-12, what='IQR')

    coef_p = np.array([s.coef for s in stage1])
    vcov_p = np.array([s.vcov for s in stage1])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mv = mvmeta(y=coef_p, S=vcov_p, X=np.column_stack([np.ones(n), avg, iqr]), method='reml')
    blups = py_blup(mv, vcov=True)

    np2r('wk_COEF', coef_p)
    np2r('wk_AVG', avg)
    np2r('wk_IQR', iqr)
    r('wk_VC <- lapply(wk_regs, function(l) NULL)')
    for i, v in enumerate(vcov_p):
        np2r('wk_v', v)
        r(f'wk_VC[[{i + 1}]] <- wk_v')
    for label, control, tol in (('tight control', R_TIGHT, TOL_BLUP),
                                ('code.R control (igls.inititer = 10)', 'list(igls.inititer = 10)', 1e-4)):
        r(f'wk_mv <- mixmeta(wk_COEF ~ wk_AVG + wk_IQR, wk_VC, method = "reml", control = {control}); '
          f'wk_bl <- blup(wk_mv, vcov = TRUE)')
        blup_r, bvcov_r = _r_blup_table(n)
        assert_close(np.array([b['blup'] for b in blups]), blup_r, rtol=tol, what=f'BLUP vs R ({label})')
        assert_close(np.array([b['vcov'] for b in blups]), bvcov_r, rtol=tol, what=f'BLUP vcov vs R ({label})')

    for i, reg in enumerate(regions):                # MMT search with the SAME BLUP on both sides
        r(f'wk_d <- wk_region(wk_regs[{i + 1}])')
        np2r('wk_b', blups[i]['blup'])
        np2r('wk_bv', blups[i]['vcov'])
        mmt_r = float(rget('wk_mmt_blup(wk_d, wk_b, wk_bv)')[0])
        at = stage1[i].at
        ob = OneBasis(at, fun='ns', knots=reg.knots, Boundary_knots=reg.bounds)
        pred = crosspred(ob, coef=blups[i]['blup'], vcov=blups[i]['vcov'], model_link='log', at=at)
        i0, i1 = int(rget('wk_i0')[0]) - 1, int(rget('wk_i1')[0])
        mmt_p = float(pred.predvar[i0 + int(np.argmin(pred.allRRfit[i0:i1]))])
        assert mmt_p == mmt_r, f'{reg.name}: BLUP-based MMT {mmt_p} vs R {mmt_r}'


def test_mixmeta_only_control_name_is_rejected_like_R_mvmeta():
    """code.R passes control = list(igls.inititer = 10) to mixmeta(); that spelling is mixmeta's.  R's mvmeta (whose
    control PyDLNM's mvmeta follows) rejects it, and so must PyDLNM (loudly, not silently ignored)."""
    from meta_analysis import mvmeta
    require_r_packages('mvmeta')
    with pytest.raises(Exception, match='unused argument'):
        r('mvmeta::mvmeta.control(igls.inititer = 10)')
    rng = np.random.default_rng(3)
    y = rng.normal(size=(12, 2))
    S = np.tile(0.04 * np.eye(2), (12, 1, 1))
    with pytest.raises(TypeError):
        mvmeta(y=y, S=S, control={'igls.inititer': 10})
