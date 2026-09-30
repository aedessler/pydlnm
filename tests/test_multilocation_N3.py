"""MultiLocationDLNM second stage (theme N3): pooled MMT, region bookkeeping, per-region model summary, list response.

Every reference is computed by R (dlnm 2.4.10 / splines / stats, at run time) on the England & Wales regions of
2015_gasparrini_Lancet_Rcodedata-master (00.prepdata.R, 01.firststage.R, 02.secondstage.R), 5 or 4 regions x 4 years
(1993-1996, n = 1461 days each), and compared with PyDLNM's MultiLocationDLNM on identical inputs.

Theme N3
  mvmeta-blup-2   calculate_pooled_mmts() calls find_mmt_blup(fun="bs", degree=2) with the default P10/75/90 knots whatever
                  basis the first stage used, and find_mmt_blup never reads `fun`.  R (02.secondstage.R:61-70) rebuilds the
                  SAME onebasis (varfun, varper, vardegree) as the first stage.  A first-stage basis that is not bs/deg 2/
                  P10-75-90 therefore gives either the wrong MMT silently (same number of columns) or -- when the number of
                  columns differs from 5 -- a swallowed shape error and the regional MEDIAN temperature reported as "MMT"
                  (method='median_fallback') and pooled with the real ones.
  mvmeta-blup-4   (nit) "pooled MMT" is the median of the regional MMT *temperatures*; R's country-level quantity
                  minperccountry = median(minperccity) is a *percentile* and is not exposed.  Plain tests pin the documented
                  keys / semantics; one known-defect test asks for the percentile statistic (fix sketch of the audit).
  mvmeta-blup-5   meta-predictors (and region_mmts) are stored in dicts keyed by region name: with duplicate labels the
                  second region silently gets the first region's avg_temp / temp_range in the meta-regression design
                  matrix and region_mmts collapses.  R aligns avgtmean / rangetmean with dlist by position.
                  Either rejecting duplicate names (ValueError) or keeping the regions distinct is acceptable.
  mvmeta-blup-11  ImprovedGLMInterface.get_model_summary() evaluates the R global `fitted_model`: after several regions were
                  fitted, every region_results[i]['glm_interface'] reports the LAST region's model.
  mvmeta-blup-12  add_region_analysis / fit_dlnm_model crash with TypeError for a response given as a Python list
                  (docstring: array-like; R accepts any numeric vector, NA included).

Tests decorated with @known_defect assert the R-faithful (or documented-correct) behaviour and fail today (strict xfail);
the plain tests guard neighbouring behaviour that is already faithful and must keep passing while the fixes land.

Design notes
  * Sibling defect basis-cont-5 (theme M1): find_mmt_blup also builds its B-spline with the boundary knots of the 1st-99th
    percentile grid instead of range(x).  To test the N3 defects independently of it, the exposure series are WINSORISED at
    their 2.5th / 97.5th percentiles: >= 2.5 % of the days then equal min(x) and max(x), so P1 == min and P99 == max and the
    (mis-set) boundary knots coincide with R's Boundary.knots = range(x).  The pooled-MMT tests below therefore go green
    with the N3 fix alone, whether or not M1 has been fixed yet (test_default_spec_... guards exactly this coincidence).
  * The MMT reference is R's own recipe applied to PyDLNM's BLUP vectors (isolates the MMT step from the optimiser-limited
    second-stage fit, which tests/test_metaanalysis_N.py covers).
  * The first stage is real (add_region_analysis -> R glm), because a fix may read the basis specification from the
    CrossBasis object handed to add_region_analysis.
"""
import contextlib
import copy
import io
import os
import warnings
from functools import lru_cache
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from rhelpers import REPO, assert_close, known_defect, np2r, r, r2np, rget      # rhelpers first: it starts R

THEME = 'N3'
EW_CSV = REPO / '2015_gasparrini_Lancet_Rcodedata-master' / 'regEngWales.csv'      # data, not code: always the real repo
YEARS = [1993, 1994, 1995, 1996]
REGIONS = ['East', 'E-Mid', 'London', 'N-East', 'N-West']
LAG = 21
DFSEAS = 8

# first-stage variable bases used below: name -> (fun, degree or None, knot percentiles)
SPECS = {
    'bs2_p10_75_90': ('bs', 2, (10, 75, 90)),          # the Lancet spec == hard-coded spec of calculate_pooled_mmts (5 columns)
    'ns_p10_75_90': ('ns', None, (10, 75, 90)),        # 4 columns: width differs from 5
    'bs3_p10_75_90': ('bs', 3, (10, 75, 90)),          # 6 columns: width differs from 5
    'bs2_p25_50_75': ('bs', 2, (25, 50, 75)),          # 5 columns, other knots
    'ns_p20_40_60_80': ('ns', None, (20, 40, 60, 80)),  # 5 columns, other function
}
WIDTH_MISMATCH = ['ns_p10_75_90', 'bs3_p10_75_90']
SAME_WIDTH = ['bs2_p25_50_75', 'ns_p20_40_60_80']


# --------------------------------------------------------------------------------------------------------------
# R home: PyDLNM modules (basis, improved_glm, ...) overwrite os.environ['R_HOME'] with the R 4.6 path while the R session
# started by rhelpers is R 4.5; R then segfaults when it lazily dlopen()s its LAPACK module (chol2inv in vcov.glm, solve,
# ...).  Pin the home of the R that is actually running around every test and load LAPACK now (audit finding Q2).
# --------------------------------------------------------------------------------------------------------------
def _running_r_home():
    p = str(r('as.character(getLoadedDLLs()[["utils"]][["path"]])')[0])
    for _ in range(4):
        p = os.path.dirname(p)
    return p


_R_HOME = _running_r_home()
os.environ['R_HOME'] = _R_HOME
r('invisible(list(chol(diag(2)), chol2inv(chol(diag(2))), solve(diag(2)), eigen(diag(2)), svd(diag(2)), qr(diag(2))))')


@pytest.fixture(autouse=True)
def _keep_r_home():
    before = os.environ.get('R_HOME')
    os.environ['R_HOME'] = _R_HOME
    try:
        yield
    finally:
        if before is None:
            os.environ.pop('R_HOME', None)
        else:
            os.environ['R_HOME'] = before


@pytest.fixture(scope='module', autouse=True)
def _single_threaded_blas():
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        yield
        return
    with threadpool_limits(limits=1, user_api='blas'):
        yield


@pytest.fixture(scope='module', autouse=True)
def _r_defs():
    r(f'''
    n3_ew <- read.csv("{EW_CSV}", row.names = 1); n3_ew$date <- as.Date(n3_ew$date)
    n3_region <- function(name, years, tmean, death = NULL) {{
      d <- n3_ew[n3_ew$regnames == name & n3_ew$year %in% years, ]
      stopifnot(nrow(d) == length(tmean))
      d$tmean <- tmean
      if (!is.null(death)) d$death <- death
      d
    }}
    n3_firststage <- function(d, per = c(10, 75, 90)) {{
      kn <- quantile(d$tmean, per / 100, na.rm = TRUE)
      cb <- crossbasis(d$tmean, lag = {LAG}, argvar = list(fun = "bs", degree = 2, knots = kn),
                       arglag = list(knots = logknots({LAG}, 3)))
      m <- glm(death ~ cb + dow + ns(date, df = {DFSEAS} * length(unique(year))), d, family = quasipoisson,
               na.action = "na.exclude")
      list(m = m, red = crossreduce(cb, m, cen = mean(d$tmean, na.rm = TRUE)))
    }}
    n3_mmt <- function(blup, tm, fun, per, degree) {{
      pv <- quantile(tm, 1:99 / 100, na.rm = TRUE)
      arg <- list(x = pv, fun = fun, knots = quantile(tm, per / 100, na.rm = TRUE),
                  Boundary.knots = range(tm, na.rm = TRUE))
      if (!is.na(degree)) arg$degree <- degree
      bv <- do.call(onebasis, arg)
      i <- which.min(bv %*% blup)
      list(percentile = (1:99)[i], temperature = as.numeric(pv[i]), basis = unclass(bv), risk = as.numeric(bv %*% blup))
    }}
    ''')
    yield


# --------------------------------------------------------------------------------------------------------------
# helpers: identical inputs for R and Python
# --------------------------------------------------------------------------------------------------------------
@contextlib.contextmanager
def _quiet():
    """Swallow the (very chatty) prints and the optimiser warnings of the PyDLNM pipeline; yield the recorded warnings."""
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        yield caught


@lru_cache(maxsize=None)
def _ew_frame():
    df = pd.read_csv(EW_CSV, index_col=0)
    df['date'] = pd.to_datetime(df['date'])
    return df


def region_data(name, winsorise=True):
    """Exposure, deaths and dates of one England & Wales region, 1993-1996.  The exposure is winsorised at P2.5/P97.5."""
    df = _ew_frame()
    d = df[(df['regnames'] == name) & (df['year'].isin(YEARS))].reset_index(drop=True)
    t = d['tmean'].to_numpy(dtype=float).copy()
    if winsorise:
        lo, hi = np.quantile(t, [0.025, 0.975])
        t = np.clip(t, lo, hi)
    return SimpleNamespace(name=name, t=t, y=d['death'].to_numpy(dtype=float).copy(), dates=d['date'].copy())


def argvar_for(spec_key, t):
    fun, degree, per = SPECS[spec_key]
    argvar = {'fun': fun, 'knots': np.quantile(t, np.array(per) / 100.0)}
    if degree is not None:
        argvar['degree'] = degree
    return argvar


def make_crossbasis(t, spec_key='bs2_p10_75_90'):
    from basis import CrossBasis
    from utils import logknots
    return CrossBasis(t, lag=LAG, argvar=argvar_for(spec_key, t), arglag={'fun': 'ns', 'knots': logknots([0, LAG], nk=3)})


def push_region(tag, reg, y=None):
    """R data frame `n3_d_<tag>` = the same rows and the same (winsorised) exposure that PyDLNM receives."""
    np2r(f'n3_t_{tag}', reg.t)
    if y is None:
        r(f'n3_d_{tag} <- n3_region("{reg.name}", c({", ".join(str(v) for v in YEARS)}), n3_t_{tag})')
    else:
        np2r(f'n3_y_{tag}', np.asarray(y, dtype=float))
        r(f'n3_d_{tag} <- n3_region("{reg.name}", c({", ".join(str(v) for v in YEARS)}), n3_t_{tag}, n3_y_{tag})')


def r_firststage(tag, reg, y=None):
    """R 01.firststage.R on the region (bs2/P10-75-90 x ns(logknots) lag 21, quasi-Poisson); model kept in R as n3_fs_<tag>."""
    push_region(tag, reg, y)
    r(f'n3_fs_{tag} <- n3_firststage(n3_d_{tag})')
    return SimpleNamespace(red_coef=rget(f'as.numeric(coef(n3_fs_{tag}$red))'),
                           cb_coef=rget(f'coef(n3_fs_{tag}$m)[grep("^cb", names(coef(n3_fs_{tag}$m)))]'),
                           dispersion=float(rget(f'summary(n3_fs_{tag}$m)$dispersion')[0]),
                           deviance=float(rget(f'deviance(n3_fs_{tag}$m)')[0]),
                           df_residual=int(rget(f'df.residual(n3_fs_{tag}$m)')[0]),
                           nobs=int(rget(f'nobs(n3_fs_{tag}$m)')[0]))


def r_mmt(blup, t, spec_key):
    """R 02.secondstage.R:61-70 (onebasis with the first-stage spec, which.min over the 1st-99th percentiles)."""
    fun, degree, per = SPECS[spec_key]
    np2r('n3_blup', np.asarray(blup, dtype=float))
    np2r('n3_tm', t)
    np2r('n3_per', np.array(per, dtype=float))
    r(f'n3_res <- n3_mmt(n3_blup, n3_tm, "{fun}", n3_per, {"NA" if degree is None else degree})')
    return SimpleNamespace(percentile=int(rget('as.numeric(n3_res$percentile)')[0]),
                           temperature=float(rget('n3_res$temperature')[0]),
                           basis=rget('n3_res$basis'), risk=rget('n3_res$risk'))


def add_regions(an, regs, labels=None, spec_key='bs2_p10_75_90', **kw):
    """add_region_analysis for every region with the given first-stage spec; output and warnings swallowed."""
    with _quiet():
        for i, reg in enumerate(regs):
            an.add_region_analysis(labels[i] if labels else reg.name, make_crossbasis(reg.t, spec_key), reg.y, reg.dates,
                                   dfseas=DFSEAS, **kw)
    return an


_RUNS = {}


def pipeline(spec_key, names=tuple(REGIONS)):
    """Full PyDLNM second stage (add regions -> meta-analysis -> BLUPs -> pooled MMTs) for one first-stage basis; cached."""
    key = (spec_key, tuple(names))
    if key not in _RUNS:
        from multi_location import MultiLocationDLNM
        regs = [region_data(n) for n in names]
        an = MultiLocationDLNM()
        add_regions(an, regs, spec_key=spec_key)
        with _quiet() as caught:
            an.fit_meta_analysis()
            an.calculate_blups()
            pm = an.calculate_pooled_mmts()
        _RUNS[key] = SimpleNamespace(an=an, pm=pm, regs=regs, names=list(names), spec=spec_key,
                                     warnings=[str(w.message) for w in caught])
    return _RUNS[key]


def r_reference(run):
    """R's MMT step for every region of a pipeline run, applied to PyDLNM's own BLUP vectors."""
    return [r_mmt(run.an.blup_results[i]['blup'], run.regs[i].t, run.spec) for i in range(len(run.names))]


def _fmt(d):
    return ', '.join(f'{k}: {v}' for k, v in d.items())


def assert_pooled_mmt_matches_r(run):
    """Per-region MMT percentile and temperature, and the pooled median / mean temperature, equal R's recipe."""
    ref = r_reference(run)
    region_mmts = run.pm['region_mmts']
    rows = {}
    for n, rf in zip(run.names, ref):
        got = region_mmts.get(n)
        rows[n] = (None if got is None else (int(got['percentile']), got.get('method')), rf.percentile)
    wrong = {n: f'Python {v[0]} vs R {v[1]}' for n, v in rows.items() if v[0] is None or v[0][0] != v[1]}
    assert not wrong, f'{run.spec}: MMT percentile differs from R (Python (percentile, method)): {_fmt(wrong)}'
    r_temps = np.array([rf.temperature for rf in ref])
    py_temps = np.array([region_mmts[n]['mmt'] for n in run.names], dtype=float)
    assert_close(py_temps, r_temps, rtol=1e-9, what=f'{run.spec}: regional MMT temperature')
    assert_close(run.pm['pooled_mmt_median'], np.median(r_temps), rtol=1e-9, what=f'{run.spec}: pooled median MMT')
    assert_close(run.pm['pooled_mmt_mean'], np.mean(r_temps), rtol=1e-9, what=f'{run.spec}: pooled mean MMT')


# ==============================================================================================================
# mvmeta-blup-2: pooled MMT must use the first-stage basis specification
# ==============================================================================================================
@known_defect(THEME, 'mvmeta-blup-2', note='bs/deg 2/P10-75-90 hard-coded; a basis of another width raises inside and '
                                          'the regional median temperature is reported as the MMT')
@pytest.mark.parametrize('spec_key', WIDTH_MISMATCH)
def test_pooled_mmt_matches_r_when_first_stage_basis_has_another_width(spec_key):
    """ns (4 columns) and bs degree 3 (6 columns): R rebuilds the first-stage onebasis for the MMT search; PyDLNM builds a
    5-column bs basis, the matmul with the BLUP fails and find_mmt_blup answers with the median temperature."""
    assert_pooled_mmt_matches_r(pipeline(spec_key))


@known_defect(THEME, 'mvmeta-blup-2', note='hard-coded knots / find_mmt_blup ignores `fun`: wrong basis, no warning')
@pytest.mark.parametrize('spec_key', SAME_WIDTH)
def test_pooled_mmt_matches_r_when_first_stage_basis_has_the_same_width(spec_key):
    """Also 5 columns, but other knots (bs P25/50/75) or another function (ns with 4 knots): the shapes fit, so the
    MMT is silently computed from the wrong basis."""
    assert_pooled_mmt_matches_r(pipeline(spec_key))


@known_defect(THEME, 'mvmeta-blup-2', note="blanket except in find_mmt_blup: method='median_fallback' pooled as a real MMT")
@pytest.mark.parametrize('spec_key', WIDTH_MISMATCH)
def test_pooled_mmt_never_reports_a_median_fallback(spec_key):
    """A valid first-stage basis must never end in the 'median fallback' (regional median temperature dressed up as MMT
    and included in the pooled median / mean / std), nor in a warning that says so."""
    run = pipeline(spec_key)
    fallbacks = [n for n in run.names if run.pm['region_mmts'].get(n, {}).get('method') == 'median_fallback']
    msgs = [m for m in run.warnings if 'fallback' in m.lower()]
    assert not fallbacks and not msgs, (f'{run.spec}: median_fallback for {fallbacks}; warnings: {msgs[:1]}; '
                                        f'percentiles {[round(float(run.pm["region_mmts"][n]["percentile"]), 1) for n in fallbacks]}')


@known_defect(THEME, 'mvmeta-blup-2', note='a failing MMT search is replaced by the median temperature instead of an error')
def test_failed_mmt_search_is_not_replaced_by_the_median_temperature():
    """R: bvar %*% blup with a BLUP of the wrong length stops ('non-conformable').  PyDLNM may raise or leave the region out
    of the pooled statistics, but must not report its median temperature as an MMT."""
    run = pipeline('bs2_p10_75_90')
    an = copy.copy(run.an)                                   # shallow copy: the cached run stays untouched
    an.blup_results = [dict(b) for b in run.an.blup_results]
    an.blup_results[2]['blup'] = np.asarray(an.blup_results[2]['blup'])[:3]          # London: 3 coefficients for a 5-column basis
    try:
        with _quiet():
            out = an.calculate_pooled_mmts()
    except (ValueError, RuntimeError):
        return                                               # raising is R-faithful
    got = out['region_mmts'].get('London')
    assert got is None or got.get('method') != 'median_fallback', \
        f"London reported as an MMT although the search failed: method={got.get('method')!r}, mmt={got.get('mmt')!r}"
    med_london = float(np.median(run.regs[2].t))
    assert not any(abs(float(v) - med_london) < 1e-12 for v in out['individual_mmts']), \
        'the median temperature of the failed region is pooled with the real MMTs'


@known_defect(THEME, 'mvmeta-blup-2', note='find_mmt_blup never reads `fun`: ns is built as a B-spline')
@pytest.mark.parametrize('per', [(10, 75, 90), (20, 40, 60, 80)], ids=['ns4cols', 'ns5cols'])
def test_find_mmt_blup_honours_fun_ns(per):
    """find_mmt_blup(fun='ns', knots=...) documents `fun` as the basis function type; R: onebasis(fun='ns', knots=...).
    (x winsorised so that the boundary-knot defect of theme M1 does not interfere.)"""
    from centering import find_mmt_blup
    x = region_data('London').t
    spec_key = 'ns_p10_75_90' if len(per) == 3 else 'ns_p20_40_60_80'
    ncol = len(per) + 1
    blup = np.random.default_rng(31 + ncol).normal(0.0, 0.1, ncol)
    ref = r_mmt(blup, x, spec_key)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
        res = find_mmt_blup(x, blup, fun='ns', knots=np.quantile(x, np.array(per) / 100.0))
    assert res.get('method') != 'median_fallback', f"median fallback instead of an ns basis: {res.get('error')}"
    assert_close(res['basis_matrix'], ref.basis, rtol=1e-9, what='ns basis on the 1st-99th percentile grid')
    assert int(res['percentile']) == ref.percentile
    assert float(res['mmt']) == pytest.approx(ref.temperature, rel=1e-9)


# ---- plain tests: what is already faithful -------------------------------------------------------------------
def test_default_spec_pooled_mmt_matches_r_when_boundary_knots_coincide():
    """Lancet default bs/deg 2/P10-75-90 == the hard-coded spec: equals R on the winsorised series (P1 = min, P99 = max).
    Must keep passing whatever the N3 fix does (and after the M1 boundary fix)."""
    run = pipeline('bs2_p10_75_90')
    assert_pooled_mmt_matches_r(run)
    assert all(run.pm['region_mmts'][n]['method'] != 'median_fallback' for n in run.names)


def test_find_mmt_blup_default_bs_spec_matches_r_on_winsorised_series():
    """find_mmt_blup(x, blup) with its documented defaults (bs, degree 2, knots P10/75/90, search 1st-99th percentile)."""
    from centering import find_mmt_blup
    x = region_data('Wales').t
    blup = np.random.default_rng(12).normal(0.0, 0.1, 5)
    ref = r_mmt(blup, x, 'bs2_p10_75_90')
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
        res = find_mmt_blup(x, blup)
    assert res['method'] == 'blup_optimization'
    assert_close(res['basis_matrix'], ref.basis, rtol=1e-9, what='bs basis')
    assert_close(res['risk_values'], ref.risk, rtol=1e-9, what='bvar %*% blup')
    assert int(res['percentile']) == ref.percentile
    assert float(res['mmt']) == pytest.approx(ref.temperature, rel=1e-9)


def test_region_mmt_percentiles_lie_on_the_1_to_99_grid_and_temperature_is_that_quantile():
    """02.secondstage.R: minperccity in 1:99 and mintempcity = quantile(tmean, minperccity/100)."""
    run = pipeline('bs2_p10_75_90')
    for n, reg in zip(run.names, run.regs):
        got = run.pm['region_mmts'][n]
        pc = int(got['percentile'])
        assert 1 <= pc <= 99
        np2r('n3_tm', reg.t)
        assert float(got['mmt']) == pytest.approx(float(rget(f'quantile(n3_tm, {pc}/100)')[0]), rel=1e-12), n


# ==============================================================================================================
# mvmeta-blup-4 (nit): what "pooled MMT" is
# ==============================================================================================================
POOLED_KEYS = ('region_mmts', 'pooled_mmt_median', 'pooled_mmt_mean', 'pooled_mmt_std', 'individual_mmts')


def test_pooled_mmt_keys_and_semantics_are_median_mean_std_of_regional_mmt_temperatures():
    """Plain (documents what the result IS): pooled_mmt_median / _mean / _std are statistics of the regional MMT
    temperatures, listed in region order in individual_mmts; pooled_mmt_median is a temperature, not a percentile."""
    run = pipeline('bs2_p10_75_90')
    pm = run.pm
    assert set(POOLED_KEYS) <= set(pm), f'missing keys: {set(POOLED_KEYS) - set(pm)}'
    assert list(pm['region_mmts']) == run.names
    temps = np.array([pm['region_mmts'][n]['mmt'] for n in run.names], dtype=float)
    np.testing.assert_allclose(np.asarray(pm['individual_mmts'], dtype=float), temps, rtol=0, atol=0)
    assert pm['pooled_mmt_median'] == pytest.approx(np.median(temps), rel=1e-12)
    assert pm['pooled_mmt_mean'] == pytest.approx(np.mean(temps), rel=1e-12)
    assert pm['pooled_mmt_std'] == pytest.approx(np.std(temps), rel=1e-12)
    pcs = np.array([pm['region_mmts'][n]['percentile'] for n in run.names], dtype=float)
    assert not np.isclose(pm['pooled_mmt_median'], np.median(pcs)), 'the median is of temperatures (deg C), not of percentiles'


@known_defect(THEME, 'mvmeta-blup-4', note="R's minperccountry = median(minperccity) (a percentile) is not exposed")
def test_pooled_result_exposes_rs_country_level_median_percentile():
    """02.secondstage.R:74 `minperccountry <- median(minperccity)`: the audit's fix sketch adds the median MMT percentile to
    the pooled result under an explicit key (e.g. 'pooled_mmt_percentile_median').  Any top-level key that names a
    percentile and holds R's value is accepted."""
    run = pipeline('bs2_p10_75_90')
    ref = r_reference(run)
    np2r('n3_pcs', np.array([rf.percentile for rf in ref], dtype=float))
    minperccountry = float(rget('median(n3_pcs)')[0])
    cand = {k: v for k, v in run.pm.items()
            if 'perc' in k.lower() and isinstance(v, (int, float, np.integer, np.floating))}
    assert cand, f'no percentile-valued key in the pooled result (keys: {sorted(run.pm)})'
    assert any(abs(float(v) - minperccountry) < 1e-9 for v in cand.values()), \
        f'R minperccountry = {minperccountry}, pooled percentile keys: {cand}'


# ==============================================================================================================
# mvmeta-blup-5: duplicate region names
# ==============================================================================================================
DUP_REGIONS = ['East', 'London', 'E-Mid', 'N-East']
DUP_LABELS = ['East', 'East', 'E-Mid', 'N-East']        # London is (mis)labelled 'East'


def r_meta_predictors(regs):
    """02.secondstage.R:22-23 by position: intercept, avgtmean, rangetmean."""
    cols = []
    for i, reg in enumerate(regs):
        np2r(f'n3_mp{i}', reg.t)
        cols.append(rget(f'c(1, mean(n3_mp{i}), diff(range(n3_mp{i})))'))
    return np.array(cols)


@lru_cache(maxsize=None)
def _duplicate_run():
    """(analysis or None, error) for the 4 regions above with London labelled 'East': the analysis has been taken through
    meta-analysis, BLUPs and pooled MMTs; None + the ValueError if a duplicate label is rejected."""
    from multi_location import MultiLocationDLNM
    an = MultiLocationDLNM()
    regs = [region_data(n) for n in DUP_REGIONS]
    try:
        add_regions(an, regs, labels=DUP_LABELS)
        with _quiet():
            an.fit_meta_analysis()
            an.calculate_blups()
            an.calculate_pooled_mmts()
    except ValueError as exc:                                # rejecting a duplicate label is an acceptable fix
        return None, exc
    return an, None


@known_defect(THEME, 'mvmeta-blup-5', note='meta-predictors keyed by region name: the second "East" gets the first one\'s')
def test_duplicate_region_name_gets_its_own_meta_predictors_or_is_rejected():
    an, _ = _duplicate_run()
    if an is None:
        return                                                # duplicate rejected with ValueError
    from meta_analysis import MVMeta
    expected = r_meta_predictors([region_data(n) for n in DUP_REGIONS])
    X = np.asarray(an.mv_model.X, dtype=float)
    assert X.shape == expected.shape
    assert_close(X, expected, rtol=1e-12,
                 what="meta-regression design [1, avg_temp, temp_range] of London labelled 'East'")
    # observable consequence: the meta-regression equals the one fitted with R's (positional) design matrix
    y = np.array([res['reduced']['coefficients'] for res in an.region_results])
    S = np.array([res['reduced']['vcov'] for res in an.region_results])
    with _quiet():
        ref = MVMeta(method='reml').fit(y, S, expected)
    assert_close(an.mv_model.coefficients, ref.coefficients, rtol=1e-10, what='meta-regression coefficients')


@known_defect(THEME, 'mvmeta-blup-5', note='region_mmts is keyed by name: the second "East" overwrites the first')
def test_duplicate_region_name_keeps_one_mmt_entry_per_region_or_is_rejected():
    an, _ = _duplicate_run()
    if an is None:
        return
    pm = an.pooled_mmts
    assert len(pm['individual_mmts']) == len(DUP_REGIONS)
    assert len(pm['region_mmts']) == len(DUP_REGIONS), \
        f'{len(DUP_REGIONS)} regions but region_mmts has {len(pm["region_mmts"])} entries: {list(pm["region_mmts"])}'


def test_unique_region_names_meta_predictors_match_r_in_input_order():
    """Plain: with unique labels X = [1, avg_temp, temp_range] of each region, in the order the regions were added
    (shuffled input order included), as 02.secondstage.R builds avgtmean / rangetmean."""
    from multi_location import MultiLocationDLNM
    order = ['N-East', 'East', 'London', 'E-Mid']
    regs = [region_data(n) for n in order]
    an = MultiLocationDLNM()
    add_regions(an, regs)
    with _quiet():
        an.fit_meta_analysis()
    assert an.region_names == order
    assert_close(np.asarray(an.mv_model.X, dtype=float), r_meta_predictors(regs), rtol=1e-12, what='meta-regression design')
    assert an.mv_model.coefficients.shape[1] == 5 and an.mv_model.coefficients.shape[0] == 3


def test_fit_meta_analysis_needs_two_regions():
    """Plain: a single region cannot be meta-analysed (ValueError, as mvmeta needs several studies)."""
    from multi_location import MultiLocationDLNM
    an = MultiLocationDLNM()
    add_regions(an, [region_data('London')])
    with pytest.raises(ValueError):
        an.fit_meta_analysis()


# ==============================================================================================================
# mvmeta-blup-11: get_model_summary() of a region must report that region's model
# ==============================================================================================================
SUMMARY_ORDER = ['London', 'Wales', 'East']


@lru_cache(maxsize=None)
def _three_region_fit():
    """London, Wales, East fitted in this order through add_region_analysis (PyDLNM) and refitted independently in R."""
    from multi_location import MultiLocationDLNM
    regs = [region_data(n) for n in SUMMARY_ORDER]
    an = MultiLocationDLNM()
    add_regions(an, regs)
    refs = [r_firststage(f's{i}', reg) for i, reg in enumerate(regs)]
    return an, regs, refs


def _summary_numbers(summary):
    return dict(dispersion=float(r2np(summary.rx2('dispersion'))[0]), deviance=float(r2np(summary.rx2('deviance'))[0]),
                df_residual=int(r2np(summary.rx2('df.residual'))[0]))


@known_defect(THEME, 'mvmeta-blup-11', note="summary(fitted_model) evaluates the R global, i.e. the last region's model")
def test_get_model_summary_returns_each_regions_own_model_after_later_fits():
    an, regs, refs = _three_region_fit()
    disp = [ref.dispersion for ref in refs]
    assert max(disp) / min(disp) > 1.01, 'test setup: the three regions must have distinguishable dispersions'
    wrong = []
    for i, (name, ref) in enumerate(zip(SUMMARY_ORDER, refs)):
        got = _summary_numbers(an.region_results[i]['glm_interface'].get_model_summary())
        ok = (abs(got['dispersion'] - ref.dispersion) <= 1e-6 * ref.dispersion
              and abs(got['deviance'] - ref.deviance) <= 1e-6 * ref.deviance and got['df_residual'] == ref.df_residual)
        if not ok:
            wrong.append(f'{name}: summary dispersion {got["dispersion"]:.6f} / deviance {got["deviance"]:.3f} vs R '
                         f'{ref.dispersion:.6f} / {ref.deviance:.3f}')
    assert not wrong, 'get_model_summary() does not describe the region\'s own model: ' + '; '.join(wrong)


def test_get_model_summary_of_the_last_fitted_region_matches_r():
    """Plain: the last fitted region is the one the R global points to, so its summary is right today (the defect needs a
    later fit); df.residual included."""
    an, regs, refs = _three_region_fit()
    got = _summary_numbers(an.region_results[-1]['glm_interface'].get_model_summary())
    ref = refs[-1]
    assert got['dispersion'] == pytest.approx(ref.dispersion, rel=1e-6)
    assert got['deviance'] == pytest.approx(ref.deviance, rel=1e-6)
    assert got['df_residual'] == ref.df_residual


def test_region_first_stage_results_match_r_for_every_region_after_later_fits():
    """Plain: cb_coef / reduced coefficients are captured right after each fit, so every region's pipeline numbers equal
    R's first stage (crossbasis + glm + crossreduce, centred at the mean) for that region."""
    an, regs, refs = _three_region_fit()
    for i, (name, ref) in enumerate(zip(SUMMARY_ORDER, refs)):
        gi = an.region_results[i]['glm_interface']
        assert_close(np.asarray(gi.cb_coef), ref.cb_coef, rtol=1e-7, what=f'{name}: cross-basis coefficients')
        assert_close(np.asarray(an.region_results[i]['reduced']['coefficients']), ref.red_coef, rtol=1e-6,
                     what=f'{name}: reduced coefficients')
    assert an.region_names == SUMMARY_ORDER


# ==============================================================================================================
# mvmeta-blup-12: response given as a list
# ==============================================================================================================
@lru_cache(maxsize=None)
def _london_reference():
    reg = region_data('London')
    return reg, r_firststage('l0', reg)


def _add_london(y, dates=None, name='London'):
    from multi_location import MultiLocationDLNM
    reg, _ = _london_reference()
    an = MultiLocationDLNM()
    with _quiet():
        res = an.add_region_analysis(name, make_crossbasis(reg.t), y, reg.dates if dates is None else dates, dfseas=DFSEAS)
    return res


@known_defect(THEME, 'mvmeta-blup-12', note='fit_dlnm_model indexes y[~nan_mask] without converting: TypeError for a list')
@pytest.mark.parametrize('kind', ['floats', 'ints', 'with_nan'])
def test_add_region_analysis_accepts_a_list_response(kind):
    """Docstring: y is array-like; R takes any numeric vector (NA included).  The reduced coefficients equal R's
    crossreduce() of glm(death ~ cb + dow + ns(date)) on the same response."""
    reg, _ = _london_reference()
    y = reg.y.copy()
    if kind == 'with_nan':
        y[np.random.default_rng(7).choice(len(y), 20, replace=False)] = np.nan
        ref = r_firststage('l_nan', reg, y)
        assert ref.nobs == len(y) - LAG - int(np.isnan(y[LAG:]).sum()), 'test setup: R drops the NA rows and the lag rows'
        ylist = [float(v) for v in y]
    else:
        ref = _london_reference()[1]
        ylist = [float(v) for v in y] if kind == 'floats' else [int(v) for v in y]
    res = _add_london(ylist)
    assert_close(np.asarray(res['reduced']['coefficients']), ref.red_coef, rtol=1e-6, what=f'reduced coefficients, list of {kind}')


@pytest.mark.parametrize('kind', ['ndarray', 'float32_as_float64', 'int64', 'pandas_series'])
def test_add_region_analysis_response_containers_that_work_match_r(kind):
    """Plain: ndarray / integer ndarray / pandas Series responses (already accepted) reproduce R's first stage."""
    reg, ref = _london_reference()
    y = {'ndarray': reg.y, 'float32_as_float64': reg.y.astype(np.float32).astype(np.float64),
         'int64': reg.y.astype(np.int64), 'pandas_series': pd.Series(reg.y)}[kind]
    res = _add_london(y)
    assert_close(np.asarray(res['reduced']['coefficients']), ref.red_coef, rtol=1e-6, what=f'reduced coefficients, y {kind}')


@pytest.mark.parametrize('kind', ['datetimeindex', 'datetime64'])
def test_add_region_analysis_date_containers_that_work_match_r(kind):
    """Plain: dates as a DatetimeIndex or a datetime64 array (besides the documented pd.Series) give R's first stage."""
    reg, ref = _london_reference()
    dates = pd.DatetimeIndex(reg.dates) if kind == 'datetimeindex' else reg.dates.to_numpy()
    res = _add_london(reg.y, dates=dates)
    assert_close(np.asarray(res['reduced']['coefficients']), ref.red_coef, rtol=1e-6, what=f'reduced coefficients, dates {kind}')
