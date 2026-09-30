"""Gap cox_clogit_link_not_inferred: the link of Cox / conditional-logit / conditional-Poisson models (getlink) and what
hangs on it in crosspred, crossreduce and attrdl.

R (dlnm:::getlink) knows the link of the models used for the standard time-stratified case-crossover and Cox DLNM designs:

    class "coxph"  -> "log"                         survival::coxph
    class "clogit" -> "logit"  (clogit is also a coxph)   survival::clogit      (time-stratified case-crossover)
    class "gnm"/"glm" -> family$link ("log")        gnm::gnm(family=poisson, eliminate=stratum)  (conditional Poisson)

so that crosspred()/crossreduce() return the exponentiated fields (allRRfit, matRRfit, cumRRfit, RRlow, RRhigh, ...) and
attrdl(model=) accepts the model. PyDLNM's model_utils.getlink() has no rule for statsmodels PHReg, ConditionalLogit or
ConditionalPoisson, so it returns None: crosspred/crossreduce then silently produce linear-scale output only
(low/high instead of RRlow/RRhigh, no allRRfit/matRRfit/cumRRfit) and attrdl(model=) raises "'model' must have a log or
logit link function".

Designs (one per statsmodels class; R fits the same model; every R number is computed at test run time):
  cox      PHReg(ties='breslow')        vs survival::coxph(ties="breslow"), exposure histories (matrix x, lags 0..7),
                                        a covariate before the cross-basis block.
  clogit   ConditionalLogit             vs survival::clogit, time-stratified case-crossover (stratum = year x month x
                                        day of week, one case per risk set, the other days of the stratum as controls).
  condpois ConditionalPoisson           vs R conditional Poisson. The gnm package is not installed here, so the R
                                        reference model is glm(family=poisson) with one dummy per stratum: the profile
                                        (conditional) likelihood of the slopes is identical and its class ("glm") gives
                                        the same getlink result ("log") as a gnm object does.

  negbin   statsmodels discrete NegativeBinomial (log link, coefficient vector ends with 'alpha') vs MASS::glm.nb
           (class negbin/glm/lm -> "log"). Only the exact tests run on it: R's vcov(glm.nb) is conditional on theta
           while statsmodels' includes alpha, so end-to-end allse differs by ~1e-3 (a separate finding,
           negbin_vcov_conditional_on_theta, not tested here).

Comparison design. The statsmodels and R fits of the same model agree only to optimiser precision (PHReg vs coxph 1e-8
in the coefficients, statsmodels' numerical Hessians 1e-6 in the vcov), so the tests are split in two:
  * EXACT tests (rtol 1e-9, deterministic arithmetic): R crosspred/crossreduce/attrdl run on the cross-basis coefficients
    and vcov of the *statsmodels fit* (handed to R as coef=/vcov=) with model.link = what R's getlink returns for the
    native R model object. They test exactly what the gap is about: does PyDLNM, given only the model object, infer the
    link R infers and produce R's output on top of it?
  * END-TO-END tests (rtol 1e-5): R on the R model object versus PyDLNM on the statsmodels model object.
The neighbouring behaviour that already works (explicit model_link=, the coef=/vcov= route of attrdl) is covered by
plain tests; tests that assert R's behaviour and fail today carry @known_defect.
"""
import contextlib
import io
import os
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from rhelpers import assert_close, chicago, known_defect, np2r, r, rget

REPO = Path(__file__).resolve().parents[1]
GAP, KEY = 'GAP', 'cox_clogit_link_not_inferred'
EXACT = 1e-9          # identical arithmetic on identical numbers (agrees to ~1e-15)
E2E = 1e-5            # statsmodels fit vs R fit of the same model (optimiser / numerical Hessian precision)
KINDS = ('cox', 'clogit', 'condpois')              # models that have an R counterpart fitted on the same data
ALL_KINDS = KINDS + ('negbin',)
NLAG = 7              # lags 0..7
P = 'cxl_'            # prefix of every R global created here


# =====================================================================================================================
# environment guards and small helpers
# =====================================================================================================================
def _good_r_home():
    return os.path.dirname(str(r('.Library')[0]))


_R_HOME = _good_r_home()


@pytest.fixture(autouse=True)
def _keep_r_home():
    """PyDLNM modules may overwrite os.environ['R_HOME'] (audit finding Q2); R loads LAPACK lazily from it."""
    os.environ['R_HOME'] = _R_HOME
    yield
    os.environ['R_HOME'] = _R_HOME


@contextlib.contextmanager
def _quiet():
    """PyDLNM prints progress lines and warns about unrelated things; statsmodels warns about dropped groups."""
    try:
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter('ignore')
            yield
    finally:
        os.environ['R_HOME'] = _R_HOME


def _require_r_packages(*pkgs):
    """Skip unless the R packages are installed, then attach them. (rhelpers.require_r_packages indexes the invisible
    result of requireNamespace, which rpy2 3.6 returns as None.)"""
    for p in pkgs:
        if not bool(r(f'isTRUE(suppressWarnings(requireNamespace("{p}", quietly=TRUE)))')[0]):
            pytest.skip(f'R package {p} not installed')
        r(f'suppressMessages(library({p}))')


def _rstr(expr):
    return str(r(expr)[0])


def _rhas(obj, field):
    return bool(r(f'!is.null({obj}[["{field}"]])')[0])


def _rfield(obj, field):
    return rget(f'as.numeric({obj}[["{field}"]])')


def _rmat(obj, field):
    return rget(f'matrix(as.numeric({obj}[["{field}"]]), nrow(as.matrix({obj}[["{field}"]])))')


def _pyfield(obj, field):
    val = getattr(obj, field, None)
    assert val is not None, (f'{type(obj).__name__} has no field {field!r} although R returns it; attributes: '
                             f'{sorted(a for a in vars(obj) if not a.startswith("_"))}')
    return np.asarray(val, dtype=float)


def _load_r_attrdl():
    _require_r_packages('tsModel')
    if not bool(r(f'exists("{P}attr", envir=globalenv(), inherits=FALSE)')[0]):
        path = REPO / '2015_gasparrini_Lancet_Rcodedata-master' / 'attrdl.R'
        r(f'{P}attr <- new.env(); sys.source("{path}", envir={P}attr)')


# =====================================================================================================================
# the three designs: data, statsmodels fit, R fit, cross-basis in R and Python
# =====================================================================================================================
_DS = {}
ARGVAR = dict(fun='ns', df=3)
ARGLAG = dict(fun='ns', df=2)
R_CB_ARGS = 'argvar=list(fun="ns", df=3), arglag=list(fun="ns", df=2)'


def _pushed_matrix(name, m):
    m = np.asarray(m, dtype=float)
    np2r(name, m)
    r(f'{name} <- matrix({name}, {m.shape[0]}, {m.shape[1]})')


def _finish(ds, cb_block_names):
    """Fill in what is common to the designs: cb block of coef/vcov (extracted here from the statsmodels fit,
    independently of PyDLNM), the R native link, the R-side copy of that coef/vcov, prediction grid and centring."""
    k = ds.kind
    params = np.asarray(ds.fit.params, dtype=float)
    cov = np.asarray(ds.fit.cov_params(), dtype=float)
    index = getattr(ds.fit.params, 'index', None)
    names = [str(n) for n in (index if index is not None else ds.fit.model.exog_names)]
    sel = [names.index(n) for n in cb_block_names]                   # by name (the negbin vector ends with 'alpha')
    ds.coef, ds.vcov = params[sel], cov[np.ix_(sel, sel)]
    ds.ncb = len(sel)
    ds.cb_r, ds.m_r = f'{P}{k}_cb', f'{P}{k}_m'
    ds.link_r = _rstr(f'dlnm:::getlink({ds.m_r}, class({ds.m_r}))')
    np2r(f'{P}{k}_coef', ds.coef)
    _pushed_matrix(f'{P}{k}_vcov', ds.vcov)
    r(f'{P}{k}_link <- "{ds.link_r}"')
    return ds


def _build_cox():
    from basis import CrossBasis
    from statsmodels.duration.hazard_regression import PHReg
    _require_r_packages('survival')
    temp = chicago()['temp']
    n = 700
    rng = np.random.default_rng(20240607)
    starts = rng.integers(NLAG, len(temp), n)
    hist = np.column_stack([temp[starts - j] for j in range(NLAG + 1)])        # column j = exposure at lag j
    age = rng.normal(size=n)
    with _quiet():
        cb = CrossBasis(hist, lag=[0, NLAG], argvar=dict(ARGVAR), arglag=dict(ARGLAG))
    beta = rng.normal(0, 0.03, cb.shape[1])
    lp = 0.2 * age + cb.basis @ beta
    t_event, t_cens = rng.exponential(1 / np.exp(lp)), rng.exponential(1.5, n)
    time, status = np.minimum(t_event, t_cens), (t_event <= t_cens).astype(float)
    X = pd.DataFrame(np.column_stack([age, cb.basis]), columns=['age'] + list(cb.colnames))
    with _quiet():
        fit = PHReg(time, X, status=status, ties='breslow').fit()
    np2r(f'{P}cox_hist', hist)
    _pushed_matrix(f'{P}cox_hist', hist)
    for nm, v in (('t', time), ('s', status), ('age', age)):
        np2r(f'{P}cox_{nm}', v)
    r(f'{P}cox_cb <- crossbasis({P}cox_hist, lag=c(0,{NLAG}), {R_CB_ARGS})')
    r(f'{P}cox_m <- survival::coxph(survival::Surv({P}cox_t, {P}cox_s) ~ {P}cox_age + {P}cox_cb, ties="breslow")')
    ds = SimpleNamespace(kind='cox', fit=fit, cbP=cb, exposure=hist, cases=status, attr_dirs=('back',),
                         at=np.linspace(np.percentile(hist, 2), np.percentile(hist, 98), 9),
                         cen=float(np.median(hist)))
    return _finish(ds, list(cb.colnames))


def _series_design(kind):
    """A 4-year slice of the Chicago series with the time-stratified strata (year x month x day of week), the PyDLNM
    cross-basis, and the R cross-basis; rows with an incomplete lag window or a single-row stratum are dropped."""
    from basis import CrossBasis
    ch = chicago()
    a, b = 800, 800 + 1461
    x, death, date = ch['temp'][a:b], ch['death'][a:b], ch['date'][a:b]
    d = pd.to_datetime(date, unit='D')
    sid = pd.factorize(d.year.astype(str) + '-' + d.month.astype(str) + '-' + d.dayofweek.astype(str))[0]
    with _quiet():
        cb = CrossBasis(x, lag=NLAG, argvar=dict(ARGVAR), arglag=dict(ARGLAG))
    complete = ~np.isnan(cb.basis).any(axis=1)
    keep = complete & (np.bincount(sid[complete], minlength=sid.max() + 1)[sid] >= 2)
    z = np.random.default_rng(7 if kind == 'clogit' else 8).normal(size=len(x))      # a covariate before the cb block
    np2r(f'{P}{kind}_x', x)
    r(f'{P}{kind}_cb <- crossbasis({P}{kind}_x, lag={NLAG}, {R_CB_ARGS})')
    return SimpleNamespace(x=x, death=death, sid=sid, cb=cb, keep=keep, z=z)


def _build_clogit():
    from statsmodels.discrete.conditional_models import ConditionalLogit
    _require_r_packages('survival')
    s = _series_design('clogit')
    rng = np.random.default_rng(99)
    days = np.flatnonzero(s.keep)
    prob = s.death[days].astype(float)
    cdays = rng.choice(days, 900, p=prob / prob.sum())              # case days, weighted by the daily death counts
    rows, cid, case = [], [], []
    for i, d in enumerate(cdays):
        stratum = days[s.sid[days] == s.sid[d]]
        rows += [d] + [c for c in stratum if c != d]                 # the case, then the controls of its stratum
        cid += [i] * len(stratum)
        case += [1.0] + [0.0] * (len(stratum) - 1)
    rows, cid, case = np.array(rows), np.array(cid), np.array(case)
    X = pd.DataFrame(np.column_stack([s.z[rows], s.cb.basis[rows]]), columns=['z'] + list(s.cb.colnames))
    with _quiet():
        fit = ConditionalLogit(case, X, groups=cid).fit(method='newton')
    for nm, v in (('rows', rows + 1.0), ('cid', cid.astype(float)), ('case', case), ('z', s.z[rows])):
        np2r(f'{P}clogit_{nm}', v)
    r(f'{P}clogit_dat <- data.frame(case={P}clogit_case, z={P}clogit_z, cid=factor({P}clogit_cid)); '
      f'{P}clogit_dat${P}clogit_cb <- {P}clogit_cb[as.integer({P}clogit_rows), ]')
    r(f'{P}clogit_m <- survival::clogit(case ~ z + {P}clogit_cb + strata(cid), data={P}clogit_dat)')
    ds = SimpleNamespace(kind='clogit', fit=fit, cbP=s.cb, exposure=s.x, cases=s.death.astype(float),
                         attr_dirs=('forw', 'back'), at=np.arange(-10.0, 31.0, 5.0), cen=15.0)
    return _finish(ds, list(s.cb.colnames))


def _build_condpois():
    from statsmodels.discrete.conditional_models import ConditionalPoisson
    s = _series_design('condpois')
    k = s.keep
    X = pd.DataFrame(np.column_stack([s.z[k], s.cb.basis[k]]), columns=['z'] + list(s.cb.colnames))
    with _quiet():
        fit = ConditionalPoisson(s.death[k].astype(float), X, groups=s.sid[k]).fit(method='newton')
    for nm, v in (('y', s.death.astype(float)), ('sid', s.sid.astype(float)), ('keep', k.astype(float)),
                  ('z', s.z)):
        np2r(f'{P}condpois_{nm}', v)
    r(f'{P}condpois_dat <- data.frame(y={P}condpois_y, z={P}condpois_z, sid=factor({P}condpois_sid), '
      f'keep={P}condpois_keep == 1); {P}condpois_dat${P}condpois_cb <- {P}condpois_cb')
    r(f'{P}condpois_m <- glm(y ~ z + {P}condpois_cb + sid, family=poisson, data={P}condpois_dat, subset=keep)')
    ds = SimpleNamespace(kind='condpois', fit=fit, cbP=s.cb, exposure=s.x, cases=s.death.astype(float),
                         attr_dirs=('forw', 'back'), at=np.arange(-10.0, 31.0, 5.0), cen=15.0)
    return _finish(ds, list(s.cb.colnames))


def _build_negbin():
    import statsmodels.api as sm
    _require_r_packages('MASS')
    s = _series_design('negbin')
    k = s.keep
    X = pd.DataFrame(np.column_stack([np.ones(k.sum()), s.z[k], s.cb.basis[k]]),
                     columns=['const', 'z'] + list(s.cb.colnames))
    with _quiet():
        fit = sm.NegativeBinomial(s.death[k].astype(float), X).fit(disp=0, maxiter=300)
    for nm, v in (('y', s.death.astype(float)), ('keep', k.astype(float)), ('z', s.z)):
        np2r(f'{P}negbin_{nm}', v)
    r(f'{P}negbin_dat <- data.frame(y={P}negbin_y, z={P}negbin_z, keep={P}negbin_keep == 1); '
      f'{P}negbin_dat${P}negbin_cb <- {P}negbin_cb')
    r(f'{P}negbin_m <- MASS::glm.nb(y ~ z + {P}negbin_cb, data={P}negbin_dat, subset=keep)')
    ds = SimpleNamespace(kind='negbin', fit=fit, cbP=s.cb, exposure=s.x, cases=s.death.astype(float),
                         attr_dirs=('forw', 'back'), at=np.arange(-10.0, 31.0, 5.0), cen=15.0)
    return _finish(ds, list(s.cb.colnames))


_BUILDERS = {'cox': _build_cox, 'clogit': _build_clogit, 'condpois': _build_condpois, 'negbin': _build_negbin}


def _get_ds(kind):
    if kind not in _DS:
        _DS[kind] = _BUILDERS[kind]()
    return _DS[kind]


LINK_NOTE = ('getlink() has no rule for statsmodels PHReg / ConditionalLogit / ConditionalPoisson (returns None): '
             'no RR fields in crosspred/crossreduce, attrdl(model=) raises')


def _mark_link_defect(request, kind):
    """The models whose link PyDLNM cannot infer carry the known-defect mark (R infers it). Applied per test
    invocation (pytest.param cannot carry the no-op mark that known_defect() returns when PYDLNM_XFAIL_OFF is set)."""
    if kind != 'negbin':
        request.applymarker(known_defect(GAP, KEY, note=LINK_NOTE))


@pytest.fixture(scope='module', params=ALL_KINDS)
def ds(request):
    """Every design, unmarked: for behaviour that already works (name selection, explicit model_link, coef= route)."""
    return _get_ds(request.param)


@pytest.fixture(params=ALL_KINDS)
def ds_link(request):
    """Every design, for tests that need the link inferred from the model object (negbin already works)."""
    _mark_link_defect(request, request.param)
    return _get_ds(request.param)


@pytest.fixture(params=KINDS)
def ds_e2e(request):
    """Link-inference tests against R's own model fit: the designs whose R model has the same vcov as the
    statsmodels fit (not negbin, see the module docstring)."""
    _mark_link_defect(request, request.param)
    return _get_ds(request.param)


# =====================================================================================================================
# R references
# =====================================================================================================================
def _r_at(ds, at):
    np2r(f'{P}at', np.asarray(at, dtype=float))
    return f'{P}at'


def r_crosspred_exact(ds, cumul=False, lag=None, at=None):
    """R crosspred on the statsmodels coef/vcov with the link R infers for the native R model object."""
    at = ds.at if at is None else at
    k = ds.kind
    extra = f', lag={lag}' if lag is not None else ''
    r(f'{P}pred <- crosspred({ds.cb_r}, coef={P}{k}_coef, vcov={P}{k}_vcov, model.link={P}{k}_link, at={_r_at(ds, at)}, '
      f'cen={ds.cen!r}, cumul={"TRUE" if cumul else "FALSE"}{extra})')
    return f'{P}pred'


def _assert_fields(py, robj, fields, rtol, what):
    for f in fields:
        assert _rhas(robj, f), f'R result lacks {f}: test bug'
        a = _pyfield(py, f)
        ref = _rmat(robj, f) if np.ndim(a) > 1 else _rfield(robj, f)
        assert_close(a, ref.reshape(a.shape) if ref.size == a.size else ref, rtol=rtol, what=f'{what} {f}')


LOG_FIELDS = ['allfit', 'allse', 'matfit', 'matse', 'allRRfit', 'allRRlow', 'allRRhigh', 'matRRfit', 'matRRlow',
              'matRRhigh']
LINEAR_ONLY_FIELDS = ['alllow', 'allhigh', 'matlow', 'mathigh']
CUM_FIELDS = ['cumfit', 'cumse', 'cumRRfit', 'cumRRlow', 'cumRRhigh']


# =====================================================================================================================
# getlink
# =====================================================================================================================
def test_getlink_matches_R_native_link(ds_link):
    """R: coxph -> "log", clogit -> "logit", gnm/glm poisson -> "log" (dlnm:::getlink on the native R model object)."""
    ds = ds_link
    from model_utils import getlink
    assert getlink(ds.fit) == ds.link_r, f'{type(ds.fit).__name__}: getlink={getlink(ds.fit)!r}, R {ds.link_r!r}'


def test_R_native_links_are_the_expected_ones(ds):
    """Guard for the reference itself: the links R infers for these designs."""
    assert ds.link_r == {'cox': 'log', 'clogit': 'logit', 'condpois': 'log', 'negbin': 'log'}[ds.kind]


def test_getlink_user_link_takes_precedence_like_R(ds):
    """R: getlink returns model.link unchanged when it is given."""
    from model_utils import getlink
    assert getlink(ds.fit, model_link='identity') == _rstr(f'dlnm:::getlink({ds.m_r}, class({ds.m_r}), "identity")')
    assert getlink(ds.fit, model_link='logit') == 'logit'


# =====================================================================================================================
# crosspred with the model object
# =====================================================================================================================
def test_crosspred_model_link_attribute_matches_R(ds_link):
    """CrossPred.model_link is what R records in $model.link for the native model (log / logit / log)."""
    ds = ds_link
    from prediction import crosspred
    with _quiet():
        pp = crosspred(ds.cbP, model=ds.fit, at=ds.at, cen=ds.cen)
    robj = r_crosspred_exact(ds)
    assert pp.model_link == _rstr(f'{robj}[["model.link"]]') == ds.link_r


def test_crosspred_returns_the_exponentiated_fields_like_R(ds_link):
    """allRRfit/allRRlow/allRRhigh/matRR* exist and equal R's; the linear-scale CI fields R does not return are absent."""
    ds = ds_link
    from prediction import crosspred
    with _quiet():
        pp = crosspred(ds.cbP, model=ds.fit, at=ds.at, cen=ds.cen)
    robj = r_crosspred_exact(ds)
    _assert_fields(pp, robj, LOG_FIELDS, EXACT, f'{ds.kind} crosspred')
    for f in LINEAR_ONLY_FIELDS:
        assert not _rhas(robj, f)
        assert getattr(pp, f, None) is None, f'PyDLNM returns linear-scale field {f} although R does not (link log/logit)'


def test_crosspred_cumulative_fields_match_R(ds_link):
    """cumul=TRUE: cumfit, cumse and the exponentiated cumRRfit/cumRRlow/cumRRhigh."""
    ds = ds_link
    from prediction import crosspred
    with _quiet():
        pp = crosspred(ds.cbP, model=ds.fit, at=ds.at, cen=ds.cen, cumul=True)
    robj = r_crosspred_exact(ds, cumul=True)
    _assert_fields(pp, robj, LOG_FIELDS + CUM_FIELDS, EXACT, f'{ds.kind} crosspred(cumul)')


def test_crosspred_lag_subperiod_fields_match_R(ds_link):
    """lag sub-period: the same inferred link gives the same exponentiated lag-specific and overall fields."""
    ds = ds_link
    from prediction import crosspred
    with _quiet():
        pp = crosspred(ds.cbP, model=ds.fit, at=ds.at, cen=ds.cen, lag=[1, 4])
    robj = r_crosspred_exact(ds, lag='c(1,4)')
    _assert_fields(pp, robj, ['allfit', 'allse', 'allRRfit', 'allRRlow', 'allRRhigh', 'matRRfit', 'matRRlow',
                              'matRRhigh'], EXACT, f'{ds.kind} crosspred(lag=1:4)')


def test_crosspred_selects_the_cb_block_by_name(ds):
    """The design has a covariate before the cross-basis block: coefficients / vcov are those of the cb block (R: grep
    on the coefficient names), for every statsmodels class (names from the exog names)."""
    from prediction import crosspred
    with _quiet():
        pp = crosspred(ds.cbP, model=ds.fit, at=ds.at, cen=ds.cen)
    assert_close(pp.coefficients, ds.coef, rtol=1e-14, what='coefficients')
    assert_close(pp.vcov, ds.vcov, rtol=1e-14, what='vcov')


def test_crosspred_model_vs_R_model_end_to_end(ds_e2e):
    """PyDLNM(model=statsmodels fit) vs R crosspred(model=R fit) on the same data: allfit/allse/RR fields (1e-5)."""
    ds = ds_e2e
    from prediction import crosspred
    with _quiet():
        pp = crosspred(ds.cbP, model=ds.fit, at=ds.at, cen=ds.cen)
    r(f'{P}pred_e2e <- crosspred({ds.cb_r}, {ds.m_r}, at={_r_at(ds, ds.at)}, cen={ds.cen!r})')
    _assert_fields(pp, f'{P}pred_e2e', LOG_FIELDS, E2E, f'{ds.kind} end-to-end')
    assert pp.model_link == _rstr(f'{P}pred_e2e[["model.link"]]')


def test_crosspred_explicit_model_link_route_is_faithful(ds):
    """Already works: model_link given explicitly with the model object (the workaround for the missing inference)."""
    from prediction import crosspred
    with _quiet():
        pp = crosspred(ds.cbP, model=ds.fit, model_link=ds.link_r, at=ds.at, cen=ds.cen, cumul=True)
    robj = r_crosspred_exact(ds, cumul=True)
    _assert_fields(pp, robj, LOG_FIELDS + CUM_FIELDS, EXACT, f'{ds.kind} crosspred(model_link=)')
    assert pp.model_link == ds.link_r


def test_crosspred_coef_vcov_route_is_faithful(ds):
    """Already works: coef=/vcov= (the cb block) with model_link."""
    from prediction import crosspred
    with _quiet():
        pp = crosspred(ds.cbP, coef=ds.coef, vcov=ds.vcov, model_link=ds.link_r, at=ds.at, cen=ds.cen)
    robj = r_crosspred_exact(ds)
    _assert_fields(pp, robj, LOG_FIELDS, EXACT, f'{ds.kind} crosspred(coef=)')


# =====================================================================================================================
# crossreduce with the model object
# =====================================================================================================================
def r_crossreduce_exact(ds, typ, value=None):
    k = ds.kind
    val = f', value={value!r}' if value is not None else ''
    r(f'{P}red <- crossreduce({ds.cb_r}, coef={P}{k}_coef, vcov={P}{k}_vcov, model.link={P}{k}_link, type="{typ}"'
      f'{val}, cen={ds.cen!r}, at={_r_at(ds, ds.at)})')
    return f'{P}red'


REDUCTIONS = [('overall', None), ('var', 'mid'), ('lag', 3)]


def _value(ds, v):
    return float(np.median(ds.at)) if v == 'mid' else v


@pytest.mark.parametrize('typ,val', REDUCTIONS)
def test_crossreduce_returns_RR_fields_like_R(ds_link, typ, val):
    """RRfit/RRlow/RRhigh (not low/high) for overall, var- and lag-specific reductions, equal to R's."""
    ds = ds_link
    from crossreduce import crossreduce
    value = _value(ds, val)
    with _quiet():
        red = crossreduce(ds.cbP, model=ds.fit, type=typ, value=value, cen=ds.cen, at=ds.at)
    robj = r_crossreduce_exact(ds, typ, value)
    _assert_fields(red, robj, ['coefficients', 'fit', 'se', 'RRfit', 'RRlow', 'RRhigh'], EXACT,
                   f'{ds.kind} crossreduce({typ})')
    for f in ('low', 'high'):
        assert not _rhas(robj, f)
        assert getattr(red, f, None) is None, f'crossreduce returns linear-scale {f} although R returns RR fields'
    assert red.model_link == _rstr(f'{robj}[["model.link"]]') == ds.link_r


@pytest.mark.parametrize('typ,val', REDUCTIONS)
def test_crossreduce_explicit_model_link_is_faithful(ds, typ, val):
    """Already works: model_link given with the model object."""
    from crossreduce import crossreduce
    value = _value(ds, val)
    with _quiet():
        red = crossreduce(ds.cbP, model=ds.fit, model_link=ds.link_r, type=typ, value=value, cen=ds.cen, at=ds.at)
    robj = r_crossreduce_exact(ds, typ, value)
    _assert_fields(red, robj, ['coefficients', 'fit', 'se', 'RRfit', 'RRlow', 'RRhigh'], EXACT,
                   f'{ds.kind} crossreduce({typ}, model_link=)')


def test_crossreduce_model_vs_R_model_end_to_end(ds_e2e):
    """PyDLNM(model=statsmodels fit) vs R crossreduce(model=R fit), overall reduction (1e-5)."""
    ds = ds_e2e
    from crossreduce import crossreduce
    with _quiet():
        red = crossreduce(ds.cbP, model=ds.fit, type='overall', cen=ds.cen, at=ds.at)
    r(f'{P}red_e2e <- crossreduce({ds.cb_r}, {ds.m_r}, type="overall", cen={ds.cen!r}, at={_r_at(ds, ds.at)})')
    _assert_fields(red, f'{P}red_e2e', ['fit', 'se', 'RRfit', 'RRlow', 'RRhigh'], E2E, f'{ds.kind} crossreduce e2e')


# =====================================================================================================================
# attrdl
# =====================================================================================================================
def _attr_r(ds, call_args, typ, dr, tot):
    _load_r_attrdl()
    k = ds.kind
    np2r(f'{P}{k}_attr_x', ds.exposure)
    if ds.exposure.ndim == 2:
        _pushed_matrix(f'{P}{k}_attr_x', ds.exposure)
    np2r(f'{P}{k}_attr_cases', ds.cases)
    r(f'{P}attr_res <- {P}attr$attrdl({P}{k}_attr_x, {ds.cb_r}, {P}{k}_attr_cases, {call_args}, type="{typ}", '
      f'dir="{dr}", tot={"TRUE" if tot else "FALSE"}, cen={ds.cen!r})')
    return np.atleast_1d(rget(f'as.numeric({P}attr_res)'))


def _attr_py(ds, typ, dr, tot, **kw):
    from attribution import attrdl
    with _quiet():
        res = attrdl(ds.exposure, ds.cbP, ds.cases, type=typ, dir=dr, tot=tot, cen=ds.cen, **kw)
    return np.atleast_1d(np.asarray(res[typ + ('_total' if tot else '')], dtype=float))


ATTR_CASES = [('an', True), ('af', True), ('an', False), ('af', False)]


@pytest.mark.parametrize('typ,tot', ATTR_CASES)
def test_attrdl_accepts_the_model_like_R(ds_link, typ, tot):
    """R attrdl(model=coxph/clogit/gnm) runs (log / logit link) and equals R attrdl on the same coef/vcov; PyDLNM
    attrdl(model=) raises "'model' must have a log or logit link function"."""
    ds = ds_link
    _load_r_attrdl()
    for dr in ds.attr_dirs:
        ref = _attr_r(ds, f'coef={P}{ds.kind}_coef, vcov={P}{ds.kind}_vcov', typ, dr, tot)
        _attr_r(ds, f'model={ds.m_r}', typ, dr, tot)                     # R accepts the native model (must not raise)
        py = _attr_py(ds, typ, dr, tot, model=ds.fit)
        assert_close(py, ref, rtol=EXACT, what=f'{ds.kind} attrdl(model=) {typ} tot={tot} dir={dr}')


@pytest.mark.parametrize('typ,tot', [('an', True), ('af', False)])
def test_attrdl_model_vs_R_model_end_to_end(ds_e2e, typ, tot):
    """attrdl(model=statsmodels fit) vs R attrdl(model=R fit) (1e-5)."""
    ds = ds_e2e
    for dr in ds.attr_dirs:
        ref = _attr_r(ds, f'model={ds.m_r}', typ, dr, tot)
        py = _attr_py(ds, typ, dr, tot, model=ds.fit)
        assert_close(py, ref, rtol=E2E, what=f'{ds.kind} attrdl e2e {typ} tot={tot} dir={dr}')


@pytest.mark.parametrize('typ,tot', ATTR_CASES)
def test_attrdl_coef_vcov_route_is_faithful(ds, typ, tot):
    """Already works: attrdl(coef=, vcov=) of the cb block (log scale) equals R's."""
    for dr in ds.attr_dirs:
        ref = _attr_r(ds, f'coef={P}{ds.kind}_coef, vcov={P}{ds.kind}_vcov', typ, dr, tot)
        py = _attr_py(ds, typ, dr, tot, coef=ds.coef, vcov=ds.vcov)
        assert_close(py, ref, rtol=EXACT, what=f'{ds.kind} attrdl(coef=) {typ} tot={tot} dir={dr}')


# =====================================================================================================================
# neighbouring getlink behaviour: user-specified link with a model whose link is inferred; linear models
# =====================================================================================================================
def _gaussian_glm_design():
    """statsmodels GLM(Gaussian, identity) and OLS of log(deaths) on a cross-basis, and the R glm/lm of the same."""
    import statsmodels.api as sm
    from basis import CrossBasis
    ch = chicago()
    x, y = ch['temp'][:2000], np.log(ch['death'][:2000])
    with _quiet():
        cb = CrossBasis(x, lag=NLAG, argvar=dict(ARGVAR), arglag=dict(ARGLAG))
    ok = ~np.isnan(cb.basis).any(axis=1)
    X = pd.DataFrame(cb.basis[ok], columns=cb.colnames)
    X.insert(0, 'const', 1.0)
    with _quiet():
        glm = sm.GLM(y[ok], X, family=sm.families.Gaussian()).fit()
        ols = sm.OLS(y[ok], X).fit()
    np2r(f'{P}g_x', x)
    np2r(f'{P}g_y', y)
    r(f'{P}g_cb <- crossbasis({P}g_x, lag={NLAG}, {R_CB_ARGS}); {P}g_glm <- glm({P}g_y ~ {P}g_cb, family=gaussian); '
      f'{P}g_lm <- lm({P}g_y ~ {P}g_cb)')
    return SimpleNamespace(cb=cb, glm=glm, ols=ols)


_G = {}


def _g():
    if not _G:
        _G['d'] = _gaussian_glm_design()
    return _G['d']


@known_defect(GAP, KEY, note='CrossPred ignores a user model_link when the model object has an inferred link: '
                             'prediction.py `model_info["link"] or model_link`; R getlink returns model.link first')
def test_crosspred_user_model_link_overrides_inferred_link_like_R():
    """R: crosspred(cb, glm_gaussian, model.link="log") -> model.link "log" with RR fields (a model of log(y) whose
    coefficients are log-RR). PyDLNM keeps the inferred 'identity' (crossreduce honours the user's link)."""
    from crossreduce import crossreduce
    from prediction import crosspred
    g = _g()
    at = np.array([0.0, 10.0, 20.0])
    r(f'{P}g_p <- crosspred({P}g_cb, {P}g_glm, model.link="log", at=c(0,10,20), cen=15)')
    assert _rstr(f'{P}g_p[["model.link"]]') == 'log' and _rhas(f'{P}g_p', 'allRRfit')
    with _quiet():
        pp = crosspred(g.cb, model=g.glm, model_link='log', at=at, cen=15.0)
        red = crossreduce(g.cb, model=g.glm, model_link='log', cen=15.0)
    assert red.model_link == 'log' and hasattr(red, 'RRfit')                   # crossreduce is faithful
    assert pp.model_link == 'log', f'CrossPred.model_link={pp.model_link!r}; R: "log"'
    assert_close(_pyfield(pp, 'allRRfit'), _rfield(f'{P}g_p', 'allRRfit'), rtol=E2E, what='allRRfit')


def test_crosspred_user_model_link_with_model_without_inferred_link_is_faithful():
    """Already works: OLS has no inferred link in PyDLNM, so the user's model.link="log" is used (R: same output)."""
    from prediction import crosspred
    g = _g()
    with _quiet():
        pp = crosspred(g.cb, model=g.ols, model_link='log', at=np.array([0.0, 10.0, 20.0]), cen=15.0)
    r(f'{P}g_p2 <- crosspred({P}g_cb, {P}g_lm, model.link="log", at=c(0,10,20), cen=15)')
    assert pp.model_link == 'log'
    assert_close(_pyfield(pp, 'allRRfit'), _rfield(f'{P}g_p2', 'allRRfit'), rtol=E2E, what='allRRfit')
    assert_close(_pyfield(pp, 'allRRlow'), _rfield(f'{P}g_p2', 'allRRlow'), rtol=E2E, what='allRRlow')


@known_defect(GAP, KEY, note='cosmetic: getlink returns None for statsmodels OLS results (class RegressionResultsWrapper); '
                             'R: lm -> "identity". Numbers are unaffected (linear-scale low/high either way)')
def test_getlink_ols_is_identity_like_R():
    from model_utils import getlink
    g = _g()
    assert getlink(g.ols) == _rstr(f'dlnm:::getlink({P}g_lm, class({P}g_lm))') == 'identity'


def test_linear_model_crosspred_has_no_RR_fields_like_R():
    """Already works: a Gaussian/identity model gives linear-scale low/high only (R: alllow/allhigh, no allRR*)."""
    from prediction import crosspred
    g = _g()
    with _quiet():
        pp = crosspred(g.cb, model=g.glm, at=np.array([0.0, 10.0, 20.0]), cen=15.0)
    r(f'{P}g_p3 <- crosspred({P}g_cb, {P}g_lm, at=c(0,10,20), cen=15)')
    assert pp.model_link == 'identity' == _rstr(f'{P}g_p3[["model.link"]]')
    assert not _rhas(f'{P}g_p3', 'allRRfit') and getattr(pp, 'allRRfit', None) is None
    assert_close(_pyfield(pp, 'alllow'), _rfield(f'{P}g_p3', 'alllow'), rtol=E2E, what='alllow')
    assert_close(_pyfield(pp, 'allhigh'), _rfield(f'{P}g_p3', 'allhigh'), rtol=E2E, what='allhigh')


# =====================================================================================================================
# the wrappers built on attrdl / crosspred: attr_heat_cold and find_mmt with a model object
# =====================================================================================================================
TS_KINDS = ('clogit', 'condpois', 'negbin')


@pytest.mark.parametrize('kind', TS_KINDS)
def test_attr_heat_cold_with_model_matches_R(request, kind):
    """attr_heat_cold(model=) = R attrdl(range=c(-Inf, cen)) for cold and attrdl(range=c(cen, Inf)) for heat (forward
    perspective, total AN / AF)."""
    from attribution import attr_heat_cold
    _mark_link_defect(request, kind)
    ds = _get_ds(kind)
    _load_r_attrdl()
    np2r(f'{P}{kind}_attr_x', ds.exposure)
    np2r(f'{P}{kind}_attr_cases', ds.cases)
    with _quiet():
        out = attr_heat_cold(ds.exposure, ds.cbP, ds.cases, model=ds.fit, cen=ds.cen)['summary']
    for part, rng in (('cold', 'c(-Inf, cen)'), ('heat', 'c(cen, Inf)')):
        for typ in ('an', 'af'):
            r(f'{P}attr_res <- {P}attr$attrdl({P}{kind}_attr_x, {ds.cb_r}, {P}{kind}_attr_cases, '
              f'coef={P}{kind}_coef, vcov={P}{kind}_vcov, type="{typ}", dir="forw", tot=TRUE, cen={ds.cen!r}, '
              f'range={rng.replace("cen", repr(ds.cen))})')
            ref = rget(f'as.numeric({P}attr_res)')
            assert_close(np.atleast_1d(out[f'{part}_{typ}_total']), ref, rtol=EXACT, what=f'{kind} {part} {typ}')


def test_find_mmt_with_model_matches_R_argmin(ds):
    """The minimum-risk exposure is the argmin of the overall fit (link independent): R which.min(allfit) on the
    statsmodels coef/vcov; the searched curve is R's overall fit up to a constant."""
    from centering import find_mmt
    grid = np.linspace(ds.at.min(), ds.at.max(), 41)
    with _quiet():
        res = find_mmt(ds.cbP, ds.fit, at=grid)
    r(f'{P}mmt_p <- suppressMessages(crosspred({ds.cb_r}, coef={P}{ds.kind}_coef, vcov={P}{ds.kind}_vcov, '
      f'model.link={P}{ds.kind}_link, at={_r_at(ds, grid)}))')
    assert res['mmt'] == grid[int(rget(f'which.min({P}mmt_p$allfit)')[0]) - 1]
    # R centres automatically (value 5 for this basis) whereas find_mmt searches the uncentred curve: compare up to that shift
    py_fit, r_fit = np.asarray(res['all_fit'], dtype=float), rget(f'as.numeric({P}mmt_p$allfit)')
    assert_close(py_fit - py_fit[0], r_fit - r_fit[0], rtol=EXACT, what='overall fit (shift-free)')
