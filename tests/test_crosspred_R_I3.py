"""Reduced-coefficient crosspred (theme R) and model-object plumbing (theme I3) versus R dlnm 2.4.10.

Every test computes its reference in R at run time (rpy2) and compares PyDLNM on identical inputs.

Theme R  -- crosspred(CrossBasis, coef=<reduced>, vcov=<reduced>) is the README / validation route for BLUP curves.
            R's equivalent is crosspred(onebasis, coef=, vcov=), whose lag range is c(0,0): matfit/matse/cumfit are
            single 'lag0' columns and cumfit == allfit.  PyDLNM keeps the CrossBasis lag range instead, so it
            returns L+1 identical columns, lag=[0,L], cumfit zero except in its last column, and print()s.
  crosspred-core-12, crosspred-grid-13, validation-audit-17
                        lag / matfit / matse / matRR* / cumfit / cumse / cumRR* of the reduced path are not R's
                        onebasis output (overall fields allfit/allse/allRR* are exact and guarded by plain tests).
                        NOT asserted: R's own lag=c(0,5) result (allfit summed 6 times) is an artefact of R's
                        onebasis path, so it is no reference for a lag sub-period request on reduced coefficients.
Theme I3 -- model objects that are not R glms
  crosspred-core-15     getvcov() 'diagonal from standard errors' fallback: crashes with a truth-value ValueError for
                        array `bse`, silently fabricates diag(se^2) for `std_err`/list `bse`; R stops with an error.
  crossreduce-4         crossreduce(cb, model=<statsmodels result>) raises pandas InvalidIndexError for every
                        pandas-backed fit (DataFrame design, formula API); the cross-basis columns come first in
                        these fits (crossreduce takes them positionally), so R's by-name extraction gives the same
                        parameters and the R result is well defined.
  crosspred-core-16     (nit, documentation gap) a statsmodels Poisson GLM has scale=1 exactly like R glm(poisson);
                        R quasipoisson needs fit(scale='X2').  Plain baseline tests pin both agreements.

Tests decorated with @known_defect assert the R-faithful behaviour and fail today (strict xfail); the plain tests
guard neighbouring behaviour that is already faithful and must keep passing while the fixes land.
"""
import os
import warnings

import numpy as np
import pytest

from rhelpers import assert_close, chicago, known_defect, np2r, r, rget


def _warm_up_r_lapack():
    """Load R's lazily loaded LAPACK module while os.environ['R_HOME'] still points at the R that is embedded.
    Workaround for audit finding Q2: basis.py (every CrossBasis construction) and the *_glm modules overwrite
    R_HOME with the /Library/Frameworks/R.framework/Resources path, and the first La_*() call afterwards
    (chol2inv in vcov.glm, solve, ...) dlopens the wrong R's modules/lapack and segfaults.  Remove once Q2 is fixed."""
    os.environ['R_HOME'] = os.path.dirname(str(r('.Library')[0]))      # the R_HOME R itself started with
    r('invisible(chol2inv(chol(diag(2) + 1))); invisible(solve(diag(2)))')


_warm_up_r_lapack()

AT = np.arange(-10.0, 30.0 + 1e-9, 2.5)          # 17 prediction points inside the Chicago temperature range
CEN = 15.0

# --------------------------------------------------------------------------------------------------------------
# helpers: identical inputs for R and Python
# --------------------------------------------------------------------------------------------------------------
# Marginal-basis configurations of the reduced route (explicit knots, so the overall curve is exact today: this
# isolates the lag-dimension defect from the resolved-argument bookkeeping of theme A1).
CONFIGS = {
    'bs2_ns_L21': dict(var='bs', degree=2, probs=(.10, .75, .90), lagfun='ns', L=21, nk=3),
    'bs3_int_L6': dict(var='bs', degree=3, probs=(.25, .75), lagfun='integer', L=6, nk=None),
    'ns_ns_L10': dict(var='ns', degree=None, probs=(.10, .50, .90), lagfun='ns', L=10, nk=2),
}
_BASES = {}


def _bases(name):
    """R crossbasis `cb_<name>` and onebasis `ob_<name>` (in the R workspace) plus the identical PyDLNM CrossBasis.
    Knots are computed once in R and given to both sides."""
    if name in _BASES:
        return _BASES[name]
    from basis import CrossBasis
    cfg = CONFIGS[name]
    temp = chicago()['temp']
    np2r('temp', temp)
    probs = ', '.join(repr(p) for p in cfg['probs'])
    r(f'kv_{name} <- quantile(temp, c({probs}))')
    kv = rget(f'kv_{name}')
    argvar = dict(fun=cfg['var'], knots=kv)
    r_argvar = f'fun="{cfg["var"]}", knots=kv_{name}'
    if cfg['degree']:
        argvar['degree'] = cfg['degree']
        r_argvar += f', degree={cfg["degree"]}'
    if cfg['lagfun'] == 'ns':
        r(f'kn_{name} <- logknots({cfg["L"]}, nk={cfg["nk"]})')
        arglag = dict(fun='ns', knots=rget(f'kn_{name}'))
        r_arglag = f'fun="ns", knots=kn_{name}'
    else:
        arglag, r_arglag = dict(fun='integer'), 'fun="integer"'
    r(f'cb_{name} <- crossbasis(temp, lag={cfg["L"]}, argvar=list({r_argvar}), arglag=list({r_arglag}))')
    r(f'ob_{name} <- do.call("onebasis", c(list(x=temp), attr(cb_{name}, "argvar")))')
    cb = CrossBasis(temp, lag=cfg['L'], argvar=argvar, arglag=arglag)
    p = int(rget(f'ncol(ob_{name})')[0])
    assert_close(np.asarray(cb.basis), rget(f'unclass(cb_{name})'), rtol=1e-12, what=f'{name} cross-basis')
    _BASES[name] = (cb, p)
    return _BASES[name]


def _rand_coef_vcov(p, seed, scale=0.05):
    """Deterministic coefficient vector and SPD covariance matrix."""
    rng = np.random.default_rng(seed)
    coef = rng.normal(0, scale, p)
    A = rng.normal(0, 1, (p, p))
    vcov = A @ A.T * scale ** 2 / p * 0.05 + np.eye(p) * 1e-5
    return coef, (vcov + vcov.T) / 2


def _rlogical(flag):
    return 'TRUE' if flag else 'FALSE'


def _r_link(link):
    return f', model.link="{link}"' if link else ''


def _reduced_pair(name, link, cumul, bylag=1.0, seed=31):
    """R crosspred(onebasis, coef, vcov) -> R object `pR`; Python crosspred(CrossBasis, reduced coef, vcov)."""
    from prediction import crosspred
    cb, p = _bases(name)
    coef, vcov = _rand_coef_vcov(p, seed)
    np2r('rc_coef', coef); np2r('rc_vcov', vcov); np2r('rc_at', AT)
    r(f'pR <- crosspred(ob_{name}, coef=rc_coef, vcov=rc_vcov{_r_link(link)}, at=rc_at, cen={CEN!r}, '
      f'bylag={bylag!r}, cumul={_rlogical(cumul)})')
    pp = crosspred(cb, coef=coef, vcov=vcov, model_link=link, at=AT, cen=CEN, bylag=bylag, cumul=cumul)
    return pp


def _fields(kind, link):
    """Names of the R / PyDLNM output fields of one kind ('all', 'mat', 'cum') for a given link."""
    fit = {'all': ['allfit', 'allse'], 'mat': ['matfit', 'matse'], 'cum': ['cumfit', 'cumse']}[kind]
    if link:
        return fit + [f'{kind}RR{s}' for s in ('fit', 'low', 'high')]
    return fit + [f'{kind}low', f'{kind}high']


def _compare(pp, fields, rtol=1e-12):
    """All fields of `pp` against R object `pR`; one assertion listing every disagreement."""
    bad = []
    for f in fields:
        py = getattr(pp, f, None)
        if py is None:
            bad.append(f'{f}: missing in Python')
            continue
        try:
            assert_close(np.atleast_1d(np.asarray(py, dtype=float)), rget(f'pR${f}'), rtol=rtol, what=f)
        except AssertionError as exc:
            bad.append(str(exc))
    assert not bad, ' | '.join(bad)


NAMES = list(CONFIGS)
NAMES_LINKS = [(n, lk) for n in NAMES for lk in ('log', None)]
ids_nl = [f'{n}-{lk or "nolink"}' for n, lk in NAMES_LINKS]

DEFECT_R = ('R', 'crosspred-core-12', 'crosspred-grid-13', 'validation-audit-17')
REDUCED_NOTE = 'reduced-coefficient crosspred keeps the CrossBasis lag range; R onebasis has lag c(0,0)'


# --------------------------------------------------------------------------------------------------------------
# theme R: reduced-coefficient crosspred versus R crosspred(onebasis, coef, vcov)
# --------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('name,link', NAMES_LINKS, ids=ids_nl)
def test_reduced_overall_fields_match_r_onebasis(name, link):
    """BASELINE (already faithful): predvar, cen, coefficients, vcov, allfit/allse and the overall CI fields of the
    reduced route equal R's onebasis crosspred at machine precision."""
    pp = _reduced_pair(name, link, cumul=False)
    _compare(pp, ['predvar', 'cen', 'coefficients', 'vcov'] + _fields('all', link))


@pytest.mark.parametrize('name', NAMES)
def test_reduced_first_lag_column_matches_r_onebasis(name):
    """BASELINE: the first (only, in R) lag-specific column matfit[:, 0] / matse[:, 0] equals R's `lag0` column."""
    pp = _reduced_pair(name, 'log', cumul=False)
    for f in ('matfit', 'matse'):
        assert_close(np.asarray(getattr(pp, f))[:, 0], rget(f'pR${f}')[:, 0], rtol=1e-12, what=f + '[:, 0]')


@pytest.mark.parametrize('name', NAMES)
def test_reduced_lag_range_is_onebasis_lag0(name):
    """R: crosspred(onebasis) has lag=c(0,0), bylag=1 and a single column named 'lag0'."""
    pp = _reduced_pair(name, 'log', cumul=False)
    assert np.asarray(pp.lag).tolist() == rget('pR$lag').tolist(), f'lag: Python {np.asarray(pp.lag).tolist()} vs R [0, 0]'
    assert list(pp.lag_names) == list(r('colnames(pR$matfit)')), f'lag names {list(pp.lag_names)[:3]}... vs R lag0'
    assert float(pp.bylag) == rget('pR$bylag')[0]


@pytest.mark.parametrize('name,link', NAMES_LINKS, ids=ids_nl)
def test_reduced_lag_specific_fields_match_r_onebasis(name, link):
    """R: matfit/matse/matRR*/matlow/mathigh have one column ('lag0'); PyDLNM returns L+1 identical copies of the
    overall effect."""
    pp = _reduced_pair(name, link, cumul=False)
    _compare(pp, _fields('mat', link))


@pytest.mark.parametrize('name,link', NAMES_LINKS, ids=ids_nl)
def test_reduced_cumulative_fields_match_r_onebasis(name, link):
    """R (cumul=TRUE): cumfit/cumse/cumRR*/cumlow/cumhigh are (n,1) with cumfit == allfit; PyDLNM returns (n, L+1)
    with zeros in every column but the last (cumRRfit == 1 for lags 0..L-1)."""
    pp = _reduced_pair(name, link, cumul=True)
    _compare(pp, _fields('cum', link) + _fields('all', link))


@pytest.mark.parametrize('bylag', [0.5, 0.25])
def test_reduced_bylag_gives_single_column_like_r(bylag):
    """R: with lag c(0,0) any bylag still gives seqlag(c(0,0), bylag) == 0, i.e. one column; PyDLNM returns
    2L/bylag-style duplicated columns (43 for L=21, bylag=0.5)."""
    pp = _reduced_pair('bs2_ns_L21', 'log', cumul=False, bylag=bylag)
    _compare(pp, _fields('mat', 'log') + _fields('all', 'log'))


def test_reduced_crosspred_prints_nothing(capsys):
    """R prints nothing; PyDLNM prints 'Detected reduced coefficients ...' / 'Variable basis created ...'."""
    from prediction import crosspred
    cb, p = _bases('bs2_ns_L21')
    coef, vcov = _rand_coef_vcov(p, 5)
    capsys.readouterr()
    crosspred(cb, coef=coef, vcov=vcov, model_link='log', at=AT, cen=CEN)
    assert capsys.readouterr().out == ''


@pytest.mark.parametrize('length', [24, 4])
def test_reduced_path_rejects_wrong_length_coefficients_like_r(length):
    """BASELINE: neither the full (25) nor the reduced (5) length: R stops ('coef/vcov not consistent with basis
    matrix'); PyDLNM raises as well instead of guessing a basis."""
    from prediction import crosspred
    cb, _ = _bases('bs2_ns_L21')
    coef, vcov = _rand_coef_vcov(length, 2)
    np2r('wl_coef', coef); np2r('wl_vcov', vcov)
    with pytest.raises(Exception, match='not consistent with basis matrix'):
        r('crosspred(cb_bs2_ns_L21, coef=wl_coef, vcov=wl_vcov, model.link="log", at=seq(-10, 30, by=2.5), cen=15)')
    with pytest.raises(ValueError):
        crosspred(cb, coef=coef, vcov=vcov, model_link='log', at=AT, cen=CEN)


@pytest.mark.parametrize('bylag', [1.0, 0.5])
def test_full_coef_crossbasis_crosspred_matches_r(bylag):
    """BASELINE: with full-length coefficients the lag-specific and cumulative fields (real lag dimension) equal R's
    crosspred(crossbasis) exactly, including bylag != 1."""
    from prediction import crosspred
    cb, _ = _bases('bs2_ns_L21')
    coef, vcov = _rand_coef_vcov(cb.shape[1], 12)
    np2r('fc_coef', coef); np2r('fc_vcov', vcov); np2r('fc_at', AT)
    r(f'pR <- crosspred(cb_bs2_ns_L21, coef=fc_coef, vcov=fc_vcov, model.link="log", at=fc_at, cen={CEN!r}, '
      f'bylag={bylag!r}, cumul=TRUE)')
    pp = crosspred(cb, coef=coef, vcov=vcov, model_link='log', at=AT, cen=CEN, bylag=bylag, cumul=True)
    assert np.asarray(pp.lag).tolist() == rget('pR$lag').tolist(), 'lag'
    _compare(pp, ['predvar', 'coefficients', 'vcov'] + _fields('mat', 'log') + _fields('all', 'log')
             + _fields('cum', 'log'))


def test_crossreduce_then_reduced_crosspred_matches_r_full_model():
    """BASELINE (the validated route): crossreduce() of full coefficients, then crosspred on the reduced
    coefficients, gives R's full-model overall effect and its standard error."""
    from crossreduce import crossreduce
    from prediction import crosspred
    cb, _ = _bases('bs2_ns_L21')
    coef, vcov = _rand_coef_vcov(cb.shape[1], 77)
    np2r('fc_coef', coef); np2r('fc_vcov', vcov); np2r('fc_at', AT)
    r(f'pR <- crosspred(cb_bs2_ns_L21, coef=fc_coef, vcov=fc_vcov, model.link="log", at=fc_at, cen={CEN!r})')
    r('redR <- crossreduce(cb_bs2_ns_L21, coef=fc_coef, vcov=fc_vcov, model.link="log", cen=15)')
    red = crossreduce(cb, coef=coef, vcov=vcov, cen=CEN)
    assert_close(red.coef, rget('unname(redR$coefficients)'), rtol=1e-12, what='reduced coef')
    assert_close(red.vcov, rget('unname(redR$vcov)'), rtol=1e-12, what='reduced vcov')
    pp = crosspred(cb, coef=red.coef, vcov=red.vcov, model_link='log', at=AT, cen=CEN)
    _compare(pp, ['allfit', 'allse', 'allRRfit', 'allRRlow', 'allRRhigh'], rtol=1e-10)


# --------------------------------------------------------------------------------------------------------------
# theme I3: statsmodels fits (cross-basis columns first, intercept last) versus R glm on the same design
# --------------------------------------------------------------------------------------------------------------
GLM_TOL = dict(tol=1e-10, maxiter=200)          # converge statsmodels IRLS tighter than the default (R side: 1e-14)
DESIGNS = ['numpy', 'dataframe', 'formula']
# 'numpy' is unnamed (statsmodels x1..xN); 'dataframe' and 'formula' carry R's own names for the cross-basis block
# ('cb_gv1.l1', ...), so crosspred(model=)/crossreduce(model=) can find it by position (cb first) or by name (as R).


@pytest.fixture(scope='module')
def glm_case():
    """Chicago quasi-Poisson / Poisson fits in R and the identical design in statsmodels (module scoped)."""
    sm = pytest.importorskip('statsmodels.api')
    pd = pytest.importorskip('pandas')
    from basis import CrossBasis
    temp = chicago()['temp']
    np2r('temp', temp)
    r('kv_g <- quantile(temp, c(.10,.75,.90)); kn_g <- logknots(21, nk=3)')
    kv, kn = rget('kv_g'), rget('kn_g')
    r('cb_g <- crossbasis(temp, lag=21, argvar=list(fun="bs", degree=2, knots=kv_g), '
      'arglag=list(fun="ns", knots=kn_g))')
    r('''
    dat_g <- chicagoNMMAPS
    dat_g$dowf <- factor(dat_g$dow)
    dat_g$tt <- seq_len(nrow(dat_g))
    ctl_g <- glm.control(epsilon=1e-14, maxit=200)
    fit_q <- glm(death ~ cb_g + dowf + ns(tt, df=10), family=quasipoisson(), data=dat_g, control=ctl_g)
    fit_p <- glm(death ~ cb_g + dowf + ns(tt, df=10), family=poisson(), data=dat_g, control=ctl_g)
    X_g <- model.matrix(fit_q); y_g <- fit_q$y
    ''')
    cb = CrossBasis(temp, lag=21, argvar={'fun': 'bs', 'degree': 2, 'knots': kv},
                    arglag={'fun': 'ns', 'knots': kn})
    n_cb = cb.shape[1]
    X, y = rget('X_g'), rget('y_g')
    r_names = list(r('colnames(X_g)'))
    order = list(range(1, n_cb + 1)) + list(range(n_cb + 1, X.shape[1])) + [0]      # cb first, intercept last
    X = X[:, order]
    cb_cols = [f'cb_g{c}' for c in cb.colnames]                    # R: cb_gv1.l1, cb_gv1.l2, ...
    assert cb_cols == r_names[1:n_cb + 1], 'cross-basis column names differ from R model.matrix names'
    cols = cb_cols + [f'x{i}' for i in range(X.shape[1] - n_cb - 1)] + ['const']
    frame = pd.DataFrame(X, columns=cols)
    frame['y'] = y

    def fit(design, scale='X2'):
        kw = dict(GLM_TOL, **({'scale': scale} if scale else {}))
        fam = sm.families.Poisson()
        if design == 'numpy':
            return sm.GLM(y, X, family=fam).fit(**kw)
        if design == 'dataframe':
            return sm.GLM(y, frame[cols], family=fam).fit(**kw)
        smf = pytest.importorskip('statsmodels.formula.api')
        formula = 'y ~ 0 + ' + ' + '.join(f'Q("{c}")' for c in cols)       # Q(): names with '.' are not identifiers
        return smf.glm(formula, data=frame, family=fam).fit(**kw)

    return dict(sm=sm, cb=cb, n_cb=n_cb, order=order, cols=cols, cb_cols=cb_cols, X=X, y=y, fit=fit)


def _r_glm_crosspred(fit, cumul=True):
    r(f'pR <- crosspred(cb_g, {fit}, at=seq(-15, 30, by=2.5), cen=21, cumul={_rlogical(cumul)})')


LINK_FREE = ['coefficients', 'vcov', 'matfit', 'matse', 'allfit', 'allse', 'cumfit', 'cumse']
GLM_AT = np.arange(-15.0, 30.0 + 1e-9, 2.5)


def test_crosspred_statsmodels_scale_x2_matches_r_quasipoisson(glm_case):
    """BASELINE (crosspred-core-16, documentation gap only): statsmodels Poisson fitted with scale='X2' is R's
    quasi-Poisson: same dispersion, coefficients, vcov and every crosspred fit/se field.  (Named, pandas-backed fit:
    the cross-basis block is found by position today and would be found by name as in R.)"""
    from prediction import crosspred
    res = glm_case['fit']('dataframe', scale='X2')
    _r_glm_crosspred('fit_q')
    assert float(res.scale) == pytest.approx(rget('summary(fit_q)$dispersion')[0], rel=1e-10)
    pp = crosspred(glm_case['cb'], model=res, at=GLM_AT, cen=21.0, cumul=True)
    _compare(pp, LINK_FREE, rtol=1e-8)


def test_crosspred_statsmodels_default_scale_matches_r_poisson(glm_case):
    """BASELINE (crosspred-core-16): the default statsmodels fit (scale=1) equals R glm(poisson), whose vcov has
    dispersion 1, so crosspred(model=) propagates the fitted model exactly like R does."""
    from prediction import crosspred
    res = glm_case['fit']('dataframe', scale=None)
    _r_glm_crosspred('fit_p')
    pp = crosspred(glm_case['cb'], model=res, at=GLM_AT, cen=21.0, cumul=True)
    _compare(pp, LINK_FREE, rtol=1e-8)


@pytest.mark.parametrize('design', DESIGNS)
def test_getvcov_getcoef_statsmodels_match_r(glm_case, design):
    """BASELINE: getcoef/getvcov of numpy- and pandas-backed statsmodels results are plain ndarrays equal to R's
    coef()/vcov() of the same quasi-Poisson fit (this is why the bse fallback is not reached for statsmodels)."""
    from model_utils import getcoef, getvcov
    res = glm_case['fit'](design)
    order = glm_case['order']
    coef_py, vcov_py = getcoef(res), getvcov(res)
    assert isinstance(coef_py, np.ndarray) and isinstance(vcov_py, np.ndarray)
    assert_close(coef_py, rget('unname(coef(fit_q))')[order], rtol=1e-8, what='getcoef')
    assert_close(vcov_py, rget('unname(vcov(fit_q))')[np.ix_(order, order)], rtol=1e-8, what='getvcov')


def _r_crossreduce(glm_case):
    r('red_g <- crossreduce(cb_g, fit_q, cen=21)')
    return rget('unname(red_g$coefficients)'), rget('unname(red_g$vcov)')


def _check_crossreduce_statsmodels(glm_case, design):
    from crossreduce import crossreduce
    res = glm_case['fit'](design)
    ref_coef, ref_vcov = _r_crossreduce(glm_case)
    red = crossreduce(glm_case['cb'], model=res)
    assert_close(red.coef, ref_coef, rtol=1e-8, what='crossreduce coef')
    assert_close(red.vcov, ref_vcov, rtol=1e-8, what='crossreduce vcov')


def test_crossreduce_numpy_statsmodels_result_matches_r_or_raises(glm_case):
    """R selects the cross-basis coefficients by NAME; a numpy-backed statsmodels result has none, so PyDLNM cannot
    identify the block among the other coefficients (theme I1: no positional guess). Either R's numbers or a
    ValueError asking for explicit coef=/vcov= are acceptable, never a silent wrong number."""
    try:
        _check_crossreduce_statsmodels(glm_case, 'numpy')
    except ValueError as exc:
        assert 'coef=' in str(exc) or 'name' in str(exc), exc


@pytest.mark.parametrize('design', ['dataframe', 'formula'])
def test_crossreduce_pandas_statsmodels_result_matches_r(glm_case, design):
    """R crossreduce(cb, model) on the same fit.  Pandas-backed statsmodels results (DataFrame exog, formula API)
    have Series params / DataFrame cov_params(); PyDLNM raises pandas InvalidIndexError.  The cross-basis columns
    come first in these fits, so the positional extraction selects the same parameters as R's by-name grep."""
    _check_crossreduce_statsmodels(glm_case, design)


def test_crossreduce_pandas_fit_by_name_workaround_matches_r(glm_case):
    """BASELINE: extracting the cross-basis parameters by name from a pandas-backed fit and passing coef=/vcov=
    (the workaround for crossreduce-4) equals R crossreduce on the same fit."""
    from crossreduce import crossreduce
    res = glm_case['fit']('dataframe')
    names = glm_case['cb_cols']
    ref_coef, ref_vcov = _r_crossreduce(glm_case)
    red = crossreduce(glm_case['cb'], coef=res.params[names].to_numpy(),
                      vcov=res.cov_params().loc[names, names].to_numpy())
    assert_close(red.coef, ref_coef, rtol=1e-8, what='crossreduce coef')
    assert_close(red.vcov, ref_vcov, rtol=1e-8, what='crossreduce vcov')


# --------------------------------------------------------------------------------------------------------------
# theme I3: getvcov 'diagonal from standard errors' fallback (crosspred-core-15)
# --------------------------------------------------------------------------------------------------------------
class _BseArray:                        # params + array bse, no vcov accessor
    def __init__(self, coef):
        self.params, self.bse = coef, np.full(len(coef), 0.1)


class _BseSeries:                       # params + pandas Series bse
    def __init__(self, coef):
        import pandas as pd
        self.params, self.bse = coef, pd.Series(np.full(len(coef), 0.1))


class _BseList:                         # params + list bse (truthy list: `or` short-circuits, no ValueError)
    def __init__(self, coef):
        self.params, self.bse = coef, [0.1] * len(coef)


class _StdErr:                          # coef_ + std_err only
    def __init__(self, coef):
        self.coef_, self.std_err = coef, np.full(len(coef), 0.1)


NO_VCOV = {'bse-ndarray': _BseArray, 'bse-series': _BseSeries, 'bse-list': _BseList, 'std_err-ndarray': _StdErr}


@pytest.mark.parametrize('kind', list(NO_VCOV))
def test_getvcov_without_vcov_accessor_raises_attribute_error(kind):
    """R getvcov() stops ('methods for coef() and vcov() must exist ...'); PyDLNM's documented contract is an
    AttributeError.  Today array/Series `bse` gives a misleading truth-value ValueError, and list `bse` /
    `std_err` return a diagonal matrix that discards all covariances."""
    from model_utils import getvcov
    obj = NO_VCOV[kind](np.linspace(-0.2, 0.2, 5))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            v = getvcov(obj)
        except AttributeError:
            return
        except Exception as exc:                                        # noqa: BLE001
            pytest.fail(f'getvcov raised {type(exc).__name__} ({exc}); expected AttributeError')
    pytest.fail(f'getvcov returned a {np.shape(v)} matrix (diag of se^2); R stops with an error')


def _check_crosspred_model_without_vcov_stops(kind):
    from prediction import crosspred
    cb, _ = _bases('bs2_ns_L21')
    coef, _ = _rand_coef_vcov(cb.shape[1], 3)
    # R reference: an S3 object with a coef() method but no vcov() method
    np2r('bo_coef', coef)
    r('bo <- structure(list(coefficients=bo_coef), class="bseonly"); '
      'coef.bseonly <- function(object, ...) object$coefficients')
    with pytest.raises(Exception, match=r'coef\(\) and vcov\(\) must exist'):
        r('crosspred(cb_bs2_ns_L21, model=bo, at=seq(-10, 30, by=2.5), cen=15)')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            pp = crosspred(cb, model=NO_VCOV[kind](coef), at=AT, cen=CEN)
        except (AttributeError, ValueError):
            return
    pytest.fail(f'crosspred returned allse[:3]={np.asarray(pp.allse)[:3]} from a fabricated diagonal vcov; R stops')


@pytest.mark.parametrize('kind', ['bse-ndarray', 'bse-series'])
def test_crosspred_model_with_array_bse_stops_like_r(kind):
    """BASELINE: R crosspred(cb, model=<object with coef() but no vcov()>) stops; PyDLNM also raises for array-like
    `bse` (today via the truth-value ValueError wrapped by validate_model_compatibility)."""
    _check_crosspred_model_without_vcov_stops(kind)


@pytest.mark.parametrize('kind', ['bse-list', 'std_err-ndarray'])
def test_crosspred_model_without_vcov_stops_like_r(kind):
    """R stops; PyDLNM must raise as well instead of returning predictions whose standard errors come from a
    fabricated diagonal vcov (allse 2-7x too large in a real DLNM, confidence intervals wrong)."""
    _check_crosspred_model_without_vcov_stops(kind)
