"""Themes A1 and A2: resolved basis arguments are not stored, so prediction rebuilds the basis from the wrong data.

R's onebasis() keeps the arguments it actually used as attributes (the knots and Boundary.knots that ns/bs derived from
the data, the poly scale, the threshold, the integer values ...). crossbasis() rewrites argvar/arglag from those
attributes and crosspred()/mkXpred() rebuild the identical basis at the prediction grid, at the centering value and for
a lag sub-period.  PyDLNM keeps the user's argument dict instead, so the basis is silently re-derived from the
prediction vector (df-based knots, poly scale, default thr), from a single centering value, or from the lag sub-period.

Every test feeds IDENTICAL random coef/vcov to R (crossbasis/onebasis + crosspred, computed at test time) and PyDLNM.

  Theme A1 (resolved arguments not stored; crosspred rebuilds the basis from the prediction grid)
    basis-cont-1      df-based ns/bs and poly through CrossBasis + crosspred; OneBasis attributes; argvar dict leak
    basis-discrete-3  same, plus default thr, strata df breaks and the lag basis (arglag) at a lag sub-period
    basis-discrete-7  crosspred(OneBasis): 'thr.value' does not round-trip; ns/bs knots not recorded (crash / wrong)
    crossbasis-9      CrossBasis.argvar/arglag lack knots and Boundary.knots (df-based bases, lag sub-period, bylag)
    attr-point-8      df-based var basis in attrdl/CrossPred/reduced-coefficient prediction: one-value centering basis
  Theme A2 (lag sub-period / single-lag prediction)
    basis-discrete-10 integer lag basis: intercept=FALSE/values ignored, lag sub-period crashes, no OneBasis 'integer'

Plain (unmarked) tests guard what is already faithful: CrossBasis matrices for df-based bases, explicit-knot crosspred
(full lag, cumul, bylag, grid wider than the data), lin/poly/strata OneBasis crosspred, poly/thr/strata attribute
values, the default integer lag basis, and the R errors that PyDLNM must also raise.
"""
import numpy as np
import pytest

from rhelpers import REPO, assert_close, chicago, np2r, r, rget

P = 'tA_'                       # prefix of every R global created here (other modules share the R session)
LAG = 6                         # lag of the spline lag-basis tests
LAG_INT = 5                     # lag of the integer lag-basis tests
CEN = 18.0
_R_NAMES = {'Boundary_knots': 'Boundary.knots', 'thr_value': 'thr.value'}
_ALL_A1 = ('basis-cont-1', 'basis-discrete-3', 'basis-discrete-7', 'crossbasis-9', 'attr-point-8')


# --------------------------------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------------------------------
def _series(n=1500, start=0):
    """A fixed slice of Chicago temperature, also pushed to R as tA_temp."""
    temp = chicago()['temp'][start:start + n]
    np2r(P + 'temp', temp)
    return temp


def _rq(*probs):
    """R quantiles (type 7) of tA_temp as exact doubles."""
    return rget(f'quantile({P}temp, c({",".join(repr(float(p)) for p in probs)}), names=FALSE)')


def _rlogknots(lag, nk):
    return rget(f'logknots({_rlag(lag)}, {nk})')


def _rlist(spec, tag):
    """Python spec dict -> text of an R list(...); array values are pushed to R under a global name."""
    parts = []
    for key, val in spec.items():
        rk = _R_NAMES.get(key, key)
        if isinstance(val, str):
            parts.append(f'{rk}="{val}"')
        elif isinstance(val, (bool, np.bool_)):
            parts.append(f'{rk}={"TRUE" if val else "FALSE"}')
        elif np.ndim(val) == 0:
            parts.append(f'{rk}={int(val) if float(val).is_integer() else repr(float(val))}')
        else:
            name = f'{P}{tag}_{key}'
            np2r(name, val)
            parts.append(f'{rk}=`{name}`')
    return 'list(' + ', '.join(parts) + ')'


def _rlag(lag):
    return f'c({int(lag[0])},{int(lag[1])})' if np.ndim(lag) else str(int(lag))


def _require_r_package(pkg):
    """(rhelpers.require_r_packages fails on R's invisible() return value under rpy2, so this is local.)"""
    if not bool(rget(f'as.numeric(requireNamespace("{pkg}", quietly=TRUE))')[0]):
        pytest.skip(f'R package {pkg} not installed')
    r(f'suppressMessages(library({pkg}))')


def _make_cb(temp, lag, argvar, arglag, name=P + 'cb'):
    """The same cross-basis in R (global `name`) and in PyDLNM; PyDLNM gets its own copy of the spec dicts."""
    from basis import CrossBasis
    r(f'{name} <- crossbasis({P}temp, lag={_rlag(lag)}, argvar={_rlist(argvar, "av")}, '
      f'arglag={_rlist(arglag, "al")})')
    cb = CrossBasis(temp, lag=lag, argvar=dict(argvar), arglag=dict(arglag))
    assert cb.basis.shape == tuple(int(v) for v in rget(f'dim({name})')), 'cross-basis shape differs from R'
    return cb


def _make_ob(x, spec, name=P + 'ob'):
    """The same onebasis in R (global `name`) and in PyDLNM; `spec` holds fun and its arguments."""
    from basis import OneBasis
    np2r(P + 'x', x)
    r(f'{name} <- do.call(onebasis, c(list(x={P}x), {_rlist(spec, "ob")}))')
    return OneBasis(x, **spec)


def _rand_coef_vcov(k, seed):
    """Deterministic random coefficients and a full (non-diagonal) positive-definite vcov."""
    rng = np.random.default_rng(seed)
    coef = rng.normal(0.0, 0.05, k)
    a = rng.normal(size=(k, k))
    return coef, 1e-4 * (a @ a.T / k + np.eye(k))


def _grid(temp, n=15, lo=.02, hi=.98):
    return np.unique(np.round(np.linspace(np.quantile(temp, lo), np.quantile(temp, hi), n), 3))


def _r_pred(obj, coef, vcov, at, cen=None, lag=None, bylag=1, cumul=False):
    """R crosspred(obj, ...) on the given coef/vcov; cen=None is R's cen=FALSE (no centering)."""
    np2r(P + 'coef', coef)
    np2r(P + 'vcov', vcov)
    np2r(P + 'at', at)
    cen_r = 'FALSE' if cen is None else repr(float(cen))
    lag_r = '' if lag is None else f', lag={_rlag(lag)}'
    r(f'{P}p <- crosspred({obj}, coef={P}coef, vcov={P}vcov, model.link="log", at={P}at, cen={cen_r}{lag_r}, '
      f'bylag={bylag}, cumul={"TRUE" if cumul else "FALSE"})')
    fields = ['matfit', 'matse', 'allfit', 'allse'] + (['cumfit', 'cumse'] if cumul else [])
    return {f: rget(f'{P}p${f}') for f in fields}


def _py_pred(basis, coef, vcov, at, cen=None, lag=None, bylag=1.0, cumul=False):
    from prediction import crosspred
    # cen=None here means 'no centering' (R: cen=FALSE); R's own NULL is the automatic mid-range centering
    return crosspred(basis, coef=coef, vcov=vcov, at=at, cen=False if cen is None else cen, lag=lag, bylag=bylag,
                     cumul=cumul)


def _compare_pred(py, ref, rtol=1e-8, what=''):
    for f, ref_val in ref.items():
        assert_close(getattr(py, f), ref_val, rtol=rtol, what=f'{what}{f}')


def _crosspred_vs_r(basis, robj, temp, seed, *, cen=None, lag=None, bylag=1, cumul=False, at=None, rtol=1e-8):
    """Random coef/vcov of the right size -> crosspred in R and in PyDLNM -> compare mat/all (and cum) fit and se."""
    k = int(rget(f'ncol({robj})')[0])
    coef, vcov = _rand_coef_vcov(k, seed)
    at = _grid(temp) if at is None else at
    ref = _r_pred(robj, coef, vcov, at, cen=cen, lag=lag, bylag=bylag, cumul=cumul)
    py = _py_pred(basis, coef, vcov, at, cen=cen, lag=lag, bylag=float(bylag), cumul=cumul)
    _compare_pred(py, ref, rtol=rtol)


def _attr(ob, name):
    """A recorded argument of a PyDLNM OneBasis, looked up under R's name and Python's (Boundary.knots/_knots)."""
    for key in (name, name.replace('.', '_')):
        if key in ob.attributes and ob.attributes[key] is not None:
            return np.atleast_1d(np.asarray(ob.attributes[key], dtype=float))
    raise AssertionError(f'attribute {name!r} not recorded (attributes: {sorted(ob.attributes)})')


def _r_attr(robj, name):
    return np.atleast_1d(rget(f'as.numeric(attr({robj}, "{name}"))'))


def _fill_knots(spec):
    """Replace the placeholders of a case table (call after _series): knots='q3' -> three quantiles of tA_temp,
    Boundary_knots='wide' -> the data range widened by 2 on both sides."""
    spec = dict(spec)
    if isinstance(spec.get('knots'), str):
        spec['knots'] = _rq(.25, .5, .75)
    if isinstance(spec.get('Boundary_knots'), str):
        spec['Boundary_knots'] = _rq(0, 1) + np.array([-2.0, 2.0])
    return spec


# --------------------------------------------------------------------------------------------------------------------
# case tables
# --------------------------------------------------------------------------------------------------------------------
DF_VAR = [pytest.param({'fun': 'ns', 'df': 4}, id='ns_df4'),
          pytest.param({'fun': 'ns', 'df': 5}, id='ns_df5'),
          pytest.param({'fun': 'bs', 'df': 5, 'degree': 2}, id='bs_df5_deg2'),
          pytest.param({'fun': 'bs', 'df': 6}, id='bs_df6_deg3')]

SPLINE_ONEBASIS = [pytest.param({'fun': 'ns', 'df': 4}, id='ns_df4'),
                   pytest.param({'fun': 'ns', 'df': 5}, id='ns_df5'),
                   pytest.param({'fun': 'ns', 'knots': 'q3'}, id='ns_knots3'),
                   pytest.param({'fun': 'bs', 'df': 5, 'degree': 2}, id='bs_df5_deg2'),
                   pytest.param({'fun': 'bs', 'knots': 'q3', 'degree': 2}, id='bs_knots3_deg2')]

# explicit-knot var bases (the validated style): nothing data-derived is lost at prediction
EXPLICIT_VAR = {'bs_knots_deg2': lambda: {'fun': 'bs', 'degree': 2, 'knots': _rq(.1, .75, .9)},
                'ns_knots': lambda: {'fun': 'ns', 'knots': _rq(.25, .5, .75)}}

# lag-basis kind, lag used to fit, lag predicted, bylag
LAG_SUB = [pytest.param('log2', (0, LAG), (0, 3), 1, id='ns_logknots_lag0-3'),
           pytest.param('log2', (0, LAG), (2, 5), 1, id='ns_logknots_lag2-5'),
           pytest.param('log2', (0, LAG), (3, 3), 1, id='ns_logknots_single_lag3'),
           pytest.param('log2', (0, LAG), (1, 4), 0.5, id='ns_logknots_lag1-4_bylag0.5'),
           pytest.param('log2', (2, 8), (3, 5), 1, id='ns_logknots_fit2-8_lag3-5'),
           pytest.param('df3', (0, LAG), (0, 3), 1, id='ns_df3_lag0-3'),
           pytest.param('df3', (0, LAG), (2, 5), 1, id='ns_df3_lag2-5'),
           pytest.param('df4', (0, LAG), (1, 3), 1, id='ns_df4_lag1-3')]


def _arglag(kind, fit_lag=(0, LAG)):
    return {'log2': lambda: {'fun': 'ns', 'knots': _rlogknots(fit_lag, 2)},
            'df3': lambda: {'fun': 'ns', 'df': 3},
            'df4': lambda: {'fun': 'ns', 'df': 4}}[kind]()


# --------------------------------------------------------------------------------------------------------------------
# A1: CrossBasis + crosspred with a var basis given by df / poly / default thr        (fixed)        
# --------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('cen', [CEN, None], ids=['cen18', 'nocen'])
@pytest.mark.parametrize('argvar', DF_VAR)
def test_crosspred_cb_df_var_basis_uses_training_knots(argvar, cen):
    temp = _series()
    cb = _make_cb(temp, LAG, argvar, {'fun': 'ns', 'knots': _rlogknots(LAG, 2)})
    _crosspred_vs_r(cb, P + 'cb', temp, seed=11, cen=cen)


@pytest.mark.parametrize('cen', [CEN, None], ids=['cen18', 'nocen'])
@pytest.mark.parametrize('degree', [2, 3])
def test_crosspred_cb_poly_scale_from_training_data(degree, cen):
    temp = _series()
    cb = _make_cb(temp, LAG, {'fun': 'poly', 'degree': degree}, {'fun': 'ns', 'knots': _rlogknots(LAG, 2)})
    _crosspred_vs_r(cb, P + 'cb', temp, seed=12, cen=cen)


@pytest.mark.parametrize('cen', [15.0, None], ids=['cen15', 'nocen'])
def test_crosspred_cb_thr_default_threshold_from_training_data(cen):
    temp = _series()
    cb = _make_cb(temp, LAG, {'fun': 'thr'}, {'fun': 'ns', 'knots': _rlogknots(LAG, 2)})
    _crosspred_vs_r(cb, P + 'cb', temp, seed=13, cen=cen)


@pytest.mark.parametrize('cen', [15.0, None], ids=['cen15', 'nocen'])
def test_crosspred_cb_strata_df_breaks_from_training_data(cen):
    temp = _series()
    cb = _make_cb(temp, LAG, {'fun': 'strata', 'df': 1}, {'fun': 'ns', 'knots': _rlogknots(LAG, 2)})
    _crosspred_vs_r(cb, P + 'cb', temp, seed=13, cen=cen)


def test_crossbasis_does_not_leak_resolved_args_between_series():
    """One argvar dict reused for two series (the usual multi-city loop): each cross-basis must equal R's."""
    from basis import CrossBasis
    argvar = {'fun': 'bs', 'df': 5, 'degree': 2}
    arglag = {'fun': 'ns', 'knots': _rlogknots(LAG, 2)}
    av_r, al_r = _rlist(argvar, 'av'), _rlist(arglag, 'al')     # R side is built from the pristine dicts
    for start in (0, 2500):                                     # two different series, the same dict objects
        temp = _series(1200, start)
        r(f'{P}cb <- crossbasis({P}temp, lag={LAG}, argvar={av_r}, arglag={al_r})')
        cb = CrossBasis(temp, lag=LAG, argvar=argvar, arglag=arglag)
        assert_close(np.asarray(cb.basis), rget(f'unclass({P}cb)'), rtol=1e-10, what=f'crossbasis (series {start})')


@pytest.mark.parametrize('argvar', DF_VAR)
def test_crosspred_reduced_coef_df_var_basis(argvar):
    """coef/vcov of length ncol(var basis) (crossreduce-style) with the cross-basis: R rebuilds the var basis from
    attr(cb,'argvar'), i.e. onebasis(x, <resolved argvar>)."""
    temp = _series()
    cb = _make_cb(temp, LAG, argvar, {'fun': 'ns', 'knots': _rlogknots(LAG, 2)})
    r(f'{P}av <- attr({P}cb, "argvar"); {P}av$cen <- NULL; '
      f'{P}bvar <- do.call(onebasis, c(list(x={P}temp), {P}av))')
    k = int(rget(f'ncol({P}bvar)')[0])
    coef, vcov = _rand_coef_vcov(k, 14)
    at = _grid(temp)
    ref = _r_pred(P + 'bvar', coef, vcov, at, cen=CEN)
    py = _py_pred(cb, coef, vcov, at, cen=CEN)
    _compare_pred(py, {f: ref[f] for f in ('allfit', 'allse')}, what='reduced coef: ')   # matfit layout differs


class ImprovedGLMInterface:
    """Minimal stand-in of PyDLNM's GLM wrapper: model_utils dispatches on this class NAME (coef/vcov, log link)."""

    def __init__(self, coef, vcov):
        self.cb_coef = np.asarray(coef)
        self.cb_vcov = np.asarray(vcov)


@pytest.mark.parametrize('argvar', DF_VAR[:3])
def test_attrdl_forward_af_df_var_basis(argvar):
    """Per-observation forward attributable fraction (Lancet attrdl.R vs PyDLNM attrdl) at the training x."""
    _require_r_package('tsModel')
    r(f'source(file.path("{REPO}", "2015_gasparrini_Lancet_Rcodedata-master", "attrdl.R"))')
    from attribution import attrdl
    lag, cen = 10, 15.0
    temp = _series(900, 1000)
    death = chicago()['death'][1000:1900]
    np2r(P + 'death', death)
    cb = _make_cb(temp, lag, argvar, {'fun': 'ns', 'knots': _rlogknots(lag, 2)})
    coef, vcov = _rand_coef_vcov(cb.basis.shape[1], 15)
    np2r(P + 'coef', coef)
    np2r(P + 'vcov', vcov)
    ref = rget(f'attrdl({P}temp, {P}cb, {P}death, coef={P}coef, vcov={P}vcov, model.link="log", type="af", '
               f'dir="forw", tot=FALSE, cen={cen})')
    py = attrdl(temp, cb, death, model=ImprovedGLMInterface(coef, vcov), type='af', dir='forw', tot=False, cen=cen)
    assert_close(np.asarray(py['af']), ref, rtol=1e-8, what='per-observation forward AF')


# --------------------------------------------------------------------------------------------------------------------
# A1: OneBasis attributes and crosspred(OneBasis)                                     (fixed)        
# --------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('spec', SPLINE_ONEBASIS)
def test_onebasis_spline_records_knots_and_boundary_knots(spec):
    x = _series(500)
    ob = _make_ob(x, _fill_knots(spec))
    assert_close(_attr(ob, 'knots'), _r_attr(P + 'ob', 'knots'), rtol=1e-12, what='attribute knots')
    assert_close(_attr(ob, 'Boundary.knots'), _r_attr(P + 'ob', 'Boundary.knots'), rtol=1e-12,
                 what='attribute Boundary.knots')


@pytest.mark.parametrize('cen', [None, CEN], ids=['nocen', 'cen18'])
@pytest.mark.parametrize('spec', SPLINE_ONEBASIS)
def test_crosspred_onebasis_spline_uses_training_knots(spec, cen):
    temp = _series(500)
    ob = _make_ob(temp, _fill_knots(spec))
    _crosspred_vs_r(ob, P + 'ob', temp, seed=21, cen=cen)


THR_CASES = [pytest.param({'thr_value': 20.0}, id='thr20'),
             pytest.param({}, id='default_median'),
             pytest.param({'thr_value': 10.0, 'side': 'l'}, id='thr10_side_l'),
             pytest.param({'thr_value': np.array([10.0, 25.0]), 'side': 'd'}, id='thr10_25_side_d')]


@pytest.mark.parametrize('cen', [None, 15.0], ids=['nocen', 'cen15'])
@pytest.mark.parametrize('spec', THR_CASES)
def test_crosspred_onebasis_thr_keeps_training_threshold(spec, cen):
    temp = _series(500)
    ob = _make_ob(temp, {'fun': 'thr', **spec})
    _crosspred_vs_r(ob, P + 'ob', temp, seed=22, cen=cen)


# --------------------------------------------------------------------------------------------------------------------
# A2: lag sub-period / single lag / bylag with a spline lag basis                     (fixed)        
# --------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('kind, fit_lag, pred_lag, bylag', LAG_SUB)
def test_crosspred_cb_lag_subperiod_uses_training_lag_boundary_knots(kind, fit_lag, pred_lag, bylag):
    temp = _series()
    var = {'fun': 'bs', 'degree': 2, 'knots': _rq(.1, .75, .9)}
    cb = _make_cb(temp, list(fit_lag), var, _arglag(kind, fit_lag))
    _crosspred_vs_r(cb, P + 'cb', temp, seed=31, cen=CEN, lag=pred_lag, bylag=bylag)


# --------------------------------------------------------------------------------------------------------------------
# A2: integer lag basis                                                               (fixed)        
# --------------------------------------------------------------------------------------------------------------------
INT_ARGLAG = [pytest.param({'fun': 'integer', 'intercept': False}, id='intercept_false'),
              pytest.param({'fun': 'integer', 'values': np.arange(0.0, 8.0)}, id='values_0to7'),
              pytest.param({'fun': 'integer', 'values': np.arange(0.0, 6.0), 'intercept': False},
                           id='values_0to5_intercept_false')]


@pytest.mark.parametrize('arglag', INT_ARGLAG)
def test_crossbasis_integer_lag_intercept_and_values(arglag):
    temp = _series(300)
    var = {'fun': 'ns', 'knots': _rq(.25, .5, .75)}
    cb = _make_cb(temp, LAG_INT, var, arglag)                 # asserts the R shape first
    ref = rget(f'unclass({P}cb)')
    assert_close(np.asarray(cb.basis), ref, rtol=1e-12, what='integer-lag cross-basis')


INT_SUB = [pytest.param((0, 5), (1, 3), {}, id='fit0-5_pred1-3'),
           pytest.param((0, 5), (0, 2), {}, id='fit0-5_pred0-2'),
           pytest.param((0, 5), (3, 5), {}, id='fit0-5_pred3-5'),
           pytest.param((0, 5), (2, 2), {}, id='fit0-5_single_lag2'),
           pytest.param((2, 6), (3, 4), {}, id='fit2-6_pred3-4'),
           pytest.param((0, 5), (1, 3), {'intercept': False}, id='intercept_false_pred1-3')]


@pytest.mark.parametrize('fit_lag, pred_lag, extra', INT_SUB)
def test_crosspred_integer_lag_subperiod(fit_lag, pred_lag, extra):
    temp = _series(400)
    var = {'fun': 'ns', 'knots': _rq(.25, .5, .75)}
    cb = _make_cb(temp, list(fit_lag), var, {'fun': 'integer', **extra})
    _crosspred_vs_r(cb, P + 'cb', temp, seed=41, cen=CEN, lag=pred_lag)


ONEBASIS_INTEGER = [pytest.param({}, id='default'),
                    pytest.param({'intercept': True}, id='intercept_true'),
                    pytest.param({'values': np.arange(-1.0, 8.0), 'intercept': True}, id='values_wider_than_x'),
                    pytest.param({'values': np.arange(1.0, 6.0)}, id='values_1to5')]


@pytest.mark.parametrize('spec', ONEBASIS_INTEGER)
def test_onebasis_integer_matches_r(spec):
    x = np.random.default_rng(5).integers(1, 6, 40).astype(float)
    ob = _make_ob(x, {'fun': 'integer', **spec})
    assert_close(np.asarray(ob.basis), rget(f'unclass({P}ob)'), rtol=1e-12, what='integer basis')
    assert_close(_attr(ob, 'values'), _r_attr(P + 'ob', 'values'), rtol=1e-12, what='attribute values')


# --------------------------------------------------------------------------------------------------------------------
# PLAIN tests: behaviour that is already faithful and must stay so while the fixes land
# --------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('argvar', DF_VAR)
def test_crossbasis_matrix_df_var_basis_matches_r(argvar):
    """Training side is exact: df-derived knots of ns/bs give R's cross-basis bit for bit (first LAG rows NaN)."""
    temp = _series()
    cb = _make_cb(temp, LAG, argvar, {'fun': 'ns', 'knots': _rlogknots(LAG, 2)})
    ref = rget(f'unclass({P}cb)')
    assert_close(np.asarray(cb.basis), ref, rtol=1e-12, what='cross-basis')
    assert int(np.isnan(cb.basis).any(axis=1).sum()) == LAG


PRED_OPTIONS = [pytest.param({'cen': CEN}, id='cen18'),
                pytest.param({'cen': None}, id='nocen'),
                pytest.param({'cen': CEN, 'cumul': True}, id='cumul'),
                pytest.param({'cen': CEN, 'bylag': 0.5}, id='bylag0.5'),
                pytest.param({'cen': CEN, 'wide': True}, id='grid_wider_than_data')]


@pytest.mark.parametrize('opt', PRED_OPTIONS)
@pytest.mark.parametrize('name', sorted(EXPLICIT_VAR))
def test_crosspred_cb_explicit_knots_matches_r(name, opt):
    """Explicit var knots + explicit lag knots over the full lag (the validated style) already agree to 1e-8."""
    temp = _series()
    opt = dict(opt)
    at = np.linspace(temp.min() - 4, temp.max() + 4, 17) if opt.pop('wide', False) else None
    cb = _make_cb(temp, LAG, EXPLICIT_VAR[name](), {'fun': 'ns', 'knots': _rlogknots(LAG, 2)})
    _crosspred_vs_r(cb, P + 'cb', temp, seed=51, at=at, **opt)


@pytest.mark.parametrize('cen', [15.0, None], ids=['cen15', 'nocen'])
@pytest.mark.parametrize('argvar', [
    pytest.param({'fun': 'thr', 'thr_value': 20.0}, id='thr20'),
    pytest.param({'fun': 'strata', 'breaks': np.array([10.0, 20.0])}, id='strata_breaks')])
def test_crosspred_cb_explicit_thr_strata_matches_r(argvar, cen):
    """Explicit thr.value / strata breaks in argvar are used as given by crosspred (R: same attributes)."""
    temp = _series()
    cb = _make_cb(temp, LAG, argvar, {'fun': 'ns', 'knots': _rlogknots(LAG, 2)})
    _crosspred_vs_r(cb, P + 'cb', temp, seed=56, cen=cen)


@pytest.mark.parametrize('name', sorted(EXPLICIT_VAR))
def test_crosspred_reduced_coef_explicit_knots_matches_r(name):
    """Reduced coef/vcov (BLUP route: crosspred(cb, coef=<var-basis size>)) with explicit knots equals
    crosspred(onebasis) in R for allfit/allse."""
    temp = _series()
    cb = _make_cb(temp, LAG, EXPLICIT_VAR[name](), {'fun': 'ns', 'knots': _rlogknots(LAG, 2)})
    r(f'{P}av <- attr({P}cb, "argvar"); {P}av$cen <- NULL; {P}bvar <- do.call(onebasis, c(list(x={P}temp), {P}av))')
    coef, vcov = _rand_coef_vcov(int(rget(f'ncol({P}bvar)')[0]), 57)
    at = _grid(temp)
    ref = _r_pred(P + 'bvar', coef, vcov, at, cen=CEN)
    _compare_pred(_py_pred(cb, coef, vcov, at, cen=CEN), {f: ref[f] for f in ('allfit', 'allse')})


@pytest.mark.parametrize('kind', ['log2', 'df3'])
@pytest.mark.parametrize('fit_lag', [(1, 7), (2, 8)], ids=['fit1-7', 'fit2-8'])
def test_crosspred_cb_minimum_lag_above_zero_full_lag_matches_r(fit_lag, kind):
    """Crossbasis with a minimum lag > 0, predicted over the fitted lag range (cumul, and bylag 0.5)."""
    temp = _series()
    cb = _make_cb(temp, list(fit_lag), {'fun': 'bs', 'degree': 2, 'knots': _rq(.1, .75, .9)}, _arglag(kind, fit_lag))
    assert_close(np.asarray(cb.basis), rget(f'unclass({P}cb)'), rtol=1e-12, what='cross-basis')
    _crosspred_vs_r(cb, P + 'cb', temp, seed=58, cen=CEN, cumul=True)
    _crosspred_vs_r(cb, P + 'cb', temp, seed=59, cen=CEN, bylag=0.5)


@pytest.mark.parametrize('bylag', [1, 0.5])
@pytest.mark.parametrize('kind', ['df3', 'df4'])
def test_crosspred_cb_df_lag_basis_full_lag_matches_r(kind, bylag):
    """Over the full lag range (also with fractional bylag) the df-derived lag knots are the training ones."""
    temp = _series()
    cb = _make_cb(temp, LAG, {'fun': 'bs', 'degree': 2, 'knots': _rq(.1, .75, .9)}, _arglag(kind))
    _crosspred_vs_r(cb, P + 'cb', temp, seed=52, cen=CEN, cumul=(bylag == 1), bylag=bylag)


ONEBASIS_FAITHFUL = [pytest.param({'fun': 'lin'}, id='lin'),
                     pytest.param({'fun': 'ns', 'knots': 'q3', 'Boundary_knots': 'wide'}, id='ns_knots_and_boundary'),
                     pytest.param({'fun': 'poly', 'degree': 2}, id='poly2'),
                     pytest.param({'fun': 'poly', 'degree': 3, 'scale': 25.0}, id='poly3_scale25'),
                     pytest.param({'fun': 'strata', 'breaks': np.array([10.0, 20.0])}, id='strata_breaks')]


@pytest.mark.parametrize('cen', [None, 15.0], ids=['nocen', 'cen15'])
@pytest.mark.parametrize('spec', ONEBASIS_FAITHFUL)
def test_crosspred_onebasis_lin_poly_strata_matches_r(spec, cen):
    """OneBasis attributes that are supplied or recorded today (lin, poly, strata breaks=, ns knots + Boundary.knots)
    already round-trip through crosspred."""
    temp = _series(500)
    ob = _make_ob(temp, _fill_knots(spec))
    _crosspred_vs_r(ob, P + 'ob', temp, seed=53, cen=cen)


def test_onebasis_attributes_recorded_today_match_r():
    """Attributes PyDLNM already records equal R's: poly scale, thr.value, strata breaks, ns explicit knots."""
    temp = _series(500)
    ob = _make_ob(temp, {'fun': 'poly', 'degree': 3})
    assert_close(_attr(ob, 'scale'), _r_attr(P + 'ob', 'scale'), rtol=1e-14, what='poly scale')
    ob = _make_ob(temp, {'fun': 'thr'})
    assert_close(_attr(ob, 'thr.value'), _r_attr(P + 'ob', 'thr.value'), rtol=1e-14, what='default thr.value (median)')
    ob = _make_ob(temp, {'fun': 'thr', 'thr_value': np.array([10.0, 25.0]), 'side': 'd'})
    assert_close(_attr(ob, 'thr.value'), _r_attr(P + 'ob', 'thr.value'), rtol=1e-14, what='double thr.value')
    ob = _make_ob(temp, {'fun': 'strata', 'breaks': np.array([10.0, 20.0])})
    assert_close(_attr(ob, 'breaks'), _r_attr(P + 'ob', 'breaks'), rtol=1e-14, what='strata breaks')
    ob = _make_ob(temp, _fill_knots({'fun': 'ns', 'knots': 'q3'}))
    assert_close(_attr(ob, 'knots'), _r_attr(P + 'ob', 'knots'), rtol=1e-14, what='ns explicit knots')


@pytest.mark.parametrize('fit_lag, kw', [((0, LAG_INT), {}), ((2, 6), {}), ((0, LAG_INT), {'cumul': True}),
                                         ((0, LAG_INT), {'wide': True})],
                         ids=['fit0-5', 'fit2-6', 'cumul', 'wide_grid'])
def test_integer_lag_default_matches_r(fit_lag, kw):
    """Default integer lag basis (intercept TRUE, values = lags): cross-basis and full-lag crosspred agree with R."""
    temp = _series(400)
    kw = dict(kw)
    at = np.linspace(temp.min() - 3, temp.max() + 3, 13) if kw.pop('wide', False) else None
    cb = _make_cb(temp, list(fit_lag), {'fun': 'ns', 'knots': _rq(.25, .5, .75)}, {'fun': 'integer'})
    assert_close(np.asarray(cb.basis), rget(f'unclass({P}cb)'), rtol=1e-12, what='integer-lag cross-basis')
    _crosspred_vs_r(cb, P + 'cb', temp, seed=54, cen=CEN, at=at, **kw)


@pytest.mark.parametrize('kwargs', [{'lag': (1, 3), 'cumul': True}, {'bylag': 0.5}],
                         ids=['cumul_with_subperiod', 'integer_lag_bylag'])
def test_crosspred_invalid_lag_requests_raise_in_r_and_python(kwargs):
    """R refuses cumul at a lag sub-period and non-integer lags for the integer basis; PyDLNM must refuse too."""
    temp = _series(400)
    cb = _make_cb(temp, LAG_INT, {'fun': 'ns', 'knots': _rq(.25, .5, .75)}, {'fun': 'integer'})
    coef, vcov = _rand_coef_vcov(cb.basis.shape[1], 55)
    at = _grid(temp)
    with pytest.raises(Exception):
        _r_pred(P + 'cb', coef, vcov, at, cen=CEN, **kwargs)
    with pytest.raises(ValueError):
        _py_pred(cb, coef, vcov, at, cen=CEN, **kwargs)
