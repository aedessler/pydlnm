"""Gap multi_crossbasis_model_selection: SEVERAL bases in ONE model (model_utils.basis_block / validate_model_compatibility).

R picks the block of the basis passed to crosspred / crossreduce / attrdl out of the coefficients of a multi-exposure
model BY THE NAME of the basis object (grep of '<name>[[:print:]]*v#.l#', deparse(substitute(basis))), so
crosspred(cb.temp, model) and crosspred(cb.o3, model) both work on glm(death ~ cb.temp + cb.o3 + dow + ns(time)) (the
dlnm vignette fits pm10 and temperature this way).  PyDLNM has no basis name: it matches only the pattern v#.l# (b# for a
one-basis) and, when the model has no informative names, takes the block only if it is the whole coefficient vector.

Every test computes its reference in R at run time (rpy2, dlnm, the Lancet attrdl.R) on the identical design matrix:
the R fit is R's own glm(quasipoisson) on the chicagoNMMAPS rows 1..2500 (temp and o3 as the two exposures), the
Python fit is statsmodels Poisson with scale='X2' (= R's quasi-Poisson dispersion) on R's model.matrix.  R's and
statsmodels' IRLS agree to ~1e-9 (tol 1e-10 / epsilon 1e-14), so the comparisons use rtol 1e-8 like the other
statsmodels tests of this suite.

Three statsmodels designs are used, because the coefficient names are what the selection depends on:
  dataframe  pandas DataFrame exog, columns named exactly as R names them ('<basis name><colname>', 'mcb_..._tv1.l1')
  numpy      plain ndarray exog (statsmodels names const/x1..xk: no information at all)
  formula    smf.glm('y ~ cbt + cbo + rest') with the bases as patsy matrix terms ('cbt[0]', 'cbt[1]', ...)
and four two-basis layouts (6+2 columns with temperature first, 4+6 with o3 first, 10+10 with equal size, 6+1 with a
one-column second basis), plus a two-one-basis model (ns(temp) and ns(humidity), kind 'one').

Findings (all fixed; every test below is an ordinary test, and the tests that used to be strict xfails assert
R's behaviour):

  MB1  crosspred / CrossPred, crossreduce (overall, var, lag), attrdl and find_mmt with model= on a model holding two
       cross-bases (or two one-bases) used to raise ValueError('... pass coef= and vcov= ...') for every design and
       every basis, although R returns the block of the basis passed.
  MB2  (was: silent wrong numbers) an ImprovedGLMInterface / Rpy2GLMInterface fitted with a second cross-basis among
       other_vars exposed only the block of ITS cross-basis; crosspred(cb2, model=interface) took that block as the
       coefficients of cb2 whenever the two bases have the same number of columns (R: the block of cb2).
  MB3  (was: silent wrong numbers) a model that holds only the block of basis A was accepted for basis B of the same
       size when its columns carry v#.l# names; R stops ('coef/vcov not consistent with basis matrix') because the
       name of B is not in the model.
  MB4  a ONE-column cross-basis (lin x the default strata(df=1) lag basis) is named by the object alone in R's model
       ('cb', not 'cbv1.l1'); R's `cond <- name` branch finds it, PyDLNM only knew the v#.l# / b# patterns.

The fix (model_utils.locate_block): R's name is not visible to Python, so the block of the basis passed is identified
by, in this order, an explicit `name=` prefix (or the `name` attribute of the basis), the columns of the basis in the
design matrix of the model (statsmodels `model.exog`, the R model of a PyDLNM GLM interface), and the coefficient
names v#.l# / b#. The model carries the basis columns, so no name is needed, and a basis that is not in the model is
rejected. The `name=` route is covered at the end of this module.

The neighbouring behaviour (block by name in a model with one cross-basis, a cross-basis next to a one-basis, explicit
coef=/vcov= slices, the coef=/vcov= advice of the ValueError) is covered by the same tests.
"""
import contextlib
import copy
import io
import warnings

import numpy as np
import pytest

from rhelpers import REPO, assert_close, np2r, r, rget

RTOL = 1e-8                      # statsmodels vs R IRLS on an identical design (both converged to ~1e-10)
N_ROWS = 2500                    # first 2500 days of chicagoNMMAPS
SP = 'mcb'                       # prefix of every R global created here
GLM_TOL = dict(tol=1e-10, maxiter=200)
DESIGNS = ['dataframe', 'numpy', 'formula']
BASES = ['t', 'o']               # t: temperature basis, o: second exposure basis (o3, or humidity for the one-bases)


def _warm_up_r_lapack():
    r('invisible(chol2inv(chol(diag(2) + 1))); invisible(solve(diag(2)))')


_warm_up_r_lapack()


@contextlib.contextmanager
def _quiet():
    """PyDLNM prints summaries and warns while it works; none of that is the topic here."""
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        yield


# ==============================================================================================================
# layouts: which bases, in which order the model lists them
# ==============================================================================================================
CONFIGS = {
    'six_two': dict(kind='cb', lag=5, order=('t', 'o'),
                    t=dict(x='temp', argvar=dict(fun='ns', df=3), arglag=dict(fun='ns', df=2)),       # 3 x 2 = 6
                    o=dict(x='o3', argvar=dict(fun='lin'), arglag=dict(fun='ns', df=2))),             # 1 x 2 = 2
    'rev_four_six': dict(kind='cb', lag=4, order=('o', 't'),                                          # o3 block FIRST
                         t=dict(x='temp', argvar=dict(fun='ns', df=3), arglag=dict(fun='poly', degree=2)),   # 3 x 2 = 6
                         o=dict(x='o3', argvar=dict(fun='poly', degree=2), arglag=dict(fun='ns', df=2))),    # 2 x 2 = 4
    'ten_ten': dict(kind='cb', lag=6, order=('t', 'o'),                                               # equal size
                    t=dict(x='temp', argvar=dict(fun='bs', degree=2, df=5), arglag=dict(fun='ns', df=2)),    # 5 x 2
                    o=dict(x='o3', argvar=dict(fun='bs', degree=2, df=5), arglag=dict(fun='ns', df=2))),     # 5 x 2
    'one_col': dict(kind='cb', lag=3, order=('t', 'o'),                                              # o: ONE column
                    t=dict(x='temp', argvar=dict(fun='ns', df=3), arglag=dict(fun='ns', df=2)),       # 3 x 2 = 6
                    o=dict(x='o3', argvar=dict(fun='lin'), arglag={})),                               # lin x strata(1)
    'one_one': dict(kind='one', order=('t', 'o'),
                    t=dict(x='temp', argvar=dict(fun='ns', df=4)),
                    o=dict(x='rhum', argvar=dict(fun='ns', df=3))),
}
CB_CONFIGS = ['six_two', 'rev_four_six', 'ten_ten', 'one_col']


def _rlist(d):
    return 'list(' + ', '.join(f'{k}="{v}"' if isinstance(v, str) else f'{k}={v!r}' for k, v in d.items()) + ')'


def _rname(cfg, tag):
    """Name of the R basis object (the name R greps for)."""
    return f'{SP}_{cfg}_{tag}'


_DATA = {}


def _data():
    """chicagoNMMAPS rows 1..N in R (global `mcb_d`) and as arrays."""
    if not _DATA:
        r(f'data(chicagoNMMAPS, package="dlnm"); {SP}_d <- chicagoNMMAPS[1:{N_ROWS},]; '
          f'{SP}_d$dowf <- factor({SP}_d$dow); {SP}_d$tt <- seq_len(nrow({SP}_d))')
        for col in ('temp', 'o3', 'rhum', 'death'):
            _DATA[col] = rget(f'{SP}_d${col}')
    return _DATA


class Case:
    """R objects and identical Python objects of one layout / set of bases in the model."""


_CASES = {}


def _case(cfg_name, terms=None):
    """R basis objects + R glm fit of the layout, the Python bases and the design matrix of the R fit (cached).

    terms: the bases in the model (default: both, in the layout's order)."""
    from basis import CrossBasis, OneBasis
    cfg = CONFIGS[cfg_name]
    terms = tuple(terms or cfg['order'])
    key = (cfg_name, terms)
    if key in _CASES:
        return _CASES[key]
    d = _data()
    c = Case()
    c.cfg_name, c.cfg, c.terms, c.kind = cfg_name, cfg, terms, cfg['kind']
    c.py, c.rn, c.x = {}, {}, {}
    for tag in cfg['order']:
        spec = cfg[tag]
        c.rn[tag] = _rname(cfg_name, tag)
        c.x[tag] = d[spec['x']]
        if c.kind == 'cb':
            r(f'{c.rn[tag]} <- crossbasis({SP}_d${spec["x"]}, lag={cfg["lag"]}, argvar={_rlist(spec["argvar"])}, '
              f'arglag={_rlist(spec["arglag"])})')
            with _quiet():
                c.py[tag] = CrossBasis(c.x[tag], lag=cfg['lag'], argvar=copy.deepcopy(spec['argvar']),
                                       arglag=copy.deepcopy(spec['arglag']))
            ref = rget(f'unclass({c.rn[tag]})')
            assert_close(np.asarray(c.py[tag].basis), ref, rtol=1e-12, what=f'{cfg_name}:{tag} cross-basis precondition')
        else:
            r(f'{c.rn[tag]} <- do.call("onebasis", c(list(x={SP}_d${spec["x"]}), {_rlist(spec["argvar"])}))')
            with _quiet():
                c.py[tag] = OneBasis(c.x[tag], **copy.deepcopy(spec['argvar']))
            assert_close(np.asarray(c.py[tag].basis), rget(f'unclass({c.rn[tag]})'), rtol=1e-12,
                         what=f'{cfg_name}:{tag} one-basis precondition')
    tag_s = ''.join(terms)
    c.fit = f'{SP}_{cfg_name}_{tag_s}_fit'
    rhs = ' + '.join(c.rn[t] for t in terms)
    r(f'{c.fit} <- glm(death ~ {rhs} + dowf + ns(tt, df=10), family=quasipoisson(), data={SP}_d, '
      f'control=glm.control(epsilon=1e-14, maxit=200))')
    c.X, c.y = rget(f'model.matrix({c.fit})'), rget(f'{c.fit}$y')
    c.names = [str(s) for s in r(f'colnames(model.matrix({c.fit}))')]
    c.idx = {}
    for t in terms:
        c.idx[t] = np.array([i for i, nm in enumerate(c.names) if nm.startswith(c.rn[t])], dtype=int)
        ncol = c.py[t].shape[1]
        assert len(c.idx[t]) == ncol, f'{cfg_name}:{t}: {len(c.idx[t])} R columns vs {ncol} basis columns'
        B = np.asarray(c.py[t].basis, dtype=float)
        complete = np.isfinite(B).all(axis=1)              # R's na.omit drops the rows of the lag-induced NaNs
        assert c.X.shape[0] == int(complete.sum()), f'{cfg_name}:{t}: R kept {c.X.shape[0]} rows'
        assert_close(c.X[:, c.idx[t]], B[complete], rtol=1e-12, what=f'{cfg_name}:{t} design columns == PyDLNM basis')
    c.rest = np.array([i for i in range(c.X.shape[1]) if i != 0 and all(i not in c.idx[t] for t in terms)], dtype=int)
    # exposure summaries used for `at` and `cen`
    c.at, c.cen = {}, {}
    for t in cfg['order']:
        x = c.x[t]
        c.at[t] = np.quantile(x, np.linspace(0.05, 0.95, 12))
        c.cen[t] = float(np.median(x))
    _CASES[key] = c
    return c


_FITS = {}


def _fit(c, design):
    """statsmodels Poisson GLM, scale='X2' (R quasi-Poisson), on R's design matrix, in the requested design flavour."""
    key = (c.cfg_name, c.terms, design)
    if key in _FITS:
        return _FITS[key]
    sm = pytest.importorskip('statsmodels.api')
    pd = pytest.importorskip('pandas')
    fam = sm.families.Poisson()
    kw = dict(GLM_TOL, scale='X2')
    if design == 'numpy':
        res = sm.GLM(c.y, c.X, family=fam).fit(**kw)
    elif design == 'dataframe':
        cols = ['const'] + c.names[1:]
        res = sm.GLM(c.y, pd.DataFrame(c.X, columns=cols), family=fam).fit(**kw)
    else:
        smf = pytest.importorskip('statsmodels.formula.api')
        data = {'y': c.y, 'rest': c.X[:, c.rest]}
        for t in c.terms:
            data[f'cb{t}'] = c.X[:, c.idx[t]]
        res = smf.glm('y ~ ' + ' + '.join(f'cb{t}' for t in c.terms) + ' + rest', data=data, family=fam).fit(**kw)
    _FITS[key] = res
    return res


def _block(c, design, res, tag):
    """(coef, vcov) of the block of basis `tag`, extracted by construction (positions known from R's model.matrix)."""
    idx = c.idx[tag]
    return np.asarray(res.params)[idx], np.asarray(res.cov_params())[np.ix_(idx, idx)]


# ==============================================================================================================
# R references and Python calls
# ==============================================================================================================
PRED_FIELDS_CB = ['coefficients', 'vcov', 'matfit', 'matse', 'allfit', 'allse', 'cumfit', 'cumse',
                  'allRRfit', 'allRRlow', 'allRRhigh', 'matRRfit']
PRED_FIELDS_ONE = ['coefficients', 'vcov', 'matfit', 'matse', 'allfit', 'allse', 'allRRfit', 'allRRlow', 'allRRhigh']


def _push(c, tag):
    np2r(f'{SP}_at', c.at[tag])
    np2r(f'{SP}_cen', [c.cen[tag]])


def r_crosspred(c, tag):
    """R crosspred(<basis>, <R fit>): the model route, block selected by the name of the basis."""
    _push(c, tag)
    cumul = 'TRUE' if c.kind == 'cb' else 'FALSE'
    r(f'{SP}_pR <- crosspred({c.rn[tag]}, {c.fit}, at={SP}_at, cen={SP}_cen, cumul={cumul})')
    fields = PRED_FIELDS_CB if c.kind == 'cb' else PRED_FIELDS_ONE
    return {f: rget(f'unname({SP}_pR${f})') for f in fields}


def _assert_pred_equals(pp, ref, what):
    bad = []
    for f, refv in ref.items():
        py = getattr(pp, f, None)
        if py is None:
            bad.append(f'{f}: missing')
            continue
        try:
            assert_close(np.asarray(py, dtype=float), refv, rtol=RTOL, what=f)
        except AssertionError as exc:
            bad.append(str(exc))
    assert not bad, f'{what}: ' + '; '.join(bad)


def py_crosspred(c, tag, **kw):
    from prediction import crosspred
    with _quiet():
        return crosspred(c.py[tag], at=c.at[tag], cen=c.cen[tag], cumul=(c.kind == 'cb'), **kw)


def r_crossreduce(c, tag, type_, value):
    _push(c, tag)
    extra = '' if value is None else f', value={value!r}'
    r(f'{SP}_rR <- crossreduce({c.rn[tag]}, {c.fit}, type="{type_}"{extra}, at={SP}_at, cen={SP}_cen)')
    return dict(coef=rget(f'unname({SP}_rR$coefficients)'), vcov=rget(f'unname({SP}_rR$vcov)'),
                fit=rget(f'unname({SP}_rR$fit)'), se=rget(f'unname({SP}_rR$se)'))


def py_crossreduce(c, tag, type_, value, **kw):
    from crossreduce import crossreduce
    with _quiet():
        return crossreduce(c.py[tag], type=type_, value=value, at=c.at[tag], cen=c.cen[tag], **kw)


def _assert_red_equals(red, ref, what):
    for f in ('coef', 'vcov', 'fit', 'se'):
        assert_close(np.asarray(getattr(red, f), dtype=float), ref[f], rtol=RTOL, what=f'{what} {f}')


_ATTR_LOADED = []


def _load_r_attrdl():
    if not _ATTR_LOADED:
        if not bool(r('nzchar(system.file(package="tsModel"))')[0]):
            pytest.skip('R package tsModel not installed')
        r('suppressMessages(library(tsModel))')
        path = REPO / '2015_gasparrini_Lancet_Rcodedata-master' / 'attrdl.R'
        r(f'{SP}_attr <- new.env(); sys.source("{path}", envir={SP}_attr)')
        _ATTR_LOADED.append(True)


ATTR_MODES = [('af', 'forw', False), ('an', 'back', True)]          # (type, dir, tot)


def r_attrdl(c, tag, mode):
    _load_r_attrdl()
    typ, dir_, tot = mode
    np2r(f'{SP}_x', c.x[tag])
    np2r(f'{SP}_cases', _data()['death'])
    np2r(f'{SP}_cen', [c.cen[tag]])
    r(f'{SP}_aR <- {SP}_attr$attrdl({SP}_x, {c.rn[tag]}, {SP}_cases, model={c.fit}, type="{typ}", dir="{dir_}", '
      f'tot={"TRUE" if tot else "FALSE"}, cen={SP}_cen)')
    return np.atleast_1d(rget(f'as.numeric({SP}_aR)'))


def py_attrdl(c, tag, mode, **kw):
    from attribution import attrdl
    typ, dir_, tot = mode
    with _quiet():
        res = attrdl(c.x[tag], c.py[tag], _data()['death'], type=typ, dir=dir_, tot=tot, cen=c.cen[tag], **kw)
    return np.atleast_1d(np.asarray(res[typ + '_total'] if tot else res[typ], dtype=float))


def _assert_attr_equals(py, ref, what):
    assert py.shape == ref.shape, f'{what}: shape Python {py.shape} vs R {ref.shape}'
    assert np.array_equal(np.isfinite(py), np.isfinite(ref)), f'{what}: finite pattern differs'
    assert_close(py, ref, rtol=RTOL, what=what)


def r_mmt(c, tag):
    """R recipe of the Lancet scripts: minimum of the overall curve on the grid (uncentred)."""
    _push(c, tag)
    r(f'{SP}_mR <- crosspred({c.rn[tag]}, {c.fit}, at={SP}_at, cen=FALSE)')
    return float(rget(f'{SP}_mR$predvar[which.min({SP}_mR$allfit)]')[0])


def py_mmt(c, tag, **kw):
    from centering import find_mmt
    with _quiet():
        return float(find_mmt(c.py[tag], at=c.at[tag], **kw)['mmt'])


def r_error(code):
    """Message of the R error raised by `code`, or None."""
    res = r(f'tryCatch({{ {code}; NULL }}, error=function(e) conditionMessage(e))')
    return str(res[0]) if len(res) else None


CB_CASES = [(cfg, d, b) for cfg in CB_CONFIGS for d in DESIGNS for b in BASES]
CB_IDS = [f'{cfg}-{d}-{b}' for cfg, d, b in CB_CASES]
ONE_CASES = [('one_one', d, b) for d in DESIGNS for b in BASES]
ONE_IDS = [f'{cfg}-{d}-{b}' for cfg, d, b in ONE_CASES]
ALL_CASES = CB_CASES + ONE_CASES
ALL_IDS = CB_IDS + ONE_IDS
# ==============================================================================================================
# MB1: model= on a model with two bases: R returns the block of the basis passed
# ==============================================================================================================
@pytest.mark.parametrize('cfg,design,tag', ALL_CASES, ids=ALL_IDS)
def test_crosspred_model_route_two_bases_matches_R(cfg, design, tag):
    """R: crosspred(cb.temp, model) and crosspred(cb.o3, model) on glm(death ~ cb.temp + cb.o3 + ...): every
    field of the prediction, for every design (named columns, plain ndarray, patsy matrix terms) and both bases."""
    c = _case(cfg)
    res = _fit(c, design)
    ref = r_crosspred(c, tag)
    pp = py_crosspred(c, tag, model=res)
    _assert_pred_equals(pp, ref, f'crosspred(model=) {cfg}/{design}/{tag}')


@pytest.mark.parametrize('cfg,design,tag', CB_CASES, ids=CB_IDS)
def test_crossreduce_model_route_two_crossbases_overall_matches_R(cfg, design, tag):
    c = _case(cfg)
    res = _fit(c, design)
    ref = r_crossreduce(c, tag, 'overall', None)
    red = py_crossreduce(c, tag, 'overall', None, model=res)
    _assert_red_equals(red, ref, f'crossreduce(model=) overall {cfg}/{design}/{tag}')


@pytest.mark.parametrize('design', DESIGNS)
@pytest.mark.parametrize('type_', ['var', 'lag'])
@pytest.mark.parametrize('tag', BASES)
def test_crossreduce_model_route_two_crossbases_var_lag_matches_R(design, type_, tag):
    c = _case('six_two')
    res = _fit(c, design)
    value = float(c.at[tag][4]) if type_ == 'var' else 2.0
    ref = r_crossreduce(c, tag, type_, value)
    red = py_crossreduce(c, tag, type_, value, model=res)
    _assert_red_equals(red, ref, f'crossreduce(model=) {type_} {design}/{tag}')


@pytest.mark.parametrize('cfg,design,tag', CB_CASES, ids=CB_IDS)
def test_attrdl_model_route_two_crossbases_matches_R(cfg, design, tag):
    """attrdl(x, cb, cases, model=): per-observation forward AF and the backward total AN, both bases."""
    c = _case(cfg)
    res = _fit(c, design)
    for mode in ATTR_MODES:
        ref = r_attrdl(c, tag, mode)
        py = py_attrdl(c, tag, mode, model=res)
        _assert_attr_equals(py, ref, f'attrdl(model=) {mode} {cfg}/{design}/{tag}')


@pytest.mark.parametrize('cfg,design,tag', CB_CASES, ids=CB_IDS)
def test_find_mmt_model_route_two_crossbases_matches_R(cfg, design, tag):
    c = _case(cfg)
    res = _fit(c, design)
    assert py_mmt(c, tag, model=res) == r_mmt(c, tag)


# ==============================================================================================================
# never a silent wrong number for a multi-basis model: R's numbers, or a ValueError with the coef=/vcov= advice
# ==============================================================================================================
def _model_route_calls(c, tag, res):
    """(label, callable) per entry point on the model route: the callable runs Python with model= and asserts that
    the result equals R's (it raises ValueError when PyDLNM refuses)."""
    def pred():
        pp = py_crosspred(c, tag, model=res)
        _assert_pred_equals(pp, r_crosspred(c, tag), 'crosspred')

    def red():
        rr = py_crossreduce(c, tag, 'overall', None, model=res)
        _assert_red_equals(rr, r_crossreduce(c, tag, 'overall', None), 'crossreduce')

    def attr():
        mode = ATTR_MODES[0]
        _assert_attr_equals(py_attrdl(c, tag, mode, model=res), r_attrdl(c, tag, mode), 'attrdl')

    def mmt():
        assert py_mmt(c, tag, model=res) == r_mmt(c, tag)

    calls = [('crosspred', pred)]
    if c.kind == 'cb':                                        # R's crossreduce / attrdl take cross-bases only
        calls += [('crossreduce', red), ('attrdl', attr), ('find_mmt', mmt)]
    return calls


@pytest.mark.parametrize('cfg,design,tag', ALL_CASES, ids=ALL_IDS)
def test_two_basis_model_route_gives_R_numbers_or_a_clear_ValueError(cfg, design, tag):
    """Whatever the design, crosspred / crossreduce / attrdl / find_mmt on a two-basis model either return R's numbers
    for the block of the basis passed or raise the ValueError that asks for coef=/vcov= -- never other numbers (the
    equality with R is the subject of the tests above; this one keeps the contract for any future design)."""
    c = _case(cfg)
    res = _fit(c, design)
    for label, call in _model_route_calls(c, tag, res):
        try:
            call()
        except ValueError as exc:
            msg = str(exc)
            assert 'coef=' in msg and 'vcov=' in msg, f'{label}: ValueError without the coef=/vcov= advice: {msg[:300]}'


# ==============================================================================================================
# the supported workaround: explicit coef=/vcov= slices reproduce R's model route exactly (plain)
# ==============================================================================================================
@pytest.mark.parametrize('cfg,design,tag', ALL_CASES, ids=ALL_IDS)
def test_coef_vcov_slice_of_two_basis_model_matches_R_model_route(cfg, design, tag):
    """The block of the basis extracted by hand (by name / position) and passed as coef=, vcov= gives R's
    crosspred(basis, model), crossreduce(basis, model) and attrdl(..., model=) for each of the two bases."""
    c = _case(cfg)
    res = _fit(c, design)
    coef, vcov = _block(c, design, res, tag)
    pp = py_crosspred(c, tag, coef=coef, vcov=vcov, model_link='log')
    _assert_pred_equals(pp, r_crosspred(c, tag), f'crosspred(coef=) {cfg}/{design}/{tag}')
    if c.kind == 'cb':
        red = py_crossreduce(c, tag, 'overall', None, coef=coef, vcov=vcov, model_link='log')
        _assert_red_equals(red, r_crossreduce(c, tag, 'overall', None), f'crossreduce(coef=) {cfg}/{design}/{tag}')
        for mode in ATTR_MODES:
            py = py_attrdl(c, tag, mode, coef=coef, vcov=vcov)
            _assert_attr_equals(py, r_attrdl(c, tag, mode), f'attrdl(coef=) {mode} {cfg}/{design}/{tag}')
        assert py_mmt(c, tag, model=None, coef=coef, vcov=vcov) == r_mmt(c, tag)


@pytest.mark.parametrize('type_,value_index', [('var', 4), ('lag', None)])
@pytest.mark.parametrize('tag', BASES)
def test_coef_vcov_slice_crossreduce_var_lag_matches_R(type_, value_index, tag):
    c = _case('rev_four_six')
    res = _fit(c, 'numpy')
    coef, vcov = _block(c, 'numpy', res, tag)
    value = float(c.at[tag][value_index]) if type_ == 'var' else 3.0
    red = py_crossreduce(c, tag, type_, value, coef=coef, vcov=vcov, model_link='log')
    _assert_red_equals(red, r_crossreduce(c, tag, type_, value), f'crossreduce(coef=) {type_}/{tag}')


# ==============================================================================================================
# one basis in the model: the block is found by name (R: same) whatever else the model holds (plain)
# ==============================================================================================================
@pytest.mark.parametrize('cfg,tag', [(cfg, tag) for cfg in CB_CONFIGS + ['one_one'] for tag in BASES
                                     if (cfg, tag) != ('one_col', 'o')])
def test_single_basis_dataframe_model_route_matches_R(cfg, tag):
    """One basis plus covariates, columns named as R names them: crosspred / crossreduce / attrdl / find_mmt with
    model= select the block by name and equal R (the case the audit fixed as theme I1)."""
    c = _case(cfg, terms=(tag,))
    res = _fit(c, 'dataframe')
    _assert_pred_equals(py_crosspred(c, tag, model=res), r_crosspred(c, tag), f'crosspred {cfg}/{tag}')
    if c.kind == 'cb':
        _assert_red_equals(py_crossreduce(c, tag, 'overall', None, model=res), r_crossreduce(c, tag, 'overall', None),
                           f'crossreduce {cfg}/{tag}')
        mode = ATTR_MODES[0]
        _assert_attr_equals(py_attrdl(c, tag, mode, model=res), r_attrdl(c, tag, mode), f'attrdl {cfg}/{tag}')
        assert py_mmt(c, tag, model=res) == r_mmt(c, tag)


def test_single_column_basis_model_route_matches_R():
    """MB4: glm(death ~ cb + dow + ns(time)) with a ONE-column cross-basis (lin x strata(df=1), the default arglag):
    R's coefficient is named 'cb' and crosspred / crossreduce / attrdl use `cond <- name`.  PyDLNM has no v#.l# / b#
    name to look for, and finds the column among the design columns of the model."""
    c = _case('one_col', terms=('o',))
    res = _fit(c, 'dataframe')
    assert c.names[c.idx['o'][0]] == c.rn['o'], 'R names a one-column basis by the object alone'
    _assert_pred_equals(py_crosspred(c, 'o', model=res), r_crosspred(c, 'o'), 'crosspred(model=)')
    _assert_red_equals(py_crossreduce(c, 'o', 'overall', None, model=res), r_crossreduce(c, 'o', 'overall', None),
                       'crossreduce(model=)')
    mode = ATTR_MODES[0]
    _assert_attr_equals(py_attrdl(c, 'o', mode, model=res), r_attrdl(c, 'o', mode), 'attrdl(model=)')


@pytest.mark.parametrize('cfg', CB_CONFIGS + ['one_one'])
@pytest.mark.parametrize('design', ['numpy', 'formula'])
@pytest.mark.parametrize('tag', BASES)
def test_single_basis_unnamed_design_gives_R_numbers_or_a_clear_ValueError(cfg, design, tag):
    """A model with ONE basis among other terms but no R-style names (ndarray exog, patsy matrix term): R's name
    lookup has no PyDLNM equivalent; R's numbers or the coef=/vcov= ValueError are acceptable, never other numbers."""
    c = _case(cfg, terms=(tag,))
    res = _fit(c, design)
    for label, call in _model_route_calls(c, tag, res):
        try:
            call()
        except ValueError as exc:
            assert 'coef=' in str(exc), f'{label}: {str(exc)[:300]}'


# ==============================================================================================================
# a cross-basis next to a one-basis (different name patterns v#.l# / b#): each is found by name (plain)
# ==============================================================================================================
def _mixed_case():
    """glm(death ~ cb(temp) + ob(rhum) + dow + ns(time)): a cross-basis and a one-basis in the same model."""
    from basis import CrossBasis, OneBasis
    key = ('mixed',)
    if key in _CASES:
        return _CASES[key]
    d = _data()
    c = Case()
    c.kind, c.cfg_name = 'mixed', 'mixed'
    r(f'{SP}_mx_cb <- crossbasis({SP}_d$temp, lag=4, argvar=list(fun="ns", df=3), arglag=list(fun="ns", df=2))')
    r(f'{SP}_mx_ob <- onebasis({SP}_d$rhum, fun="ns", df=3)')
    with _quiet():
        c.cb = CrossBasis(d['temp'], lag=4, argvar=dict(fun='ns', df=3), arglag=dict(fun='ns', df=2))
        c.ob = OneBasis(d['rhum'], fun='ns', df=3)
    r(f'{SP}_mx_fit <- glm(death ~ {SP}_mx_ob + {SP}_mx_cb + dowf + ns(tt, df=10), family=quasipoisson(), '
      f'data={SP}_d, control=glm.control(epsilon=1e-14, maxit=200))')
    c.X, c.y = rget(f'model.matrix({SP}_mx_fit)'), rget(f'{SP}_mx_fit$y')
    c.names = [str(s) for s in r(f'colnames(model.matrix({SP}_mx_fit))')]
    _CASES[key] = c
    return c


def test_crossbasis_next_to_onebasis_is_selected_by_its_pattern_like_R():
    sm = pytest.importorskip('statsmodels.api')
    pd = pytest.importorskip('pandas')
    from prediction import crosspred
    c = _mixed_case()
    res = sm.GLM(c.y, pd.DataFrame(c.X, columns=['const'] + c.names[1:]), family=sm.families.Poisson()).fit(
        scale='X2', **GLM_TOL)
    at_t = np.quantile(_data()['temp'], np.linspace(0.05, 0.95, 9))
    at_h = np.quantile(_data()['rhum'], np.linspace(0.05, 0.95, 9))
    np2r(f'{SP}_at', at_t)
    r(f'{SP}_mx_p1 <- crosspred({SP}_mx_cb, {SP}_mx_fit, at={SP}_at, cen=15, cumul=TRUE)')
    with _quiet():
        p1 = crosspred(c.cb, model=res, at=at_t, cen=15.0, cumul=True)
    _assert_pred_equals(p1, {f: rget(f'unname({SP}_mx_p1${f})') for f in PRED_FIELDS_CB}, 'cross-basis next to one-basis')
    np2r(f'{SP}_at', at_h)
    r(f'{SP}_mx_p2 <- crosspred({SP}_mx_ob, {SP}_mx_fit, at={SP}_at, cen=60)')
    with _quiet():
        p2 = crosspred(c.ob, model=res, at=at_h, cen=60.0)
    _assert_pred_equals(p2, {f: rget(f'unname({SP}_mx_p2${f})') for f in PRED_FIELDS_ONE}, 'one-basis next to cross-basis')


# ==============================================================================================================
# MB3: a basis that is NOT in the model must not be accepted (R stops: the name is not among the coefficients)
# ==============================================================================================================
def _check_model_without_the_basis_passed(cfg, design):
    """The model holds the temperature basis only; crosspred / crossreduce are asked for the o3 basis.  R: stops
    ('coef/vcov not consistent with basis matrix') since no coefficient carries the name of the o3 basis."""
    c = _case(cfg, terms=('t',))
    res = _fit(c, design)
    c2 = _case(cfg)                                       # R objects of both bases; the R fit below has only `t`
    _push(c2, 'o')
    msg = r_error(f'crosspred({c2.rn["o"]}, {c.fit}, at={SP}_at, cen={SP}_cen)')
    assert msg is not None and 'not consistent' in msg, f'R did not stop: {msg}'
    with pytest.raises(ValueError):
        py_crosspred(c2, 'o', model=res)
    with pytest.raises(ValueError):
        py_crossreduce(c2, 'o', 'overall', None, model=res)


@pytest.mark.parametrize('cfg,design', [(cfg, d) for cfg in ('ten_ten', 'six_two', 'rev_four_six') for d in DESIGNS])
def test_model_without_the_basis_passed_is_rejected_like_R(cfg, design):
    """The model holds the temperature block only: the o3 basis is refused for every design.  The 10+10 layout with
    R's own column names is the dangerous one (the temperature block has the size and the v#.l# names of o3's: it was
    used to predict o3 with temperature's coefficients); the other layouts differ in size or carry no names."""
    _check_model_without_the_basis_passed(cfg, design)


# ==============================================================================================================
# MB2: a fitted PyDLNM GLM interface holding a second cross-basis (other_vars): silently the WRONG block
# ==============================================================================================================
_IFACE = {}


def _iface_case():
    """ImprovedGLMInterface(cb_t) fitted with cb_o among other_vars (R names 'ifo<colname>'), and the R model it holds.

    The interface's own cross-basis columns are named 'cb.v1.l1' ...; R greps the name of the basis object, so the
    R calls below run in local() environments where the basis is called `cb` resp. `ifo` (neither name occurs inside
    the other one's column names)."""
    if _IFACE:
        return _IFACE
    from improved_glm import ImprovedGLMInterface
    c = _case('ten_ten')                                   # both bases have 10 columns
    d = _data()
    seas = rget(f'unclass(ns({SP}_d$tt, df=10))')
    names = [f'ifo{nm}' for nm in c.py['o'].colnames] + [f'seas{i}' for i in range(seas.shape[1])]   # R name: 'ifo'
    other = np.column_stack([np.asarray(c.py['o'].basis), seas])
    with _quiet():
        iface = ImprovedGLMInterface(c.py['t'])
        iface.fit_glm(d['death'], other_vars=other, formula_vars=names)
    import rpy2.robjects as ro
    ro.globalenv[f'{SP}_ifit'] = iface.r_model
    _IFACE.update(c=c, iface=iface)
    return _IFACE


def test_interface_first_crossbasis_matches_R():
    """PLAIN: the cross-basis the interface was built on is selected correctly (block 1 = its own coefficients):
    crosspred(cb_t, model=interface) equals R crosspred on the R model the interface holds (R name 'cb')."""
    from prediction import crosspred
    s = _iface_case()
    c, iface = s['c'], s['iface']
    _push(c, 't')
    r(f'{SP}_pIf <- local({{ cb <- {c.rn["t"]}; crosspred(cb, {SP}_ifit, at={SP}_at, cen={SP}_cen, cumul=TRUE) }})')
    ref = {f: rget(f'unname({SP}_pIf${f})') for f in PRED_FIELDS_CB}
    with _quiet():
        pp = crosspred(c.py['t'], model=iface, at=c.at['t'], cen=c.cen['t'], cumul=True)
    _assert_pred_equals(pp, ref, 'crosspred(cb_t, model=interface)')


def test_interface_second_crossbasis_crosspred_matches_R():
    """The interface holds cb_t (own block) and cb_o (in other_vars).  R: crosspred(ifo, model) = the cb_o block.
    The two bases have the same number of columns: crosspred(cb_o, model=interface) used to predict cb_o with the
    coefficients of cb_t (silently wrong numbers); the block of cb_o is now found among the columns of the R model."""
    from prediction import crosspred
    s = _iface_case()
    c, iface = s['c'], s['iface']
    _push(c, 'o')
    r(f'{SP}_pIo <- local({{ ifo <- {c.rn["o"]}; crosspred(ifo, {SP}_ifit, at={SP}_at, cen={SP}_cen, cumul=TRUE) }})')
    ref = {f: rget(f'unname({SP}_pIo${f})') for f in PRED_FIELDS_CB}
    with _quiet():
        pp = crosspred(c.py['o'], model=iface, at=c.at['o'], cen=c.cen['o'], cumul=True)
    _assert_pred_equals(pp, ref, 'crosspred(cb_o, model=interface)')


def test_interface_second_crossbasis_crossreduce_matches_R():
    from crossreduce import crossreduce
    s = _iface_case()
    c, iface = s['c'], s['iface']
    _push(c, 'o')
    r(f'{SP}_rIo <- local({{ ifo <- {c.rn["o"]}; '
      f'crossreduce(ifo, {SP}_ifit, type="overall", at={SP}_at, cen={SP}_cen) }})')
    ref = dict(coef=rget(f'unname({SP}_rIo$coefficients)'), vcov=rget(f'unname({SP}_rIo$vcov)'),
               fit=rget(f'unname({SP}_rIo$fit)'), se=rget(f'unname({SP}_rIo$se)'))
    with _quiet():
        red = crossreduce(c.py['o'], model=iface, type='overall', at=c.at['o'], cen=c.cen['o'])
    _assert_red_equals(red, ref, 'crossreduce(cb_o, model=interface)')


def test_interface_second_crossbasis_block_by_hand_matches_R():
    """PLAIN workaround: the cb_o block taken from the R model the interface holds (by name) and passed as coef=/vcov=
    equals R's crosspred(ifo, model)."""
    from prediction import crosspred
    s = _iface_case()
    c = s['c']
    r(f'{SP}_ifi <- grep("^ifo", names(coef({SP}_ifit)))')
    coef = rget(f'unname(coef({SP}_ifit)[{SP}_ifi])')
    vcov = rget(f'unname(vcov({SP}_ifit)[{SP}_ifi, {SP}_ifi])')
    _push(c, 'o')
    r(f'{SP}_pIo <- local({{ ifo <- {c.rn["o"]}; crosspred(ifo, {SP}_ifit, at={SP}_at, cen={SP}_cen, cumul=TRUE) }})')
    ref = {f: rget(f'unname({SP}_pIo${f})') for f in PRED_FIELDS_CB}
    with _quiet():
        pp = crosspred(c.py['o'], coef=coef, vcov=vcov, model_link='log', at=c.at['o'], cen=c.cen['o'], cumul=True)
    _assert_pred_equals(pp, ref, 'crosspred(cb_o, coef=block from interface R model)')


# ==============================================================================================================
# the name= route (R: the name of the basis object) and the wording of the errors
# ==============================================================================================================
@pytest.mark.parametrize('design', DESIGNS)
@pytest.mark.parametrize('tag', BASES)
def test_name_argument_selects_the_block_like_R(design, tag):
    """name= is the R object name that crosspred / crossreduce / attrdl / find_mmt grep for: with it every entry point
    equals R, also for a model without informative names (ndarray, patsy matrix terms), where the name matches
    no coefficient and the columns of the basis are found in the design matrix instead."""
    c = _case('six_two')
    res = _fit(c, design)
    name = c.rn[tag]
    _assert_pred_equals(py_crosspred(c, tag, model=res, name=name), r_crosspred(c, tag), f'crosspred(name=) {design}/{tag}')
    _assert_red_equals(py_crossreduce(c, tag, 'overall', None, model=res, name=name),
                       r_crossreduce(c, tag, 'overall', None), f'crossreduce(name=) {design}/{tag}')
    for mode in ATTR_MODES:
        _assert_attr_equals(py_attrdl(c, tag, mode, model=res, name=name), r_attrdl(c, tag, mode),
                            f'attrdl(name=) {mode} {design}/{tag}')
    assert py_mmt(c, tag, model=res, name=name) == r_mmt(c, tag)


def test_name_attribute_of_the_basis_is_the_default_name():
    """A basis carrying a `name` attribute uses it as the prefix; the name= argument overrides the attribute."""
    from prediction import crosspred
    c = _case('ten_ten')
    res = _fit(c, 'dataframe')
    named = copy.copy(c.py['o'])
    named.name = c.rn['o']
    with _quiet():
        pp = crosspred(named, model=res, at=c.at['o'], cen=c.cen['o'], cumul=True)
    _assert_pred_equals(pp, r_crosspred(c, 'o'), 'crosspred(basis.name)')
    named.name = c.rn['t']                                  # the attribute names the temperature block ...
    with _quiet():
        pp_t = crosspred(named, model=res, at=c.at['o'], cen=c.cen['o'])
        pp_o = crosspred(named, model=res, name=c.rn['o'], at=c.at['o'], cen=c.cen['o'])   # ... the argument wins
    assert np.array_equal(pp_t.coefficients, _block(c, 'dataframe', res, 't')[0])
    assert np.array_equal(pp_o.coefficients, _block(c, 'dataframe', res, 'o')[0])


def test_name_matching_the_wrong_number_of_coefficients_raises():
    """R's grep of a name that is a prefix of both bases selects 20 coefficients for a 10-column basis and stops."""
    c = _case('ten_ten')
    res = _fit(c, 'dataframe')
    with pytest.raises(ValueError, match=r'coef=.*vcov='):
        py_crosspred(c, 'o', model=res, name=f'{SP}_ten_ten_')
    with pytest.raises(ValueError, match=r'coef=.*vcov='):
        py_crossreduce(c, 'o', 'overall', None, model=res, name=f'{SP}_ten_ten_')


class _NamedFit:
    """A fitted model that exposes only params (named) and cov_params(): no design matrix to find a basis in."""

    def __init__(self, names, coef, vcov):
        import pandas as pd
        self.params = pd.Series(coef, index=names)
        self._vcov = vcov

    def cov_params(self):
        return self._vcov


def test_name_selects_the_block_of_a_model_without_design_matrix_like_R():
    """Without a design matrix the two bases can only be told apart by name (R's only route): name= selects each
    block and equals R; without name= the ValueError asks for name= or coef=/vcov=."""
    c = _case('ten_ten')
    res = _fit(c, 'dataframe')
    model = _NamedFit(c.names, np.asarray(res.params), np.asarray(res.cov_params()))
    for tag in BASES:
        pp = py_crosspred(c, tag, model=model, name=c.rn[tag], model_link='log')     # the stand-in has no family
        _assert_pred_equals(pp, r_crosspred(c, tag), f'name={tag}')
    with pytest.raises(ValueError, match=r'name=.*coef=.*vcov=|coef=.*vcov=.*name='):
        py_crosspred(c, 't', model=model)


def test_errors_of_the_model_route_are_short_and_not_repeated():
    """The messages used to embed the multi-line summary of the basis several times (7 lines each)."""
    c = _case('ten_ten')
    c_t = _case('ten_ten', terms=('t',))
    cases = [(py_crosspred, (c, 'o'), dict(model=_fit(c_t, 'dataframe'))),        # the basis is not in the model
             (py_crossreduce, (c, 'o', 'overall', None), dict(model=_fit(c_t, 'numpy'))),
             (py_crosspred, (c, 't'), dict(model=_NamedFit(c.names, np.asarray(_fit(c, 'dataframe').params),
                                                          np.asarray(_fit(c, 'dataframe').cov_params()))))]
    for call, args, kw in cases:
        with pytest.raises(ValueError) as exc:
            call(*args, **kw)
        msg = str(exc.value)
        assert '\n' not in msg and len(msg) < 350, msg
        assert 'coef=' in msg and 'vcov=' in msg and 'Lag range' not in msg, msg


@pytest.mark.parametrize('how', ['subset', 'shuffled', 'nan_rows_filled_with_zero'])
def test_block_is_found_when_the_model_rows_are_not_the_basis_rows(how):
    """The design of the model need not have the rows of the basis: observations dropped for other reasons, reordered
    rows, or the lag-induced NaN rows filled with zeros.  The block found is the one at the known position."""
    sm = pytest.importorskip('statsmodels.api')
    from prediction import crosspred
    c = _case('ten_ten')
    rng = np.random.default_rng(11)
    X, y = c.X.copy(), c.y.copy()
    if how == 'subset':
        keep = rng.uniform(size=len(y)) > 0.15
        X, y = X[keep], y[keep]
    elif how == 'shuffled':
        perm = rng.permutation(len(y))
        X, y = X[perm], y[perm]
    else:
        complete = np.isfinite(np.asarray(c.py['t'].basis)).all(axis=1)
        full = np.zeros((len(complete), X.shape[1]))
        full[complete] = X
        yfull = np.zeros(len(complete))
        yfull[complete] = y
        X, y = full, yfull
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = sm.GLM(y, X, family=sm.families.Poisson()).fit(scale='X2', **GLM_TOL)
    for tag in BASES:
        coef, vcov = _block(c, 'numpy', res, tag)
        with _quiet():
            pp = crosspred(c.py[tag], model=res, at=c.at[tag], cen=c.cen[tag])
        assert np.array_equal(pp.coefficients, coef), f'{how}/{tag}: wrong block'
        assert np.array_equal(pp.vcov, vcov)


def test_interface_name_argument_selects_the_second_crossbasis():
    """name= also selects a cross-basis that was passed to the interface among other_vars (R names it 'ifo')."""
    from prediction import crosspred
    s = _iface_case()
    c, iface = s['c'], s['iface']
    _push(c, 'o')
    r(f'{SP}_pIn <- local({{ ifo <- {c.rn["o"]}; crosspred(ifo, {SP}_ifit, at={SP}_at, cen={SP}_cen, cumul=TRUE) }})')
    ref = {f: rget(f'unname({SP}_pIn${f})') for f in PRED_FIELDS_CB}
    with _quiet():
        pp = crosspred(c.py['o'], model=iface, name='ifo', at=c.at['o'], cen=c.cen['o'], cumul=True)
    _assert_pred_equals(pp, ref, 'crosspred(cb_o, model=interface, name="ifo")')


def test_interface_rejects_a_basis_that_is_not_in_its_model():
    """A cross-basis that is neither the interface's own nor among its covariates is refused (it used to be given
    the block of the interface's own cross-basis when the sizes agreed)."""
    from basis import CrossBasis
    from prediction import crosspred
    s = _iface_case()
    c, iface = s['c'], s['iface']
    with _quiet():
        stranger = CrossBasis(c.x['o'][::-1].copy(), lag=6, argvar=dict(fun='bs', degree=2, df=5),
                              arglag=dict(fun='ns', df=2))
    assert stranger.shape[1] == c.py['t'].shape[1]
    with pytest.raises(ValueError, match=r'not in the model.*coef=.*vcov='):
        crosspred(stranger, model=iface, at=c.at['o'], cen=c.cen['o'])
