"""CrossBasis time-series path vs R crossbasis(): argvar / arglag `fun` and `intercept` must be honoured (theme B).

Defect (basis.py, CrossBasis._create_time_series_basis): CrossBasis builds the correct marginal bases with OneBasis
(self.basisvar / self.basislag) but then throws them away and re-derives both with hand-written R calls. The variable
side only knows `ns` and "everything else is bs" (df default 3, `intercept` dropped), the lag side only knows `ns`
and `integer`. R's crossbasis() multiplies the lagged columns of onebasis(x, argvar) by onebasis(seqlag(lag), arglag)
whatever `fun` was asked for. Consequences seen against R: var fun lin/poly/strata/thr/callable silently computed as
a cubic B-spline, var `intercept=True` ignored (IndexError or wrong values), argvar without `fun` computed as bs
(R: ns), lag fun bs/poly/lin/strata/thr silently ns or a broadcast ValueError, bare argvar/ns/bs crash on the df
default (OneBasis df=4 vs R df=NULL). Coefficients fitted on such a matrix belong to a different basis than the one
crosspred / attribution evaluate.

Findings covered
  basis-cont-2      var fun lin/poly/strata/thr/callable and var intercept ignored
  basis-discrete-1  same, plus argvar without fun, bare argvar, lag fun lin/poly/strata/thr/bs
  crossbasis-3      var and lag `fun` other than ns/bs and ns/integer ignored (grid of combinations)
  crossbasis-4      var intercept ignored, df defaults differ from R (bare argvar / ns / bs)
  attr-point-11     coefficients fitted on the CrossBasis matrix are for the wrong basis (fit and crosspred vs R)

Known-defect tests (strict xfail) assert the R-faithful behaviour; the plain tests guard the sub-space that is already
faithful (ns/bs var with knots or df, ns/integer lag, NaN handling, lag ranges, marginal OneBasis objects, crosspred
after CrossBasis with given coefficients) so that a fix which merely swaps in basisvar.basis / basislag.basis cannot
regress it (in particular it must keep storing the training Boundary_knots for ns, not only for bs).

Two fixes are needed for everything to pass: using basisvar.basis / basislag.basis in _create_time_series_basis
(var/lag `fun`, intercept, argvar without fun, lag funs, fit/crosspred) and R's df=NULL default for bare ns/bs in
basis_functions.py (test_var_bare_defaults_match_r, test_lag_bare_defaults_match_r). Strict xfail is per test case, so a
partial fix makes the cases it repairs XPASS until their markers are removed.

Findings that are NOT covered here although they show up in the same sweeps: StrataBasis with df and no breaks as
exposure basis (shape differs from R), default arglag (R: strata df=1), the matrix-of-lags input path and the
resolved argvar bookkeeping (CrossBasis.argvar does not record the training poly `scale`, so crosspred rebuilds poly
with the scale of the prediction grid; poly is therefore left out of the crosspred comparisons).

The reference is always computed by R at run time (crossbasis / onebasis / glm / crosspred); no numbers are hard-coded.
"""
import copy
import os

import numpy as np
import pytest

from rhelpers import assert_close, chicago, known_defect, np2r, r, rget

THEME = 'B'
RTOL = 1e-10          # cross-basis matrices agree with R to ~1e-15 whenever the right basis is built

_CH = chicago()
X = _CH['temp'][:400]                     # time series used for the matrix comparisons
KV = np.quantile(X, [.10, .75, .90])      # explicit interior knots, so no df->knots machinery is involved
Q25, Q50, Q75 = (float(v) for v in np.quantile(X, [.25, .50, .75]))
LAG = 5
LAG_KNOTS = np.array([1.0, 3.0])          # explicit lag-ns knots
LAG_NS = {'fun': 'ns', 'knots': LAG_KNOTS}
VAR_BS2 = {'fun': 'bs', 'degree': 2, 'knots': KV}     # the validated exposure basis (used while varying the lag basis)

_R_HOME = str(r('R.home()')[0])           # home of the R that rhelpers started (before any module can overwrite $R_HOME)


@pytest.fixture(autouse=True)
def _true_r_home(monkeypatch):
    """In-process R locates LAPACK through $R_HOME at first use; the unfixed CrossBasis time-series path (and, on
    import, improved_glm / rpy2_glm) overwrite it with a hard-coded R path, which crashes the later glm fits."""
    monkeypatch.setenv('R_HOME', _R_HOME)


# a user-supplied callable exposure basis; the R twin is a global R function of the same name
r('.cbB_shift_quad <- function(x, k = 2, ...) cbind((x - 15) / 10, ((x - 15) / 10)^k)')


def _shift_quad(x, k=2, **kwargs):
    z = (np.asarray(x, dtype=float) - 15.0) / 10.0
    return np.column_stack([z, z ** k])


_R_FUN_NAME = {_shift_quad: '.cbB_shift_quad'}


# ---------------------------------------------------------------------------------------------- helpers
def _rlit(v):
    """Python value -> R literal (callables map to the name of their R twin)."""
    if callable(v):
        return f'"{_R_FUN_NAME[v]}"'
    if isinstance(v, (bool, np.bool_)):
        return 'TRUE' if v else 'FALSE'
    if isinstance(v, str):
        return f'"{v}"'
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    a = np.atleast_1d(np.asarray(v, dtype=float))
    if a.size == 1 and np.ndim(v) == 0:
        return repr(float(a[0]))
    return 'c(' + ', '.join(repr(float(t)) for t in a) + ')'


def _rlist(spec):
    """Python argvar/arglag dict -> R list(...): key_with_underscores -> key.with.dots (thr_value, Boundary_knots)."""
    return 'list(' + ', '.join(f'{k.replace("_", ".")}={_rlit(v)}' for k, v in spec.items()) + ')'


def r_crossbasis(x, lag, argvar, arglag):
    np2r('cbB_x', x)
    r(f'cbB_cb <- crossbasis(cbB_x, lag={_rlit(lag)}, argvar={_rlist(argvar)}, arglag={_rlist(arglag)})')
    return rget('unclass(cbB_cb)')


def py_crossbasis(x, lag, argvar, arglag):
    """PyDLNM CrossBasis object; $R_HOME is restored afterwards because the unfixed time-series path overwrites it
    (see _true_r_home)."""
    from basis import CrossBasis
    rhome = os.environ.get('R_HOME')
    try:
        return CrossBasis(x, lag=lag, argvar=copy.deepcopy(argvar), arglag=copy.deepcopy(arglag))
    finally:
        if rhome is not None:
            os.environ['R_HOME'] = rhome


def check_crossbasis(x, lag, argvar, arglag, rtol=RTOL):
    """Python cross-basis matrix must equal R's (shape, NaN rows, values); a crash in Python is a failure."""
    cb_r = r_crossbasis(x, lag, argvar, arglag)
    cb_p = np.asarray(py_crossbasis(x, lag, argvar, arglag).basis)
    assert_close(cb_p, cb_r, rtol=rtol, what=f'crossbasis argvar={argvar} arglag={arglag}')
    return cb_p


def _with_lag_intercept(arglag):
    """R crossbasis() adds intercept=TRUE to arglag when the lag function has that argument and none was given."""
    out = dict(arglag)
    if out.get('fun', 'ns') != 'integer':
        out.setdefault('intercept', True)
    return out


# ---------------------------------------------------------------------------------------------- case tables
# variable dimension: `fun` other than ns/bs, with and without intercept (lag basis fixed to ns with explicit knots)
VAR_FUN = [
    ('lin', {'fun': 'lin'}),
    ('lin-intercept', {'fun': 'lin', 'intercept': True}),
    ('poly2', {'fun': 'poly', 'degree': 2}),
    ('poly3-scale30', {'fun': 'poly', 'degree': 3, 'scale': 30.0}),
    ('poly2-intercept', {'fun': 'poly', 'degree': 2, 'intercept': True}),
    ('strata-breaks', {'fun': 'strata', 'breaks': np.array([Q25, Q75])}),
    ('strata-breaks-intercept', {'fun': 'strata', 'breaks': np.array([Q25, Q75]), 'intercept': True}),
    ('thr-h', {'fun': 'thr', 'thr_value': Q50, 'side': 'h'}),
    ('thr-l', {'fun': 'thr', 'thr_value': Q50, 'side': 'l'}),
    ('thr-d', {'fun': 'thr', 'thr_value': np.array([Q25, Q75]), 'side': 'd'}),
    ('thr-h-intercept', {'fun': 'thr', 'thr_value': Q50, 'side': 'h', 'intercept': True}),
]

# splines with intercept=True: R's onebasis gives one more column, the hand-written call did not
VAR_SPLINE_INTERCEPT = [
    ('ns-knots', {'fun': 'ns', 'knots': KV, 'intercept': True}),
    ('ns-df3', {'fun': 'ns', 'df': 3, 'intercept': True}),
    ('ns-df5', {'fun': 'ns', 'df': 5, 'intercept': True}),
    ('bs-knots-deg3', {'fun': 'bs', 'knots': KV, 'intercept': True}),
    ('bs-knots-deg2', {'fun': 'bs', 'degree': 2, 'knots': KV, 'intercept': True}),
    ('bs-df5', {'fun': 'bs', 'df': 5, 'intercept': True}),
]

# argvar without `fun`: R's onebasis default is ns (the time-series path defaulted to bs)
VAR_NO_FUN = [
    ('df4', {'df': 4}),
    ('df5', {'df': 5}),
    ('knots', {'knots': KV}),
    ('knots-intercept', {'knots': KV, 'intercept': True}),
]

# bare argvar: R has no df default (ns(x) is one column, bs(x) is degree columns), PyDLNM used df=4
VAR_BARE = [
    ('empty', {}),
    ('ns', {'fun': 'ns'}),
    ('bs', {'fun': 'bs'}),
    ('bs-deg2', {'fun': 'bs', 'degree': 2}),
]

# lag dimension (var = validated bs deg2 knots): fun other than ns/integer, intercept default TRUE as in R
LAG_FUN = [
    ('bs-df5', {'fun': 'bs', 'df': 5}),
    ('bs-df4-deg2', {'fun': 'bs', 'df': 4, 'degree': 2}),
    ('bs-df6', {'fun': 'bs', 'df': 6}),
    ('bs-bare', {'fun': 'bs'}),                     # bare bs with the lag intercept: no interior knots, as in R
    ('bs-knots', {'fun': 'bs', 'knots': LAG_KNOTS}),
    ('bs-knots-deg1', {'fun': 'bs', 'knots': LAG_KNOTS, 'degree': 1}),
    ('bs-knots-deg2-nointercept', {'fun': 'bs', 'knots': LAG_KNOTS, 'degree': 2, 'intercept': False}),
    ('poly2', {'fun': 'poly', 'degree': 2}),
    ('poly3', {'fun': 'poly', 'degree': 3}),
    ('poly2-nointercept', {'fun': 'poly', 'degree': 2, 'intercept': False}),
    ('lin', {'fun': 'lin'}),
    ('lin-nointercept', {'fun': 'lin', 'intercept': False}),
    ('poly1', {'fun': 'poly', 'degree': 1}),
    ('strata-breaks', {'fun': 'strata', 'breaks': np.array([2.0, 4.0])}),
    ('strata-breaks-nointercept', {'fun': 'strata', 'breaks': np.array([2.0, 4.0]), 'intercept': False}),
    ('strata-df2', {'fun': 'strata', 'df': 2}),
    ('strata-df3', {'fun': 'strata', 'df': 3}),
    ('thr-h', {'fun': 'thr', 'thr_value': 2.0, 'side': 'h'}),
    ('thr-d', {'fun': 'thr', 'thr_value': np.array([1.0, 3.0]), 'side': 'd'}),
]

# bare lag basis: R's ns(x, intercept=TRUE) has 2 columns, bs(x, degree=2, intercept=TRUE) has 3 (df=NULL, no interior knots)
LAG_BARE = [
    ('ns', {'fun': 'ns'}),
    ('bs-deg2', {'fun': 'bs', 'degree': 2}),
]

# grid var x lag; (ns|bs with knots) x (ns|integer) is the sub-space that is already faithful today
GRID_VAR = {
    'lin': {'fun': 'lin'},
    'poly2': {'fun': 'poly', 'degree': 2},
    'thr': {'fun': 'thr', 'thr_value': Q50, 'side': 'h'},
    'strata': {'fun': 'strata', 'breaks': np.array([Q25, Q75])},
    'ns': {'fun': 'ns', 'knots': KV},
    'bs': {'fun': 'bs', 'degree': 2, 'knots': KV},
}
GRID_LAG = {
    'ns': LAG_NS,
    'integer': {'fun': 'integer'},
    'bs': {'fun': 'bs', 'df': 5},
    'poly3': {'fun': 'poly', 'degree': 3},
    'strata': {'fun': 'strata', 'breaks': np.array([2.0, 4.0])},
}


def _defect_marks(*finding_ids, note=''):
    """known_defect(...) for use in pytest.param(marks=...); with PYDLNM_XFAIL_OFF=1 the harness returns a no-op
    usefixtures mark, which pytest.param refuses, so no mark is added then."""
    return [] if os.environ.get('PYDLNM_XFAIL_OFF') else [known_defect(THEME, *finding_ids, note=note)]


def _cases(table, defect_ids, note=''):
    """pytest params (id, spec) all marked as known defects of theme B."""
    return [pytest.param(name, spec, id=name, marks=_defect_marks(*defect_ids, note=note)) for name, spec in table]


# ---------------------------------------------------------------------------------------------- known defects
@pytest.mark.parametrize('name,argvar', _cases(VAR_FUN, ('basis-cont-2', 'basis-discrete-1', 'crossbasis-3',
                                                         'attr-point-11'), 'var fun computed as bs'))
def test_var_fun_lin_poly_strata_thr_matches_r(name, argvar):
    check_crossbasis(X, LAG, argvar, LAG_NS)


@known_defect(THEME, 'basis-cont-2', 'crossbasis-3', note='callable var fun computed as bs')
@pytest.mark.parametrize('k', [2, 3])
def test_var_callable_fun_matches_r(k):
    # R can only name a function; the Python twin is passed as a callable with the same extra argument
    check_crossbasis(X, LAG, {'fun': _shift_quad, 'k': k}, LAG_NS)


@pytest.mark.parametrize('name,argvar', _cases(VAR_SPLINE_INTERCEPT, ('basis-cont-2', 'basis-discrete-1',
                                                                      'crossbasis-4'), 'var intercept ignored'))
def test_var_spline_intercept_true_matches_r(name, argvar):
    check_crossbasis(X, LAG, argvar, LAG_NS)


@pytest.mark.parametrize('name,argvar', _cases(VAR_NO_FUN, ('basis-discrete-1', 'crossbasis-3', 'crossbasis-4'),
                                               'no fun: R default is ns, time-series path used bs'))
def test_var_without_fun_defaults_to_ns_like_r(name, argvar):
    check_crossbasis(X, LAG, argvar, LAG_NS)


@pytest.mark.parametrize('name,argvar', _cases(VAR_BARE, ('crossbasis-4', 'basis-discrete-1'),
                                               'bare ns/bs: R df=NULL, PyDLNM df=4'))
def test_var_bare_defaults_match_r(name, argvar):
    check_crossbasis(X, LAG, argvar, LAG_NS)


@pytest.mark.parametrize('name,arglag', _cases(LAG_FUN, ('basis-discrete-1', 'crossbasis-3', 'attr-point-11',
                                                         'basis-cont-2'), 'lag fun computed as ns / crash'))
def test_lag_fun_matches_r(name, arglag):
    check_crossbasis(X, LAG, VAR_BS2, arglag)


@pytest.mark.parametrize('name,arglag', _cases(LAG_BARE, ('crossbasis-4', 'basis-discrete-1'),
                                               'bare lag ns/bs: R df=NULL, PyDLNM df=4'))
def test_lag_bare_defaults_match_r(name, arglag):
    check_crossbasis(X, LAG, VAR_BS2, arglag)


def _grid():
    params = []
    for vn, vs in GRID_VAR.items():
        for ln, ls in GRID_LAG.items():
            faithful = vn in ('ns', 'bs') and ln in ('ns', 'integer')
            marks = [] if faithful else _defect_marks('attr-point-11', 'crossbasis-3', 'basis-discrete-1',
                                                      note='var or lag fun other than ns/bs, ns/integer')
            params.append(pytest.param(vs, ls, id=f'var-{vn}_lag-{ln}', marks=marks))
    return params


@pytest.mark.parametrize('argvar,arglag', _grid())
def test_var_by_lag_grid_matches_r(argvar, arglag):
    check_crossbasis(X, LAG, argvar, arglag)


# --- attr-point-11: what goes wrong downstream. The GLM is fitted on CrossBasis.basis, crosspred/attribution evaluate
# --- OneBasis(argvar): the fitted coefficients must be those of R's fit on R's cross-basis.
_DEATH = _CH['death'][:1500]
_TEMP = _CH['temp'][:1500]
FIT_CASES = [
    ('lin', {'fun': 'lin'}, LAG_NS),
    ('poly2', {'fun': 'poly', 'degree': 2}, LAG_NS),
    ('thr-h', {'fun': 'thr', 'thr_value': float(np.median(_TEMP)), 'side': 'h'}, LAG_NS),
    ('strata', {'fun': 'strata', 'breaks': np.quantile(_TEMP, [.25, .75])}, LAG_NS),
    ('bs2-x-poly3lag', {'fun': 'bs', 'degree': 2, 'knots': np.quantile(_TEMP, [.10, .75, .90])},
     {'fun': 'poly', 'degree': 3}),
]
# crosspred of the fitted model is compared only where crosspred itself is already faithful for the given basis
# (poly: argvar lacks the training scale, a different finding)
FIT_PRED_CASES = [c for c in FIT_CASES if c[0] != 'poly2']


def _fit_both(argvar, arglag):
    """Poisson glm in R on R's crossbasis (model cbB_mr) and on the PyDLNM CrossBasis matrix (cbB_mp).
    Returns the CrossBasis object and the cross-basis coefficients / vcov of the Python-matrix fit."""
    np2r('cbB_death', _DEATH)
    r('cbB_tt <- seq_along(cbB_death)')
    r_crossbasis(_TEMP, LAG, argvar, arglag)                 # leaves the R crossbasis object in R as cbB_cb
    r('cbB_cbr <- cbB_cb')
    cb_py = py_crossbasis(_TEMP, LAG, argvar, arglag)
    np2r('cbB_cbp', np.asarray(cb_py.basis))
    r('cbB_mr <- glm(cbB_death ~ cbB_cbr + cbB_tt, family = poisson, na.action = na.exclude)')
    r('cbB_mp <- glm(cbB_death ~ cbB_cbp + cbB_tt, family = poisson, na.action = na.exclude)')
    r('cbB_ir <- grep("^cbB_cbr", names(coef(cbB_mr))); cbB_ip <- grep("^cbB_cbp", names(coef(cbB_mp)))')
    return (cb_py, rget('unname(coef(cbB_mr)[cbB_ir])'), rget('unname(coef(cbB_mp)[cbB_ip])'),
            rget('unname(vcov(cbB_mp)[cbB_ip, cbB_ip])'))


@known_defect(THEME, 'attr-point-11', note='GLM coefficients fitted on the wrong cross-basis')
@pytest.mark.parametrize('name,argvar,arglag', FIT_CASES, ids=[c[0] for c in FIT_CASES])
def test_glm_coefficients_on_python_crossbasis_match_r(name, argvar, arglag):
    _, coef_r, coef_p, _ = _fit_both(argvar, arglag)
    assert_close(coef_p, coef_r, rtol=1e-8, what=f'glm coefficients ({name}) fitted on Python vs R cross-basis')


@known_defect(THEME, 'attr-point-11', note='crosspred from a model fitted on the wrong cross-basis')
@pytest.mark.parametrize('name,argvar,arglag', FIT_PRED_CASES, ids=[c[0] for c in FIT_PRED_CASES])
def test_crosspred_of_model_fitted_on_python_crossbasis_matches_r(name, argvar, arglag):
    from prediction import crosspred
    cb_py, _, coef_p, vcov_p = _fit_both(argvar, arglag)
    cen = float(np.median(_TEMP))
    at = np.round(np.linspace(*np.quantile(_TEMP, [.05, .95]), 7), 4)
    np2r('cbB_at', at)
    r(f'cbB_cpr <- crosspred(cbB_cbr, coef=coef(cbB_mr)[cbB_ir], vcov=vcov(cbB_mr)[cbB_ir, cbB_ir], '
      f'model.link="log", at=cbB_at, cen={cen!r}, bylag=1)')
    cp = crosspred(cb_py, coef=coef_p, vcov=vcov_p, model_link='log', at=at, cen=cen, bylag=1.0)
    for attr in ('allfit', 'matfit', 'allse', 'matse'):
        assert_close(np.asarray(getattr(cp, attr)), rget(f'cbB_cpr${attr}'), rtol=1e-8, what=f'crosspred {attr} ({name})')


# ---------------------------------------------------------------------------------------------- already faithful (plain)
FAITHFUL_VAR = {
    'ns-knots': {'fun': 'ns', 'knots': KV},
    'ns-knots-nointercept': {'fun': 'ns', 'knots': KV, 'intercept': False},
    'ns-df4': {'fun': 'ns', 'df': 4},
    'ns-df6': {'fun': 'ns', 'df': 6},
    'ns-knots-boundary': {'fun': 'ns', 'knots': KV, 'Boundary_knots': np.array([-30.0, 40.0])},
    'bs-knots-deg2': {'fun': 'bs', 'degree': 2, 'knots': KV},
    'bs-knots-deg3': {'fun': 'bs', 'knots': KV},
    'bs-df5': {'fun': 'bs', 'df': 5},
    'bs-df6-deg2': {'fun': 'bs', 'df': 6, 'degree': 2},
    'bs-knots-boundary': {'fun': 'bs', 'degree': 2, 'knots': KV, 'Boundary_knots': np.array([-30.0, 40.0])},
}
FAITHFUL_LAG = {
    'ns-knots': LAG_NS,
    'ns-knots-nointercept': {'fun': 'ns', 'knots': LAG_KNOTS, 'intercept': False},
    'ns-df3': {'fun': 'ns', 'df': 3},
    'integer': {'fun': 'integer'},
}


@pytest.mark.parametrize('lag_name', list(FAITHFUL_LAG))
@pytest.mark.parametrize('var_name', list(FAITHFUL_VAR))
def test_ns_bs_var_by_ns_integer_lag_matches_r(var_name, lag_name):
    check_crossbasis(X, LAG, FAITHFUL_VAR[var_name], FAITHFUL_LAG[lag_name])


_X_NAN = X.copy()
_X_NAN[[20, 150, 151, 300]] = np.nan


@pytest.mark.parametrize('lag,arglag', [
    (LAG, LAG_NS), (8, {'fun': 'ns', 'df': 4}), ([2, 6], LAG_NS), ([1, 4], {'fun': 'ns', 'knots': np.array([2.0])}),
    ([0, 4], {'fun': 'integer'}), ([2, 5], {'fun': 'integer'})],
    ids=['lag5-ns', 'lag8-nsdf4', 'lag2to6-ns', 'lag1to4-ns', 'lag0to4-integer', 'lag2to5-integer'])
@pytest.mark.parametrize('var_name', ['ns-knots', 'bs-knots-deg2'])
@pytest.mark.parametrize('with_nan', [False, True], ids=['clean', 'nan'])
def test_lag_ranges_and_missing_exposure_match_r(with_nan, var_name, lag, arglag):
    # NaN exposure: every row whose lag window touches a NaN, and the first max(lag) rows, must be NaN as in R
    cb = check_crossbasis(_X_NAN if with_nan else X, lag, FAITHFUL_VAR[var_name], arglag)
    assert np.isnan(cb).any(axis=1).sum() >= max(np.atleast_1d(lag))


@pytest.mark.parametrize('name,argvar', VAR_FUN + VAR_SPLINE_INTERCEPT, ids=[n for n, _ in VAR_FUN + VAR_SPLINE_INTERCEPT])
def test_marginal_var_basis_matches_r_onebasis(name, argvar):
    """The exposure basis CrossBasis builds first (OneBasis, kept as basisvar) is already R's onebasis; the tensor
    step only has to use it."""
    from basis import OneBasis
    np2r('cbB_x', X)
    r(f'cbB_ob <- do.call(onebasis, c(list(x=cbB_x), {_rlist(argvar)}))')
    assert_close(np.asarray(OneBasis(X, **copy.deepcopy(argvar)).basis), rget('unclass(cbB_ob)'), rtol=RTOL,
                 what=f'exposure basis {name}')


@pytest.mark.parametrize('name,arglag', LAG_FUN + [(k, v) for k, v in FAITHFUL_LAG.items() if v['fun'] != 'integer'],
                         ids=[n for n, _ in LAG_FUN] + ['faithful-' + k for k, v in FAITHFUL_LAG.items()
                                                       if v['fun'] != 'integer'])
def test_marginal_lag_basis_matches_r_onebasis(name, arglag):
    from basis import OneBasis
    full = _with_lag_intercept(arglag)                 # R crossbasis() defaults intercept=TRUE on the lag basis
    np2r('cbB_lagseq', np.arange(LAG + 1.0))
    r(f'cbB_lb <- do.call(onebasis, c(list(x=cbB_lagseq), {_rlist(full)}))')
    assert_close(np.asarray(OneBasis(np.arange(LAG + 1.0), **copy.deepcopy(full)).basis), rget('unclass(cbB_lb)'),
                 rtol=RTOL, what=f'lag basis {name}')


def test_marginal_integer_lag_basis_is_identity():
    cb = py_crossbasis(X, LAG, VAR_BS2, {'fun': 'integer'})
    assert_close(np.asarray(cb.basislag.basis), np.eye(LAG + 1), rtol=0, what='integer lag basis')


@pytest.mark.parametrize('name,argvar', VAR_FUN, ids=[n for n, _ in VAR_FUN])
def test_crossbasis_df_attribute_matches_r(name, argvar):
    """CrossBasis.df (columns of the exposure / lag bases) already follows onebasis, also where the matrix values do not."""
    cb = py_crossbasis(X, LAG, argvar, LAG_NS)
    r_crossbasis(X, LAG, argvar, LAG_NS)
    assert tuple(int(v) for v in cb.df) == tuple(int(v) for v in rget('attr(cbB_cb, "df")'))
    assert cb.basis.shape == (len(X), cb.df[0] * cb.df[1])


@pytest.mark.parametrize('argvar', [
    {'fun': 'ns', 'knots': KV}, {'fun': 'bs', 'degree': 2, 'knots': KV}, {'fun': 'ns', 'knots': KV, 'intercept': False},
    {'fun': 'lin'}, {'fun': 'thr', 'thr_value': Q50, 'side': 'h'}, {'fun': 'strata', 'breaks': np.array([Q25, Q75])}],
    ids=['ns-knots', 'bs-knots-deg2', 'ns-knots-nointercept', 'lin', 'thr-h', 'strata-breaks'])
def test_crosspred_after_crossbasis_with_given_coefficients_matches_r(argvar):
    """crosspred rebuilds the exposure basis from CrossBasis.argvar, on a grid that does not span the training range.
    A fix that swaps in basisvar.basis must keep storing the training Boundary_knots for ns (not only bs), and must
    not store spurious ones for other funs. Coefficients are synthetic, so this does not depend on the matrix values."""
    from prediction import crosspred
    x = _TEMP
    at = np.round(np.linspace(*np.quantile(x, [.10, .90]), 6), 4)
    cen = float(np.median(x))
    np2r('cbB_at', at)
    cb_r = r_crossbasis(x, LAG, argvar, LAG_NS)
    n_coef = cb_r.shape[1]
    coef = np.linspace(0.02, -0.03, n_coef)
    vcov = np.diag(np.linspace(1.0, 2.0, n_coef)) * 1e-4
    np2r('cbB_coef', coef)
    np2r('cbB_vcov', vcov)
    r(f'cbB_cpr <- crosspred(cbB_cb, coef=cbB_coef, vcov=matrix(cbB_vcov, {n_coef}), model.link="log", at=cbB_at, '
      f'cen={cen!r}, bylag=1)')
    cp = crosspred(py_crossbasis(x, LAG, argvar, LAG_NS), coef=coef, vcov=vcov, model_link='log', at=at, cen=cen,
                   bylag=1.0)
    for attr in ('allfit', 'matfit', 'allse'):
        assert_close(np.asarray(getattr(cp, attr)), rget(f'cbB_cpr${attr}'), rtol=1e-8, what=f'crosspred {attr}')


@pytest.mark.parametrize('where', ['argvar', 'arglag'])
def test_unknown_fun_raises_like_r(where):
    """R stops on a non-existent basis function; PyDLNM must not silently substitute another one."""
    bad = {'fun': 'no_such_basis_fun'}
    argvar, arglag = (bad, LAG_NS) if where == 'argvar' else (VAR_BS2, dict(bad))
    with pytest.raises(Exception):
        r_crossbasis(X, LAG, argvar, arglag)
    with pytest.raises(Exception):
        py_crossbasis(X, LAG, argvar, arglag)
