"""CrossBasis / OneBasis front end: caller dicts (D), matrix / group input (E), swallowed keywords (F).

Every test computes the reference in R (dlnm 2.4.10, tsModel) at run time and compares it with PyDLNM on identical
inputs; nothing is copied from Python output.  Tests marked ``known_defect`` assert the R-faithful behaviour that
PyDLNM does not have yet (strict xfail: they must fail today and pass once the defect is fixed, at which point the
marker has to be removed in the fix commit).  Unmarked tests guard neighbouring behaviour that is already faithful.

Themes and findings (audit_handoff_2026-09-29/findings/root_cause_themes.md)

  D  CrossBasis writes into the caller's argvar/arglag dicts; reusing one dict for a second dataset silently uses the
     first dataset's Boundary_knots
        basis-discrete-12, crossbasis-8
  E  lag-matrix (exposure history) x crashes / is silently wrong, group= is ignored, OneBasis flattens 2-D x row-major
        basis-discrete-11, crossbasis-5   matrix x (CrossBasis)
        crossbasis-6                      group= (R: lags computed inside each group, NaN rows per group, checkgroup)
        basis-discrete-17                 OneBasis with 2-D x (R: as.vector, column-major)
  F  unknown / R-spelled keywords (Boundary.knots, thr.value, bound, type, typos) are swallowed by **kwargs
        basis-cont-4, basis-discrete-8, crossbasis-15

Contract chosen for F.  In R an R-style spelling is honoured (or renamed with a warning) and an unknown argument is an
error ("unused argument").  A Python port may honour the R spelling or reject it loudly (TypeError / ValueError);
what it must not do is return numbers that differ from R without complaint.  The F tests therefore accept either
outcome for R spellings and require an exception for arguments that R rejects.

Not covered here (other root causes): CrossBasis ignoring the ``fun`` of argvar/arglag in its time-series path (so an
arglag 'Boundary.knots' is still dropped after a keyword fix), the default arglag, negative lags, and crosspred
bookkeeping for df-based bases.
"""
import copy
import warnings

import numpy as np
import pytest

from rhelpers import assert_close, chicago, known_defect, np2r, r, rget

from basis import CrossBasis, OneBasis          # noqa: E402  (after rhelpers: R must start first)
from prediction import crosspred                # noqa: E402
from utils import logknots                      # noqa: E402

TOL = 1e-10     # PyDLNM and R run the same splines / BLAS code; agreement is ~1e-16 when the algorithm is right


# --------------------------------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------------------------------
def push(**arrays):
    """Push numpy arrays into R's global environment under the given names."""
    for name, arr in arrays.items():
        np2r(name, arr)


def r_lag_arg(lag):
    """R 'lag=...' argument text; empty when lag is None (R then uses c(0, NCOL(x)-1))."""
    if lag is None:
        return ''
    if np.ndim(lag) == 0:
        return f'lag={int(lag)}, '
    return 'lag=c(%s), ' % ', '.join(str(int(v)) for v in lag)


def r_crossbasis(x, argvar, arglag, lag=None, group=None):
    """dlnm::crossbasis in R.  argvar / arglag are R expressions (strings).
    Returns the matrix (NaN for NA) and the attributes df, lag and group."""
    push(xin=x)
    grp = ''
    if group is not None:
        push(grp=group)
        grp = ', group=grp'
    r(f'cb_ref <- suppressWarnings(crossbasis(xin, {r_lag_arg(lag)}argvar={argvar}, arglag={arglag}{grp}))')
    attrs = {'df': rget('attr(cb_ref, "df")'), 'lag': rget('attr(cb_ref, "lag")'),
             'group': rget('attr(cb_ref, "group")') if group is not None else None}
    return rget('unclass(cb_ref)'), attrs


def r_onebasis(x, fun, rargs=''):
    """dlnm::onebasis in R (rargs = extra arguments as R text) as a plain matrix."""
    push(xin=x)
    sep = ', ' if rargs else ''
    r(f'ob_ref <- suppressWarnings(onebasis(xin, fun="{fun}"{sep}{rargs}))')
    return rget('matrix(as.numeric(ob_ref), nrow=nrow(ob_ref))')


def r_raises(code):
    """True if the R code raises an error (R is the reference for 'must be rejected')."""
    try:
        r(code)
    except Exception:            # rpy2 RRuntimeError
        return True
    return False


def dict_differences(after, before):
    """Sorted keys whose presence or value differs between two argument dicts (arrays compared by value)."""
    bad = list(set(after) ^ set(before))
    for k in set(after) & set(before):
        a, b = after[k], before[k]
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            if not (np.shape(a) == np.shape(b) and np.array_equal(a, b)):
                bad.append(k)
        elif a != b:
            bad.append(k)
    return sorted(bad)


def names_key(exc, key):
    """The error message names the offending argument (as written, or with '.' spelled '_')."""
    msg = str(exc)
    return key in msg or key.replace('.', '_') in msg


def honoured_or_rejected(py_fn, ref, what, key, rtol=TOL):
    """R honours the argument `key`.  Python must either reject it loudly (TypeError/ValueError naming the argument)
    or reproduce R; returning different numbers without complaint is the defect."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out = np.asarray(py_fn(), dtype=float)
    except (TypeError, ValueError) as exc:
        assert names_key(exc, key), f'{what}: rejected, but the message does not name {key!r}: {exc}'
        return
    assert_close(out, ref, rtol=rtol, what=what)


def must_be_rejected(py_fn, r_code, what, key):
    """R rejects the call ('unused argument'); Python must raise too, naming the offending argument `key`."""
    assert r_raises(r_code), f'{what}: expected R itself to reject this call'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with pytest.raises((TypeError, ValueError)) as info:
            py_fn()
    assert names_key(info.value, key), f'{what}: the error does not name the unused argument {key!r}: {info.value}'


def nan_rows(a):
    return np.flatnonzero(np.isnan(np.asarray(a, dtype=float)).any(axis=1))


@pytest.fixture(scope='module')
def temp():
    return chicago()['temp']


@pytest.fixture(autouse=True)
def leave_r_globals_as_found():
    """The R session is shared by all test modules: put its global environment back after every test here."""
    chicago()                        # loads chicagoNMMAPS into R (if not yet) *before* the snapshot, so it is kept
    r('.dEF_saved <- as.list(globalenv(), all.names=TRUE)')
    yield
    r('rm(list=setdiff(ls(globalenv(), all.names=TRUE), ".dEF_saved"), envir=globalenv());'
      'invisible(list2env(.dEF_saved, envir=globalenv())); rm(.dEF_saved, envir=globalenv())')


# --------------------------------------------------------------------------------------------------------------------
# Theme D: CrossBasis must not write into the caller's argvar / arglag dicts
# --------------------------------------------------------------------------------------------------------------------
D_CASES = [
    pytest.param({'fun': 'bs', 'df': 5, 'degree': 2}, 'list(fun="bs", df=5, degree=2)', id='bs-df5-deg2'),
    pytest.param({'fun': 'ns', 'df': 4}, 'list(fun="ns", df=4)', id='ns-df4'),
    pytest.param({'fun': 'bs', 'degree': 2, 'knots': np.array([10., 15., 20.])},
                 'list(fun="bs", degree=2, knots=c(10, 15, 20))', id='bs-knots'),
    pytest.param({'fun': 'ns', 'knots': np.array([9., 12., 18.])},
                 'list(fun="ns", knots=c(9, 12, 18))', id='ns-knots'),
]
ARGLAG_D = {'fun': 'ns', 'df': 3}
ARGLAG_D_R = 'list(fun="ns", df=3)'


def two_datasets(temp):
    """Two 'cities' with clearly different exposure ranges (A: -16.7..30.6, B: 7.2..39.2)."""
    return temp[:300].copy(), temp[300:600] * 0.6 + 20.0


@pytest.mark.parametrize('argvar, argvar_r', D_CASES)
def test_caller_dicts_are_not_mutated(temp, argvar, argvar_r):
    """R lists are copy-on-modify: crossbasis() never changes the argvar/arglag it was given."""
    argvar, arglag = copy.deepcopy(argvar), copy.deepcopy(ARGLAG_D)
    argvar0, arglag0 = copy.deepcopy(argvar), copy.deepcopy(arglag)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        CrossBasis(two_datasets(temp)[0], lag=3, argvar=argvar, arglag=arglag)
    changed_var, changed_lag = dict_differences(argvar, argvar0), dict_differences(arglag, arglag0)
    assert not (changed_var or changed_lag), \
        f'CrossBasis modified the caller dicts: argvar keys {changed_var}, arglag keys {changed_lag}'


@pytest.mark.parametrize('argvar, argvar_r', D_CASES)
def test_dicts_reused_for_second_dataset_match_r(temp, argvar, argvar_r):
    """Multi-city loop with the dicts built once: dataset B must get its own boundary knots, as in R."""
    a, b = two_datasets(temp)
    argvar, arglag = copy.deepcopy(argvar), copy.deepcopy(ARGLAG_D)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        CrossBasis(a, lag=3, argvar=argvar, arglag=arglag)                 # dataset A, dicts now (mis)used
        cb_b = CrossBasis(b, lag=3, argvar=argvar, arglag=arglag)          # dataset B, same dict objects
    ref, _ = r_crossbasis(b, argvar_r, ARGLAG_D_R, lag=3)
    assert_close(cb_b.basis, ref, rtol=TOL, what='CrossBasis(B) with dicts reused after A')


@pytest.mark.parametrize('argvar, argvar_r', D_CASES)
def test_fresh_dicts_match_r_on_both_datasets(temp, argvar, argvar_r):
    """Control for the reuse test: with a fresh dict per call both datasets agree with R (faithful today)."""
    for name, x in zip('AB', two_datasets(temp)):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            cb = CrossBasis(x, lag=3, argvar=copy.deepcopy(argvar), arglag=copy.deepcopy(ARGLAG_D))
        ref, attrs = r_crossbasis(x, argvar_r, ARGLAG_D_R, lag=3)
        assert_close(cb.basis, ref, rtol=TOL, what=f'CrossBasis(dataset {name})')
        assert tuple(cb.df) == tuple(int(v) for v in attrs['df']), f'df attribute, dataset {name}'


@pytest.mark.parametrize('fun', ['bs', 'ns'])
def test_caller_arrays_not_changed_in_place_and_call_is_repeatable(temp, fun):
    """A fix for D (copying the dicts) must not break this: knots arrays are untouched and the same dict on the
    same data gives the same cross-basis twice."""
    x = temp[:300]
    knots = np.array([5., 15., 22.])
    argvar = {'fun': fun, 'knots': knots.copy()}
    if fun == 'bs':
        argvar['degree'] = 2
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        first = CrossBasis(x, lag=4, argvar=argvar, arglag={'fun': 'ns', 'df': 3}).basis.copy()
        second = CrossBasis(x, lag=4, argvar=argvar, arglag={'fun': 'ns', 'df': 3}).basis
    assert np.array_equal(argvar['knots'], knots)
    assert_close(second, first, rtol=0, what='second call with the same dict')


@pytest.mark.parametrize('fun, kw', [('bs', {'degree': 2}), ('ns', {})], ids=['bs', 'ns'])
def test_crosspred_outside_training_range_uses_training_boundary_knots(temp, fun, kw):
    """CrossBasis records the training Boundary_knots (in its own argvar) so that crosspred evaluates the exposure
    basis outside the training range with the training boundary, as R does.  A fix for D that stops storing them
    altogether would break this."""
    x = temp[:400]
    kv = np.quantile(x, [.1, .75, .9])
    lk = logknots([0, 5], nk=2)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        cb = CrossBasis(x, lag=5, argvar={'fun': fun, 'knots': kv, **kw}, arglag={'fun': 'ns', 'knots': lk})
        p = cb.basis.shape[1]
        rng = np.random.default_rng(3)
        coef = rng.normal(0, 0.05, p)
        a = rng.normal(size=(p, p))
        vcov = 1e-4 * (a @ a.T)
        at = np.linspace(x.min() - 4, x.max() + 4, 9)
        pred = crosspred(cb, coef=coef, vcov=vcov, model_link='log', at=at, cen=15.0)
    kw_r = ', degree=2' if fun == 'bs' else ''
    push(kv=kv, lk=lk, coef=coef, vcov=vcov, at=at)
    push(xin=x)
    r(f'cbR <- suppressWarnings(crossbasis(xin, lag=5, argvar=list(fun="{fun}", knots=kv{kw_r}), '
      f'arglag=list(fun="ns", knots=lk)))')
    r('pR <- suppressWarnings(crosspred(cbR, coef=coef, vcov=vcov, model.link="log", at=at, cen=15))')
    assert_close(pred.allfit, rget('pR$allfit'), rtol=1e-8, what='allfit')
    assert_close(pred.matfit, rget('pR$matfit'), rtol=1e-8, what='matfit')
    assert_close(pred.allse, rget('pR$allse'), rtol=1e-8, what='allse')


# --------------------------------------------------------------------------------------------------------------------
# Theme E: lag-matrix (exposure history) input to CrossBasis
# --------------------------------------------------------------------------------------------------------------------
_MATRICES = {}


def exposure_matrix(key):
    """Exposure-history matrices (rows = observations, columns = lags), built once."""
    if key not in _MATRICES:
        t = chicago()['temp']
        rng = np.random.default_rng({'random60x6': 1, 'random40x5': 4, 'square6x6': 2}.get(key, 0))
        if key == 'exphist':                                   # dlnm::exphist, lag 0..5, no missing cells
            push(xs=t[:120])
            r('Hm <- unclass(exphist(xs, lag=c(0, 5), fill=0))')
            _MATRICES[key] = rget('Hm')
        elif key == 'tslag':                                   # tsModel::Lag: NaN cells in the first 5 rows
            push(xs=t[:120])
            r('Hm <- unclass(tsModel::Lag(xs, 0:5))')
            _MATRICES[key] = rget('Hm')
        elif key == 'random60x6':
            _MATRICES[key] = rng.normal(15, 8, (60, 6))
        elif key == 'random40x5':
            _MATRICES[key] = rng.normal(15, 8, (40, 5))
        elif key == 'square6x6':                               # n_obs == ncol: no crash today, silently wrong
            _MATRICES[key] = rng.normal(15, 8, (6, 6))
    return _MATRICES[key].copy()


BS2 = ('bs2-knots', lambda kv: {'fun': 'bs', 'degree': 2, 'knots': kv}, 'list(fun="bs", degree=2, knots=kv)')
NS4 = ('ns-df4', lambda kv: {'fun': 'ns', 'df': 4}, 'list(fun="ns", df=4)')
NSK = ('ns-knots', lambda kv: {'fun': 'ns', 'knots': kv}, 'list(fun="ns", knots=kv)')
LIN = ('lin', lambda kv: {'fun': 'lin'}, 'list(fun="lin")')
STR = ('strata', lambda kv: {'fun': 'strata', 'breaks': np.array([10., 20.])}, 'list(fun="strata", breaks=c(10, 20))')
BS2Q = ('bs2-2knots', lambda kv: {'fun': 'bs', 'degree': 2, 'knots': kv[[0, 2]]},
        'list(fun="bs", degree=2, knots=kv[c(1, 3)])')
NS3L = ('ns-df3', {'fun': 'ns', 'df': 3}, 'list(fun="ns", df=3)')
NSKL = ('ns-knots13', {'fun': 'ns', 'knots': np.array([1., 3.])}, 'list(fun="ns", knots=c(1, 3))')
INTL = ('integer', {'fun': 'integer'}, 'list(fun="integer")')

MATRIX_CASES = [
    # matrix, var basis, lag basis, lag (None = omitted: R uses c(0, ncol-1))
    pytest.param('exphist', BS2, NS3L, 5, id='exphist-bs2_x_ns3'),
    pytest.param('exphist', NS4, NS3L, 5, id='exphist-nsdf4_x_ns3'),
    pytest.param('exphist', NSK, INTL, 5, id='exphist-nsknots_x_integer'),
    pytest.param('exphist', LIN, NS3L, 5, id='exphist-lin_x_ns3'),
    pytest.param('exphist', STR, NS3L, 5, id='exphist-strata_x_ns3'),
    pytest.param('exphist', BS2, NSKL, 5, id='exphist-bs2_x_nsknots'),
    pytest.param('exphist', BS2, NS3L, None, id='exphist-lag-omitted'),
    pytest.param('random60x6', BS2, NS3L, 5, id='random60x6-bs2_x_ns3'),
    pytest.param('tslag', BS2, NS3L, 5, id='NaN-cells-bs2_x_ns3'),
    pytest.param('random40x5', NSK, NS3L, [2, 6], id='40x5-lag2to6-nsknots_x_ns3'),
]


def run_matrix_case(matrix_key, var, lag_basis, lag):
    x = exposure_matrix(matrix_key)
    kv = np.nanquantile(x, [.1, .75, .9])
    push(kv=kv)
    (_, var_py, var_r), (_, lag_py, lag_r) = var, lag_basis
    argvar = var_py(kv)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        cb = CrossBasis(x, argvar=copy.deepcopy(argvar), arglag=copy.deepcopy(lag_py),
                        **({} if lag is None else {'lag': lag}))
    ref, attrs = r_crossbasis(x, var_r, lag_r, lag=lag)
    return cb, ref, attrs


@pytest.mark.parametrize('matrix_key, var, lag_basis, lag', MATRIX_CASES)
def test_matrix_x_matches_r(matrix_key, var, lag_basis, lag):
    """x with several columns is a matrix of lagged occurrences: R evaluates the basis on as.numeric(x)
    (column-major), reshapes each basis column back to n x nlag and multiplies by the lag basis."""
    cb, ref, attrs = run_matrix_case(matrix_key, var, lag_basis, lag)
    assert_close(cb.basis, ref, rtol=TOL, what='cross-basis from matrix x')
    assert tuple(cb.df) == tuple(int(v) for v in attrs['df'])
    assert list(np.asarray(cb.lag).ravel()) == list(attrs['lag'].ravel())


def test_square_matrix_x_matches_r():
    """Square exposure matrix (n_obs == ncol): the broadcast crash is masked and the numbers are simply wrong."""
    cb, ref, _ = run_matrix_case('square6x6', BS2Q, NS3L, 5)
    assert_close(cb.basis, ref, rtol=TOL, what='cross-basis from a square matrix')


@pytest.mark.parametrize('fun, kw', [('bs', {'degree': 2}), ('ns', {})], ids=['bs', 'ns'])
def test_matrix_x_crosspred_matches_r(fun, kw):
    """A matrix cross-basis must be usable downstream: crosspred outside the training range needs the training
    Boundary_knots in the recorded argvar (R keeps them as an attribute)."""
    x = exposure_matrix('exphist')
    kv = np.quantile(x, [.1, .75, .9])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        cb = CrossBasis(x, lag=5, argvar={'fun': fun, 'knots': kv, **kw}, arglag={'fun': 'ns', 'df': 3})
        p = cb.basis.shape[1]
        rng = np.random.default_rng(5)
        coef = rng.normal(0, 0.05, p)
        a = rng.normal(size=(p, p))
        vcov = 1e-4 * (a @ a.T)
        at = np.linspace(x.min() - 5, x.max() + 5, 7)
        pred = crosspred(cb, coef=coef, vcov=vcov, model_link='log', at=at, cen=15.0)
    kw_r = ', degree=2' if fun == 'bs' else ''
    push(kv=kv, coef=coef, vcov=vcov, at=at, Hin=x)
    r(f'cbM <- suppressWarnings(crossbasis(Hin, lag=5, argvar=list(fun="{fun}", knots=kv{kw_r}), '
      f'arglag=list(fun="ns", df=3)))')
    r('pM <- suppressWarnings(crosspred(cbM, coef=coef, vcov=vcov, model.link="log", at=at, cen=15))')
    assert_close(pred.allfit, rget('pM$allfit'), rtol=1e-8, what='allfit')
    assert_close(pred.matfit, rget('pM$matfit'), rtol=1e-8, what='matfit')


def test_matrix_x_with_wrong_number_of_columns_is_rejected():
    """R: 'NCOL(x) must be equal to 1 ... otherwise to the lag period'.  PyDLNM already raises (faithful)."""
    x = exposure_matrix('exphist')[:, :4]
    assert r_raises('crossbasis(matrix(0, 10, 4), lag=5, argvar=list(fun="lin"), arglag=list(fun="ns", df=3))')
    with pytest.raises(ValueError):
        CrossBasis(x, lag=5, argvar={'fun': 'lin'}, arglag={'fun': 'ns', 'df': 3})


# --------------------------------------------------------------------------------------------------------------------
# Theme E: OneBasis with 2-D input (R: x <- as.vector(x), column-major)
# --------------------------------------------------------------------------------------------------------------------
def matrix_input():
    return np.round(np.random.default_rng(11).normal(15, 8, (8, 3)), 2)


ONEBASIS_2D_BAD = [
    pytest.param('lin', '', {}, id='lin'),
    pytest.param('lin', 'intercept=TRUE', {'intercept': True}, id='lin-intercept'),
    pytest.param('poly', 'degree=2', {'degree': 2}, id='poly2'),
    pytest.param('thr', 'thr.value=15', {'thr_value': 15.0}, id='thr'),
    pytest.param('thr', 'thr.value=c(10, 20), side="d"', {'thr_value': np.array([10., 20.]), 'side': 'd'},
                 id='thr-double'),
    pytest.param('strata', 'breaks=c(10, 20)', {'breaks': np.array([10., 20.])}, id='strata'),
]
ONEBASIS_2D_OK = [
    pytest.param('ns', 'df=3', {'df': 3}, id='ns-df3'),
    pytest.param('bs', 'df=4', {'df': 4}, id='bs-df4'),
    pytest.param('ns', 'knots=c(10, 20)', {'knots': np.array([10., 20.])}, id='ns-knots'),
]


@pytest.mark.parametrize('fun, rargs, kw', ONEBASIS_2D_BAD)
def test_onebasis_2d_input_is_flattened_column_major(fun, rargs, kw):
    x = matrix_input()
    ref = r_onebasis(x, fun, rargs)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ob = OneBasis(x, fun=fun, **kw)
    assert_close(ob.basis, ref, rtol=TOL, what=f'OneBasis({fun}) on a 2-D array')


@pytest.mark.parametrize('fun, rargs, kw', ONEBASIS_2D_OK)
def test_onebasis_2d_input_spline_funs_match_r(fun, rargs, kw):
    """ns / bs already flatten like as.vector (faithful today); guards the flatten fix."""
    x = matrix_input()
    ref = r_onebasis(x, fun, rargs)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ob = OneBasis(x, fun=fun, **kw)
    assert_close(ob.basis, ref, rtol=TOL, what=f'OneBasis({fun}) on a 2-D array')


# --------------------------------------------------------------------------------------------------------------------
# Theme E: group= (independent series stacked in x; R computes the lags inside each group)
# --------------------------------------------------------------------------------------------------------------------
GROUP_AV = lambda kv: {'fun': 'bs', 'degree': 2, 'knots': kv}                      # noqa: E731
GROUP_AV_R = 'list(fun="bs", degree=2, knots=kv)'
GROUP_CASES = [
    # id: group labels, lag, index of an injected NaN in x (or None), lag basis
    pytest.param(np.repeat([1, 2, 3], 100), 5, None, NS3L, id='3-blocks'),
    pytest.param(np.tile([1, 2, 3], 100), 5, None, NS3L, id='interleaved-labels'),
    pytest.param(np.repeat([1, 2, 3], [50, 120, 130]), 5, None, NS3L, id='unequal-blocks'),
    pytest.param(np.repeat([1, 2, 3], 100), [2, 5], None, NS3L, id='3-blocks-lag2to5'),
    pytest.param(np.repeat([1, 2, 3], 100), 5, 50, NS3L, id='3-blocks-NaN-in-x'),
    pytest.param(np.repeat([1, 2, 3], 100), 5, None, INTL, id='3-blocks-integer-lag'),
]


def run_group_case(temp, group, lag, nan_at, lag_basis):
    x = temp[:300].copy()
    if nan_at is not None:
        x[nan_at] = np.nan
    kv = np.nanquantile(temp[:300], [.1, .75, .9])
    push(kv=kv)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        cb = CrossBasis(x, lag=lag, argvar=GROUP_AV(kv), arglag=copy.deepcopy(lag_basis[1]), group=group)
    ref, attrs = r_crossbasis(x, GROUP_AV_R, lag_basis[2], lag=lag, group=group)
    return cb, ref, attrs


@pytest.mark.parametrize('group, lag, nan_at, lag_basis', GROUP_CASES)
def test_group_lags_stay_inside_groups(temp, group, lag, nan_at, lag_basis):
    cb, ref, _ = run_group_case(temp, group, lag, nan_at, lag_basis)
    assert list(nan_rows(cb.basis)) == list(nan_rows(ref)), 'rows with NaN differ from R (first rows of each group)'
    assert_close(cb.basis, ref, rtol=TOL, what='grouped cross-basis')


def test_group_attribute_is_number_of_groups(temp):
    """R: attr(crossbasis, "group") is the number of groups."""
    group = np.repeat([1, 2, 3], 100)
    cb, _, attrs = run_group_case(temp, group, 5, None, NS3L)
    n_groups_r = int(attrs['group'][0])
    assert n_groups_r == 3
    assert np.ndim(cb.group) == 0, f'group attribute has shape {np.shape(cb.group)}, expected a scalar count'
    assert int(cb.group) == n_groups_r


@pytest.mark.parametrize('n, lag, sizes', [
    pytest.param(20, 10, [5, 5, 5, 5], id='groups-of-5-lag10'),
    pytest.param(20, 10, [10, 10], id='groups-of-10-lag10-length-equals-diff'),
], )
def test_group_shorter_than_lag_is_rejected(temp, n, lag, sizes):
    """R checkgroup: 'each group must have length > diff(lag)'."""
    x = temp[:n]
    group = np.repeat(np.arange(len(sizes)), sizes).astype(float)
    push(xin=x, grp=group)
    code = f'crossbasis(xin, lag={lag}, argvar=list(fun="lin"), arglag=list(fun="ns", df=3), group=grp)'
    assert r_raises(code), 'expected R to reject the short groups'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with pytest.raises((ValueError, TypeError), match=r'(?i)group'):
            CrossBasis(x, lag=lag, argvar={'fun': 'lin'}, arglag={'fun': 'ns', 'df': 3}, group=group)


def test_group_with_matrix_x_is_rejected_for_the_right_reason():
    x = exposure_matrix('exphist')
    group = np.repeat([1., 2.], 60)
    push(Hin=x, grp=group)
    assert r_raises('crossbasis(Hin, lag=5, argvar=list(fun="lin"), arglag=list(fun="ns", df=3), group=grp)')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with pytest.raises((ValueError, TypeError), match=r'(?i)group'):
            CrossBasis(x, lag=5, argvar={'fun': 'lin'}, arglag={'fun': 'ns', 'df': 3}, group=group)


def test_single_group_equals_ungrouped_and_matches_r(temp):
    """One group label = one series (faithful today): 5 NaN rows, values as R; guards the group-aware lag rewrite."""
    group = np.ones(300)
    cb, ref, _ = run_group_case(temp, group, 5, None, NS3L)
    assert list(nan_rows(cb.basis)) == [0, 1, 2, 3, 4]
    assert_close(cb.basis, ref, rtol=TOL, what='cross-basis with a single group')


@pytest.mark.parametrize('nan_at', [[50], [0, 299], [3, 120, 121]], ids=['interior', 'both-ends', 'runs'])
def test_nan_in_exposure_series_matches_r_without_group(temp, nan_at):
    """NaN in x propagates to the next max-lag rows, and the first max-lag rows are NaN (faithful today)."""
    x = temp[:300].copy()
    x[nan_at] = np.nan
    kv = np.nanquantile(temp[:300], [.1, .75, .9])
    push(kv=kv)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        cb = CrossBasis(x, lag=5, argvar=GROUP_AV(kv), arglag={'fun': 'ns', 'df': 3})
    ref, _ = r_crossbasis(x, GROUP_AV_R, 'list(fun="ns", df=3)', lag=5)
    assert list(nan_rows(cb.basis)) == list(nan_rows(ref))
    assert_close(cb.basis, ref, rtol=TOL, what='cross-basis with NaN in x')
    assert np.allclose(cb.range, (np.nanmin(x), np.nanmax(x)))


# --------------------------------------------------------------------------------------------------------------------
# Theme F: R-spelled / unknown / misspelled keyword arguments
# --------------------------------------------------------------------------------------------------------------------
_RNG = np.random.default_rng(1)
X1 = np.round(_RNG.normal(15, 8, 300), 2)                          # OneBasis input
KN = np.quantile(X1, [.25, .5, .75])
BK = np.array([X1.min() - 5, X1.max() + 5])

# R spellings that R honours (dotted names, or renamed with a warning: bound, old knots for thr / strata)
ONEBASIS_R_SPELLED = [
    pytest.param('ns', 'knots=kn, Boundary.knots=bk', {'knots': KN, 'Boundary.knots': BK}, 'Boundary.knots',
                 id='ns-Boundary.knots'),
    pytest.param('bs', 'knots=kn, Boundary.knots=bk', {'knots': KN, 'Boundary.knots': BK}, 'Boundary.knots',
                 id='bs-Boundary.knots'),
    pytest.param('ns', 'knots=kn, bound=bk', {'knots': KN, 'bound': BK}, 'bound', id='ns-bound'),
    pytest.param('bs', 'knots=kn, bound=bk', {'knots': KN, 'bound': BK}, 'bound', id='bs-bound'),
    pytest.param('thr', 'thr.value=25', {'thr.value': 25.0}, 'thr.value', id='thr-thr.value'),
    pytest.param('thr', 'knots=25', {'knots': 25.0}, 'knots', id='thr-old-knots'),
    pytest.param('strata', 'knots=c(10, 20)', {'knots': np.array([10., 20.])}, 'knots', id='strata-old-knots'),
]
# Python spellings of the same arguments: faithful today
ONEBASIS_PY_SPELLED = [
    pytest.param('ns', 'knots=kn, Boundary.knots=bk', {'knots': KN, 'Boundary_knots': BK}, id='ns-Boundary_knots'),
    pytest.param('bs', 'knots=kn, Boundary.knots=bk', {'knots': KN, 'Boundary_knots': BK}, id='bs-Boundary_knots'),
    pytest.param('thr', 'thr.value=25', {'thr_value': 25.0}, id='thr-thr_value'),
    pytest.param('strata', 'breaks=c(10, 20)', {'breaks': np.array([10., 20.])}, id='strata-breaks'),
]
# arguments the function does not have: R stops with 'unused argument'
ONEBASIS_UNUSED = [
    pytest.param('ns', 'knots=kn, degree=2', {'knots': KN, 'degree': 2}, 'degree', id='ns-degree'),
    pytest.param('ns', 'df=4, foo=1', {'df': 4, 'foo': 1}, 'foo', id='ns-foo'),
    pytest.param('ns', 'df=4, scale=5', {'df': 4, 'scale': 5.0}, 'scale', id='ns-scale'),
    pytest.param('bs', 'df=4, dgree=2', {'df': 4, 'dgree': 2}, 'dgree', id='bs-typo-dgree'),
    pytest.param('bs', 'df=4, scale=3', {'df': 4, 'scale': 3.0}, 'scale', id='bs-scale'),
    pytest.param('lin', 'df=4', {'df': 4}, 'df', id='lin-df'),
    pytest.param('lin', 'knots=kn', {'knots': KN}, 'knots', id='lin-knots'),
    pytest.param('poly', 'df=3', {'df': 3}, 'df', id='poly-df'),
    pytest.param('thr', 'thr.value=25, degree=2', {'thr_value': 25.0, 'degree': 2}, 'degree', id='thr-degree'),
    pytest.param('strata', 'breaks=c(10, 20), degree=2', {'breaks': np.array([10., 20.]), 'degree': 2}, 'degree',
                 id='strata-degree'),
]


@pytest.mark.parametrize('fun, rargs, kw, key', ONEBASIS_R_SPELLED)
def test_onebasis_r_spelled_args_honoured_or_rejected(fun, rargs, kw, key):
    push(kn=KN, bk=BK)
    ref = r_onebasis(X1, fun, rargs)
    honoured_or_rejected(lambda: OneBasis(X1, fun=fun, **copy.deepcopy(kw)).basis, ref,
                         what=f'OneBasis({fun}, {list(kw)}) vs R', key=key)


@pytest.mark.parametrize('fun, rargs, kw, key', ONEBASIS_UNUSED)
def test_onebasis_unused_arguments_are_rejected(fun, rargs, kw, key):
    push(kn=KN, xin=X1)
    must_be_rejected(lambda: OneBasis(X1, fun=fun, **copy.deepcopy(kw)),
                     f'suppressWarnings(onebasis(xin, fun="{fun}", {rargs}))', f'OneBasis({fun}, {list(kw)})', key)


@pytest.mark.parametrize('fun, rargs, kw', ONEBASIS_PY_SPELLED)
def test_onebasis_python_spellings_match_r(fun, rargs, kw):
    push(kn=KN, bk=BK)
    ref = r_onebasis(X1, fun, rargs)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ob = OneBasis(X1, fun=fun, **copy.deepcopy(kw))
    assert_close(ob.basis, ref, rtol=TOL, what=f'OneBasis({fun}, {list(kw)})')


# every legitimate argument of every built-in function must survive a stricter argument check
ONEBASIS_VALID = [
    pytest.param('ns', 'df=4, intercept=TRUE', {'df': 4, 'intercept': True}, id='ns-df-intercept'),
    pytest.param('ns', 'df=4, cen=15', {'df': 4, 'cen': 15.0}, id='ns-cen'),
    pytest.param('bs', 'df=6, degree=2, intercept=TRUE', {'df': 6, 'degree': 2, 'intercept': True},
                 id='bs-df-degree-intercept'),
    pytest.param('bs', 'knots=kn, degree=2, Boundary.knots=bk', {'knots': KN, 'degree': 2, 'Boundary_knots': BK},
                 id='bs-knots-degree-boundary'),
    pytest.param('lin', 'intercept=TRUE', {'intercept': True}, id='lin-intercept'),
    pytest.param('poly', 'degree=3, scale=30, intercept=TRUE', {'degree': 3, 'scale': 30.0, 'intercept': True},
                 id='poly-degree-scale-intercept'),
    pytest.param('strata', 'breaks=c(10, 20), ref=2, intercept=TRUE',
                 {'breaks': np.array([10., 20.]), 'ref': 2, 'intercept': True}, id='strata-breaks-ref-intercept'),
    pytest.param('thr', 'thr.value=25, side="l", intercept=TRUE', {'thr_value': 25.0, 'side': 'l', 'intercept': True},
                 id='thr-single-low-intercept'),
    pytest.param('thr', 'thr.value=c(10, 20), side="d", intercept=TRUE',
                 {'thr_value': np.array([10., 20.]), 'side': 'd', 'intercept': True}, id='thr-double-intercept'),
]


@pytest.mark.parametrize('fun, rargs, kw', ONEBASIS_VALID)
def test_onebasis_valid_arguments_still_accepted(fun, rargs, kw):
    """Guard against an over-strict argument check: all documented arguments must keep working."""
    push(kn=KN, bk=BK)
    ref = r_onebasis(X1, fun, rargs)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ob = OneBasis(X1, fun=fun, **copy.deepcopy(kw))
    assert_close(ob.basis, ref, rtol=TOL, what=f'OneBasis({fun}, {list(kw)})')


def test_onebasis_unknown_fun_name_is_rejected():
    push(xin=X1)
    assert r_raises('onebasis(xin, fun="foo")')
    with pytest.raises((ValueError, TypeError)):
        OneBasis(X1, fun='foo')


# ---- CrossBasis argvar / arglag keywords ---------------------------------------------------------------------------
CB_LAG = 6
ARGLAG_F = {'fun': 'ns', 'df': 3}
ARGLAG_F_R = 'list(fun="ns", df=3)'


def cb_series():
    """(x, var knots, wide Boundary.knots) for the CrossBasis keyword tests."""
    x = chicago()['temp'][:300]
    return x, np.quantile(x, [.1, .75, .9]), np.array([-30., 40.])


CB_R_SPELLED = [
    pytest.param(lambda kv, bk: {'fun': 'bs', 'degree': 2, 'knots': kv, 'Boundary.knots': bk},
                 'list(fun="bs", degree=2, knots=kv, Boundary.knots=bk)', 'Boundary.knots', id='bs-Boundary.knots'),
    pytest.param(lambda kv, bk: {'fun': 'ns', 'knots': kv, 'Boundary.knots': bk},
                 'list(fun="ns", knots=kv, Boundary.knots=bk)', 'Boundary.knots', id='ns-Boundary.knots'),
    pytest.param(lambda kv, bk: {'fun': 'bs', 'degree': 2, 'knots': kv, 'bound': bk},
                 'list(fun="bs", degree=2, knots=kv, bound=bk)', 'bound', id='bs-bound'),
    pytest.param(lambda kv, bk: {'type': 'bs', 'degree': 2, 'knots': kv},
                 'list(type="bs", degree=2, knots=kv)', 'type', id='deprecated-type-is-fun'),
]
CB_PY_SPELLED = [
    pytest.param(lambda kv, bk: {'fun': 'bs', 'degree': 2, 'knots': kv, 'Boundary_knots': bk},
                 'list(fun="bs", degree=2, knots=kv, Boundary.knots=bk)', id='bs-Boundary_knots'),
    pytest.param(lambda kv, bk: {'fun': 'ns', 'knots': kv, 'Boundary_knots': bk},
                 'list(fun="ns", knots=kv, Boundary.knots=bk)', id='ns-Boundary_knots'),
    pytest.param(lambda kv, bk: {'fun': 'bs', 'degree': 2, 'knots': kv, 'cen': 15.0},
                 'list(fun="bs", degree=2, knots=kv, cen=15)', id='bs-cen-kept-out-of-basis'),
]
CB_UNUSED = [
    pytest.param(lambda kv, bk: {'fun': 'bs', 'degree': 2, 'knots': kv, 'foo': 1}, ARGLAG_F,
                 'list(fun="bs", degree=2, knots=kv, foo=1)', ARGLAG_F_R, 'foo', id='argvar-bs-foo'),
    pytest.param(lambda kv, bk: {'fun': 'ns', 'knots': kv, 'foo': 1}, ARGLAG_F,
                 'list(fun="ns", knots=kv, foo=1)', ARGLAG_F_R, 'foo', id='argvar-ns-foo'),
    pytest.param(lambda kv, bk: {'fun': 'bs', 'dgree': 2, 'knots': kv}, ARGLAG_F,
                 'list(fun="bs", dgree=2, knots=kv)', ARGLAG_F_R, 'dgree', id='argvar-bs-typo-dgree'),
    pytest.param(lambda kv, bk: {'fun': 'lin', 'df': 3}, ARGLAG_F,
                 'list(fun="lin", df=3)', ARGLAG_F_R, 'df', id='argvar-lin-df'),
    pytest.param(lambda kv, bk: {'fun': 'ns', 'knots': kv}, {'fun': 'ns', 'df': 3, 'foo': 1},
                 'list(fun="ns", knots=kv)', 'list(fun="ns", df=3, foo=1)', 'foo', id='arglag-ns-foo'),
]


@pytest.mark.parametrize('argvar_fn, argvar_r, key', CB_R_SPELLED)
def test_crossbasis_r_spelled_argvar_honoured_or_rejected(argvar_fn, argvar_r, key):
    x, kv, bk = cb_series()
    push(kv=kv, bk=bk)
    ref, _ = r_crossbasis(x, argvar_r, ARGLAG_F_R, lag=CB_LAG)
    honoured_or_rejected(
        lambda: CrossBasis(x, lag=CB_LAG, argvar=argvar_fn(kv, bk), arglag=copy.deepcopy(ARGLAG_F)).basis, ref,
        what='CrossBasis(argvar with R spelling) vs R', key=key)


@pytest.mark.parametrize('argvar_fn, arglag, argvar_r, arglag_r, key', CB_UNUSED)
def test_crossbasis_unused_argument_is_rejected(argvar_fn, arglag, argvar_r, arglag_r, key):
    x, kv, bk = cb_series()
    push(kv=kv, bk=bk, xin=x)
    must_be_rejected(
        lambda: CrossBasis(x, lag=CB_LAG, argvar=argvar_fn(kv, bk), arglag=copy.deepcopy(arglag)),
        f'suppressWarnings(crossbasis(xin, lag={CB_LAG}, argvar={argvar_r}, arglag={arglag_r}))',
        'CrossBasis with an unused argument', key)


@pytest.mark.parametrize('argvar_fn, argvar_r', CB_PY_SPELLED)
def test_crossbasis_python_spelled_argvar_matches_r(argvar_fn, argvar_r):
    """Python spelling Boundary_knots, and cen (kept out of the basis, applied at prediction) agree with R today."""
    x, kv, bk = cb_series()
    push(kv=kv, bk=bk)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        cb = CrossBasis(x, lag=CB_LAG, argvar=argvar_fn(kv, bk), arglag=copy.deepcopy(ARGLAG_F))
    ref, attrs = r_crossbasis(x, argvar_r, ARGLAG_F_R, lag=CB_LAG)
    assert_close(cb.basis, ref, rtol=TOL, what='CrossBasis with valid argvar')
    assert tuple(cb.df) == tuple(int(v) for v in attrs['df'])


def test_crossbasis_europe_pattern_ns_knots_boundary_integer_lag(temp):
    """The validated Europe call pattern: ns with explicit knots and Boundary_knots, integer lag basis, lag [0, 3]."""
    x = temp[:400]
    kv = np.quantile(x, [.10, .50, .90])
    bk = np.array([x.min(), x.max()])
    push(kv=kv, bk=bk)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        cb = CrossBasis(x, lag=[0, 3], argvar={'fun': 'ns', 'knots': kv, 'Boundary_knots': bk},
                        arglag={'fun': 'integer'})
    ref, attrs = r_crossbasis(x, 'list(fun="ns", knots=kv, Boundary.knots=bk)', 'list(fun="integer")', lag=[0, 3])
    assert_close(cb.basis, ref, rtol=TOL, what='Europe-pattern cross-basis')
    assert tuple(cb.df) == tuple(int(v) for v in attrs['df'])
    assert len(nan_rows(cb.basis)) == 3


@pytest.mark.parametrize('fun, kw, rargs, coef', [
    pytest.param('lin', {}, 'fun="lin"', [0.03], id='lin'),
    pytest.param('poly', {'degree': 2}, 'fun="poly", degree=2', [0.03, -0.01], id='poly2'),
    pytest.param('poly', {'degree': 2, 'scale': 30.0}, 'fun="poly", degree=2, scale=30', [0.03, -0.01],
                 id='poly2-scale'),
], )
def test_crosspred_on_onebasis_round_trips_the_stored_attributes(temp, fun, kw, rargs, coef):
    """crosspred rebuilds the basis with OneBasis(newx, **basis.attributes) (attributes include the data 'range').
    A strict unused-argument check must not reject those bookkeeping attributes."""
    x = temp[:400]
    coef = np.asarray(coef)
    vcov = 1e-4 * np.eye(len(coef))
    at = np.arange(-10., 30., 5.)
    push(xin=x, coef=coef, vcov=vcov, at=at)
    r(f'obR <- onebasis(xin, {rargs})')
    r('pOB <- suppressWarnings(crosspred(obR, coef=coef, vcov=vcov, model.link="log", at=at, cen=15))')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ob = OneBasis(x, fun=fun, **kw)
        pred = crosspred(ob, coef=coef, vcov=vcov, model_link='log', at=at, cen=15.0)
    assert_close(pred.allfit, rget('pOB$allfit'), rtol=1e-8, what='allfit')
