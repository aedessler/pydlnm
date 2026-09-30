"""crossreduce output surface (theme T) and API parity with the dlnm NAMESPACE (theme S3) versus R dlnm 2.4.10.

Every test computes its reference in R at run time (rpy2) and compares PyDLNM on identical inputs, or asserts a
precisely documented R behaviour (dlnm NAMESPACE / crossreduce.R / mkcen.R / coef.crosspred.R).

Theme T  -- PyDLNM crossreduce() ports only  newcoef = M coef, newvcov = M vcov M'  of type="overall" (both already
            equal to R to ~1e-16) while its docstring advertises R's interface and "identical to R's crossreduce object".
  crossreduce-5   `type` is documented but the parameter is `reduction_type`; type "var"/"lag" raise ValueError;
                  value, at, lag (sub-period), bylag, ci_level, model_link raise TypeError; the result lacks basis,
                  predvar, fit, se, RRfit/RRlow/RRhigh (or low/high), lag, bylag, type, value, ci_level, summary().
  crossreduce-6   cen is only echoed: with cen unset R resolves mkcen() (median(pretty(range)) for bs/ns/poly, NULL for
                  thr/strata/integer/lin and intercept=TRUE, cen=TRUE/FALSE handled) and reports it in $cen.
  centering-19    the reduced basis, fit, se and RR/CI are all centred at cen in R (fit == 0, RRfit == 1 at cen).
Theme S3 -- API parity gaps versus the dlnm NAMESPACE (validation-audit-19); only what a Python port sensibly provides
            is tested:
  coef() of a CrossPred (R coef.crosspred), onebasis() / crossbasis() functions next to the classes, logknots /
  equalknots exported by the package, summary() of a CrossReduce (R summary.crossreduce) and a crosspred `at` given as an
  exposure-history matrix, which is silently reduced to its first column (R: history-aware prediction).

Tests decorated with @known_defect assert the R-faithful behaviour and fail today (strict xfail); the plain tests guard
neighbouring behaviour that is already faithful (reduced coef/vcov, cen independence of the reduction, vcov() of a
CrossPred, the classes' numbers, utils helpers, ...) and must keep passing while the fixes land.

Not tested here (see the module-level notes at the bottom): plotting, ps/cr/cbPen/smooth constructor, datasets,
GAM crosspred, and the sub-items of validation-audit-19 that other modules already cover (integer OneBasis, group=/matrix
x, __init__.py syntax).

Test design
* The cross-basis is the validated design: bs(degree 2, knots at P10/75/90 of R's chicagoNMMAPS temperature) x
  ns(logknots(21,3)) with explicit knots; its coef/vcov are a fixed-seed random draw pushed to R (25 parameters), so
  every difference is due to crossreduce alone.
* Python keyword names for the unsupported options follow the crosspred() spelling (`model_link`, `ci_level`); the
  reduction-type keyword is looked up in the signature (`type` as documented, or `reduction_type`), and `model_link`
  is passed only when the signature has it, so the plain tests run on either side of the fix.
"""
import ast
import contextlib
import inspect
import io
import os
import re
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from rhelpers import SRC, assert_close, chicago, known_defect, np2r, r, rget

RTOL = 1e-10

# --------------------------------------------------------------------------------------------------------------
# environment guards (audit finding Q2: CrossBasis construction overwrites os.environ['R_HOME'])
# --------------------------------------------------------------------------------------------------------------
_GOOD_R_HOME = os.path.dirname(str(r('.Library')[0]))


def _pin_r_home():
    os.environ['R_HOME'] = _GOOD_R_HOME


_pin_r_home()
r('invisible(chol2inv(chol(diag(2) + 1))); invisible(solve(diag(2))); invisible(qr(diag(2)))')


@pytest.fixture(autouse=True)
def _keep_r_home():
    _pin_r_home()
    yield
    _pin_r_home()


@contextlib.contextmanager
def _quiet():
    """PyDLNM prints progress lines; R messages (mkcen) go to stderr and are harmless."""
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        yield


# --------------------------------------------------------------------------------------------------------------
# helpers: identical inputs for R and Python
# --------------------------------------------------------------------------------------------------------------
AT = np.arange(-10.0, 30.0 + 1e-9, 2.5)        # 17 points inside the Chicago range; contains 5, 10, 15, 20
CEN = 15.0
_DESIGN = {}


def _rquantile(x, probs):
    np2r('crapi_xq', x)
    return rget(f'quantile(crapi_xq, c({", ".join(repr(float(p)) for p in probs)}))')


def design():
    """bs2(P10/75/90) x ns(logknots(21,3)) cross-basis of Chicago temperature + fixed-seed coef/vcov, in R and Python."""
    if not _DESIGN:
        from basis import CrossBasis
        from utils import logknots
        temp = chicago()['temp']
        kv = _rquantile(temp, [.10, .75, .90])
        np2r('crapi_temp', temp)
        np2r('crapi_kv', kv)
        r('crapi_cb <- crossbasis(crapi_temp, lag=21, argvar=list(fun="bs", degree=2, knots=crapi_kv),'
          ' arglag=list(fun="ns", knots=logknots(21, 3)))')
        with _quiet():
            cb = CrossBasis(temp, lag=21, argvar={'fun': 'bs', 'degree': 2, 'knots': kv},
                            arglag={'fun': 'ns', 'knots': logknots([0, 21], nk=3)})
        p = int(cb.shape[1])
        assert p == int(rget('ncol(crapi_cb)')[0]) == 25
        rng = np.random.default_rng(42)
        coef = rng.normal(0, .05, p)
        A = rng.normal(0, .02, (p, p))
        vcov = A @ A.T + 1e-4 * np.eye(p)
        np2r('crapi_coef', coef)
        np2r('crapi_vcov', vcov)
        r(f'crapi_vcov <- matrix(crapi_vcov, {p}, {p})')
        _DESIGN['d'] = SimpleNamespace(temp=temp, cb=cb, p=p, coef=coef, vcov=vcov, nvar=5, nlag=5)
    return _DESIGN['d']


def _rnum(v):
    if isinstance(v, (bool, np.bool_)):
        return 'TRUE' if v else 'FALSE'
    return repr(float(v))


def r_reduce(name, typ='overall', value=None, cen=None, at=None, lag=None, bylag=None, ci_level=None, link='log'):
    """R crossreduce() on the design's coef/vcov, stored in the R global `name`."""
    design()
    args = ['crapi_cb', 'coef=crapi_coef', 'vcov=crapi_vcov', f'type="{typ}"']
    if link is not None:
        args.append(f'model.link="{link}"')
    if value is not None:
        args.append(f'value={_rnum(value)}')
    if cen is not None:
        args.append(f'cen={_rnum(cen)}')
    if at is not None:
        np2r('crapi_at', at)
        args.append('at=crapi_at')
    if lag is not None:
        args.append(f'lag=c({_rnum(lag[0])}, {_rnum(lag[1])})')
    if bylag is not None:
        args.append(f'bylag={_rnum(bylag)}')
    if ci_level is not None:
        args.append(f'ci.level={_rnum(ci_level)}')
    r(f'{name} <- suppressMessages(crossreduce({", ".join(args)}))')
    return name


def r_field(name, field):
    """Numeric content of one component of an R crossreduce object (exact, not partial, name matching)."""
    if field in ('basis', 'vcov'):
        return rget(f'matrix(as.numeric({name}[["{field}"]]), nrow({name}[["{field}"]]))')
    return rget(f'as.numeric({name}[["{field}"]])')


def r_has(name, field):
    return bool(r(f'!is.null({name}[["{field}"]])')[0])


_ALIASES = {'coefficients': ('coefficients', 'coef'), 'ci.level': ('ci_level', 'ci.level'),
            'type': ('type', 'reduction_type')}


def py_reduce(typ=None, link='log', **kw):
    """PyDLNM crossreduce() on the design's coef/vcov.  The reduction-type keyword is `type` (documented) or
    `reduction_type` (actual signature today); model_link is only passed when the signature has it."""
    from crossreduce import crossreduce
    d = design()
    params = inspect.signature(crossreduce).parameters
    if typ is not None:
        kw['type' if 'type' in params else 'reduction_type'] = typ
    if link is not None and 'model_link' in params:
        kw['model_link'] = link
    with _quiet():
        return crossreduce(d.cb, coef=d.coef, vcov=d.vcov, **kw)


def py_field(red, field):
    """Component of the PyDLNM CrossReduce as a float array; a missing component is the defect, say so."""
    for nm in _ALIASES.get(field, (field,)):
        if hasattr(red, nm):
            val = getattr(red, nm)
            val = getattr(val, 'basis', val) if field == 'basis' else val
            return np.atleast_1d(np.asarray(val, dtype=float))
    raise AssertionError(f'CrossReduce has no component {field!r} (R crossreduce returns it); '
                         f'attributes: {sorted(a for a in vars(red) if not a.startswith("_"))}')


def py_field_type(red):
    for nm in _ALIASES['type']:
        if hasattr(red, nm):
            return getattr(red, nm)
    raise AssertionError('CrossReduce does not record the reduction type (R: $type)')


def require_outputs(red, *fields):
    """The centred quantities R's crossreduce returns must exist at all (checked before any keyword-specific call)."""
    for f in fields:
        py_field(red, f)


def assert_fields(red, rname, fields, rtol=RTOL, what=''):
    for f in fields:
        assert_close(py_field(red, f), r_field(rname, f), rtol=rtol, what=f'{what} {f}'.strip())


OVERALL_POINT = ('coefficients', 'vcov', 'predvar', 'basis', 'fit', 'se')
RR_FIELDS = ('RRfit', 'RRlow', 'RRhigh')


# ==============================================================================================================
# Theme T / crossreduce-5: type="overall" outputs (fit, se, RR, CI, basis, predvar) of R's crossreduce
# ==============================================================================================================
def test_overall_reduced_basis_fit_and_se_match_r():
    """R: newbasis = onebasis(at) - onebasis(cen), fit = newbasis newcoef, se = sqrt(rowSums((newbasis newvcov) newbasis))."""
    ref = r_reduce('crapi_ov', at=AT, cen=CEN)
    require_outputs(py_reduce(cen=CEN), 'basis', 'predvar', 'fit', 'se')
    red = py_reduce(at=AT, cen=CEN)
    assert_fields(red, ref, OVERALL_POINT, what='overall')
    assert_close(py_field(red, 'lag'), r_field(ref, 'lag'), what='lag')


def test_overall_rr_and_ci_match_r():
    ref = r_reduce('crapi_rr', at=AT, cen=CEN)
    require_outputs(py_reduce(cen=CEN), *RR_FIELDS)
    red = py_reduce(at=AT, cen=CEN)
    assert_fields(red, ref, RR_FIELDS, what='overall')
    assert np.all(py_field(red, 'RRlow') <= py_field(red, 'RRfit')) and np.all(py_field(red, 'RRfit') <= py_field(red, 'RRhigh'))


def test_reduced_coef_through_onebasis_reproduces_r_fit_and_se():
    """The building blocks a full crossreduce needs already agree with R: PyDLNM's reduced coef/vcov and PyDLNM's
    OneBasis at (at, cen) with the cross-basis' recorded variable-basis arguments give R's fit and se to 1e-10."""
    from basis import OneBasis
    d = design()
    ref = r_reduce('crapi_blocks', at=AT, cen=CEN)
    red = py_reduce(cen=CEN)
    argvar = {k: v for k, v in d.cb.argvar.items() if k != 'cen'}
    with _quiet():
        newbasis = np.asarray(OneBasis(AT, **argvar).basis) - np.asarray(OneBasis(np.array([CEN]), **argvar).basis)
    assert_close(newbasis, r_field(ref, 'basis'), rtol=RTOL, what='centred reduced basis')
    coef, vcov = py_field(red, 'coefficients'), py_field(red, 'vcov')
    assert_close(newbasis @ coef, r_field(ref, 'fit'), rtol=RTOL, what='fit')
    se = np.sqrt(np.maximum(0, np.einsum('ij,jk,ik->i', newbasis, vcov, newbasis)))
    assert_close(se, r_field(ref, 'se'), rtol=RTOL, what='se')


def test_overall_default_prediction_grid_matches_r():
    """at omitted: R mkat() gives pretty(range, n=50) inside the observed range; fit/se follow on that grid."""
    ref = r_reduce('crapi_dflt', cen=CEN)
    red = py_reduce(cen=CEN)
    assert_fields(red, ref, ('predvar', 'fit', 'se', 'RRfit'), what='default grid')


def test_var_reduction_matches_r():
    """type='var': lag-response at a fixed exposure value; M = (onebasis(value) - onebasis(cen)) (x) I."""
    ref = r_reduce('crapi_var', typ='var', value=10.0, cen=CEN)
    red = py_reduce('var', value=10.0, cen=CEN)
    assert_fields(red, ref, ('coefficients', 'vcov', 'basis', 'fit', 'se') + RR_FIELDS, what="type='var'")
    assert_close(py_field(red, 'lag'), r_field(ref, 'lag'), what='lag')
    assert py_field(red, 'fit').size == 22 and not r_has(ref, 'predvar')     # one value per lag 0..21
    assert str(py_field_type(red)) == 'var'


def test_lag_reduction_matches_r():
    """type='lag': exposure-response at a fixed lag; M = I (x) onebasis_lag(value)."""
    ref = r_reduce('crapi_lag', typ='lag', value=3.0, cen=CEN, at=AT)
    red = py_reduce('lag', value=3.0, cen=CEN, at=AT)
    assert_fields(red, ref, OVERALL_POINT + RR_FIELDS, what="type='lag'")
    assert str(py_field_type(red)) == 'lag'


@pytest.mark.parametrize('typ, value, lag, bylag', [
    ('overall', None, (2, 10), None),
    ('var', 10.0, (0, 10), 2),
    ('var', 10.0, (3, 12), 3),
], ids=['overall_lag2_10', 'var_lag0_10_by2', 'var_lag3_12_by3'])
def test_lag_subperiod_and_bylag_match_r(typ, value, lag, bylag):
    """R: overall sums the lag basis over seqlag(lag); type='var' evaluates the lag basis at seqlag(lag, bylag)."""
    ref = r_reduce('crapi_sub', typ=typ, value=value, lag=lag, bylag=bylag, cen=CEN, at=AT)
    red = py_reduce(None if typ == 'overall' else typ, **{k: v for k, v in
                    dict(value=value, lag=list(lag), bylag=bylag, cen=CEN, at=AT).items() if v is not None})
    fields = ['coefficients', 'vcov', 'basis', 'fit', 'se', 'lag'] + (['bylag'] if bylag else [])
    assert_fields(red, ref, fields, what=f'{typ} lag={lag} bylag={bylag}')


@pytest.mark.parametrize('link, ci_level, fields, absent', [
    ('log', 0.90, ('RRfit', 'RRlow', 'RRhigh'), ()),
    ('logit', 0.99, ('RRfit', 'RRlow', 'RRhigh'), ()),
    ('identity', 0.95, ('low', 'high'), ('RRfit', 'RRlow', 'RRhigh')),
], ids=['log_90', 'logit_99', 'identity_95'])
def test_link_and_ci_level_select_the_confidence_interval_like_r(link, ci_level, fields, absent):
    """R: exp() of fit -/+ qnorm(1-(1-ci.level)/2) se for the log/logit links, plain low/high otherwise (an RR of an
    identity-link fit would be meaningless, so R reports none)."""
    ref = r_reduce('crapi_ci', at=AT, cen=CEN, link=link, ci_level=ci_level)
    red = py_reduce(at=AT, cen=CEN, link=link, ci_level=ci_level)
    assert_fields(red, ref, ('fit', 'se') + fields, what=f'link={link} ci={ci_level}')
    for f in absent:
        assert not r_has(ref, f) and not hasattr(red, f), f'{f} must not be reported for link {link} (R does not)'
    assert np.isclose(float(py_field(red, 'ci.level')[0]), ci_level, rtol=0, atol=1e-15)


def _documented(func):
    """(names of the documented parameters, {name: documentation text}) of a numpydoc docstring."""
    doc = inspect.cleandoc(func.__doc__ or '')
    m = re.search(r'^Parameters\n-+\n(.*?)(?:^\w[^\n]*\n-+\n|\Z)', doc, re.S | re.M)
    body = m.group(1) if m else ''
    entries = re.split(r'^(?=\w+ : )', body, flags=re.M)
    out = {}
    for e in entries:
        mm = re.match(r'(\w+) : ', e)
        if mm:
            out[mm.group(1)] = e
    return list(out), out


def test_documented_parameters_exist_in_the_signature():
    """Either fix of the finding (implement `type` or correct the docstring) makes the documentation and the signature agree."""
    from crossreduce import crossreduce
    names, _ = _documented(crossreduce)          # no Parameters section = nothing documented wrongly
    params = set(inspect.signature(crossreduce).parameters)
    missing = [n for n in names if n not in params]
    assert not missing, f'documented but not accepted: {missing}; signature: {sorted(params)}'


def test_every_documented_reduction_type_is_implemented():
    """Whatever reduction types the docstring names must actually run (R: overall, var, lag)."""
    from crossreduce import crossreduce
    _, docs = _documented(crossreduce)
    text = docs.get('type') or docs.get('reduction_type') or ''
    types = [t for t in dict.fromkeys(re.findall(r'["\']([a-z]+)["\']', text)) if t in ('overall', 'var', 'lag')]
    has_value = 'value' in inspect.signature(crossreduce).parameters
    for t in types:
        extra = {'value': {'var': 10.0, 'lag': 3.0}[t]} if (t != 'overall' and has_value) else {}
        red = py_reduce(t, cen=CEN, **extra)
        assert red is not None, t


# ==============================================================================================================
# Theme T / crossreduce-6: the centring value (R mkcen)
# ==============================================================================================================
SERIES = {'chicago': slice(None), 'summer': slice(150, 250), 'mild': slice(3000, 3100)}   # R cen: 5, 22.5, 15
# (R text, Python dict, quantile probabilities of the knots / breaks / threshold; None = no data-derived argument)
_CEN_FUNS = {
    'bs2': ('fun="bs", degree=2', {'fun': 'bs', 'degree': 2}, ('knots', [.25, .50, .90])),
    'ns': ('fun="ns"', {'fun': 'ns'}, ('knots', [.25, .50, .90])),
    'thr': ('fun="thr"', {'fun': 'thr'}, ('thr_value', [.50])),
    'lin': ('fun="lin"', {'fun': 'lin'}, None),
    'strata': ('fun="strata"', {'fun': 'strata'}, ('breaks', [.33, .66])),
}


def _cen_case(fun, series, cen):
    """(R's $cen or None, PyDLNM CrossReduce) for a lag-3 cross-basis of `fun` (arguments at data quantiles)."""
    from basis import CrossBasis
    from crossreduce import crossreduce
    from utils import logknots
    x = chicago()['temp'][SERIES[series]]
    rtxt, argvar, derived = _CEN_FUNS[fun]
    argvar = dict(argvar)
    if derived is not None:
        key, probs = derived
        vals = _rquantile(x, probs)
        argvar[key] = vals if len(vals) > 1 or key != 'thr_value' else float(vals[0])
        np2r('crapi_cenarg', vals)
        rtxt += f', {key.replace("_", ".")}=crapi_cenarg'       # PyDLNM thr_value = R thr.value
    np2r('crapi_cenx', x)
    r(f'crapi_cencb <- crossbasis(crapi_cenx, lag=3, argvar=list({rtxt}), arglag=list(fun="ns", knots=logknots(3, 1)))')
    p = int(rget('ncol(crapi_cencb)')[0])
    cen_txt = '' if cen is None else f', cen={_rnum(cen)}'
    r(f'crapi_cenred <- suppressMessages(crossreduce(crapi_cencb, coef=rep(.01, {p}), vcov=diag({p}) * 1e-3,'
      f' model.link="log"{cen_txt}))')
    ref = None if bool(r('is.null(crapi_cenred[["cen"]])')[0]) else float(rget('crapi_cenred[["cen"]]')[0])
    with _quiet():
        cb = CrossBasis(x, lag=3, argvar=argvar, arglag={'fun': 'ns', 'knots': logknots([0, 3], nk=1)})
    assert int(cb.shape[1]) == p
    params = inspect.signature(crossreduce).parameters
    kw = {'model_link': 'log'} if 'model_link' in params else {}
    with _quiet():
        red = crossreduce(cb, coef=np.full(p, .01), vcov=np.eye(p) * 1e-3, cen=cen, **kw)
    return ref, red


def _assert_cen(ref, red, what):
    got = red.cen
    if ref is None:
        assert got is None, f'{what}: R reports no centring value, PyDLNM cen = {got!r}'
    else:
        assert got is not None and not isinstance(got, (bool, np.bool_)) and abs(float(got) - ref) <= 1e-12, \
            f'{what}: R resolved cen = {ref}, PyDLNM cen = {got!r}'


@pytest.mark.parametrize('series', list(SERIES))
@pytest.mark.parametrize('fun, cen', [('bs2', None), ('ns', None), ('ns', True), ('ns', False), ('thr', True),
                                      ('lin', True), ('strata', False)],
                         ids=['bs2-unset', 'ns-unset', 'ns-True', 'ns-False', 'thr-True', 'lin-True', 'strata-False'])
def test_resolved_centring_value_matches_r_mkcen(fun, cen, series):
    """cen unset/TRUE -> median(pretty(range)) for bs/ns/poly, FALSE -> NULL; logical cen is dropped for thr/lin/strata."""
    ref, red = _cen_case(fun, series, cen)
    _assert_cen(ref, red, f'{fun} cen={cen} on {series}')


@pytest.mark.parametrize('series', list(SERIES))
@pytest.mark.parametrize('fun', ['thr', 'lin', 'strata'])
def test_functions_without_default_centring_keep_cen_none(fun, series):
    """R mkcen: thr/strata/integer/lin get no automatic centring value (cen stays NULL / None)."""
    ref, red = _cen_case(fun, series, None)
    assert ref is None
    _assert_cen(ref, red, f'{fun} on {series}')


def test_unset_cen_centres_the_reduced_fit_at_the_automatic_value():
    """R centres the Chicago bs reduction at mkcen() = median(pretty(range)) when cen is omitted and reports it."""
    ref = r_reduce('crapi_auto', at=AT)
    cen_r = float(r_field(ref, 'cen')[0])
    assert cen_r in AT                                   # premise: the automatic value is one of the grid points
    require_outputs(py_reduce(), 'fit', 'se', 'basis', 'predvar', 'RRfit')
    red = py_reduce(at=AT)
    assert_fields(red, ref, ('fit', 'se', 'RRfit', 'basis'), what='automatic cen')
    i = int(np.flatnonzero(AT == cen_r)[0])
    assert py_field(red, 'fit')[i] == 0.0 and py_field(red, 'RRfit')[i] == 1.0
    assert float(red.cen) == cen_r


@pytest.mark.parametrize('cen', [None, 10.0, 20.0, 32.0], ids=['unset', '10', '20', '32'])
def test_reduced_coef_and_vcov_do_not_depend_on_cen_and_explicit_cen_is_reported(cen):
    """R: coefficients/vcov of crossreduce are identical for every cen (only basis/fit/se change); an explicit numeric
    cen is reported back unchanged."""
    ref = r_reduce('crapi_cenind', at=AT, cen=cen)
    red = py_reduce(cen=cen)
    assert_close(py_field(red, 'coefficients'), r_field(ref, 'coefficients'), rtol=1e-13, what='reduced coef')
    assert_close(py_field(red, 'vcov'), r_field(ref, 'vcov'), rtol=1e-13, what='reduced vcov')
    base = py_reduce(cen=None)
    assert np.array_equal(py_field(red, 'coefficients'), py_field(base, 'coefficients'))
    assert np.array_equal(py_field(red, 'vcov'), py_field(base, 'vcov'))
    if cen is not None:
        assert float(red.cen) == cen


# ==============================================================================================================
# Theme T / centering-19: everything is centred at cen
# ==============================================================================================================
@pytest.mark.parametrize('cen', [10.0, 18.5], ids=['cen_on_grid', 'cen_off_grid'])
def test_reduced_fit_se_and_rr_are_centred_at_cen_like_r(cen):
    ref = r_reduce('crapi_c19', at=AT, cen=cen)
    require_outputs(py_reduce(cen=cen), 'fit', 'se', 'RRfit', 'RRlow', 'RRhigh')
    red = py_reduce(at=AT, cen=cen)
    assert_fields(red, ref, ('fit', 'se') + RR_FIELDS, what=f'cen={cen}')
    if cen in AT:                               # the reference point itself: log-RR 0, RR 1, no uncertainty
        i = int(np.flatnonzero(AT == cen)[0])
        assert py_field(red, 'fit')[i] == 0.0 and py_field(red, 'se')[i] == 0.0
        assert py_field(red, 'RRfit')[i] == 1.0 and py_field(red, 'RRlow')[i] == 1.0 == py_field(red, 'RRhigh')[i]


@pytest.mark.parametrize('cen', [10.0, 18.5], ids=['cen_on_grid', 'cen_off_grid'])
def test_centred_reduced_basis_matches_r(cen):
    """R: basis = scale(onebasis(at, argvar), center = onebasis(cen, argvar), scale = FALSE); its row at cen is zero."""
    ref = r_reduce('crapi_c19b', at=AT, cen=cen)
    require_outputs(py_reduce(cen=cen), 'basis', 'predvar')
    red = py_reduce(at=AT, cen=cen)
    basis = py_field(red, 'basis')
    assert_close(basis, r_field(ref, 'basis'), rtol=RTOL, what='centred reduced basis')
    assert_close(py_field(red, 'predvar'), AT, rtol=0, what='predvar')
    if cen in AT:
        assert np.all(basis[int(np.flatnonzero(AT == cen)[0])] == 0.0)


def test_changing_cen_shifts_fit_by_a_constant_and_keeps_se_of_the_difference():
    """fit(cen=a) - fit(cen=b) = (onebasis(b) - onebasis(a)) newcoef is the same at every exposure value (R identity)."""
    a, b = 10.0, 20.0
    ra, rb = r_reduce('crapi_cshift_a', at=AT, cen=a), r_reduce('crapi_cshift_b', at=AT, cen=b)
    shift_r = r_field(ra, 'fit') - r_field(rb, 'fit')
    assert np.ptp(shift_r) < 1e-12
    require_outputs(py_reduce(cen=a), 'fit')
    pa, pb = py_reduce(at=AT, cen=a), py_reduce(at=AT, cen=b)
    shift_py = py_field(pa, 'fit') - py_field(pb, 'fit')
    assert_close(shift_py, shift_r, rtol=1e-9, what='fit(a) - fit(b)')


# ==============================================================================================================
# Theme S3 / validation-audit-19: API parity gaps versus the dlnm NAMESPACE
# ==============================================================================================================
def _crosspred_pair():
    """R and PyDLNM crosspred on the design's coef/vcov (full cross-basis coefficients)."""
    from prediction import crosspred
    d = design()
    r(f'crapi_pred <- crosspred(crapi_cb, coef=crapi_coef, vcov=crapi_vcov, model.link="log", at=crapi_at2, cen={CEN})')
    with _quiet():
        pred = crosspred(d.cb, coef=d.coef, vcov=d.vcov, model_link='log', at=AT, cen=CEN)
    return pred


np2r('crapi_at2', AT)


def test_coef_of_a_crosspred_returns_its_coefficients_like_r():
    """R coef.crosspred(object) returns object$coef: the coefficients the prediction was made with."""
    from crossreduce import coef
    pred = _crosspred_pair()
    ref = rget('as.numeric(coef(crapi_pred))')
    assert_close(np.asarray(coef(pred), dtype=float), ref, rtol=1e-13, what='coef(crosspred)')
    assert_close(ref, design().coef, rtol=1e-13, what='R coef(crosspred) == input')


def test_vcov_of_a_crosspred_returns_its_vcov_like_r():
    """R vcov.crosspred(object) returns object$vcov; PyDLNM's vcov() works because the attribute is also `vcov`."""
    from crossreduce import vcov
    pred = _crosspred_pair()
    ref = rget('unname(vcov(crapi_pred))')
    assert_close(np.asarray(vcov(pred), dtype=float), ref, rtol=1e-13, what='vcov(crosspred)')


def test_coef_and_vcov_of_a_crossreduce_match_r():
    """R coef.crossreduce / vcov.crossreduce: the reduced coefficients and their covariance (faithful today)."""
    from crossreduce import coef, vcov
    ref = r_reduce('crapi_cv', at=AT, cen=CEN)
    red = py_reduce(cen=CEN)
    assert_close(np.asarray(coef(red), dtype=float), r_field(ref, 'coefficients'), rtol=1e-13, what='coef(crossreduce)')
    assert_close(np.asarray(vcov(red), dtype=float), r_field(ref, 'vcov'), rtol=1e-13,
                 what='vcov(crossreduce)')


def test_crossreduce_has_a_summary():
    """R summary.crossreduce prints the reduction type, the reduced df and the centring value; OneBasis, CrossBasis and
    CrossPred have a summary() in PyDLNM, CrossReduce does not."""
    red = py_reduce(cen=CEN)
    assert callable(getattr(red, 'summary', None)), 'CrossReduce has no summary()'
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = red.summary()
    text = (out if isinstance(out, str) else '') + buf.getvalue()
    assert 'overall' in text.lower(), text
    assert re.search(rf'(?<![\w.]){design().nvar}(?![\w.])', text) and '15' in text, \
        f'reduced df / centring value missing: {text!r}'


def _bases_inputs():
    x = chicago()['temp'][:300]
    kn = _rquantile(x, [.25, .75])
    np2r('crapi_bx', x)
    np2r('crapi_bkn', kn)
    return x, kn


def _matrix(obj):
    return np.asarray(getattr(obj, 'basis', obj), dtype=float)


def test_onebasis_class_matches_r_onebasis():
    """PyDLNM's OneBasis(x, fun, ...) equals R's onebasis(x, fun, ...) (bs degree 2 and ns with explicit knots)."""
    from basis import OneBasis
    x, kn = _bases_inputs()
    for rtxt, kw in [('fun="bs", degree=2, knots=crapi_bkn', dict(fun='bs', degree=2, knots=kn)),
                     ('fun="ns", knots=crapi_bkn', dict(fun='ns', knots=kn))]:
        ref = rget(f'matrix(as.numeric(onebasis(crapi_bx, {rtxt})), nrow=length(crapi_bx))')
        with _quiet():
            ob = OneBasis(x, **kw)
        assert_close(_matrix(ob), ref, rtol=1e-12, what=f'OneBasis {kw["fun"]}')


def test_onebasis_function_exists_and_matches_r():
    """R exports onebasis(x, fun, ...); the port offers only the class, so `from basis import onebasis` fails."""
    from basis import onebasis
    x, kn = _bases_inputs()
    ref = rget('matrix(as.numeric(onebasis(crapi_bx, fun="bs", degree=2, knots=crapi_bkn)),'
               ' nrow=length(crapi_bx))')
    with _quiet():
        ob = onebasis(x, fun='bs', degree=2, knots=kn)
    assert_close(_matrix(ob), ref, rtol=1e-12, what='onebasis()')


def _cross_inputs():
    from utils import logknots
    x, kn = _bases_inputs()
    lk = logknots([0, 5], nk=2)
    np2r('crapi_blk', lk)
    return x, kn, lk


def test_crossbasis_class_matches_r_crossbasis():
    from basis import CrossBasis
    x, kn, lk = _cross_inputs()
    ref = rget('unclass(crossbasis(crapi_bx, lag=5, argvar=list(fun="ns", knots=crapi_bkn),'
               ' arglag=list(fun="ns", knots=crapi_blk)))')
    with _quiet():
        cb = CrossBasis(x, lag=5, argvar={'fun': 'ns', 'knots': kn}, arglag={'fun': 'ns', 'knots': lk})
    assert_close(_matrix(cb), ref, rtol=1e-12, what='CrossBasis')


def test_crossbasis_function_exists_and_matches_r():
    """R exports crossbasis(x, lag, argvar, arglag); the port offers only the class."""
    from basis import crossbasis
    x, kn, lk = _cross_inputs()
    ref = rget('unclass(crossbasis(crapi_bx, lag=5, argvar=list(fun="ns", knots=crapi_bkn),'
               ' arglag=list(fun="ns", knots=crapi_blk)))')
    with _quiet():
        cb = crossbasis(x, lag=5, argvar={'fun': 'ns', 'knots': kn}, arglag={'fun': 'ns', 'knots': lk})
    assert_close(_matrix(cb), ref, rtol=1e-12, what='crossbasis()')


def _init_public_api():
    """(names in __all__, names bound by imports) of the package __init__.py, tolerant of the known SyntaxError
    (audit Q1, covered by test_packaging_P1_Q) so that this test isolates the export question."""
    lines = (SRC / '__init__.py').read_text().splitlines()
    tree = None
    for _ in range(10):
        try:
            tree = ast.parse('\n'.join(lines))
            break
        except SyntaxError as exc:
            lines[exc.lineno - 1] = 'pass'
    assert tree is not None, '__init__.py could not be parsed'
    exported, bound = [], set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            bound.update(a.asname or a.name for a in node.names)
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == '__all__':
                    exported = list(ast.literal_eval(node.value))
    return exported, bound


def test_logknots_and_equalknots_are_exported_by_the_package():
    """R exports logknots() and equalknots() (NAMESPACE line 2); PyDLNM's __init__ exports only mklag, seqlag, exphist."""
    exported, bound = _init_public_api()
    missing = [n for n in ('logknots', 'equalknots') if n not in exported or n not in bound]
    assert not missing, f'not exported by the package: {missing}'


def test_utils_helpers_the_package_exports_today_stay_exported():
    exported, bound = _init_public_api()
    for n in ('mklag', 'seqlag', 'exphist', 'crossreduce', 'coef', 'vcov', 'crosspred', 'OneBasis', 'CrossBasis'):
        assert n in exported and n in bound, n


def test_utils_logknots_seqlag_mklag_match_r():
    """The exported helpers' numbers (logknots 21/3, seqlag with a dividing step, mklag of a scalar and a pair)."""
    from utils import logknots, mklag, seqlag
    assert_close(logknots([0, 21], nk=3), rget('logknots(21, 3)'), rtol=1e-12, what='logknots(21,3)')
    assert_close(logknots([2, 30], nk=2), rget('logknots(c(2, 30), 2)'), rtol=1e-12, what='logknots(c(2,30),2)')
    assert_close(seqlag([0, 10], 2), rget('dlnm:::seqlag(c(0, 10), 2)'), rtol=0, what='seqlag by 2')
    assert_close(seqlag([3, 12]), rget('dlnm:::seqlag(c(3, 12))'), rtol=0, what='seqlag')
    assert_close(mklag(7), rget('dlnm:::mklag(7)'), rtol=0, what='mklag(7)')
    assert_close(mklag([2, 9]), rget('dlnm:::mklag(c(2, 9))'), rtol=0, what='mklag(c(2,9))')


def _history_case():
    from basis import CrossBasis
    from utils import logknots
    x = chicago()['temp'][:600]
    kn = _rquantile(x, [.25, .75])
    lk = logknots([0, 4], nk=2)
    with _quiet():
        cb = CrossBasis(x, lag=4, argvar={'fun': 'ns', 'knots': kn}, arglag={'fun': 'ns', 'knots': lk})
    p = int(cb.shape[1])
    rng = np.random.default_rng(7)
    coef = rng.normal(0, .03, p)
    A = rng.normal(0, .02, (p, p))
    vcov = A @ A.T + 1e-4 * np.eye(p)
    H = np.vstack([np.full(5, 10.0), np.linspace(5.0, 25.0, 5), np.full(5, 20.0)])       # 3 histories, lags 0..4
    np2r('crapi_hx', x)
    np2r('crapi_hkn', kn)
    np2r('crapi_hlk', lk)
    np2r('crapi_hcoef', coef)
    np2r('crapi_hvcov', vcov)
    np2r('crapi_H', H)
    r(f'crapi_hvcov <- matrix(crapi_hvcov, {p}, {p})')
    r('crapi_hcb <- crossbasis(crapi_hx, lag=4, argvar=list(fun="ns", knots=crapi_hkn),'
      ' arglag=list(fun="ns", knots=crapi_hlk))')
    r('crapi_hpred <- crosspred(crapi_hcb, coef=crapi_hcoef, vcov=crapi_hvcov, model.link="log", at=crapi_H, cen=10)')
    return cb, coef, vcov, H


def test_crosspred_exposure_history_matrix_is_honoured_or_rejected():
    """R crosspred(at = matrix with diff(lag)+1 columns) predicts the effect of each exposure HISTORY.  PyDLNM accepts
    the matrix without complaint and returns the prediction for the first column held constant over all lags (a wrong
    number).  Acceptable: R's numbers, or a clear error for the unimplemented input."""
    from prediction import crosspred
    cb, coef, vcov, H = _history_case()
    ref_fit, ref_se = rget('as.numeric(crapi_hpred$allfit)'), rget('as.numeric(crapi_hpred$allse)')
    assert ref_fit.size == 3
    try:
        with _quiet():
            pred = crosspred(cb, coef=coef, vcov=vcov, model_link='log', at=H, cen=10.0)
    except (ValueError, NotImplementedError, TypeError):
        return
    assert_close(np.asarray(pred.allfit, dtype=float), ref_fit, rtol=1e-9, what='allfit of exposure histories')
    assert_close(np.asarray(pred.allse, dtype=float), ref_se, rtol=1e-9, what='allse of exposure histories')


def test_crosspred_vector_at_equals_constant_exposure_history_in_r():
    """Anchor for the test above: a constant history c(v, v, v, v, v) is the same prediction as the scalar exposure v held
    over all lags (R, and PyDLNM's vector `at`), so the two routes must agree once histories are implemented."""
    from prediction import crosspred
    cb, coef, vcov, H = _history_case()
    r('crapi_hconst <- crosspred(crapi_hcb, coef=crapi_hcoef, vcov=crapi_hvcov, model.link="log",'
      ' at=rbind(rep(10, 5), rep(20, 5)), cen=10)')
    r('crapi_hvec <- crosspred(crapi_hcb, coef=crapi_hcoef, vcov=crapi_hvcov, model.link="log", at=c(10, 20), cen=10)')
    assert_close(rget('as.numeric(crapi_hconst$allfit)'), rget('as.numeric(crapi_hvec$allfit)'), rtol=1e-12,
                 what='R constant history vs vector at')
    with _quiet():
        pred = crosspred(cb, coef=coef, vcov=vcov, model_link='log', at=np.array([10.0, 20.0]), cen=10.0)
    assert_close(np.asarray(pred.allfit, dtype=float), rget('as.numeric(crapi_hvec$allfit)'), rtol=1e-12,
                 what='PyDLNM vector at')


# ==============================================================================================================
# Notes: validation-audit-19 sub-items that are NOT tested in this module
#   * summary/plot/lines/points: plot.* / lines.* / points.* (R graphics; no numeric reference, a port may legitimately
#     scope them out; the finding's alternative fix is to state the supported subset).
#   * ps, cr, cbPen, smooth.construct.cb.smooth.spec, Predict.matrix.cb.smooth, crosspred('name') for a GAM: mgcv-based,
#     penalized.py is a different implementation (seasonal-penalized-* findings).
#   * datasets chicagoNMMAPS / drug / nested and the broken 'from  import data': test_packaging_P1_Q.py.
#   * OneBasis(fun='integer'): test_basis_S1_discrete.py (basis-discrete-14 / crossbasis-20).
#   * CrossBasis(group=...) and matrix x: test_crossbasis_D_E_F.py (theme E).
# ==============================================================================================================
