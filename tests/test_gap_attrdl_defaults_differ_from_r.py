"""Gap attrdl_defaults_differ_from_r (completeness critic, minor): attrdl's default arguments.

R: ``attrdl(x, basis, cases, model, coef, vcov, model.link, type = "af", dir = "back", tot = TRUE, cen, range, sim, nsim)``.
PyDLNM defaulted to ``type = "an"`` and ``dir = "forw"``, so a call translated literally from R (``attrdl(x, cb, cases,
model = m, cen = c)``) gave the forward perspective, not R's backward one (the type only labels the result: it holds both).
The defaults now are R's. The convenience wrappers (attr_heat_cold, attr_by_percentiles, AttributionManager) keep the
FORWARD perspective of the Lancet script by default (they accept ``dir``): the backward perspective cannot be used with
reduced (BLUP) coefficients, which is their typical input, in R either.

R computes the reference at run time (the defaults are read from ``formals()``; the all-defaults call is made in R).
"""
import inspect
import os

import numpy as np
import pytest

from rhelpers import assert_close, chicago, np2r, r, rget
from test_attr_algorithm_K import ImprovedGLMInterface, Scenario, _load_r_attrdl, _quiet

_R_HOME = os.path.dirname(str(r('.Library')[0]))


@pytest.fixture(autouse=True)
def _r_home_guard():
    os.environ['R_HOME'] = _R_HOME
    yield
    os.environ['R_HOME'] = _R_HOME


N, CEN = 600, 18.0


@pytest.fixture(scope='module')
def scn():
    ch = chicago()
    x, cases = ch['temp'][:N], ch['death'][:N].astype(float)
    sc = Scenario(x, cases, 5, {'fun': 'bs', 'degree': 2, 'knots': np.quantile(x, [.25, .5, .75])}, {'fun': 'ns', 'df': 3})
    sc.cen = CEN
    return sc


def test_default_arguments_are_rs():
    import attribution
    _load_r_attrdl()
    params = inspect.signature(attribution.attrdl).parameters
    assert params['type'].default == str(r('formals(aK_env$attrdl)$type')[0]) == 'af'
    assert params['dir'].default == str(r('formals(aK_env$attrdl)$dir')[0]) == 'back'
    assert params['tot'].default is True and bool(r('formals(aK_env$attrdl)$tot')[0])
    assert params['nsim'].default == int(r('formals(aK_env$attrdl)$nsim')[0])


def test_a_call_with_only_the_required_arguments_equals_rs(scn):
    import attribution
    _load_r_attrdl()
    scn.push()
    ref = float(rget(f'aK_env$attrdl(aK_x, aK_cb, aK_cases, coef = aK_coef, vcov = aK_vcov, cen = {CEN!r})')[0])
    res = _quiet(attribution.attrdl, scn.x, scn.cb, scn.cases, model=ImprovedGLMInterface(scn.coef, scn.vcov), cen=CEN)
    assert res['metadata']['direction'] == 'back' and res['metadata']['type'] == 'af'
    assert_close(res['af_total'], ref, rtol=1e-10, what='total AF with all-default arguments (R: af, back)')


def test_default_backward_perspective_is_refused_for_reduced_coefficients_as_in_r(scn):
    """R: dir = "back" (the default) stops for reduced estimates; the forward perspective works."""
    import attribution
    _load_r_attrdl()
    k = scn.cb.basisvar.basis.shape[1]                       # reduced coefficients: one per column of the exposure basis
    coef, vcov = np.linspace(-.05, .08, k), np.eye(k) * 1e-4
    np2r('aD_coef', coef)
    np2r('aD_vcov', vcov)
    scn.push()
    r('aD_coef <- as.numeric(aD_coef)')
    with pytest.raises(Exception, match='forw'):
        r(f'aK_env$attrdl(aK_x, aK_cb, aK_cases, coef = aD_coef, vcov = aD_vcov, cen = {CEN!r})')
    with pytest.raises(ValueError, match='forw'):
        attribution.attrdl(scn.x, scn.cb, scn.cases, coef=coef, vcov=vcov, cen=CEN)
    ref = float(rget(f'aK_env$attrdl(aK_x, aK_cb, aK_cases, coef = aD_coef, vcov = aD_vcov, dir = "forw", cen = {CEN!r})')[0])
    res = attribution.attrdl(scn.x, scn.cb, scn.cases, coef=coef, vcov=vcov, dir='forw', cen=CEN, model=None)
    assert_close(res['af_total'], ref, rtol=1e-10, what='forward AF from reduced coefficients')


def test_wrappers_keep_the_forward_default_so_that_reduced_coefficients_work(scn):
    import attribution
    k = scn.cb.basisvar.basis.shape[1]
    coef, vcov = np.linspace(-.05, .08, k), np.eye(k) * 1e-4
    res = _quiet(attribution.attr_heat_cold, scn.x, scn.cb, scn.cases, coef=coef, vcov=vcov, cen=CEN)
    assert np.isfinite(res['summary']['heat_an_total']) and np.isfinite(res['summary']['cold_an_total'])
    pct = _quiet(attribution.attr_by_percentiles, scn.x, scn.cb, scn.cases, coef=coef, vcov=vcov, cen=CEN)
    assert np.isfinite(pct['summary_table']['an_total']).all()
    mgr = attribution.AttributionManager(scn.x, scn.cb, scn.cases, coef=coef, vcov=vcov)
    assert np.isfinite(_quiet(mgr.total_attribution, cen=CEN)['an_total'])
    # and the perspective can be chosen
    with pytest.raises(ValueError, match='forw'):
        _quiet(attribution.attr_heat_cold, scn.x, scn.cb, scn.cases, coef=coef, vcov=vcov, cen=CEN, dir='back')
