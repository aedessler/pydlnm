"""Gap wrapper_default_mmt_grid (completeness critic, minor): the minimum-risk exposure that attr_heat_cold,
attr_by_percentiles and AttributionManager use when ``cen`` is not given.

R's attrdl() has no default (``'cen' must be provided``). The R scripts of the project that do find a minimum-mortality
value (Gasparrini et al. 2015, 02.secondstage.R) search the PERCENTILE grid ``quantile(x, 1:99/100)`` of the observed
exposure and take ``which.min`` of the overall curve there, which "excludes low and very hot temperature". The wrappers
instead called find_mmt() with its default grid (crosspred's: ``pretty`` points over the whole range, extremes included),
so for a curve that keeps falling towards the hot tail they used the hottest grid point as the counterfactual.

R computes the reference at run time: crosspred(cb, coef, vcov, at = quantile(x, 1:99/100)) and which.min(allfit).
"""
import os

import numpy as np
import pytest

from rhelpers import chicago, np2r, r, rget
from test_attr_algorithm_K import ImprovedGLMInterface, Scenario, _quiet

_R_HOME = os.path.dirname(str(r('.Library')[0]))


@pytest.fixture(autouse=True)
def _r_home_guard():
    os.environ['R_HOME'] = _R_HOME
    yield
    os.environ['R_HOME'] = _R_HOME


N, LAG = 800, 5


@pytest.fixture(scope='module')
def scn():
    """A linear exposure effect with a negative sum over the lags: the overall curve falls monotonically over the whole
    range, so the unrestricted minimum is the hottest exposure and the Lancet-recipe minimum is the 99th percentile."""
    ch = chicago()
    x, cases = ch['temp'][:N], ch['death'][:N].astype(float)
    sc = Scenario(x, cases, LAG, {'fun': 'lin'}, {'fun': 'integer'})
    sc.coef = np.full(LAG + 1, -0.01)
    sc.vcov = np.eye(LAG + 1) * 1e-6
    return sc


@pytest.fixture(scope='module')
def lancet_mmt(scn):
    """R: the minimum of the overall curve over quantile(x, 1:99/100), as 02.secondstage.R does."""
    scn.push()
    r('aK_grid <- unname(quantile(aK_x, 1:99/100, na.rm = TRUE))')
    r('aK_cpm <- crosspred(aK_cb, coef = aK_coef, vcov = aK_vcov, model.link = "log", at = aK_grid, cen = aK_grid[50])')
    # crosspred() sorts and de-duplicates `at` (tied percentiles of a discretised series), so read predvar from the object
    mmt = float(rget('aK_cpm$predvar[which.min(aK_cpm$allfit)]')[0])
    r('aK_cpf <- crosspred(aK_cb, coef = aK_coef, vcov = aK_vcov, model.link = "log", at = seq(min(aK_x), max(aK_x), length = 400), cen = median(aK_x))')
    unrestricted = float(rget('aK_cpf$predvar[which.min(aK_cpf$allfit)]')[0])
    return mmt, unrestricted


def test_premise_the_two_recipes_differ(scn, lancet_mmt):
    mmt, unrestricted = lancet_mmt
    assert unrestricted == pytest.approx(float(np.max(scn.x)))             # the curve falls to the hottest day
    assert mmt == pytest.approx(float(np.quantile(scn.x, .99)))            # the percentile grid stops at the 99th
    assert mmt < unrestricted - 0.5


def test_attr_heat_cold_default_cen_is_the_percentile_grid_minimum(scn, lancet_mmt):
    res = scn.py_call('attr_heat_cold')
    assert res['metadata']['centering'] == pytest.approx(lancet_mmt[0], abs=1e-9)


def test_attr_by_percentiles_default_cen_is_the_percentile_grid_minimum(scn, lancet_mmt):
    res = scn.py_call('attr_by_percentiles')
    assert res['metadata']['centering'] == pytest.approx(lancet_mmt[0], abs=1e-9)


def test_attribution_manager_default_cen_is_the_percentile_grid_minimum(scn, lancet_mmt):
    import attribution
    mgr = attribution.AttributionManager(scn.x, scn.cb, scn.cases, model=ImprovedGLMInterface(scn.coef, scn.vcov))
    assert _quiet(mgr.get_mmt) == pytest.approx(lancet_mmt[0], abs=1e-9)


def test_a_given_or_stored_cen_still_wins(scn):
    """Guard (true before and after the fix): an explicit cen is used as given."""
    res = scn.py_call('attr_heat_cold', cen=18.0)
    assert res['metadata']['centering'] == 18.0
