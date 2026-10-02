"""Gap attrdl_sub_splices_series (completeness critic, minor): attrdl(sub=...) cut the series to the kept rows BEFORE
building the lag windows, so for a non-contiguous subset (e.g. the summers of several years) the lag windows jumped
across the gaps, and at the start of a contiguous block the first rows lost their history.

R's attrdl has no ``sub`` argument; the R-faithful meaning (the one the Europe-2022 script uses for its sub-periods) is:
the lagged exposures and the forward moving average of the cases come from the FULL series, and ``sub`` only selects
the rows that are attributed. The reference below is therefore R's own attrdl on the whole series (tot=FALSE),
restricted to the rows of ``sub``, and R's own total formulas applied to those rows:

    af = sum(an[ok]) / sum(cases_used[ok]),   an_total = af * sum(cases[sub])      (ok = sub rows with a complete window)

R computes the reference at run time.
"""
import os

import numpy as np
import pytest

from rhelpers import assert_close, chicago, np2r, r, rget
from test_attr_algorithm_K import Scenario, _load_r_attrdl

_R_HOME = os.path.dirname(str(r('.Library')[0]))


@pytest.fixture(autouse=True)
def _r_home_guard():
    os.environ['R_HOME'] = _R_HOME
    yield
    os.environ['R_HOME'] = _R_HOME


N, LAG, CEN = 700, 6, 18.0


@pytest.fixture(scope='module')
def scn():
    ch = chicago()
    x, cases = ch['temp'][:N], ch['death'][:N].astype(float)
    sc = Scenario(x, cases, LAG, {'fun': 'bs', 'degree': 2, 'knots': np.quantile(x, [.25, .5, .75])},
                  {'fun': 'ns', 'df': 3})
    sc.cen = CEN
    return sc


def _subsets():
    """A contiguous block and a non-contiguous set (two separated 'seasons'): lag windows of the latter cross gaps."""
    block = np.zeros(N, bool)
    block[200:420] = True
    seasons = np.zeros(N, bool)
    seasons[60:140] = True
    seasons[330:410] = True
    seasons[600:690] = True
    return {'block': block, 'seasons': seasons}


def _r_per_obs(sc, dir):
    return sc.r_attrdl(type='an', dir=dir, tot=False, cen=CEN)


def _r_sub_total(sc, sub, dir):
    """R's total AF / AN formulas on the rows of `sub`, with the lag windows of the full series."""
    sc.push()
    _load_r_attrdl()
    np2r('aS_sub', sub.astype(float))
    r('aS_sub <- as.logical(as.numeric(aS_sub))')
    r(f'aS_an <- aK_env$attrdl(aK_x, aK_cb, aK_cases, coef=aK_coef, vcov=aK_vcov, type="an", dir="{dir}", '
      f'tot=FALSE, cen={CEN!r})')
    r('aS_lag <- attr(aK_cb, "lag")')
    r(f'aS_cu <- if ("{dir}" == "forw") rowMeans(as.matrix(tsModel:::Lag(aK_cases, -seq(aS_lag[1], aS_lag[2])))) '
      f'else aK_cases')
    r('aS_ok <- aS_sub & !is.na(aS_an)')
    r('aS_af <- sum(aS_an[aS_ok]) / sum(aS_cu[aS_ok]); aS_ant <- aS_af * sum(aK_cases[aS_sub], na.rm = TRUE)')
    return float(rget('aS_ant')[0]), float(rget('aS_af')[0])


@pytest.mark.parametrize('dir', ['back', 'forw'])
@pytest.mark.parametrize('which', ['block', 'seasons'])
def test_per_observation_values_use_the_lag_windows_of_the_full_series(scn, which, dir):
    sub = _subsets()[which]
    res = scn.py_attrdl(cen=CEN, type='an', dir=dir, tot=False, sub=sub)
    ref = _r_per_obs(scn, dir)[sub]
    assert_close(res['an'], ref, rtol=1e-10, what=f'per-observation AN, sub={which}, dir={dir}')


@pytest.mark.parametrize('dir', ['back', 'forw'])
@pytest.mark.parametrize('which', ['block', 'seasons'])
def test_totals_apply_r_formulas_to_the_kept_rows(scn, which, dir):
    sub = _subsets()[which]
    res = scn.py_attrdl(cen=CEN, type='an', dir=dir, tot=True, sub=sub)
    an_ref, af_ref = _r_sub_total(scn, sub, dir)
    assert_close(res['af_total'], af_ref, rtol=1e-10, what='total AF of the sub-period')
    assert_close(res['an_total'], an_ref, rtol=1e-10, what='total AN of the sub-period')


@pytest.mark.parametrize('dir', ['back', 'forw'])
def test_all_rows_kept_is_the_unsubset_call(scn, dir):
    """Guard (true before and after the fix): sub = every row changes nothing, and equals R."""
    every = np.ones(N, bool)
    a = scn.py_attrdl(cen=CEN, type='an', dir=dir, tot=True, sub=every)
    b = scn.py_attrdl(cen=CEN, type='an', dir=dir, tot=True)
    assert_close(a['an_total'], b['an_total'], rtol=1e-14, what='sub=all vs no sub')
    ref = scn.r_attrdl(type='an', dir=dir, tot=True, cen=CEN)
    assert_close(a['an_total'], ref[0], rtol=1e-10, what='sub=all vs R attrdl')


def test_sub_must_be_a_logical_vector_of_the_series_length(scn):
    with pytest.raises(ValueError, match='sub'):
        scn.py_attrdl(cen=CEN, type='an', dir='back', tot=True, sub=np.ones(N - 1, bool))
