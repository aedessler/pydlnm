"""Gap recenter_basis_drops_group (completeness critic, minor): centering.recenter_basis (and so
CenteringManager.recenter_at_mmt / recenter_at_value) rebuilt the cross-basis from x, lag, argvar and arglag but forgot
``group``. For stacked series (panel data) the new basis then built its lags across the group boundaries, so a basis
that only had a centering value changed came out with different numbers.

In R the centering value is metadata of ``argvar`` (``mkcen``); it does not change the cross-basis matrix, which keeps its
``group`` handling. The reference is R's crossbasis(..., group = g) with and without ``cen`` in argvar.
R computes the reference at run time.
"""
import os

import numpy as np
import pytest

from rhelpers import assert_close, chicago, np2r, r, r2np

_R_HOME = os.path.dirname(str(r('.Library')[0]))


@pytest.fixture(autouse=True)
def _r_home_guard():
    os.environ['R_HOME'] = _R_HOME
    yield
    os.environ['R_HOME'] = _R_HOME


GROUPS = np.repeat([1., 2., 3.], 100)
LAG = 4


@pytest.fixture(scope='module')
def grouped():
    from basis import CrossBasis
    x = chicago()['temp'][:300]
    knots = np.quantile(x, [.25, .5, .75])
    cb = CrossBasis(x, lag=LAG, argvar={'fun': 'bs', 'degree': 2, 'knots': knots}, arglag={'fun': 'ns', 'df': 3},
                    group=GROUPS)
    np2r('rg_x', x)
    np2r('rg_knots', knots)
    np2r('rg_g', GROUPS)
    r('rg_x <- as.numeric(rg_x); rg_knots <- as.numeric(rg_knots); rg_g <- as.numeric(rg_g)')
    ref = r2np(r(f'unclass(crossbasis(rg_x, lag={LAG}, argvar=list(fun="bs", degree=2, knots=rg_knots, cen=15), '
                 f'arglag=list(fun="ns", df=3), group=rg_g))[, ]'))
    return cb, ref


def test_the_grouped_basis_itself_matches_r(grouped):
    """Guard (true before and after the fix)."""
    cb, ref = grouped
    assert_close(np.asarray(cb.basis), ref, rtol=1e-10, what='grouped cross-basis')


def test_recenter_basis_keeps_the_group_structure(grouped):
    from centering import recenter_basis
    cb, ref = grouped
    new, info = recenter_basis(cb, None, cen=15.0)
    assert info['value'] == 15.0 and new.argvar['cen'] == 15.0
    assert new.group == cb.group == 3
    assert_close(np.asarray(new.basis), ref, rtol=1e-10, what='re-centred grouped cross-basis vs R group=')
    assert_close(np.asarray(new.basis), np.asarray(cb.basis), rtol=1e-14, what='centering is metadata, not a new basis')


def test_centering_manager_recentre_keeps_the_group_structure(grouped):
    from centering import CenteringManager
    cb, ref = grouped
    new, _ = CenteringManager(cb, None).recenter_at_value(15.0)
    assert new.group == 3
    assert_close(np.asarray(new.basis), ref, rtol=1e-10, what='CenteringManager.recenter_at_value vs R group=')
