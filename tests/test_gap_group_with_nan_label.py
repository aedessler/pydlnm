"""Gap group_with_nan_label (completeness critic, minor): CrossBasis(group=...) with a missing group label.

R's crossbasis(x, lag, group = g) accepts NA labels: checkgroup() measures the length of the real groups only
(``tapply`` leaves the NA group out), tsModel::Lag leaves the rows with an NA label as NA, the windows of the other rows
stay inside their own group, and ``attr(, "group")`` is ``length(unique(group))``, which counts the NA label as one more
group. PyDLNM raised "each group must have length > diff(lag)" for float NaN labels (np.unique returns the NaN rows as
a tiny group of their own) and a TypeError for None labels in an object array.

R computes the reference at run time.
"""
import os

import numpy as np
import pandas as pd
import pytest

from rhelpers import assert_close, np2r, r, r2np

_R_HOME = os.path.dirname(str(r('.Library')[0]))


@pytest.fixture(autouse=True)
def _r_home_guard():
    os.environ['R_HOME'] = _R_HOME
    yield
    os.environ['R_HOME'] = _R_HOME


LAG = 3
N = 60
BAD = [4, 44]                      # rows (0-based) with a missing label


def _data():
    x = np.random.default_rng(1).normal(10, 4, N)
    g = np.repeat([1., 2., 3.], 20)
    g[BAD] = np.nan
    return x, g


def _r_reference(x, g, fun):
    np2r('gn_x', x)
    np2r('gn_g', g)
    r('gn_x <- as.numeric(gn_x); gn_g <- as.numeric(gn_g); gn_g[is.nan(gn_g)] <- NA')
    arg = 'list(fun="lin")' if fun == 'lin' else 'list(fun="bs", degree=2, knots=quantile(gn_x, c(.25, .5, .75)))'
    lagarg = 'list(fun="integer")' if fun == 'lin' else 'list(fun="ns", df=3)'
    r(f'gn_cb <- crossbasis(gn_x, lag={LAG}, argvar={arg}, arglag={lagarg}, group=gn_g)')
    return r2np(r('unclass(gn_cb)[, ]')), int(r('attr(gn_cb, "group")')[0])


def _py_argvar(x, fun):
    return ({'fun': 'lin'}, {'fun': 'integer'}) if fun == 'lin' else \
        ({'fun': 'bs', 'degree': 2, 'knots': np.quantile(x, [.25, .5, .75])}, {'fun': 'ns', 'df': 3})


@pytest.mark.parametrize('fun', ['lin', 'bs'])
@pytest.mark.parametrize('kind', ['float_nan', 'object_none', 'pandas_nullable'])
def test_missing_labels_give_na_rows_and_keep_the_windows_inside_the_real_groups(fun, kind):
    from basis import CrossBasis
    x, g = _data()
    if kind == 'float_nan':
        labels = g
    elif kind == 'object_none':
        labels = np.where(np.isnan(g), None, g).astype(object)
    else:
        labels = pd.Series(np.nan_to_num(g).astype(int), dtype='Int64')
        labels.iloc[BAD] = pd.NA
    ref, ref_group = _r_reference(x, g, fun)
    var, lagb = _py_argvar(x, fun)
    cb = CrossBasis(x, lag=LAG, argvar=var, arglag=lagb, group=labels)
    assert_close(np.asarray(cb.basis), ref, rtol=1e-10, what=f'cross-basis with missing group labels ({kind}, {fun})')
    assert cb.group == ref_group == 4, 'R counts the NA label as one more group in attr(, "group")'


def test_the_length_check_still_applies_to_the_real_groups():
    """R: each non-missing group must be longer than diff(lag); the NA rows do not count as a group for that."""
    from basis import CrossBasis
    x = np.random.default_rng(2).normal(size=12)
    g = np.array([1.] * 3 + [2.] * 9)                      # group 1 has 3 rows, diff(lag) = 3
    with pytest.raises(ValueError, match='each group must have length'):
        CrossBasis(x, lag=3, argvar={'fun': 'lin'}, arglag={'fun': 'integer'}, group=g)
    g2 = np.array([1.] * 4 + [2.] * 6 + [np.nan] * 2)      # the NA rows (2 < 3) must not trigger the error
    cb = CrossBasis(x, lag=3, argvar={'fun': 'lin'}, arglag={'fun': 'integer'}, group=g2)
    assert np.isnan(np.asarray(cb.basis)[-2:]).all()
