"""Gap fractional_df_truncated (completeness critic, minor): a non-integer ``df`` was cut with int() by the Python side
before R's own arithmetic could see it.

R keeps the fraction. In ns()/bs()/ps()/cr() it feeds ``seq.int(length.out = ...)``, which rounds up
(df = 4.2 gives the columns of df = 5, df = 3.5 those of df = 4), and in dlnm's strata()
``quantile(x, 1/(df - intercept + 1) * 1:(df - intercept))`` the denominator keeps the fraction while ``:`` truncates the
number of breaks; mgcv's cr() additionally *fails* for some fractions (the knots are made with ceiling(), k with round()).
Whatever R does, including raising an error, is the reference; it is computed at run time.
"""
import os
import warnings

import numpy as np
import pytest

from rhelpers import assert_close, np2r, r, r2np

_R_HOME = os.path.dirname(str(r('.Library')[0]))


@pytest.fixture(autouse=True)
def _r_home_guard():
    os.environ['R_HOME'] = _R_HOME
    yield
    os.environ['R_HOME'] = _R_HOME


X = np.random.default_rng(3).normal(15, 6, 150)
DFS = [3.5, 4.2, 4.9, 5.5, 6.0, 7.3]


@pytest.fixture(scope='module', autouse=True)
def _x_in_r():
    np2r('gf_x', X)
    r('gf_x <- as.numeric(gf_x)')


def _r_basis(fun, df, intercept):
    """R's onebasis(x, fun, df, intercept) as a matrix, or None when R raises an error."""
    r(f'gf_res <- try(suppressWarnings(unclass(onebasis(gf_x, fun="{fun}", df={df!r}, '
      f'intercept={"TRUE" if intercept else "FALSE"}))[, , drop=FALSE]), silent=TRUE)')
    if bool(r('inherits(gf_res, "try-error")')[0]):
        return None
    return r2np(r('gf_res'))


def _py_basis(fun, df, intercept):
    from basis import OneBasis
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return np.asarray(OneBasis(X, fun=fun, df=df, intercept=intercept).basis)


@pytest.mark.parametrize('intercept', [False, True])
@pytest.mark.parametrize('df', DFS)
@pytest.mark.parametrize('fun', ['ns', 'bs', 'ps', 'cr', 'strata'])
def test_fractional_df_is_handled_as_r_does(fun, df, intercept):
    ref = _r_basis(fun, df, intercept)
    if ref is None:                                  # R itself stops for this fraction (mgcv cr): so must PyDLNM
        with pytest.raises(Exception):
            _py_basis(fun, df, intercept)
        return
    got = _py_basis(fun, df, intercept)
    assert got.shape == ref.shape, f'{fun}(df={df}, intercept={intercept}): {got.shape[1]} columns, R has {ref.shape[1]}'
    assert_close(got, ref, rtol=1e-8, what=f'{fun}(df={df}, intercept={intercept})')


@pytest.mark.parametrize('fun', ['ns', 'bs', 'ps', 'strata'])
def test_integer_valued_float_df_is_the_integer_df(fun):
    """Guard (true before and after the fix): df = 5.0 is df = 5."""
    assert_close(_py_basis(fun, 5.0, False), _py_basis(fun, 5, False), rtol=1e-14, what=f'{fun}: df 5.0 vs 5')
