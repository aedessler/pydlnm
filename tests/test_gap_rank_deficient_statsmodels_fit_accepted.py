"""Gap rank_deficient_statsmodels_fit_accepted (completeness critic, minor): a statsmodels fit whose design is rank
deficient inside the cross-basis block.

R's glm() gives NA to the aliased columns of a rank-deficient design (LINPACK dqrdc2: a column is aliased when, after the
columns before it are removed, its norm is below 1e-11 times its own), and crosspred()/crossreduce() then stop with
"coef/vcov not consistent with basis matrix" because the block has missing coefficients. statsmodels solves such a design
with a pseudo-inverse and returns finite numbers, so PyDLNM predicted from arbitrary minimum-norm coefficients without a word.
An aliased column OUTSIDE the block (a collinear confounder) is no problem in R, and must not be one here.

R computes the reference (the NA pattern and the stop) at run time.
"""
import os
import warnings

import numpy as np
import pytest

from rhelpers import assert_close, chicago, np2r, r, rget

sm = pytest.importorskip('statsmodels.api')

_R_HOME = os.path.dirname(str(r('.Library')[0]))


@pytest.fixture(autouse=True)
def _r_home_guard():
    os.environ['R_HOME'] = _R_HOME
    yield
    os.environ['R_HOME'] = _R_HOME


N = 700


def _data():
    ch = chicago()
    return ch['temp'][:N], ch['death'][:N], ch['dow'][:N].astype(int)


def _cb(x, lag, arglag):
    from basis import CrossBasis
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return CrossBasis(x, lag=lag, argvar={'fun': 'bs', 'degree': 2, 'knots': np.quantile(x, [.25, .5, .75])},
                          arglag=arglag)


def _fit_statsmodels(cb, y, dow, duplicate_confounder=False):
    ok = ~np.isnan(np.asarray(cb.basis)).any(axis=1)
    dummies = np.column_stack([(dow == k).astype(float) for k in range(2, 8)])
    cols = [np.ones(N), np.asarray(cb.basis), dummies]
    if duplicate_confounder:
        cols.append(dummies[:, [0]])                     # exact copy of a day-of-week dummy, AFTER the block
    X = np.column_stack(cols)[ok]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return sm.GLM(y[ok], X, family=sm.families.Poisson()).fit(scale='X2')


def _r_model(x, y, dow, lag, arglag_r, duplicate_confounder=False):
    np2r('rd_x', x)
    np2r('rd_y', y.astype(float))
    np2r('rd_dow', dow.astype(float))
    r('rd_x <- as.numeric(rd_x); rd_y <- as.numeric(rd_y); rd_dow <- factor(as.integer(rd_dow))')
    r(f'rd_cb <- crossbasis(rd_x, lag={lag}, argvar=list(fun="bs", degree=2, knots=quantile(rd_x, c(.25,.5,.75))), '
      f'arglag={arglag_r})')
    r('rd_dd <- as.numeric(rd_dow == "2")')
    extra = ' + rd_dd' if duplicate_confounder else ''
    r(f'rd_m <- glm(rd_y ~ rd_cb + rd_dow{extra}, family=quasipoisson)')


@pytest.fixture(scope='module')
def deficient():
    """Cross-basis with an intercept-ful 4-column lag basis over only 3 lags: rank 3 of 4, so aliased columns."""
    x, y, dow = _data()
    arglag = {'fun': 'ns', 'df': 4, 'intercept': True}
    cb = _cb(x, 2, arglag)
    _r_model(x, y, dow, 2, 'list(fun="ns", df=4, intercept=TRUE)')
    return cb, _fit_statsmodels(cb, y, dow)


def test_r_gives_na_coefficients_and_stops(deficient):
    """Premise: R aliases columns of this cross-basis, and crosspred / crossreduce stop."""
    assert bool(r('any(is.na(coef(rd_m)[grep("rd_cb", names(coef(rd_m)))]))')[0])
    with pytest.raises(Exception):
        r('crosspred(rd_cb, rd_m, at=seq(-5, 30, length=6), cen=15)')
    with pytest.raises(Exception):
        r('crossreduce(rd_cb, rd_m, cen=15)')


def test_crosspred_stops_for_an_aliased_cross_basis_block(deficient):
    from prediction import crosspred
    cb, res = deficient
    with pytest.raises(ValueError, match='consistent|aliased|missing'):
        crosspred(cb, model=res, at=np.linspace(-5, 30, 6), cen=15.0)


def test_crossreduce_stops_for_an_aliased_cross_basis_block(deficient):
    from crossreduce import crossreduce
    cb, res = deficient
    with pytest.raises(ValueError, match='consistent|aliased|missing'):
        crossreduce(cb, model=res, cen=15.0)


def test_an_aliased_confounder_outside_the_block_is_no_problem():
    """R: the NA is in a column after the block, so crosspred works; so must PyDLNM, with the same result as without
    the duplicated column."""
    from prediction import crosspred
    x, y, dow = _data()
    cb = _cb(x, 4, {'fun': 'ns', 'df': 3})
    at = np.linspace(-5, 30, 6)
    _r_model(x, y, dow, 4, 'list(fun="ns", df=3)', duplicate_confounder=True)
    assert bool(r('any(is.na(coef(rd_m)))')[0]), 'premise: R aliases the duplicated confounder'
    r('rd_cp <- crosspred(rd_cb, rd_m, at=seq(-5, 30, length=6), cen=15)')            # R does not stop
    ref = rget('rd_cp$allfit')
    plain = crosspred(cb, model=_fit_statsmodels(cb, y, dow), at=at, cen=15.0)
    dup = crosspred(cb, model=_fit_statsmodels(cb, y, dow, duplicate_confounder=True), at=at, cen=15.0)
    assert_close(dup.allfit, plain.allfit, rtol=1e-6, what='aliased confounder outside the block changes nothing')
    assert_close(dup.allfit, ref, rtol=1e-5, what='allfit vs R with an aliased confounder')
