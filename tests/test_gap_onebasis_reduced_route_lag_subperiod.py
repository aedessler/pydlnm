"""Gap onebasis_reduced_route_lag_subperiod (completeness critic, minor): crosspred with a OneBasis, or with reduced
(overall-effect) coefficients of a CrossBasis, and a ``lag=`` argument.

R's crosspred(onebasis, coef, vcov, lag = c(a, b), bylag) treats the one-basis as having the single original lag [0, 0] and
accepts a sub-period: the exposure basis is repeated for every requested lag (``matfit`` / ``matse`` get one identical
column per lag, ``lag`` / ``bylag`` are echoed), the overall effect sums the design over the INTEGER lags
(``allfit = n_lags * basis %*% coef``, ``allse = n_lags * se``), and ``cumul = TRUE`` is refused for a sub-period.
PyDLNM warned "OneBasis prediction ignores lag" and then crashed on a shape mismatch (OneBasis), or raised
"'lag' is not applicable" (reduced coefficients).

R computes the reference at run time.
"""
import os
import warnings

import numpy as np
import pytest

from rhelpers import assert_close, chicago, np2r, r, r2np

_R_HOME = os.path.dirname(str(r('.Library')[0]))


@pytest.fixture(autouse=True)
def _r_home_guard():
    os.environ['R_HOME'] = _R_HOME
    yield
    os.environ['R_HOME'] = _R_HOME


X = None
KNOTS = None
AT = np.linspace(-5.0, 30.0, 8)
CEN = 15.0
LAGS = [([0, 3], 1.0), ([1, 3], 1.0), (2, 1.0), ([0, 2], 0.5), ([0, 0], 1.0)]


@pytest.fixture(scope='module', autouse=True)
def _data():
    global X, KNOTS
    X = chicago()['temp'][:400]
    KNOTS = np.quantile(X, [.25, .5, .75])
    np2r('lg_x', X)
    np2r('lg_k', KNOTS)
    np2r('lg_at', AT)
    r('lg_x <- as.numeric(lg_x); lg_k <- as.numeric(lg_k); lg_at <- as.numeric(lg_at)')


def _lag_r(lag):
    return f'c(0, {int(lag)})' if np.ndim(lag) == 0 else f'c({int(lag[0])}, {int(lag[1])})'


def _r_pred(coef, vcov, lag, bylag, cumul=False, cen=CEN):
    np2r('lg_coef', coef)
    np2r('lg_vcov', vcov)
    r('lg_coef <- as.numeric(lg_coef)')
    r('lg_ob <- onebasis(lg_x, fun="ns", knots=lg_k)')
    r(f'lg_cp <- crosspred(lg_ob, coef=lg_coef, vcov=lg_vcov, model.link="log", at=lg_at, lag={_lag_r(lag)}, '
      f'bylag={bylag!r}, cen={cen!r}, cumul={"TRUE" if cumul else "FALSE"})')
    return {k: r2np(r(f'unclass(lg_cp${k})')) for k in ('matfit', 'matse', 'allfit', 'allse', 'matRRfit', 'allRRlow', 'lag')}


def _py_pred_onebasis(coef, vcov, lag, bylag, cumul=False, cen=CEN):
    from basis import OneBasis
    from prediction import crosspred
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return crosspred(OneBasis(X, fun='ns', knots=KNOTS), coef=coef, vcov=vcov, model_link='log', at=AT, lag=lag,
                         bylag=bylag, cen=cen, cumul=cumul)


def _py_pred_reduced(coef, vcov, lag, bylag, cumul=False, cen=CEN):
    from basis import CrossBasis
    from prediction import crosspred
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        cb = CrossBasis(X, lag=4, argvar={'fun': 'ns', 'knots': KNOTS}, arglag={'fun': 'ns', 'df': 3})
        return crosspred(cb, coef=coef, vcov=vcov, model_link='log', at=AT, lag=lag, bylag=bylag, cen=cen, cumul=cumul)


def _coef_vcov():
    k = 4                                                     # ns with 3 interior knots: 4 columns
    rng = np.random.default_rng(5)
    a = rng.normal(0, .3, (k, k))
    return np.linspace(-.05, .08, k), a @ a.T * 1e-3 + np.eye(k) * 1e-4


@pytest.mark.parametrize('route', ['onebasis', 'reduced'])
@pytest.mark.parametrize('lag,bylag', LAGS)
def test_lag_subperiod_matches_r(route, lag, bylag):
    coef, vcov = _coef_vcov()
    ref = _r_pred(coef, vcov, lag, bylag)
    pred = (_py_pred_onebasis if route == 'onebasis' else _py_pred_reduced)(coef, vcov, lag, bylag)
    for key in ('matfit', 'matse', 'allfit', 'allse'):
        assert_close(getattr(pred, key), ref[key].reshape(np.shape(getattr(pred, key))), rtol=1e-10,
                     what=f'{route}: {key} for lag={lag}, bylag={bylag}')
    assert_close(pred.matRRfit, ref['matRRfit'], rtol=1e-10, what=f'{route}: matRRfit')
    assert_close(pred.allRRlow, ref['allRRlow'], rtol=1e-10, what=f'{route}: allRRlow')
    n_int = len(np.arange(*(np.array(lag if np.ndim(lag) else [0, lag]) + [0, 1])))
    assert_close(pred.allfit, n_int * (pred.matfit[:, 0]), rtol=1e-10, what=f'{route}: allfit = n_lags * lag effect')
    assert_close(np.asarray(pred.lag, dtype=float), ref['lag'].ravel(), rtol=1e-14, what=f'{route}: lag echoed')


@pytest.mark.parametrize('route', ['onebasis', 'reduced'])
def test_cumulative_prediction_is_refused_for_a_sub_period(route):
    coef, vcov = _coef_vcov()
    with pytest.raises(Exception):
        _r_pred(coef, vcov, [0, 3], 1.0, cumul=True)         # R stops ...
    with pytest.raises(ValueError, match='[Cc]umulative'):      # ... so does PyDLNM
        (_py_pred_onebasis if route == 'onebasis' else _py_pred_reduced)(coef, vcov, [0, 3], 1.0, cumul=True)


@pytest.mark.parametrize('route', ['onebasis', 'reduced'])
def test_default_lag_is_unchanged(route):
    """Guard (true before and after the fix): without lag the single lag [0, 0] gives R's numbers."""
    coef, vcov = _coef_vcov()
    ref = _r_pred(coef, vcov, [0, 0], 1.0, cumul=True)
    pred = (_py_pred_onebasis if route == 'onebasis' else _py_pred_reduced)(coef, vcov, None, 1.0, cumul=True)
    assert_close(pred.allfit, ref['allfit'], rtol=1e-10, what='allfit')
    assert_close(pred.matse, ref['matse'].reshape(pred.matse.shape), rtol=1e-10, what='matse')
