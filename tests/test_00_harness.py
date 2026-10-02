"""Smoke test of the harness: the validated cross-basis (bs deg2 x ns logknots) must equal R's bit for bit."""
import numpy as np

from rhelpers import SRC, assert_close, chicago, np2r, r, rget


def test_source_under_test_is_importable():
    from basis import CrossBasis  # noqa: F401
    assert (SRC / 'basis.py').exists()


def test_validated_crossbasis_matches_r():
    from basis import CrossBasis
    from utils import logknots
    temp = chicago()['temp']
    np2r('temp', temp)
    r('kv <- quantile(temp, c(.10,.75,.90));'
      'cbR <- crossbasis(temp, lag=21, argvar=list(fun="bs", degree=2, knots=kv),'
      ' arglag=list(fun="ns", knots=logknots(21,3)))')
    cb_r = rget('unclass(cbR)')
    kv = rget('kv')
    cb_p = np.asarray(CrossBasis(temp, lag=21, argvar={'fun': 'bs', 'degree': 2, 'knots': kv},
                                 arglag={'fun': 'ns', 'knots': logknots([0, 21], nk=3)}).basis)
    assert_close(cb_p, cb_r, rtol=1e-13, what='crossbasis')
    assert int(np.isnan(cb_p).any(axis=1).sum()) == 21
