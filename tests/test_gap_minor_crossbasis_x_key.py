"""Gap minor: an 'x' entry in argvar / arglag.  R's crossbasis() does modifyList(argvar, list(x = x)) and
modifyList(arglag, list(x = seqlag(lag))), so the entry is overridden and never used; PyDLNM used to raise
"got multiple values for argument 'x'"."""
import numpy as np

from rhelpers import np2r, r, r2np


def test_x_entry_in_argvar_and_arglag_is_overridden_like_r():
    from basis import CrossBasis
    x = np.linspace(0, 20, 60)
    np2r('xk_x', x)
    cb = CrossBasis(x, lag=3, argvar={'fun': 'ns', 'df': 3, 'x': np.arange(5.0)},
                    arglag={'fun': 'poly', 'degree': 2, 'x': np.arange(4.0)})
    ref = r2np(r('unclass(dlnm::crossbasis(xk_x, lag = 3, argvar = list(fun = "ns", df = 3, x = 1:5), '
                 'arglag = list(fun = "poly", degree = 2, x = 0:3)))[, ]'))
    got = np.asarray(cb)
    assert got.shape == ref.shape
    assert np.array_equal(np.isnan(got), np.isnan(ref))
    np.testing.assert_allclose(got[~np.isnan(got)], ref[~np.isnan(ref)], rtol=1e-12, atol=1e-14)
    assert 'x' not in cb.argvar and 'x' not in cb.arglag
