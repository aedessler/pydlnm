"""Shared harness for R-dlnm-vs-PyDLNM differential tests.

R is driven in-process through rpy2. R MUST be initialised before any PyDLNM module is imported (several modules
overwrite os.environ['R_HOME']), so import this module first; conftest.py does.

Source under test: the repo root, or the directory named by the PYDLNM_SRC environment variable (used to check a
test against a patched copy of the modules).
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(os.environ.get('PYDLNM_SRC') or Path(__file__).resolve().parents[1]).resolve()
REPO = Path(__file__).resolve().parents[1]

import rpy2.robjects as ro                      # noqa: E402  (R starts here)
from rpy2.robjects import numpy2ri              # noqa: E402
from rpy2.robjects.conversion import localconverter   # noqa: E402

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

ro.r('suppressMessages({library(dlnm); library(splines)})')
r = ro.r


def r2np(x):
    """R object -> float numpy array (dim attribute kept, NULL -> empty array)."""
    with localconverter(ro.default_converter + numpy2ri.converter):
        return np.array(x)


def np2r(name, arr):
    """Push a numeric array into the R global environment under `name`."""
    with localconverter(ro.default_converter + numpy2ri.converter):
        ro.globalenv[name] = np.asarray(arr, dtype=float)


def rget(expr):
    """Evaluate an R expression and return it as a numpy array."""
    return r2np(ro.r(expr))


def require_r_packages(*pkgs):
    """Skip the calling test if an R package is missing; otherwise load it quietly."""
    for p in pkgs:
        ok = bool(ro.r(f'suppressWarnings(requireNamespace("{p}", quietly=TRUE))')[0])
        if not ok:
            pytest.skip(f'R package {p} not installed')
        ro.r(f'suppressMessages(library({p}))')


_CHICAGO = {}


def chicago():
    """R's chicagoNMMAPS as numpy arrays: temp, death, dow (1..7), date (days since epoch), year."""
    if not _CHICAGO:
        ro.r('data(chicagoNMMAPS, package="dlnm")')
        _CHICAGO['temp'] = rget('chicagoNMMAPS$temp')
        _CHICAGO['death'] = rget('chicagoNMMAPS$death')
        _CHICAGO['dow'] = rget('as.integer(chicagoNMMAPS$dow)')
        _CHICAGO['date'] = rget('as.numeric(chicagoNMMAPS$date)')
        _CHICAGO['year'] = rget('as.integer(chicagoNMMAPS$year)')
    return {k: v.copy() for k, v in _CHICAGO.items()}


def max_abs_diff(py, ref):
    py, ref = np.asarray(py, dtype=float), np.asarray(ref, dtype=float)
    m = np.isfinite(py) & np.isfinite(ref)
    return float(np.abs(py[m] - ref[m]).max()) if m.any() else 0.0


def max_rel_diff(py, ref):
    """max |py-ref| / max|ref| over jointly finite entries (scale-free, robust to entries near 0)."""
    py, ref = np.asarray(py, dtype=float), np.asarray(ref, dtype=float)
    m = np.isfinite(py) & np.isfinite(ref)
    if not m.any():
        return 0.0
    return float(np.abs(py[m] - ref[m]).max() / max(np.abs(ref[m]).max(), 1e-300))


def assert_close(py, ref, rtol=1e-8, what='', check_nan=True):
    """Same shape, same NaN pattern, and max|py-ref| <= rtol * max|ref| on the finite entries."""
    py, ref = np.asarray(py, dtype=float), np.asarray(ref, dtype=float)
    assert py.shape == ref.shape, f'{what}: shape Python {py.shape} vs R {ref.shape}'
    if check_nan:
        assert np.array_equal(np.isnan(py), np.isnan(ref)), f'{what}: NaN pattern differs'
    d = max_rel_diff(py, ref)
    assert d <= rtol, f'{what}: max relative diff {d:.3e} > {rtol:.1e}'


def known_defect(theme, *finding_ids, note=''):
    """xfail(strict) mark for a verified defect. When the fix lands the test XPASSes, strict mode turns that into a
    failure, which forces the marker to be removed in the same commit as the fix.
    Set PYDLNM_XFAIL_OFF=1 to run these tests as ordinary tests and see the raw failures."""
    if os.environ.get('PYDLNM_XFAIL_OFF'):
        return pytest.mark.usefixtures()
    reason = f'{theme}: ' + ','.join(finding_ids) + (f' ({note})' if note else '')
    return pytest.mark.xfail(strict=True, reason=reason)
