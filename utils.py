"""
Utility functions for PyDLNM

This module contains core utility functions that support the main DLNM functionality,
including lag parameter validation, sequence generation, and exposure history construction.
"""

import math
import numpy as np
from typing import Union, List, Tuple, Optional, Any
import warnings


def warn_experimental(feature: str) -> None:
    """Warn that a feature has not been validated against R (penalized and seasonality modules)."""
    warnings.warn(
        f"{feature} is experimental: it has no validated R counterpart in dlnm and has known open issues "
        "(see README, 'Experimental modules'). Do not rely on it for published analyses.",
        UserWarning, stacklevel=3)


def mklag(lag: Union[int, List[int], Tuple[int, ...], np.ndarray]) -> np.ndarray:
    """
    Validate and standardize lag specifications for distributed lag models.
    
    This function takes a lag parameter and converts it to a standardized 
    2-element array [min_lag, max_lag].
    
    Parameters
    ----------
    lag : int, list, tuple, or ndarray
        Lag specification. Can be:
        - Single integer: converted to [0, lag] if positive, [lag, 0] if negative
        - Two integers: [min_lag, max_lag]
        
    Returns
    -------
    np.ndarray
        Two-element array [min_lag, max_lag]
        
    Raises
    ------
    ValueError
        If lag specification is invalid or min_lag > max_lag
        
    Examples
    --------
    >>> mklag(5)
    array([0, 5])
    
    >>> mklag([2, 8])
    array([2, 8])
    
    >>> mklag(-3)
    array([-3, 0])
    """
    # Convert to numpy array
    lag = np.asarray(lag)
    
    # Validate input
    if lag.size == 0:
        raise ValueError("lag cannot be empty")
    if lag.size > 2:
        raise ValueError("lag must have 1 or 2 elements")
    if lag.dtype.kind not in 'iuf':
        raise TypeError("'lag' must be a numeric vector of length 2 or 1")
    lag = lag.astype(float).flatten()
    if not np.all(np.isfinite(lag)):
        raise ValueError("missing or infinite value in 'lag'")
    
    if lag.size == 1:
        lag = np.array([lag[0], 0.0]) if lag[0] < 0 else np.array([0.0, lag[0]])
    if lag[0] > lag[1]:
        raise ValueError(f"min_lag ({lag[0]:g}) must be <= max_lag ({lag[1]:g})")
    
    # R: round(lag[1:2]) (halves go to the even integer, as numpy does)
    return np.round(lag).astype(np.int64)


def seqlag(lag: Union[np.ndarray, List[int], Tuple[int, ...]], 
           by: float = 1.0) -> np.ndarray:
    """
    Create sequences of lag values.
    
    Parameters
    ----------
    lag : array-like
        Two-element array [min_lag, max_lag]
    by : float, default=1.0
        Step size for sequence
        
    Returns
    -------
    np.ndarray
        Sequence from lag[0] to lag[1] with step size 'by'
        
    Examples
    --------
    >>> seqlag([0, 5])
    array([0, 1, 2, 3, 4, 5])
    
    >>> seqlag([0, 5], by=0.5)
    array([0. , 0.5, 1. , 1.5, 2. , 2.5, 3. , 3.5, 4. , 4.5, 5. ])
    """
    lag = np.asarray(lag, dtype=float)
    if lag.size != 2:
        raise ValueError("lag must have exactly 2 elements")
    
    # R: seq(from=lag[1], to=lag[2], by=by). seq.default never overshoots 'to': the number of steps is
    # floor((to - from)/by + 1e-10) and the last value is capped at 'to'.
    start, stop = float(lag[0]), float(lag[1])
    delta = stop - start
    if delta == 0 and stop == 0:
        return np.array([stop])
    with np.errstate(divide='ignore', invalid='ignore'):
        n = delta / by
    if not np.isfinite(n):
        if by == 0 and delta == 0:
            return np.array([start])
        raise ValueError("invalid '(to - from)/by' in seq")
    if n < 0:
        raise ValueError("wrong sign in 'by' argument")
    if abs(delta) / max(abs(stop), abs(start)) < 100 * np.finfo(float).eps:
        return np.array([start])
    values = start + np.arange(int(n + 1e-10) + 1) * by
    return np.minimum(values, stop) if by > 0 else np.maximum(values, stop)


_DBL_EPS = np.finfo(float).eps
_DBL_MIN = np.finfo(float).tiny


def pretty(x, n=5, min_n=None, shrink_sml=0.75, high_u_bias=1.5, u5_bias=None, eps_correct=0):
    """
    Port of R's pretty() (pretty.default / R_pretty): about n + 1 equally spaced "round" values that
    cover range(x). Used for the default prediction grid (mkat) and the automatic centering value
    (mkcen) of crosspred. Agrees with R to rounding error (3005 of 3006 random ranges and n tested); only
    degenerate ranges narrower than ~1e-13 relative to their magnitude can differ in length.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return x
    lo, up = float(x.min()), float(x.max())
    ndiv = int(n)
    if ndiv < 0:
        raise ValueError("invalid 'n' argument")
    if min_n is None:
        min_n = ndiv // 3
    min_n = int(min_n)
    if shrink_sml <= 0:
        raise ValueError("'shrink.sml' must be positive")
    if min_n < 0 or min_n > ndiv:
        raise ValueError("invalid 'min.n' argument")
    h = float(high_u_bias)
    h5 = 0.5 + 1.5 * h if u5_bias is None else float(u5_bias)
    f_min = 2.0 ** -20
    rounding_eps = 1e-10

    dx = up - lo
    if dx == 0 and up == 0:
        cell = 1.0
        i_small = True
    else:
        cell = max(abs(lo), abs(up))
        U = 1 + ((1 / (1 + h)) if h5 >= 1.5 * h + 0.5 else 1.5 / (1 + h5))
        U *= max(1, ndiv) * _DBL_EPS
        i_small = dx < cell * U * 3
    if i_small:
        if cell > 10:
            cell = 9 + cell / 10
        cell *= shrink_sml
        if min_n > 1:
            cell /= min_n
    else:
        cell = dx
        if ndiv > 1:
            cell /= ndiv
    subsmall = f_min * _DBL_MIN
    if subsmall == 0.0:
        subsmall = _DBL_MIN
    if cell < subsmall:
        cell = subsmall
    elif cell > np.finfo(float).max / 1.25:
        cell = np.finfo(float).max / 1.25

    base = 10.0 ** math.floor(math.log10(cell))
    unit = base
    ns_ = 2 * base
    if ns_ - cell < h * (cell - unit):
        unit = ns_
        ns_ = 5 * base
        if ns_ - cell < h5 * (cell - unit):
            unit = ns_
            ns_ = 10 * base
            if ns_ - cell < h * (cell - unit):
                unit = ns_
    ns = math.floor(lo / unit + rounding_eps)
    nu = math.ceil(up / unit - rounding_eps)
    if eps_correct and (eps_correct > 1 or not i_small):
        lo = lo * (1 - _DBL_EPS) if lo != 0.0 else -_DBL_MIN
        up = up * (1 + _DBL_EPS) if up != 0.0 else _DBL_MIN
    while ns * unit > lo + rounding_eps * unit:
        ns -= 1
    while not math.isfinite(ns * unit):
        ns += 1
    while nu * unit < up - rounding_eps * unit:
        nu += 1
    while not math.isfinite(nu * unit):
        nu -= 1
    k = int(0.5 + nu - ns)
    if k < min_n:
        k = min_n - k
        if ns >= 0.0:
            nu += k // 2
            ns -= k // 2 + k % 2
        else:
            ns -= k // 2
            nu += k // 2 + k % 2
        ndiv = min_n
    else:
        ndiv = k
    # return_bounds = TRUE: the bounds cover the original range
    if ns * unit < lo:
        lo = ns * unit
    if nu * unit > up:
        up = nu * unit
    s = np.linspace(lo, up, ndiv + 1) if ndiv > 0 else np.array([lo])
    if not eps_correct and ndiv:
        delta = (up - lo) / ndiv
        s[np.abs(s) < 1e-14 * delta] = 0.0
    return s


def quantile7(x, probs) -> np.ndarray:
    """
    ``quantile(x, probs, type = 7, na.rm = TRUE)`` with R's own arithmetic.

    numpy's default quantile is type 7 too, but interpolates with a different floating-point formula, and R's
    probability vectors are built differently (``1/(k+1)*1:k`` is not ``(1:k)/(k+1)``). The results differ by ~1e-15,
    which matters wherever a value is compared with a quantile (strata breaks: ``cut(x, right=FALSE)`` puts an
    observation equal to a break into the upper stratum).
    """
    x = np.asarray(x, dtype=float).ravel()
    x = np.sort(x[~np.isnan(x)])
    probs = np.atleast_1d(np.asarray(probs, dtype=float))
    n = len(x)
    if n == 0:
        return np.full(probs.shape, np.nan)
    index = 1 + max(n - 1, 0) * probs
    lo = np.floor(index).astype(int)
    hi = np.ceil(index).astype(int)
    qs = x[lo - 1].copy()
    upper = x[hi - 1]
    sel = (index > lo) & (upper != qs)
    h = (index - lo)[sel]
    qs[sel] = (1 - h) * qs[sel] + h * upper[sel]
    return qs


def lagmatrix(values: Union[np.ndarray, List[float]], lags: Union[np.ndarray, List[int]]) -> np.ndarray:
    """
    Matrix whose column j is ``values`` shifted by ``lags[j]`` (port of tsModel::Lag).

    A positive lag uses past values (the first rows are NaN), a negative lag uses future values (the last rows
    are NaN). Raises, like R, when the largest absolute lag is not smaller than the series length.
    """
    values = np.asarray(values, dtype=float).ravel()
    lags = np.atleast_1d(np.asarray(lags)).astype(int)
    n = len(values)
    if lags.size == 0:
        raise ValueError("'lags' must not be empty")
    if np.max(np.abs(lags)) >= n:
        raise ValueError("largest lag must be less than the length of the series")
    out = np.full((n, len(lags)), np.nan)
    for j, k in enumerate(lags):
        if k > 0:
            out[k:, j] = values[:n - k]
        elif k < 0:
            out[:n + k, j] = values[-k:]
        else:
            out[:, j] = values
    return out


def exphist(exposure: Union[np.ndarray, List[float]], 
            times: Optional[Union[float, np.ndarray, List[float]]] = None,
            lag: Optional[Union[int, np.ndarray, List[int], Tuple[int, ...]]] = None,
            fill: float = 0.0) -> np.ndarray:
    """
    Define exposure histories from an exposure profile (port of R's ``exphist()``).

    The exposure profile ``exposure`` is assumed defined forward in time at equally spaced units from time 1
    (``exposure[0]`` is time 1). The history over the lag period ``lag`` is evaluated backward in time from each
    entry of ``times``. Positions outside the profile are filled with ``fill``.

    Parameters
    ----------
    exposure : array-like
        Exposure profile (R: ``exp``)
    times : scalar or array-like, optional
        Time points at which the histories are evaluated: rounded, 1-based positions in ``exposure`` (they may be
        of any length and may lie outside 1..len(exposure)). Default: every time point, 1..len(exposure).
    lag : int or array-like of length 2, optional
        Maximum lag or lag range (see ``mklag``). Default: ``[0, len(exposure) - 1]``, as in R.
    fill : float, default=0.0
        Value used for the positions of a history that fall before the start or after the end of the profile
        
    Returns
    -------
    np.ndarray
        Matrix with one row per entry of ``times`` and one column per lag ``lag[0]..lag[1]``: entry ``(i, j)`` is
        ``exposure`` at time ``times[i] - lag[0] - j``.
        
    Examples
    --------
    >>> exposure = [1, 2, 3, 4, 5]
    >>> exphist(exposure, lag=[0, 3])
    array([[1., 0., 0., 0.],
           [2., 1., 0., 0.],
           [3., 2., 1., 0.],
           [4., 3., 2., 1.],
           [5., 4., 3., 2.]])
    """
    exposure = np.asarray(exposure, dtype=float).ravel()
    n = len(exposure)
    
    # R: lag <- if(missing(lag)) c(0, length(exp)-1) else mklag(lag)
    lag = np.array([0, n - 1]) if lag is None else mklag(lag)
    
    # R: times <- if(missing(times)) seq(length(exp)) else round(times)
    if times is None:
        times = np.arange(1, n + 1)
    else:
        times = np.round(np.atleast_1d(np.asarray(times, dtype=float)).ravel())
        if times.size == 0 or not np.all(np.isfinite(times)):
            raise ValueError("'times' must be a non-empty vector of finite values")
        times = times.astype(np.int64)
    
    # Extend the profile with `fill` on both sides, as much as the histories need
    left = max(0, int(lag[1]) + 1 - int(times.min()))
    right = max(0, int(times.max()) - n - int(lag[0]))
    extended = np.concatenate([np.full(left, fill, dtype=float), exposure, np.full(right, fill, dtype=float)])
    
    # History of time t over lag[0]..lag[1]: the profile at t - lag (1-based), shifted by the left padding
    lags = np.arange(int(lag[0]), int(lag[1]) + 1)
    return extended[times[:, None] - lags[None, :] + left - 1]


_KNOT_FUNCTIONS = ("ns", "bs", "strata")


def _match_knot_function(fun: str) -> str:
    """R's ``match.arg(fun, c("ns", "bs", "strata"))``: exact or unique partial match, otherwise an error."""
    if fun in _KNOT_FUNCTIONS:
        return fun
    matches = [f for f in _KNOT_FUNCTIONS if isinstance(fun, str) and fun != "" and f.startswith(fun)]
    if len(matches) != 1:
        raise ValueError(f"'fun' should be one of {', '.join(repr(f) for f in _KNOT_FUNCTIONS)}")
    return matches[0]


def _number_of_knots(fun: str, df: float, degree: float, intercept: bool) -> float:
    """Number of knots (or cut-offs) implied by fun, df, degree and intercept (R: the ``switch`` in
    equalknots()/logknots())."""
    fun = _match_knot_function(fun)
    if fun == "ns":
        return df - 1 - intercept
    if fun == "bs":
        return df - degree - intercept
    return df - intercept


def equalknots(x: np.ndarray, 
               nk: Optional[int] = None,
               fun: str = "ns", 
               df: int = 1, 
               degree: int = 3,
               intercept: bool = False) -> np.ndarray:
    """
    Place knots (or cut-offs) at equally spaced values along the range of x (port of R's ``equalknots()``).
    
    The knots are equally spaced *values* between min(x) and max(x), not quantiles of x:
    ``min + (max - min) / (nk + 1) * (1, ..., nk)``.
    
    Parameters
    ----------
    x : array-like
        Input vector. Missing values are ignored when taking the range (a vector of length 1 or 2 is NOT a lag
        range here).
    nk : int, optional
        Number of knots. If None, it is defined by ``fun``, ``df``, ``degree`` and ``intercept``
    fun : {"ns", "bs", "strata"}, default="ns"
        Function the knots are for (partial matching, as R's ``match.arg``); used only when ``nk`` is None
    df : int, default=1
        Degrees of freedom (used only when ``nk`` is None)
    degree : int, default=3
        Degree of the piecewise polynomial (``fun="bs"``)
    intercept : bool, default=False
        Whether an intercept is included in the basis function
        
    Returns
    -------
    np.ndarray
        Knot positions

    Raises
    ------
    ValueError
        If the arguments define no knots (``nk < 1``; ns: ``df - 1 - intercept``, bs: ``df - degree - intercept``,
        strata: ``df - intercept`` knots), or ``fun`` is not one of "ns", "bs", "strata".
    """
    x = np.asarray(x, dtype=float).ravel()
    x_clean = x[~np.isnan(x)]
    
    if len(x_clean) == 0:
        raise ValueError("No valid (non-NaN) values in x")
    lower, upper = x_clean.min(), x_clean.max()
    
    # Choose number of knots if not provided
    if nk is None:
        nk = _number_of_knots(fun, df, degree, bool(intercept))
    
    # Define knots at equally spaced values along the range (R: seq(nk) is 1..nk)
    if nk < 1:
        raise ValueError("choice of arguments defines no knots")
    return lower + ((upper - lower) / (nk + 1)) * np.arange(1, int(np.floor(nk)) + 1)


def logknots(x: Union[int, List[int], np.ndarray], 
             nk: Optional[int] = None, 
             fun: str = "ns", 
             df: int = 1, 
             degree: int = 3, 
             intercept: bool = True) -> np.ndarray:
    """
    Place knots at log-spaced values along a lag range (port of R's ``logknots()``).
    
    Interior knots at equally spaced log-values, which is useful for lag-response relationships where effects
    decay with the lag.
    
    Parameters
    ----------
    x : int, list, or array
        Lag range. If of length 1 or 2, it is interpreted as a lag range (see ``mklag``: a single value is
        ``[0, x]``, or ``[x, 0]`` if negative, and the range is rounded to integers); otherwise the range of the
        vector (missing values ignored) is used.
    nk : int, optional
        Number of knots. If None, it is defined by ``fun``, ``df``, ``degree`` and ``intercept``
    fun : {"ns", "bs", "strata"}, default="ns"
        Function the knots are for (partial matching, as R's ``match.arg``); used only when ``nk`` is None
    df : int, default=1
        Degrees of freedom (used only when ``nk`` is None; R's default of 1 defines no knots)
    degree : int, default=3
        Degree of the piecewise polynomial (``fun="bs"``)
    intercept : bool, default=True
        Whether an intercept is included in the basis function
        
    Returns
    -------
    np.ndarray
        Log-spaced interior knot positions

    Raises
    ------
    ValueError
        If the range is null or the arguments define no knots (``nk < 1``)
        
    Examples
    --------
    >>> logknots(21, df=3)  # R: logknots(21, df=3)
    array([1.01119306, 2.77947331, 7.63995648])
    """
    x = np.asarray(x, dtype=float).ravel()
    
    # If length of x is 1 or 2, interpret as lag range, otherwise take the range (R: range(x, na.rm=TRUE))
    if len(x) < 3:
        lag_range = mklag(x)
    else:
        x_clean = x[~np.isnan(x)]
        if len(x_clean) == 0:
            raise ValueError("No valid (non-NaN) values in x")
        lag_range = np.array([x_clean.min(), x_clean.max()])
    
    if np.diff(lag_range)[0] == 0:
        raise ValueError("range must be > 0")
    
    # Choose number of knots if not provided
    if nk is None:
        nk = _number_of_knots(fun, df, degree, bool(intercept))
    
    if nk < 1:
        raise ValueError("choice of arguments defines no knots")
    
    # Define knots at equally spaced log-values along lag:
    # R: range[1] + exp(((1+log(diff(range)))/(nk+1))*seq(nk)-1)
    range_diff = float(np.diff(lag_range)[0])
    seq_nk = np.arange(1, int(np.floor(nk)) + 1)
    return lag_range[0] + np.exp(((1 + np.log(range_diff)) / (nk + 1)) * seq_nk - 1)
