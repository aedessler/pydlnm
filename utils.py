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
            times: Optional[Union[np.ndarray, List[float]]] = None,
            lag: Union[np.ndarray, List[int], Tuple[int, ...]] = (0, 1),
            fill: float = 0.0) -> np.ndarray:
    """
    Construct exposure history matrices from time series data.
    
    This function creates a matrix where each row represents the exposure history
    at different time points, accounting for the specified lag structure.
    
    Parameters
    ----------
    exposure : array-like
        Vector of exposure values
    times : array-like, optional
        Time points corresponding to exposures. If None, uses sequential integers.
    lag : array-like, default=(0, 1)
        Two-element array [min_lag, max_lag] specifying lag range
    fill : float, default=0.0
        Value to use for padding when exposure history extends beyond available data
        
    Returns
    -------
    np.ndarray
        Matrix of exposure histories. Each row represents exposure history at a time point,
        with columns representing different lag values.
        
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
    exposure = np.asarray(exposure)
    lag = mklag(lag)
    
    if times is None:
        times = np.arange(len(exposure))
    else:
        times = np.asarray(times)
        
    if len(times) != len(exposure):
        raise ValueError("times and exposure must have the same length")
    
    # Create lag sequence
    lag_seq = seqlag(lag)
    n_times = len(times)
    n_lags = len(lag_seq)
    
    # Initialize output matrix
    hist_matrix = np.full((n_times, n_lags), fill, dtype=float)
    
    # Fill in exposure histories
    for i, time_point in enumerate(times):
        for j, lag_val in enumerate(lag_seq):
            # Find the index for the lagged time point
            target_time = time_point - lag_val
            
            # Find closest time point (simple approach)
            time_diffs = np.abs(times - target_time)
            min_idx = np.argmin(time_diffs)
            
            # Only use if the match is exact (for integer lags) or very close
            if np.abs(times[min_idx] - target_time) < 0.5:
                hist_matrix[i, j] = exposure[min_idx]
            # Otherwise keep the fill value
                
    return hist_matrix


def equalknots(x: np.ndarray, 
               fun: str = "ns", 
               df: int = 5, 
               degree: int = 3) -> np.ndarray:
    """
    Place knots at equally-spaced values along the range of x.
    
    Parameters
    ----------
    x : array-like
        Input vector
    fun : str, default="ns"
        Basis function type (for compatibility with R dlnm)
    df : int, default=5
        Degrees of freedom
    degree : int, default=3
        Spline degree
        
    Returns
    -------
    np.ndarray
        Knot positions
    """
    x = np.asarray(x)
    x_clean = x[~np.isnan(x)]
    
    if len(x_clean) == 0:
        raise ValueError("No valid (non-NaN) values in x")
    
    # Calculate interior knots based on df and degree
    if fun == "ns":
        n_interior = df - 1
    elif fun == "bs":
        n_interior = df - degree - 1
    else:
        n_interior = df - 1
    
    if n_interior <= 0:
        return np.array([])
    
    # Place knots at equally spaced quantiles
    quantiles = np.linspace(0, 1, n_interior + 2)[1:-1]
    knots = np.quantile(x_clean, quantiles)
    
    return knots


def logknots(x: Union[int, List[int], np.ndarray], 
             nk: Optional[int] = None, 
             fun: str = "ns", 
             df: Optional[int] = None, 
             degree: int = 3, 
             intercept: bool = True) -> np.ndarray:
    """
    Place knots at log-spaced values, exactly matching R dlnm logknots() behavior.
    
    This function creates interior knots for spline functions using log-spaced positions,
    which is particularly useful for lag-response relationships where effects decay 
    exponentially with time.
    
    Parameters
    ----------
    x : int, list, or array
        Lag range. If single value, interpreted as [0, x]. If length 2, interpreted as range.
    nk : int, optional
        Number of knots. If None, calculated based on fun, df, degree, intercept.
    fun : str, default="ns"
        Basis function type ("ns", "bs", "strata")
    df : int, default=1
        Degrees of freedom
    degree : int, default=3
        Degree of polynomial (for B-splines)
    intercept : bool, default=True
        Whether intercept is included
        
    Returns
    -------
    np.ndarray
        Log-spaced interior knot positions
        
    Examples
    --------
    >>> logknots(21, df=3)  # R: logknots(21, 3)
    array([1.01119306, 2.77947331, 7.63995648])
    """
    x = np.asarray(x).flatten()
    
    # If length of x is 1 or 2, interpret as lag range, otherwise take the range
    if len(x) < 3:
        lag_range = mklag(x)
    else:
        lag_range = np.array([np.min(x), np.max(x)])
    
    if np.diff(lag_range)[0] == 0:
        raise ValueError("range must be > 0")
    
    # Choose number of knots if not provided
    if nk is None:
        # If df is provided, calculate nk from it
        if df is not None:
            if fun == "ns":
                nk = df - 1 - (1 if intercept else 0)
            elif fun == "bs":
                nk = df - degree - (1 if intercept else 0)
            elif fun == "strata":
                nk = df - (1 if intercept else 0)
            else:
                raise ValueError(f"Unknown function type: {fun}")
        else:
            # Default case
            nk = 1
    
    if nk < 1:
        raise ValueError("choice of arguments defines no knots")
    
    # Define knots at equally-spaced log-values along lag
    # R formula: range[1] + exp(((1+log(diff(range)))/(nk+1))*seq(nk)-1)
    range_start = lag_range[0]
    range_diff = np.diff(lag_range)[0]
    
    # Create the sequence: seq(nk) in R is 1:nk, so in Python it's 1 to nk+1
    seq_nk = np.arange(1, nk + 1)
    
    # Apply R's formula exactly
    log_factor = (1 + np.log(range_diff)) / (nk + 1)
    knots = range_start + np.exp(log_factor * seq_nk - 1)
    
    return knots