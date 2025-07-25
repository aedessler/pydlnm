"""
Utility functions for PyDLNM

This module contains core utility functions that support the main DLNM functionality,
including lag parameter validation, sequence generation, and exposure history construction.
"""

import numpy as np
from typing import Union, List, Tuple, Optional, Any
import warnings


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
    
    if lag.size == 1:
        lag_val = lag.item()
        if lag_val >= 0:
            return np.array([0, lag_val])
        else:
            return np.array([lag_val, 0])
    elif lag.size == 2:
        lag_array = lag.flatten()
        if lag_array[0] > lag_array[1]:
            raise ValueError(f"min_lag ({lag_array[0]}) must be <= max_lag ({lag_array[1]})")
        return lag_array
    else:
        raise ValueError("lag must have 1 or 2 elements")


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
    lag = np.asarray(lag)
    if lag.size != 2:
        raise ValueError("lag must have exactly 2 elements")
    
    return np.arange(lag[0], lag[1] + by, by)


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


def logknots(maxlag: int, df: int) -> np.ndarray:
    """
    Place knots at log-spaced values for lag dimension.
    
    Parameters
    ----------
    maxlag : int
        Maximum lag value
    df : int
        Degrees of freedom
        
    Returns
    -------
    np.ndarray
        Log-spaced knot positions
    """
    if df <= 1:
        return np.array([])
    
    # Create log-spaced positions
    log_positions = np.logspace(0, np.log10(maxlag + 1), df - 1)
    # Subtract 1 to get 0-based indexing and ensure we don't exceed maxlag
    knots = np.minimum(log_positions - 1, maxlag - 1)
    
    return knots[knots > 0]  # Remove any knots at or below 0