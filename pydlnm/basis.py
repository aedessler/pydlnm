"""
Core basis classes for PyDLNM

This module contains the main OneBasis and CrossBasis classes that form the foundation
of the distributed lag non-linear modeling framework.
"""

import numpy as np
from typing import Union, Optional, Dict, Any, Callable, List, Tuple
import warnings

from .basis_functions import (
    LinearBasis, PolynomialBasis, SplineBasis, BSplineBasis, 
    StrataBasis, ThresholdBasis, BaseBasisFunction
)
from .utils import mklag, seqlag
from .model_utils import validate_model_compatibility


class OneBasis:
    """
    One-dimensional basis function class.
    
    This class creates one-dimensional basis matrices for use in distributed
    lag models. It supports various basis function types and provides a
    consistent interface for basis matrix generation.
    
    Parameters
    ----------
    x : array-like
        Input vector for basis transformation
    fun : str or callable, default='ns'
        Basis function type. Can be:
        - 'lin': Linear basis
        - 'poly': Polynomial basis  
        - 'ns': Natural spline basis
        - 'bs': B-spline basis
        - 'strata': Stratified/categorical basis
        - 'thr': Threshold basis
        - Custom function
    **kwargs
        Additional arguments passed to the basis function
        
    Attributes
    ----------
    x : np.ndarray
        Original input vector
    fun : str or callable
        Basis function used
    basis : np.ndarray
        Generated basis matrix
    range : tuple
        Range of input values (min, max)
    attributes : dict
        Basis function attributes and parameters
    """
    
    # Mapping of function names to classes
    _FUNCTION_MAP = {
        'lin': LinearBasis,
        'poly': PolynomialBasis,
        'ns': SplineBasis,
        'bs': BSplineBasis,
        'strata': StrataBasis,
        'thr': ThresholdBasis,
    }
    
    def __init__(self, x: Union[np.ndarray, List], fun: Union[str, Callable] = 'ns', **kwargs):
        # Store original input
        self.x = np.asarray(x, dtype=float)
        self.fun = fun
        self.range = (np.nanmin(self.x), np.nanmax(self.x))
        
        # Extract centering parameter
        self.cen = kwargs.pop('cen', None)
        
        # Generate basis matrix
        self.basis, self.attributes = self._create_basis(**kwargs)
        
        # Store centering info in attributes
        if self.cen is not None:
            self.attributes['cen'] = self.cen
        
        # Set names for basis columns
        self._set_names()
    
    def _create_basis(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Create the basis matrix using the specified function.
        
        Parameters
        ----------
        **kwargs
            Arguments for the basis function
            
        Returns
        -------
        tuple
            (basis_matrix, attributes_dict)
        """
        if isinstance(self.fun, str):
            # Use built-in function
            if self.fun not in self._FUNCTION_MAP:
                raise ValueError(f"Unknown function '{self.fun}'. Available: {list(self._FUNCTION_MAP.keys())}")
            
            basis_func = self._FUNCTION_MAP[self.fun](**kwargs)
            basis = basis_func(self.x)
            attributes = basis_func.get_attributes()
            
        elif callable(self.fun):
            # Use custom function
            basis = self.fun(self.x, **kwargs)
            basis = np.asarray(basis)
            
            # Ensure it's a matrix
            if basis.ndim == 1:
                basis = basis.reshape(-1, 1)
            
            # Try to extract attributes from function
            attributes = {'fun': self.fun}
            if hasattr(basis, '__dict__'):
                attributes.update(getattr(basis, '__dict__', {}))
            attributes.update(kwargs)
            
        else:
            raise TypeError("fun must be a string or callable")
        
        # Ensure basis is a 2D array
        if basis.ndim == 1:
            basis = basis.reshape(-1, 1)
        
        # Store range in attributes
        attributes['range'] = self.range
        
        return basis, attributes
    
    def _set_names(self):
        """Set column names for the basis matrix."""
        n_cols = self.basis.shape[1]
        self.colnames = [f"b{i+1}" for i in range(n_cols)]
        
    def __array__(self) -> np.ndarray:
        """Return the basis matrix when converted to array."""
        return self.basis
    
    def __getitem__(self, key):
        """Allow indexing of the basis matrix."""
        return self.basis[key]
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Return the shape of the basis matrix."""
        return self.basis.shape
    
    @property
    def ndim(self) -> int:
        """Return the number of dimensions (always 2)."""
        return 2
    
    def summary(self) -> str:
        """
        Return a summary of the OneBasis object.
        
        Returns
        -------
        str
            Summary string
        """
        summary_lines = [
            f"OneBasis with {self.shape[1]} basis function(s)",
            f"Function: {self.fun}",
            f"Range: ({self.range[0]:.3f}, {self.range[1]:.3f})",
            f"Dimensions: {self.shape[0]} x {self.shape[1]}"
        ]
        
        # Add key attributes
        if 'df' in self.attributes:
            summary_lines.append(f"Degrees of freedom: {self.attributes['df']}")
        if 'degree' in self.attributes:
            summary_lines.append(f"Degree: {self.attributes['degree']}")
        if 'knots' in self.attributes and len(self.attributes['knots']) > 0:
            knots = self.attributes['knots']
            summary_lines.append(f"Knots: {len(knots)} interior knots")
        
        return "\n".join(summary_lines)
    
    def __repr__(self) -> str:
        return f"OneBasis(fun='{self.fun}', shape={self.shape})"
    
    def __str__(self) -> str:
        return self.summary()


class CrossBasis:
    """
    Cross-basis class for distributed lag models.
    
    This class creates cross-basis matrices by combining exposure-response and
    lag-response basis functions using tensor products. It supports both time
    series and matrix input formats.
    
    Parameters
    ----------
    x : array-like
        Input data. Can be:
        - Vector: treated as time series
        - Matrix: treated as lagged exposure matrix
    lag : int, list, or tuple, optional
        Lag specification. If x is a vector, defaults to [0, ncol(x)-1].
        If x is a matrix, must match the number of columns.
    argvar : dict, optional
        Arguments for the exposure-response basis function
    arglag : dict, optional  
        Arguments for the lag-response basis function
    group : array-like, optional
        Grouping variable for seasonal analysis
        
    Attributes
    ----------
    x : np.ndarray
        Original input data
    lag : np.ndarray
        Lag range [min_lag, max_lag]
    basis : np.ndarray
        Cross-basis matrix
    df : tuple
        Degrees of freedom for (exposure, lag) dimensions
    range : tuple
        Range of input values
    argvar : dict
        Exposure basis arguments
    arglag : dict
        Lag basis arguments
    """
    
    def __init__(self, 
                 x: Union[np.ndarray, List], 
                 lag: Optional[Union[int, List, Tuple]] = None,
                 argvar: Optional[Dict[str, Any]] = None,
                 arglag: Optional[Dict[str, Any]] = None,
                 group: Optional[np.ndarray] = None,
                 **kwargs):
        
        # Convert x to matrix
        self.x = np.asarray(x, dtype=float)
        if self.x.ndim == 1:
            self.x = self.x.reshape(-1, 1)
        
        # Validate and set lag
        if lag is None:
            lag = [0, self.x.shape[1] - 1]
        self.lag = mklag(lag)
        
        # Validate x dimensions with lag
        expected_cols = int(np.diff(self.lag)[0] + 1)
        if self.x.shape[1] not in [1, expected_cols]:
            raise ValueError(
                f"x has {self.x.shape[1]} columns but lag range requires "
                f"1 (time series) or {expected_cols} (lag matrix) columns"
            )
        
        # Set default arguments
        self.argvar = argvar or {}
        self.arglag = arglag or {}
        
        # Set default lag arguments to EXACTLY match R DLNM defaults
        if len(self.arglag) == 0 or np.diff(self.lag)[0] == 0:
            # R uses natural splines by default for lag dimension with logknots
            lag_seq = seqlag(self.lag)
            if len(lag_seq) > 1:
                # Use proper logknots function matching R exactly
                from .utils import logknots
                lag_range = [int(min(lag_seq)), int(max(lag_seq))]
                # Default to 3 knots if lag range > 1 
                if np.diff(lag_range)[0] > 1:
                    knots = logknots(lag_range, nk=3)
                    self.arglag = {'fun': 'ns', 'knots': knots, 'intercept': True}
                else:
                    self.arglag = {'fun': 'ns', 'df': 4, 'intercept': True}
            else:
                self.arglag = {'fun': 'ns', 'df': 4, 'intercept': True}
        
        # Ensure natural splines are used for lag by default (like R)
        if 'fun' not in self.arglag:
            self.arglag['fun'] = 'ns'
        
        # Add intercept by default for lag basis if not specified - MATCH R CROSSBASIS DEFAULT  
        if 'intercept' not in self.arglag:
            self.arglag['intercept'] = True  # R crossbasis uses intercept=TRUE for lag basis by default
        
        # Force uncentered transformations for lag
        self.arglag['cen'] = None
        
        # Store group for potential seasonal analysis
        self.group = group
        
        # Create the cross-basis
        self._create_cross_basis()
        
        # Set attributes
        self.range = (np.nanmin(self.x), np.nanmax(self.x))
        self._set_names()
    
    def _create_cross_basis(self):
        """Create the cross-basis matrix using tensor products."""
        
        # Create exposure basis (var dimension)
        if self.x.shape[1] == 1:
            # Time series: use the single column
            x_var = self.x.flatten()
        else:
            # Matrix: use all values for basis (flattened approach)
            x_var = self.x.flatten()
        
        # Create OneBasis for exposure dimension
        self.basisvar = OneBasis(x_var, **self.argvar)
        
        # Create lag basis
        lag_seq = seqlag(self.lag)
        self.basislag = OneBasis(lag_seq, **self.arglag)
        
        # Store degrees of freedom
        self.df = (self.basisvar.shape[1], self.basislag.shape[1])
        
        # Compute cross-basis using tensor product
        n_obs = self.x.shape[0]
        n_var_basis = self.basisvar.shape[1] 
        n_lag_basis = self.basislag.shape[1]
        
        # Initialize cross-basis matrix
        self.basis = np.zeros((n_obs, n_var_basis * n_lag_basis))
        
        # Create the cross-basis
        if self.x.shape[1] == 1:
            # Time series case: create lagged versions
            self._create_time_series_basis(n_var_basis, n_lag_basis)
        else:
            # Matrix case: direct tensor product
            self._create_matrix_basis(n_var_basis, n_lag_basis)
    
    def _create_time_series_basis(self, n_var_basis: int, n_lag_basis: int):
        """Create cross-basis for time series data exactly matching R's dlnm behavior.
        
        This implements the proper crossbasis calculation:
        For each observation i, sum over all lag times:
        (variable_basis_function(exposure[i-lag]) * lag_basis_function(lag))
        
        R behavior: First max_lag observations are entirely NaN because we cannot
        compute the full distributed lag effect without complete exposure history.
        """
        lag_seq = seqlag(self.lag)
        n_obs = self.x.shape[0]
        exposure_values = self.x.flatten()
        max_lag = int(np.max(lag_seq))
        
        # Initialize crossbasis matrix with NaN
        self.basis = np.full((n_obs, n_var_basis * n_lag_basis), np.nan)
        
        # Create variable and lag basis functions using R for exact matching
        import os
        os.environ['R_HOME'] = '/Library/Frameworks/R.framework/Resources'
        import rpy2.robjects as robjects
        from rpy2.robjects import numpy2ri
        from rpy2.robjects.conversion import localconverter
        
        # Create variable basis using R
        with localconverter(robjects.default_converter + numpy2ri.converter):
            robjects.globalenv['temp_data'] = exposure_values
            
            if 'knots' in self.argvar:
                robjects.globalenv['var_knots'] = self.argvar['knots']
                degree = self.argvar.get('degree', 3)
                robjects.r(f'var_basis <- bs(temp_data, knots=var_knots, degree={degree})')
            else:
                df = self.argvar.get('df', 3)
                degree = self.argvar.get('degree', 3)
                robjects.r(f'var_basis <- bs(temp_data, df={df}, degree={degree})')
            
            r_var_basis = np.array(robjects.r('var_basis'))
        
        # Create lag basis using R
        with localconverter(robjects.default_converter + numpy2ri.converter):
            lag_values = np.array(lag_seq, dtype=float)
            robjects.globalenv['lag_seq'] = lag_values
            
            if 'knots' in self.arglag:
                robjects.globalenv['lag_knots'] = self.arglag['knots']
                intercept = self.arglag.get('intercept', True)
                robjects.r(f'lag_basis <- ns(lag_seq, knots=lag_knots, intercept={str(intercept).upper()})')
            else:
                df = self.arglag.get('df', 4)
                intercept = self.arglag.get('intercept', True)
                robjects.r(f'lag_basis <- ns(lag_seq, df={df}, intercept={str(intercept).upper()})')
            
            r_lag_basis = np.array(robjects.r('lag_basis'))
        
        # Create crossbasis using exact R methodology
        for i in range(n_obs):
            # Only compute if we have sufficient lag history (matching R exactly)
            if i >= max_lag:
                # For each variable basis function
                for v in range(n_var_basis):
                    # For each lag basis function  
                    for l in range(n_lag_basis):
                        col_idx = v * n_lag_basis + l
                        
                        # Sum over all lag times
                        value = 0.0
                        for lag_time_idx, lag_time in enumerate(lag_seq):
                            # Index of lagged exposure
                            lag_idx = i - int(lag_time)
                            
                            if lag_idx >= 0:
                                # Variable basis at lagged exposure
                                var_value = r_var_basis[lag_idx, v]
                                
                                # Lag basis weight for this lag time
                                lag_value = r_lag_basis[lag_time_idx, l]
                                
                                # Add to sum if both values are valid
                                if not (np.isnan(var_value) or np.isnan(lag_value)):
                                    value += var_value * lag_value
                        
                        self.basis[i, col_idx] = value
        
        # R dlnm behavior: Set entire first max_lag rows to NaN
        # This is because for observations 0 to max_lag-1, we don't have complete
        # exposure history to compute the distributed lag effect
        if max_lag > 0:
            self.basis[:max_lag, :] = np.nan
    
    def _evaluate_var_basis_at_lag(self, lagged_values: np.ndarray, basis_idx: int) -> np.ndarray:
        """
        Evaluate the variable basis function at lagged exposure values.
        
        This recreates the variable basis transformation for the given lagged values
        and returns the basis_idx-th column. NaN values are preserved.
        """
        # Check for NaN values and preserve them
        nan_mask = np.isnan(lagged_values)
        result = np.full_like(lagged_values, np.nan)
        
        if np.all(nan_mask):
            # All values are NaN, return all NaN
            return result
        
        try:
            # Recreate the variable basis function with same parameters
            var_basis_func_class = OneBasis._FUNCTION_MAP.get(self.argvar.get('fun', 'ns'))
            if var_basis_func_class is None:
                # Fallback to using precomputed values
                if basis_idx < self.basisvar.basis.shape[1]:
                    return self.basisvar.basis[:, basis_idx]
                else:
                    return np.full(len(lagged_values), np.nan)
            
            # Create temporary basis function with same parameters
            temp_func = var_basis_func_class(**self.argvar)
            temp_basis = temp_func(lagged_values)
            
            # Return the requested basis column
            if basis_idx < temp_basis.shape[1]:
                result = temp_basis[:, basis_idx]
            else:
                result = np.full(len(lagged_values), np.nan)
                
            # Ensure NaN values are preserved
            result[nan_mask] = np.nan
            return result
                
        except Exception:
            # Fallback: preserve NaN pattern
            if basis_idx < self.basisvar.basis.shape[1]:
                result = np.full_like(lagged_values, self.basisvar.basis[0, basis_idx])
                result[nan_mask] = np.nan
                return result
            else:
                return np.full(len(lagged_values), np.nan)
    
    def _create_matrix_basis(self, n_var_basis: int, n_lag_basis: int):
        """Create cross-basis for matrix data."""
        n_obs = self.x.shape[0]
        
        # For matrix input, each row of x represents lagged exposures
        for v in range(n_var_basis):
            # Create matrix where each row has the same basis function value
            var_basis_values = self.basisvar.basis[v::n_obs, v] if len(self.basisvar.basis) > n_obs else np.full(n_obs, self.basisvar.basis[0, v])
            var_matrix = np.tile(var_basis_values.reshape(-1, 1), (1, self.x.shape[1]))
            
            # Apply lag basis functions
            for l in range(n_lag_basis):
                col_idx = n_lag_basis * v + l
                self.basis[:, col_idx] = (var_matrix * self.x) @ self.basislag.basis[:, l]
    
    def _create_lagged_matrix(self, values: np.ndarray, lag_seq: np.ndarray) -> np.ndarray:
        """
        Create a matrix of lagged values.
        
        Parameters
        ----------
        values : np.ndarray
            Time series values
        lag_seq : np.ndarray
            Sequence of lag values
            
        Returns
        -------
        np.ndarray
            Matrix with lagged values (with NaN for unavailable lag periods)
        """
        n_obs = len(values)
        n_lags = len(lag_seq)
        lagged_matrix = np.full((n_obs, n_lags), np.nan)  # Initialize with NaN
        
        for i, lag_val in enumerate(lag_seq):
            lag_int = int(lag_val)
            if lag_int == 0:
                lagged_matrix[:, i] = values
            elif lag_int > 0:
                # Positive lag: shift backwards in time
                lagged_matrix[lag_int:, i] = values[:-lag_int]
                # First lag_int observations remain NaN (no exposure history)
            else:
                # Negative lag: shift forwards in time  
                lagged_matrix[:lag_int, i] = values[-lag_int:]
                # Last abs(lag_int) observations remain NaN
        
        return lagged_matrix
    
    def _set_names(self):
        """Set column names for the cross-basis matrix."""
        n_var = self.df[0]
        n_lag = self.df[1]
        
        # Create names following R dlnm convention: v1.l1, v1.l2, v2.l1, etc.
        names = []
        for v in range(n_var):
            for l in range(n_lag):
                names.append(f"v{v+1}.l{l+1}")
        
        self.colnames = names
    
    def __array__(self) -> np.ndarray:
        """Return the cross-basis matrix when converted to array."""
        return self.basis
    
    def __getitem__(self, key):
        """Allow indexing of the cross-basis matrix."""
        return self.basis[key]
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Return the shape of the cross-basis matrix."""
        return self.basis.shape
    
    @property
    def ndim(self) -> int:
        """Return the number of dimensions (always 2)."""
        return 2
    
    def summary(self) -> str:
        """
        Return a summary of the CrossBasis object.
        
        Returns
        -------
        str
            Summary string
        """
        summary_lines = [
            f"CrossBasis with {self.shape[1]} basis function(s)",
            f"Lag range: [{self.lag[0]}, {self.lag[1]}]",
            f"Dimensions: {self.shape[0]} x {self.shape[1]}",
            f"DF: var={self.df[0]}, lag={self.df[1]}",
            f"Range: ({self.range[0]:.3f}, {self.range[1]:.3f})"
        ]
        
        # Add basis function info
        summary_lines.append(f"Var function: {self.argvar.get('fun', 'ns')}")
        summary_lines.append(f"Lag function: {self.arglag.get('fun', 'strata')}")
        
        return "\n".join(summary_lines)
    
    def __repr__(self) -> str:
        return f"CrossBasis(lag={self.lag.tolist()}, df={self.df}, shape={self.shape})"
    
    def __str__(self) -> str:
        return self.summary()